#!/usr/bin/env python3
"""retarget_kit.py: retarget a DeepSeek-V4.1 TensorFold kernel kit (sm_121a) to another CUDA arch (default sm_110a).

Every cubin that carries embedded PTX (.nv_debug_ptx_txt — all Triton AOT cubins plus experts.cubin and
experts_cb.cubin) is retargeted directly from the pinned kit, preserving aot.json (hashes, launch metadata,
and all PDL flags) untouched.

The three cubins without embedded PTX are handled cleanly without pulling a full kit from another source:
  - cubins/linear.cubin and cubins/linear_grouped.cubin: compiled for the target arch from EXL3 CUDA sources
    (--exl3-src DIR) using standalone nvcc (no PyTorch headers needed), or taken from --prebuilt-kit DIR.
  - vision/torch_fmha_sm120.cubin: extracted from a target-arch libtorch_cuda.so (--libtorch PATH), or taken
    from --prebuilt-kit DIR.

Subcommands:
  convert    SRC DST [--arch sm_110a] [--exl3-src DIR] [--prebuilt-kit DIR] [--libtorch PATH]
                     [--ptxas PATH] [--nvcc PATH] [--max-ptx-version X.Y] [--jobs N] [--verify-gpu]
  build-exl3 EXL3_SRC OUT_DIR [--arch sm_110] [--nvcc PATH]
  verify     KIT [--arch sm_110a] [--gpu]
"""
from __future__ import annotations

import argparse
import concurrent.futures
import ctypes
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

FMHA_KERNEL_NAME = (
    "_Z39fmha_cutlassF_f32_aligned_64x64_rf_sm80"
    "N22PyTorchMemEffAttention15AttentionKernel"
    "IfN7cutlass4arch4Sm80ELb1ELi64ELi64ELi64ELb1ELb1EE6ParamsE"
)

EXL3_BASE_REQUIRED = (
    "18group_count_kernelE",
    "18group_place_kernelE",
    "16work_list_kernelE",
    "13rot_in_kernelI13__nv_bfloat16E",
    "22gateup_epilogue_kernelE",
    "19down_combine_kernelE",
    "18decode_prep_kernelI13__nv_bfloat16E",
)

# CUDA toolkit minor -> highest PTX ISA version supported by ptxas.
CUDA_TO_MAX_PTX = {
    (12, 6): (8, 5),
    (12, 8): (8, 7),
    (12, 9): (8, 8),
    (13, 0): (9, 0),
    (13, 1): (9, 1),
    (13, 2): (9, 2),
    (13, 3): (9, 3),
}

ATEN_STUB_H = """#pragma once
#include <cstdint>
#include <optional>
#include <stdexcept>
namespace c10 {
template <typename T> using optional = std::optional<T>;
}
namespace at {
enum ScalarType : int { kHalf = 0, kBFloat16 = 1, kFloat = 2 };
struct Tensor {
    ScalarType scalar_type() const { return kHalf; }
    int64_t size(int) const { return 0; }
    int64_t stride(int) const { return 0; }
    void* data_ptr() const { return nullptr; }
    template <typename T> T* data_ptr() const { return nullptr; }
};
}
#define TORCH_CHECK(cond, ...) do { if (!(cond)) throw std::runtime_error("TORCH_CHECK"); } while (0)
"""

CUDA_CONTEXT_STUB_H = """#pragma once
#include <cuda_runtime.h>
namespace at { namespace cuda {
inline cudaStream_t getCurrentCUDAStream() { return nullptr; }
}}
#define C10_CUDA_CHECK(expr) do { (void)(expr); } while (0)
#define C10_CUDA_KERNEL_LAUNCH_CHECK() do {} while (0)
"""

CUDA_GUARD_STUB_H = "#pragma once\n"


def fail(msg: str, code: int = 1) -> None:
    print(f"retarget_kit: ERROR: {msg}", file=sys.stderr)
    raise SystemExit(code)


def parse_arch(arch: str) -> tuple[str, int, str]:
    """Return (normalized_arch, sm_num, base_arch), e.g. ('sm_110a', 110, 'sm_110')."""
    s = arch.strip()
    if s.startswith("sm_"):
        s = s[3:]
    m = re.fullmatch(r"([1-9][0-9]*)(a?)", s)
    if not m:
        raise ValueError(f"invalid CUDA arch {arch!r} (expected e.g. sm_110a, 110a, sm_110)")
    num = int(m.group(1))
    suffix = m.group(2)
    return f"sm_{num}{suffix}", num, f"sm_{num}"


def find_tool(explicit: str | None, name: str) -> str:
    if explicit:
        if not os.path.isfile(explicit) or not os.access(explicit, os.X_OK):
            raise FileNotFoundError(f"{name} not executable: {explicit}")
        return explicit
    for cand in (shutil.which(name), f"/usr/local/cuda/bin/{name}"):
        if cand and os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    raise FileNotFoundError(f"cannot find {name} on PATH or /usr/local/cuda/bin")


def detect_tool_cuda_version(tool: str) -> tuple[int, int] | None:
    res = subprocess.run([tool, "--version"], capture_output=True, text=True, check=False)
    m = re.search(r"release\s+(\d+)\.(\d+)", res.stdout + "\n" + res.stderr)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def detect_max_ptx_version(ptxas: str) -> tuple[int, int] | None:
    ver = detect_tool_cuda_version(ptxas)
    if ver is None:
        return None
    if ver in CUDA_TO_MAX_PTX:
        return CUDA_TO_MAX_PTX[ver]
    if ver[0] == 13:
        return (9, ver[1])
    return None


def parse_version_pair(text: str) -> tuple[int, int]:
    m = re.fullmatch(r"(\d+)\.(\d+)", text.strip())
    if not m:
        raise ValueError(f"invalid version {text!r} (expected X.Y)")
    return int(m.group(1)), int(m.group(2))


# -- Pure-Python ELF64 cubin inspection --------------------------------------------------------------------------------


def elf_parse_header(data: bytes) -> dict:
    if len(data) < 64 or data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:
        raise ValueError("not a 64-bit little-endian ELF cubin")
    e_flags = struct.unpack_from("<I", data, 48)[0]
    e_shoff = struct.unpack_from("<Q", data, 40)[0]
    e_shentsize, e_shnum, e_shstrndx = struct.unpack_from("<HHH", data, 58)
    if e_shoff == 0 or e_shnum == 0 or e_shentsize < 64:
        raise ValueError("ELF has no section header table")
    sections = []
    for i in range(e_shnum):
        off = e_shoff + i * e_shentsize
        sh_name, sh_type, sh_flags, sh_addr, sh_offset, sh_size, sh_link, sh_info, sh_addralign, sh_entsize = (
            struct.unpack_from("<IIQQQQIIQQ", data, off)
        )
        sections.append(
            {
                "name_off": sh_name,
                "type": sh_type,
                "offset": sh_offset,
                "size": sh_size,
                "link": sh_link,
                "entsize": sh_entsize,
            }
        )
    shstr = sections[e_shstrndx]
    strtab = data[shstr["offset"] : shstr["offset"] + shstr["size"]]
    for sec in sections:
        end = strtab.find(b"\x00", sec["name_off"])
        sec["name"] = strtab[sec["name_off"] : (len(strtab) if end < 0 else end)].decode("latin1")
    return {"e_flags": e_flags, "sections": sections}


def elf_cuda_sm(data: bytes) -> int:
    hdr = elf_parse_header(data)
    flags = hdr["e_flags"]
    low = flags & 0xFF
    mid = (flags >> 8) & 0xFF
    # CUDA 13+ ABI v2 encodes EF_CUDA_SM in bits 8..15 (low byte holds ABI version e.g. 2);
    # older CUDA toolkits put SM directly in bits 0..7.
    if low < 30 and mid >= 30:
        return mid
    return low


def extract_ptx(data: bytes) -> str | None:
    hdr = elf_parse_header(data)
    for sec in hdr["sections"]:
        if sec["name"] == ".nv_debug_ptx_txt":
            raw = data[sec["offset"] : sec["offset"] + sec["size"]]
            if not raw or not raw.strip(b"\x00\r\n\t "):
                return None
            return raw.replace(b"\x00", b"\n").decode("latin1")
    return None


STO_CUDA_ENTRY = 0x10


def elf_func_symbols(data: bytes, *, entries_only: bool = True) -> list[str]:
    hdr = elf_parse_header(data)
    out: list[str] = []
    for sec in hdr["sections"]:
        if sec["type"] != 2:  # SHT_SYMTAB
            continue
        str_sec = hdr["sections"][sec["link"]]
        strtab = data[str_sec["offset"] : str_sec["offset"] + str_sec["size"]]
        entsize = sec["entsize"] or 24
        count = sec["size"] // entsize
        for i in range(count):
            off = sec["offset"] + i * entsize
            st_name, st_info, st_other, st_shndx, _st_val, _st_size = struct.unpack_from("<IBBHQQ", data, off)
            if (st_info & 0xF) != 2 or st_shndx == 0:  # STT_FUNC defined
                continue
            if entries_only and (st_other & STO_CUDA_ENTRY) == 0:
                continue
            end = strtab.find(b"\x00", st_name)
            name = strtab[st_name : (len(strtab) if end < 0 else end)].decode("latin1")
            if name:
                out.append(name)
    return sorted(set(out))


def normalize_anon_symbol(sym: str) -> str:
    """Strip compiler-generated anonymous-namespace hash so symbols compare across builds."""
    return re.sub(r"_ZN\d+_GLOBAL__N__[0-9a-f]+_\d+_([A-Za-z0-9_]+)_[0-9a-f]+", r"_ZN_ANON_\1_", sym)


def comparable_kernel_symbols(data: bytes) -> list[str]:
    """Normalized FUNC symbols excluding ATen/CUB EmptyKernel stubs that encode SM in their namespace."""
    return sorted(
        normalize_anon_symbol(s)
        for s in elf_func_symbols(data)
        if not (s.startswith("_ZN3cub") and "EmptyKernel" in s)
    )


# -- PTX retargeting ---------------------------------------------------------------------------------------------------


def retarget_ptx(ptx: str, target_arch: str, max_ptx_version: tuple[int, int] | None = None) -> tuple[str, dict]:
    target_re = re.compile(r"^([ \t]*\.target[ \t]+)sm_\d+a?\b", re.MULTILINE)
    new_ptx, n_target = target_re.subn(rf"\g<1>{target_arch}", ptx)
    if n_target == 0:
        raise ValueError("PTX has no .target sm_<arch> directive")

    clamped_from = None
    clamped_to = None
    if max_ptx_version is not None:
        ver_re = re.compile(r"^([ \t]*\.version[ \t]+)(\d+)\.(\d+)\b", re.MULTILINE)

        def repl_ver(m: re.Match[str]) -> str:
            nonlocal clamped_from, clamped_to
            cur = (int(m.group(2)), int(m.group(3)))
            if cur > max_ptx_version:
                clamped_from = f"{cur[0]}.{cur[1]}"
                clamped_to = f"{max_ptx_version[0]}.{max_ptx_version[1]}"
                return f"{m.group(1)}{clamped_to}"
            return m.group(0)

        new_ptx = ver_re.sub(repl_ver, new_ptx)

    return new_ptx, {
        "targets_rewritten": n_target,
        "clamped_from": clamped_from,
        "clamped_to": clamped_to,
    }


def assemble_ptx(ptx_text: str, out_cubin: Path, target_arch: str, ptxas: str) -> None:
    out_cubin.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="retarget-ptx-") as td:
        in_ptx = Path(td) / "kernel.ptx"
        tmp_cubin = Path(td) / "kernel.cubin"
        in_ptx.write_text(ptx_text, encoding="latin1")
        cmd = [ptxas, f"-arch={target_arch}", "-o", str(tmp_cubin), str(in_ptx)]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode != 0:
            raise RuntimeError(f"ptxas failed ({res.returncode}): {(res.stderr or res.stdout).strip()}")
        shutil.copyfile(tmp_cubin, out_cubin)


# -- EXL3 linear / linear_grouped standalone compilation ---------------------------------------------------------------


def build_exl3_cubins(src_dir: Path, out_dir: Path, arch: str, nvcc: str) -> dict[str, Path]:
    _, sm_num, base_arch = parse_arch(arch)
    # linear.cu / linear_grouped.cu use standard sm_<N> (not arch-specific 'a')
    compile_arch = base_arch
    for req in ("linear.cu", "linear_grouped.cu", "linear_common.cuh", "decode.cuh"):
        if not (src_dir / req).is_file():
            raise FileNotFoundError(f"EXL3 source file missing: {src_dir / req}")
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="exl3-stubs-") as td:
        stub = Path(td)
        (stub / "ATen" / "cuda").mkdir(parents=True)
        (stub / "c10" / "cuda").mkdir(parents=True)
        (stub / "ATen" / "ATen.h").write_text(ATEN_STUB_H, encoding="utf-8")
        (stub / "ATen" / "cuda" / "CUDAContext.h").write_text(CUDA_CONTEXT_STUB_H, encoding="utf-8")
        (stub / "c10" / "cuda" / "CUDAGuard.h").write_text(CUDA_GUARD_STUB_H, encoding="utf-8")

        base_flags = [
            nvcc,
            "-cubin",
            "-O3",
            "-std=c++20",
            "--expt-relaxed-constexpr",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
            f"-arch={compile_arch}",
            f"-I{src_dir}",
            f"-I{stub}",
        ]

        built: dict[str, Path] = {}

        def compile_one(cu_name: str, cubin_name: str) -> tuple[str, Path]:
            out_path = out_dir / cubin_name
            tmp_path = out_dir / f".{cubin_name}.tmp"
            cmd = [*base_flags, "-o", str(tmp_path), str(src_dir / cu_name)]
            res = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if res.returncode != 0:
                tmp_path.unlink(missing_ok=True)
                raise RuntimeError(f"nvcc {cu_name} failed ({res.returncode}):\n{res.stderr or res.stdout}")
            data = tmp_path.read_bytes()
            got_sm = elf_cuda_sm(data)
            if got_sm != sm_num:
                tmp_path.unlink(missing_ok=True)
                raise RuntimeError(f"{cubin_name}: compiled arch sm_{got_sm}, expected sm_{sm_num}")
            os.replace(tmp_path, out_path)
            return cubin_name, out_path

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            futs = [
                ex.submit(compile_one, "linear.cu", "linear.cubin"),
                ex.submit(compile_one, "linear_grouped.cu", "linear_grouped.cubin"),
            ]
            for fut in futs:
                name, path = fut.result()
                built[name] = path

    check_exl3_linear_symbols(built["linear.cubin"], built["linear_grouped.cubin"])
    return built


def check_exl3_linear_symbols(linear_cubin: Path, grouped_cubin: Path) -> None:
    lin_syms = elf_func_symbols(linear_cubin.read_bytes())
    if not any("rot_in_kernel" in s for s in lin_syms):
        raise RuntimeError(f"{linear_cubin}: missing rot_in_kernel")
    unpack_mul1 = {
        int(m.group(1))
        for s in lin_syms
        if (m := re.search(r"unpack_kernelILi(\d+)ELi2EE", s))
    }
    if not unpack_mul1 or 4 not in unpack_mul1 or 6 not in unpack_mul1:
        raise RuntimeError(f"{linear_cubin}: missing mul1 unpack_kernel specializations (found K2={sorted(unpack_mul1)})")

    grp_syms = elf_func_symbols(grouped_cubin.read_bytes())
    if not any("rot_many_kernel" in s for s in grp_syms):
        raise RuntimeError(f"{grouped_cubin}: missing rot_many_kernel")
    glin_mul1 = {
        (int(m.group(1)), int(m.group(2)))
        for s in grp_syms
        if (m := re.search(r"glinear_kernelILi(\d+)ELi2ELi(\d+)EE", s))
    }
    for req in ((4, 4), (4, 8), (6, 4), (6, 8)):
        if req not in glin_mul1:
            raise RuntimeError(f"{grouped_cubin}: missing glinear_kernel<K2={req[0]}, CB=2, WK={req[1]}>")


def check_exl3_experts_symbols(experts_cubin: Path, experts_cb_cubin: Path) -> None:
    base_syms = elf_func_symbols(experts_cubin.read_bytes())
    for pat in EXL3_BASE_REQUIRED:
        if not any(pat in s for s in base_syms):
            raise RuntimeError(f"{experts_cubin}: missing required symbol matching {pat}")
    cb_syms = elf_func_symbols(experts_cb_cubin.read_bytes())
    for pat in ("18grouped_mma_kernelI", "19grouped_rows_kernelI", "17grouped_cp_kernelI"):
        if not any(pat in s for s in cb_syms):
            raise RuntimeError(f"{experts_cb_cubin}: missing required symbol matching {pat}")


def extract_fmha_cubin(libtorch_path: Path, out_path: Path, sm_num: int, cuobjdump: str) -> None:
    if not libtorch_path.is_file():
        raise FileNotFoundError(f"libtorch_cuda.so not found: {libtorch_path}")
    with tempfile.TemporaryDirectory(prefix="extract-fmha-") as td:
        res = subprocess.run(
            [cuobjdump, "-xelf", "all", str(libtorch_path)],
            cwd=td,
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode != 0:
            raise RuntimeError(f"cuobjdump failed on {libtorch_path}: {res.stderr or res.stdout}")
        needle = FMHA_KERNEL_NAME.encode("ascii")
        for cand in sorted(Path(td).glob(f"*.sm_{sm_num}.cubin")):
            raw = cand.read_bytes()
            if needle in raw and FMHA_KERNEL_NAME in elf_func_symbols(raw):
                out_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(cand, out_path)
                return
    raise RuntimeError(f"{libtorch_path} has no sm_{sm_num} cubin exporting {FMHA_KERNEL_NAME}")


# -- MANIFEST handling -------------------------------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_manifest(path: Path) -> list[tuple[str, str]]:
    """Parse both 2-column ('<sha>  <path>') and 5-column ('<sha> <bytes> <orig> <tp2> <path>') manifests."""
    entries: list[tuple[str, str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2 or not re.fullmatch(r"[0-9a-f]{64}", parts[0]):
            raise ValueError(f"malformed MANIFEST line in {path}: {raw!r}")
        entries.append((parts[0], parts[-1]))
    return entries


def write_manifest(kit_dir: Path, src_manifest: Path | None) -> int:
    """Write standard 2-column sha256sum MANIFEST (what scripts/prepare.sh reads) and keep MANIFEST.base."""
    entries: dict[str, str] = {}
    if src_manifest is not None and src_manifest.is_file():
        (kit_dir / "MANIFEST.base").write_bytes(src_manifest.read_bytes())
        for old_sha, rel in parse_manifest(src_manifest):
            p = kit_dir / rel
            entries[rel] = sha256_file(p) if p.is_file() else old_sha
    else:
        for p in sorted(kit_dir.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(kit_dir).as_posix()
            if rel in ("MANIFEST", "MANIFEST.base", "RETARGET.json") or ".git" in p.parts:
                continue
            entries[rel] = sha256_file(p)

    lines = [f"{entries[rel]}  {rel}\n" for rel in sorted(entries)]
    tmp = kit_dir / ".MANIFEST.tmp"
    tmp.write_text("".join(lines), encoding="utf-8")
    os.replace(tmp, kit_dir / "MANIFEST")
    return len(lines)


# -- Kit resolution, conversion, and verification ----------------------------------------------------------------------


def resolve_kit_root(path: Path) -> Path:
    if (path / "aot" / "aot.json").is_file():
        return path
    if (path / "kit" / "aot" / "aot.json").is_file():
        return path / "kit"
    raise FileNotFoundError(f"no aot/aot.json under {path} or {path / 'kit'}")


def convert_kit(
    src_path: Path,
    dst_path: Path,
    *,
    arch: str = "sm_110a",
    ptxas_path: str | None = None,
    nvcc_path: str | None = None,
    cuobjdump_path: str | None = None,
    max_ptx_version: str | None = None,
    exl3_src: Path | None = None,
    prebuilt_kit: Path | None = None,
    libtorch: Path | None = None,
    jobs: int | None = None,
    verify_gpu: bool = False,
) -> dict:
    src = resolve_kit_root(src_path).resolve()
    dst = dst_path.resolve()
    if dst == src or src in dst.parents or dst in src.parents:
        raise ValueError(f"DST ({dst}) must not overlap SRC ({src})")

    target_arch, sm_num, base_arch = parse_arch(arch)
    ptxas = find_tool(ptxas_path, "ptxas")
    max_ptx = parse_version_pair(max_ptx_version) if max_ptx_version else detect_max_ptx_version(ptxas)
    workers = jobs or max(1, min(os.cpu_count() or 1, 16))

    prebuilt_root = resolve_kit_root(prebuilt_kit).resolve() if prebuilt_kit else None
    if prebuilt_kit and prebuilt_root is None:
        raise FileNotFoundError(f"invalid --prebuilt-kit: {prebuilt_kit}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{dst.name}.stage.", dir=dst.parent))
    try:
        ptx_jobs: list[tuple[Path, str, str]] = []
        no_ptx_rels: list[str] = []
        copied_files = 0

        for p in sorted(src.rglob("*")):
            if ".git" in p.parts:
                continue
            rel = p.relative_to(src).as_posix()
            if rel in ("MANIFEST", "MANIFEST.base", "RETARGET.json"):
                continue
            out = stage / rel
            if p.is_dir():
                out.mkdir(parents=True, exist_ok=True)
                continue
            if p.suffix != ".cubin":
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, out)
                copied_files += 1
                continue

            raw = p.read_bytes()
            ptx = extract_ptx(raw)
            if ptx is not None:
                ptx_jobs.append((p, rel, ptx))
            else:
                no_ptx_rels.append(rel)

        # 1. Retarget all PTX-bearing cubins in parallel.
        retargeted = 0
        clamped = 0

        def do_one(item: tuple[Path, str, str]) -> tuple[str, bool]:
            src_cubin, rel, ptx = item
            new_ptx, info = retarget_ptx(ptx, target_arch, max_ptx)
            out_cubin = stage / rel
            assemble_ptx(new_ptx, out_cubin, target_arch, ptxas)
            out_bytes = out_cubin.read_bytes()
            got_sm = elf_cuda_sm(out_bytes)
            if got_sm != sm_num:
                raise RuntimeError(f"{rel}: assembled arch sm_{got_sm}, expected sm_{sm_num}")
            src_syms = comparable_kernel_symbols(src_cubin.read_bytes())
            dst_syms = comparable_kernel_symbols(out_bytes)
            if src_syms != dst_syms:
                raise RuntimeError(f"{rel}: symbol set changed after PTX retarget ({len(src_syms)} -> {len(dst_syms)})")
            return rel, info["clamped_from"] is not None

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            for _rel, did_clamp in ex.map(do_one, ptx_jobs):
                retargeted += 1
                if did_clamp:
                    clamped += 1

        # 2. Handle cubins without embedded PTX.
        built_exl3: list[str] = []
        extracted_fmha: list[str] = []
        borrowed: list[str] = []

        need_exl3 = [r for r in no_ptx_rels if r in ("cubins/linear.cubin", "cubins/linear_grouped.cubin")]
        if need_exl3 and exl3_src is not None:
            nvcc = find_tool(nvcc_path, "nvcc")
            build_exl3_cubins(exl3_src, stage / "cubins", base_arch, nvcc)
            for rel in need_exl3:
                src_syms = comparable_kernel_symbols((src / rel).read_bytes())
                dst_syms = comparable_kernel_symbols((stage / rel).read_bytes())
                if src_syms != dst_syms:
                    raise RuntimeError(
                        f"{rel}: built symbol set ({len(dst_syms)}) differs from source kit ({len(src_syms)})"
                    )
                built_exl3.append(rel)

        for rel in no_ptx_rels:
            if (stage / rel).is_file():
                continue
            if rel == "vision/torch_fmha_sm120.cubin" and libtorch is not None:
                cuobjdump = find_tool(cuobjdump_path, "cuobjdump")
                extract_fmha_cubin(libtorch, stage / rel, sm_num, cuobjdump)
                extracted_fmha.append(rel)
                continue
            if prebuilt_root is not None and (prebuilt_root / rel).is_file():
                cand = prebuilt_root / rel
                raw = cand.read_bytes()
                got_sm = elf_cuda_sm(raw)
                if got_sm != sm_num:
                    raise RuntimeError(f"{cand}: prebuilt arch is sm_{got_sm}, expected sm_{sm_num}")
                src_syms = comparable_kernel_symbols((src / rel).read_bytes())
                dst_syms = comparable_kernel_symbols(raw)
                if src_syms != dst_syms:
                    raise RuntimeError(
                        f"{rel}: prebuilt symbol set ({len(dst_syms)}) differs from source kit ({len(src_syms)})"
                    )
                (stage / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(cand, stage / rel)
                borrowed.append(rel)
                continue
            raise RuntimeError(
                f"{rel}: cubin has no embedded PTX; pass --exl3-src DIR (for linear*.cubin), "
                f"--libtorch PATH (for vision/torch_fmha_sm120.cubin), or --prebuilt-kit DIR"
            )

        # 3. Regenerate MANIFEST and record RETARGET.json.
        src_manifest = src / "MANIFEST" if (src / "MANIFEST").is_file() else None
        manifest_count = write_manifest(stage, src_manifest)

        aot_data = json.loads((stage / "aot" / "aot.json").read_text(encoding="utf-8"))
        summary = {
            "target_arch": target_arch,
            "sm": sm_num,
            "ptxas": ptxas,
            "ptxas_cuda": list(detect_tool_cuda_version(ptxas) or ()),
            "max_ptx_version": f"{max_ptx[0]}.{max_ptx[1]}" if max_ptx else None,
            "aot_variants": len(aot_data["kernels"]),
            "aot_pdl_variants": sum(1 for k in aot_data["kernels"] if k.get("pdl")),
            "cubins_retargeted": retargeted,
            "cubins_ptx_version_clamped": clamped,
            "cubins_built_exl3": sorted(built_exl3),
            "cubins_extracted_fmha": sorted(extracted_fmha),
            "cubins_borrowed": sorted(borrowed),
            "non_cubin_files": copied_files,
            "manifest_entries": manifest_count,
        }
        (stage / "RETARGET.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        # 4. Verify staged kit before swapping into DST.
        verify_report = verify_kit(stage, arch=target_arch, gpu=verify_gpu)
        verify_report["kit"] = str(dst)
        summary["verify"] = verify_report
        (stage / "RETARGET.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        if dst.exists():
            backup = dst.with_name(f".{dst.name}.old.{os.getpid()}")
            os.replace(dst, backup)
            try:
                os.replace(stage, dst)
            except BaseException:
                os.replace(backup, dst)
                raise
            shutil.rmtree(backup, ignore_errors=True)
        else:
            os.replace(stage, dst)
        return summary
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def verify_cubins_on_gpu(kit: Path, aot_kernels: list[dict]) -> dict:
    """Load every cubin in KIT onto GPU 0 via libcuda.so.1 and verify its entry symbols + opt-in SMEM."""
    lib = ctypes.CDLL("libcuda.so.1")
    cuInit = lib.cuInit
    cuInit.argtypes = [ctypes.c_uint]
    cuInit.restype = ctypes.c_int

    cuDeviceGet = lib.cuDeviceGet
    cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    cuDeviceGet.restype = ctypes.c_int

    cuDeviceGetAttribute = lib.cuDeviceGetAttribute
    cuDeviceGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
    cuDeviceGetAttribute.restype = ctypes.c_int

    cuDevicePrimaryCtxRetain = lib.cuDevicePrimaryCtxRetain
    cuDevicePrimaryCtxRetain.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
    cuDevicePrimaryCtxRetain.restype = ctypes.c_int

    cuDevicePrimaryCtxRelease = lib.cuDevicePrimaryCtxRelease_v2
    cuDevicePrimaryCtxRelease.argtypes = [ctypes.c_int]
    cuDevicePrimaryCtxRelease.restype = ctypes.c_int

    cuCtxSetCurrent = lib.cuCtxSetCurrent
    cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
    cuCtxSetCurrent.restype = ctypes.c_int

    cuModuleLoadData = lib.cuModuleLoadData
    cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
    cuModuleLoadData.restype = ctypes.c_int

    cuModuleUnload = lib.cuModuleUnload
    cuModuleUnload.argtypes = [ctypes.c_void_p]
    cuModuleUnload.restype = ctypes.c_int

    cuModuleGetFunction = lib.cuModuleGetFunction
    cuModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
    cuModuleGetFunction.restype = ctypes.c_int

    cuFuncGetAttribute = lib.cuFuncGetAttribute
    cuFuncGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_void_p]
    cuFuncGetAttribute.restype = ctypes.c_int

    cuGetErrorName = lib.cuGetErrorName
    cuGetErrorName.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
    cuGetErrorName.restype = ctypes.c_int

    def check(rc: int, what: str) -> None:
        if rc != 0:
            name = ctypes.c_char_p()
            cuGetErrorName(rc, ctypes.byref(name))
            err = name.value.decode("ascii", "replace") if name.value else str(rc)
            raise RuntimeError(f"CUDA error {err} ({rc}) in {what}")

    check(cuInit(0), "cuInit")
    dev = ctypes.c_int(0)
    check(cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")
    cc_major, cc_minor, max_optin = ctypes.c_int(0), ctypes.c_int(0), ctypes.c_int(0)
    check(cuDeviceGetAttribute(ctypes.byref(cc_major), 75, dev), "CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR")
    check(cuDeviceGetAttribute(ctypes.byref(cc_minor), 76, dev), "CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR")
    check(cuDeviceGetAttribute(ctypes.byref(max_optin), 97, dev), "CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN")

    ctx = ctypes.c_void_p()
    check(cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev), "cuDevicePrimaryCtxRetain")
    try:
        check(cuCtxSetCurrent(ctx), "cuCtxSetCurrent")
        aot_by_rel = {f"aot/cubins/{k['hash']}.cubin": k for k in aot_kernels}
        cubins = sorted(kit.rglob("*.cubin"))
        loaded_modules = 0
        loaded_funcs = 0
        for cubin_path in cubins:
            rel = cubin_path.relative_to(kit).as_posix()
            raw = cubin_path.read_bytes()
            buf = ctypes.create_string_buffer(raw, len(raw))
            mod = ctypes.c_void_p()
            check(cuModuleLoadData(ctypes.byref(mod), ctypes.cast(buf, ctypes.c_void_p)), f"cuModuleLoadData({rel})")
            try:
                loaded_modules += 1
                if rel in aot_by_rel:
                    k = aot_by_rel[rel]
                    fn = ctypes.c_void_p()
                    check(cuModuleGetFunction(ctypes.byref(fn), mod, k["name"].encode("ascii")), f"cuModuleGetFunction({rel}:{k['name']})")
                    loaded_funcs += 1
                    static_smem = ctypes.c_int(0)
                    check(cuFuncGetAttribute(ctypes.byref(static_smem), 1, fn), f"cuFuncGetAttribute(SHARED_SIZE_BYTES, {rel})")
                    if k.get("shared", 0) + static_smem.value > max_optin.value:
                        raise RuntimeError(
                            f"{rel}: needs {k['shared']} + {static_smem.value} B shared memory, device opt-in max is {max_optin.value} B"
                        )
                elif rel == "vision/torch_fmha_sm120.cubin":
                    fn = ctypes.c_void_p()
                    check(cuModuleGetFunction(ctypes.byref(fn), mod, FMHA_KERNEL_NAME.encode("ascii")), f"cuModuleGetFunction({rel})")
                    loaded_funcs += 1
                else:
                    # Verify every exported entry function in EXL3 cubins loads cleanly.
                    for sym in elf_func_symbols(raw):
                        fn = ctypes.c_void_p()
                        check(cuModuleGetFunction(ctypes.byref(fn), mod, sym.encode("ascii")), f"cuModuleGetFunction({rel}:{sym})")
                        loaded_funcs += 1
            finally:
                cuModuleUnload(mod)
        return {
            "device_cc": f"{cc_major.value}.{cc_minor.value}",
            "max_shared_optin_bytes": max_optin.value,
            "modules_loaded": loaded_modules,
            "functions_loaded": loaded_funcs,
        }
    finally:
        cuDevicePrimaryCtxRelease(dev)


def verify_kit(kit_path: Path, *, arch: str = "sm_110a", gpu: bool = False) -> dict:
    kit = resolve_kit_root(kit_path).resolve()
    _, sm_num, _ = parse_arch(arch)

    aot_path = kit / "aot" / "aot.json"
    aot = json.loads(aot_path.read_text(encoding="utf-8"))
    kernels = aot["kernels"]
    for k in kernels:
        cubin = kit / "aot" / "cubins" / f"{k['hash']}.cubin"
        if not cubin.is_file():
            raise RuntimeError(f"missing AOT cubin for {k['fn']} ({k['hash']})")
        raw = cubin.read_bytes()
        if elf_cuda_sm(raw) != sm_num:
            raise RuntimeError(f"{cubin.name}: arch sm_{elf_cuda_sm(raw)}, expected sm_{sm_num}")
        if k["name"] not in elf_func_symbols(raw):
            raise RuntimeError(f"{cubin.name}: missing kernel symbol {k['name']}")

    cubins = sorted(kit.rglob("*.cubin"))
    for c in cubins:
        got = elf_cuda_sm(c.read_bytes())
        if got != sm_num:
            raise RuntimeError(f"{c.relative_to(kit)}: arch sm_{got}, expected sm_{sm_num}")

    check_exl3_linear_symbols(kit / "cubins" / "linear.cubin", kit / "cubins" / "linear_grouped.cubin")
    check_exl3_experts_symbols(kit / "cubins" / "experts.cubin", kit / "cubins" / "experts_cb.cubin")
    fmha = kit / "vision" / "torch_fmha_sm120.cubin"
    if FMHA_KERNEL_NAME not in elf_func_symbols(fmha.read_bytes()):
        raise RuntimeError(f"{fmha}: missing {FMHA_KERNEL_NAME}")

    manifest_checked = 0
    manifest_unmaterialized: list[str] = []
    mf = kit / "MANIFEST"
    if mf.is_file():
        for expected_sha, rel in parse_manifest(mf):
            p = kit / rel
            if not p.is_file():
                manifest_unmaterialized.append(rel)
                continue
            actual = sha256_file(p)
            if actual != expected_sha:
                raise RuntimeError(f"MANIFEST mismatch for {rel}: expected {expected_sha}, got {actual}")
            manifest_checked += 1

    report: dict = {
        "kit": str(kit),
        "sm": sm_num,
        "total_cubins": len(cubins),
        "aot_variants": len(kernels),
        "aot_pdl_variants": sum(1 for k in kernels if k.get("pdl")),
        "manifest_checked": manifest_checked,
        "manifest_unmaterialized": manifest_unmaterialized,
    }
    if gpu:
        report["gpu"] = verify_cubins_on_gpu(kit, kernels)
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_conv = sub.add_parser("convert", help="Convert an sm_121a kit into a target-arch kit")
    p_conv.add_argument("src", type=Path, help="Source kit directory (or repo root containing kit/)")
    p_conv.add_argument("dst", type=Path, help="Destination kit directory")
    p_conv.add_argument("--arch", default="sm_110a", help="Target CUDA arch (default: sm_110a)")
    p_conv.add_argument("--ptxas", default=None, help="Path to ptxas")
    p_conv.add_argument("--nvcc", default=None, help="Path to nvcc")
    p_conv.add_argument("--cuobjdump", default=None, help="Path to cuobjdump")
    p_conv.add_argument("--max-ptx-version", default=None, help="Clamp PTX .version above X.Y (default: auto from ptxas)")
    p_conv.add_argument("--exl3-src", type=Path, default=None, help="Directory holding EXL3 linear*.cu sources")
    p_conv.add_argument("--prebuilt-kit", type=Path, default=None, help="Target-arch kit supplying cubins without PTX")
    p_conv.add_argument("--libtorch", type=Path, default=None, help="Target-arch libtorch_cuda.so for vision fmha cubin")
    p_conv.add_argument("--jobs", type=int, default=None, help="Parallel ptxas workers")
    p_conv.add_argument("--verify-gpu", action="store_true", help="Load every converted cubin on GPU 0 via libcuda.so.1")

    p_exl3 = sub.add_parser("build-exl3", help="Compile linear.cubin and linear_grouped.cubin from EXL3 sources")
    p_exl3.add_argument("exl3_src", type=Path, help="Directory holding linear.cu, linear_grouped.cu, linear_common.cuh, decode.cuh")
    p_exl3.add_argument("out_dir", type=Path, help="Output directory")
    p_exl3.add_argument("--arch", default="sm_110", help="Target CUDA arch (default: sm_110)")
    p_exl3.add_argument("--nvcc", default=None, help="Path to nvcc")

    p_ver = sub.add_parser("verify", help="Verify a converted kit (MANIFEST, arch, symbols, optional GPU load)")
    p_ver.add_argument("kit", type=Path, help="Kit directory to verify")
    p_ver.add_argument("--arch", default="sm_110a", help="Expected CUDA arch (default: sm_110a)")
    p_ver.add_argument("--gpu", action="store_true", help="Load every cubin on GPU 0 via libcuda.so.1")

    args = ap.parse_args(argv)
    try:
        if args.cmd == "convert":
            out = convert_kit(
                args.src,
                args.dst,
                arch=args.arch,
                ptxas_path=args.ptxas,
                nvcc_path=args.nvcc,
                cuobjdump_path=args.cuobjdump,
                max_ptx_version=args.max_ptx_version,
                exl3_src=args.exl3_src,
                prebuilt_kit=args.prebuilt_kit,
                libtorch=args.libtorch,
                jobs=args.jobs,
                verify_gpu=args.verify_gpu,
            )
        elif args.cmd == "build-exl3":
            nvcc = find_tool(args.nvcc, "nvcc")
            built = build_exl3_cubins(args.exl3_src, args.out_dir, args.arch, nvcc)
            out = {k: str(v) for k, v in built.items()}
        else:
            out = verify_kit(args.kit, arch=args.arch, gpu=args.gpu)
    except Exception as exc:  # noqa: BLE001
        fail(str(exc))
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
