#!/usr/bin/env python3
"""Offline unit tests for scripts/retarget_kit.py (no GPU, CUDA toolkit, or network required).

Run: python3 -m unittest scripts/test_retarget_kit.py
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import retarget_kit as rk  # noqa: E402


def make_elf_cubin(
    sm: int,
    funcs: list[str],
    ptx: bytes | None = None,
    *,
    legacy_flags: bool = False,
    device_helpers: list[str] | None = None,
) -> bytes:
    """Build a minimal valid 64-bit little-endian NVIDIA-style ELF cubin for offline tests."""
    e_flags = sm if legacy_flags else (0x09000002 | ((sm & 0xFF) << 8))

    shstrtab = bytearray(b"\x00")
    strtab = bytearray(b"\x00")
    symtab = bytearray(b"\x00" * 24)  # STN_UNDEF

    def add_str(buf: bytearray, s: str) -> int:
        off = len(buf)
        buf.extend(s.encode("latin1") + b"\x00")
        return off

    for fn in funcs:
        name_off = add_str(strtab, fn)
        # st_name (I), st_info (B: STB_GLOBAL<<4 | STT_FUNC=2), st_other (B: STO_CUDA_ENTRY=0x10), st_shndx (H=1)
        symtab.extend(struct.pack("<IBBHQQ", name_off, (1 << 4) | 2, 0x10, 1, 0, 16))
    for helper in device_helpers or []:
        name_off = add_str(strtab, helper)
        symtab.extend(struct.pack("<IBBHQQ", name_off, 2, 0x00, 1, 0, 16))

    sec_defs: list[tuple[str, int, bytes, int, int]] = [
        ("", 0, b"", 0, 0),
        (".text", 1, b"\x00" * 16, 0, 0),
        (".strtab", 3, bytes(strtab), 0, 0),
        (".symtab", 2, bytes(symtab), 2, 24),  # sh_link = 2 (.strtab)
    ]
    if ptx is not None:
        sec_defs.append((".nv_debug_ptx_txt", 1, ptx, 0, 0))
    shstr_idx = len(sec_defs)
    name_offs = [add_str(shstrtab, name) if name else 0 for name, *_ in sec_defs]
    shstr_name_off = add_str(shstrtab, ".shstrtab")
    name_offs.append(shstr_name_off)
    sec_defs.append((".shstrtab", 3, bytes(shstrtab), 0, 0))

    body = bytearray(64)  # placeholder ELF64 header
    shdrs = bytearray()
    for idx, (_name, sh_type, content, sh_link, sh_entsize) in enumerate(sec_defs):
        if idx == 0:
            offset, size = 0, 0
        else:
            while len(body) % 8:
                body.append(0)
            offset = len(body)
            body.extend(content)
            size = len(content)
        shdrs.extend(
            struct.pack(
                "<IIQQQQIIQQ",
                name_offs[idx],
                sh_type,
                0,
                0,
                offset,
                size,
                sh_link,
                0,
                8 if size else 0,
                sh_entsize,
            )
        )

    while len(body) % 8:
        body.append(0)
    e_shoff = len(body)
    body.extend(shdrs)

    ehdr = struct.pack(
        "<16sHHIQQQIHHHHHH",
        b"\x7fELF\x02\x01\x01" + b"\x00" * 9,
        2,  # ET_EXEC
        190,  # EM_CUDA
        1,
        0,
        0,
        e_shoff,
        e_flags,
        64,
        0,
        0,
        64,
        len(sec_defs),
        shstr_idx,
    )
    body[:64] = ehdr
    return bytes(body)


FAKE_PTXAS = f"""#!{sys.executable}
import re, sys
from pathlib import Path
sys.path.insert(0, {str(Path(__file__).resolve().parent)!r})
from test_retarget_kit import make_elf_cubin

args = sys.argv[1:]
if "--version" in args:
    print("Cuda compilation tools, release 13.2, V13.2.86")
    sys.exit(0)
arch = next(a.split("=", 1)[1] for a in args if a.startswith("-arch="))
out = Path(args[args.index("-o") + 1])
ptx = Path(args[-1]).read_text(encoding="latin1")
m_ver = re.search(r"\\.version\\s+(\\d+)\\.(\\d+)", ptx)
if not m_ver or (int(m_ver.group(1)), int(m_ver.group(2))) > (9, 2):
    print("ptxas fatal: unsupported PTX version", file=sys.stderr)
    sys.exit(1)
if f".target {{arch}}" not in ptx:
    print(f"ptxas fatal: expected .target {{arch}}", file=sys.stderr)
    sys.exit(1)
sm = int(re.search(r"sm_(\\d+)", arch).group(1))
funcs = re.findall(r"\\.entry\\s+([A-Za-z0-9_$.]+)", ptx)
out.write_bytes(make_elf_cubin(sm, funcs))
"""

FAKE_NVCC = f"""#!{sys.executable}
import re, sys
from pathlib import Path
sys.path.insert(0, {str(Path(__file__).resolve().parent)!r})
from test_retarget_kit import make_elf_cubin

args = sys.argv[1:]
if "--version" in args:
    print("Cuda compilation tools, release 13.2, V13.2.86")
    sys.exit(0)
arch = next(a.split("=", 1)[1] for a in args if a.startswith("-arch="))
sm = int(re.search(r"sm_(\\d+)", arch).group(1))
out = Path(args[args.index("-o") + 1])
src = Path(args[-1]).name
if src == "linear.cu":
    funcs = [
        "_ZN41_GLOBAL__N__11111111_9_linear_cu_2222222213rot_in_kernelEv",
        "_ZN41_GLOBAL__N__11111111_9_linear_cu_2222222213unpack_kernelILi4ELi2EEEvPKjP6__halfill",
        "_ZN41_GLOBAL__N__11111111_9_linear_cu_2222222213unpack_kernelILi6ELi2EEEvPKjP6__halfill",
    ]
elif src == "linear_grouped.cu":
    funcs = [
        "_ZN50_GLOBAL__N__11111111_17_linear_grouped_cu_2222222215rot_many_kernelEv",
        "_ZN50_GLOBAL__N__11111111_17_linear_grouped_cu_2222222214glinear_kernelILi4ELi2ELi4EEEvv",
        "_ZN50_GLOBAL__N__11111111_17_linear_grouped_cu_2222222214glinear_kernelILi4ELi2ELi8EEEvv",
        "_ZN50_GLOBAL__N__11111111_17_linear_grouped_cu_2222222214glinear_kernelILi6ELi2ELi4EEEvv",
        "_ZN50_GLOBAL__N__11111111_17_linear_grouped_cu_2222222214glinear_kernelILi6ELi2ELi8EEEvv",
    ]
else:
    sys.exit(2)
out.write_bytes(make_elf_cubin(sm, funcs))
"""


class RetargetKitTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="tf-retarget-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def write_exec(self, name: str, content: str) -> Path:
        p = self.tmp / name
        p.write_text(content, encoding="utf-8")
        p.chmod(p.stat().st_mode | stat.S_IEXEC)
        return p

    def make_sample_kit(self, root: Path, *, manifest_cols: int = 2) -> Path:
        kit = root / "kit"
        (kit / "aot" / "cubins").mkdir(parents=True)
        (kit / "cubins").mkdir(parents=True)
        (kit / "vision").mkdir(parents=True)

        aot_hash = "a" * 64
        aot_ptx = b".version 9.3\n.target sm_121a\n.address_size 64\x00.visible .entry _rowmm_parts() { ret; }\n"
        (kit / "aot" / "cubins" / f"{aot_hash}.cubin").write_bytes(
            make_elf_cubin(121, ["_rowmm_parts"], ptx=aot_ptx)
        )
        aot_json = {
            "kernels": [
                {
                    "fn": "_rowmm_parts",
                    "hash": aot_hash,
                    "name": "_rowmm_parts",
                    "num_warps": 2,
                    "num_ctas": 1,
                    "shared": 32,
                    "pdl": True,
                    "params": [],
                    "consts": {},
                }
            ]
        }
        (kit / "aot" / "aot.json").write_text(json.dumps(aot_json, indent=1) + "\n", encoding="utf-8")

        exp_funcs = [f"_ZN42_GLOBAL__N__aaaaaaaa_10_experts_cu_bbbbbbbb{s}v" for s in rk.EXL3_BASE_REQUIRED]
        exp_ptx = (
            ".version 9.3\n.target sm_121\n"
            + "\n".join(f".visible .entry {f}() {{ ret; }}" for f in exp_funcs)
            + "\n"
        ).encode("latin1")
        (kit / "cubins" / "experts.cubin").write_bytes(make_elf_cubin(121, exp_funcs, ptx=exp_ptx))

        cb_funcs = [
            "_ZN45_GLOBAL__N__aaaaaaaa_13_experts_cb_cu_bbbbbbbb18grouped_mma_kernelILi2ELi2ELi10ELb1ELi4EEEvv",
            "_ZN45_GLOBAL__N__aaaaaaaa_13_experts_cb_cu_bbbbbbbb19grouped_rows_kernelILi2ELi8EEEvv",
            "_ZN45_GLOBAL__N__aaaaaaaa_13_experts_cb_cu_bbbbbbbb17grouped_cp_kernelILi2ELi8EEEvv",
        ]
        cb_ptx = (
            ".version 9.3\n.target sm_121\n"
            + "\n".join(f".visible .entry {f}() {{ ret; }}" for f in cb_funcs)
            + "\n"
        ).encode("latin1")
        (kit / "cubins" / "experts_cb.cubin").write_bytes(make_elf_cubin(121, cb_funcs, ptx=cb_ptx))

        lin_funcs = [
            "_ZN3cub17_V_300304_SM_12106detail11EmptyKernelIvEEvv",
            "_ZN41_GLOBAL__N__5b811ffc_9_linear_cu_8819841b13rot_in_kernelEv",
            "_ZN41_GLOBAL__N__5b811ffc_9_linear_cu_8819841b13unpack_kernelILi4ELi2EEEvPKjP6__halfill",
            "_ZN41_GLOBAL__N__5b811ffc_9_linear_cu_8819841b13unpack_kernelILi6ELi2EEEvPKjP6__halfill",
        ]
        (kit / "cubins" / "linear.cubin").write_bytes(make_elf_cubin(121, lin_funcs, ptx=None))

        grp_funcs = [
            "_ZN50_GLOBAL__N__913a4562_17_linear_grouped_cu_52d7933215rot_many_kernelEv",
            "_ZN50_GLOBAL__N__913a4562_17_linear_grouped_cu_52d7933214glinear_kernelILi4ELi2ELi4EEEvv",
            "_ZN50_GLOBAL__N__913a4562_17_linear_grouped_cu_52d7933214glinear_kernelILi4ELi2ELi8EEEvv",
            "_ZN50_GLOBAL__N__913a4562_17_linear_grouped_cu_52d7933214glinear_kernelILi6ELi2ELi4EEEvv",
            "_ZN50_GLOBAL__N__913a4562_17_linear_grouped_cu_52d7933214glinear_kernelILi6ELi2ELi8EEEvv",
        ]
        (kit / "cubins" / "linear_grouped.cubin").write_bytes(make_elf_cubin(121, grp_funcs, ptx=None))

        (kit / "vision" / "torch_fmha_sm120.cubin").write_bytes(
            make_elf_cubin(120, [rk.FMHA_KERNEL_NAME], ptx=None)
        )
        (kit / "engram.json").write_text('{"ok": true}\n', encoding="utf-8")

        lines = []
        if manifest_cols == 5:
            lines.append("# header comment\n")
        for p in sorted(kit.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(kit).as_posix()
            sha = hashlib.sha256(p.read_bytes()).hexdigest()
            if manifest_cols == 5:
                lines.append(f"{sha}  {p.stat().st_size:8d}  git  -  {rel}\n")
            else:
                lines.append(f"{sha}  {rel}\n")
        # Include an unmaterialized RoPE table entry (generated later by make_rope.py)
        rope_sha = "f" * 64
        if manifest_cols == 5:
            lines.append(f"{rope_sha}  142606336  rope  tp2  rope-plain-cos.f32\n")
        else:
            lines.append(f"{rope_sha}  rope-plain-cos.f32\n")
        (kit / "MANIFEST").write_text("".join(lines), encoding="utf-8")
        return kit

    def make_prebuilt_kit(self, root: Path, sm: int = 110) -> Path:
        kit = root / "prebuilt"
        (kit / "aot").mkdir(parents=True)
        (kit / "vision").mkdir(parents=True)
        (kit / "aot" / "aot.json").write_text('{"kernels": []}\n', encoding="utf-8")
        (kit / "vision" / "torch_fmha_sm120.cubin").write_bytes(
            make_elf_cubin(sm, [rk.FMHA_KERNEL_NAME], ptx=None)
        )
        return kit

    def make_exl3_src(self, root: Path) -> Path:
        d = root / "exl3"
        d.mkdir(parents=True)
        for name in ("linear.cu", "linear_grouped.cu", "linear_common.cuh", "decode.cuh"):
            (d / name).write_text("// stub\n", encoding="utf-8")
        return d

    def test_elf_parsing_and_arch(self):
        c13 = make_elf_cubin(
            121,
            ["k1", "k2"],
            ptx=b".version 9.3\x00.target sm_121a\n",
            device_helpers=["$__internal_0_$__cuda_sm3x_div_rn_noftz_f32_slowpath"],
        )
        self.assertEqual(rk.elf_cuda_sm(c13), 121)
        self.assertEqual(rk.elf_func_symbols(c13), ["k1", "k2"])
        self.assertIn("$__internal_0_$__cuda_sm3x_div_rn_noftz_f32_slowpath", rk.elf_func_symbols(c13, entries_only=False))
        self.assertEqual(rk.extract_ptx(c13), ".version 9.3\n.target sm_121a\n")

        legacy = make_elf_cubin(89, ["k_old"], ptx=None, legacy_flags=True)
        self.assertEqual(rk.elf_cuda_sm(legacy), 89)
        self.assertIsNone(rk.extract_ptx(legacy))

    def test_retarget_ptx_version_clamping(self):
        ptx = ".version 9.3\n.target sm_121a\n.address_size 64\n"
        out, info = rk.retarget_ptx(ptx, "sm_110a", max_ptx_version=(9, 2))
        self.assertIn(".version 9.2", out)
        self.assertIn(".target sm_110a", out)
        self.assertEqual(info["clamped_from"], "9.3")
        self.assertEqual(info["clamped_to"], "9.2")

        out2, info2 = rk.retarget_ptx(ptx, "sm_110a", max_ptx_version=(9, 3))
        self.assertIn(".version 9.3", out2)
        self.assertIsNone(info2["clamped_from"])

        old = ".version 8.5\n.target sm_121\n"
        out3, info3 = rk.retarget_ptx(old, "sm_110a", max_ptx_version=(9, 2))
        self.assertIn(".version 8.5", out3)
        self.assertIn(".target sm_110a", out3)
        self.assertIsNone(info3["clamped_from"])

        with self.assertRaises(ValueError):
            rk.retarget_ptx(".version 9.3\n", "sm_110a")

    def test_convert_and_verify_2col_and_5col_manifests(self):
        ptxas = self.write_exec("fake-ptxas", FAKE_PTXAS)
        nvcc = self.write_exec("fake-nvcc", FAKE_NVCC)
        exl3_src = self.make_exl3_src(self.tmp)
        prebuilt = self.make_prebuilt_kit(self.tmp, sm=110)

        for cols in (2, 5):
            with self.subTest(manifest_cols=cols):
                src = self.make_sample_kit(self.tmp / f"src{cols}", manifest_cols=cols)
                dst = self.tmp / f"dst{cols}" / "kit"
                summary = rk.convert_kit(
                    src,
                    dst,
                    arch="sm_110a",
                    ptxas_path=str(ptxas),
                    nvcc_path=str(nvcc),
                    exl3_src=exl3_src,
                    prebuilt_kit=prebuilt,
                    jobs=2,
                )
                self.assertEqual(summary["cubins_retargeted"], 3)
                self.assertEqual(summary["cubins_ptx_version_clamped"], 3)
                self.assertEqual(summary["cubins_built_exl3"], ["cubins/linear.cubin", "cubins/linear_grouped.cubin"])
                self.assertEqual(summary["cubins_borrowed"], ["vision/torch_fmha_sm120.cubin"])
                self.assertEqual(summary["aot_pdl_variants"], 1)

                # aot.json preserved byte-for-byte
                self.assertEqual((dst / "aot" / "aot.json").read_bytes(), (src / "aot" / "aot.json").read_bytes())
                # MANIFEST.base preserved, MANIFEST is standard 2-column format with unmaterialized rope kept
                self.assertEqual((dst / "MANIFEST.base").read_bytes(), (src / "MANIFEST").read_bytes())
                mf_lines = (dst / "MANIFEST").read_text(encoding="utf-8").splitlines()
                self.assertTrue(all(len(line.split()) == 2 for line in mf_lines))
                self.assertIn(f"{'f' * 64}  rope-plain-cos.f32", mf_lines)

                rep = rk.verify_kit(dst, arch="sm_110a", gpu=False)
                self.assertEqual(rep["total_cubins"], 6)
                self.assertEqual(rep["aot_pdl_variants"], 1)
                self.assertEqual(rep["manifest_unmaterialized"], ["rope-plain-cos.f32"])

    def test_refuses_missing_no_ptx_source_and_wrong_arch(self):
        ptxas = self.write_exec("fake-ptxas", FAKE_PTXAS)
        src = self.make_sample_kit(self.tmp / "src_fail")

        with self.assertRaisesRegex(RuntimeError, "no embedded PTX"):
            rk.convert_kit(src, self.tmp / "dst_fail", arch="sm_110a", ptxas_path=str(ptxas), jobs=1)

        bad_prebuilt = self.make_prebuilt_kit(self.tmp / "bad", sm=121)
        nvcc = self.write_exec("fake-nvcc", FAKE_NVCC)
        exl3_src = self.make_exl3_src(self.tmp / "e2")
        with self.assertRaisesRegex(RuntimeError, "prebuilt arch is sm_121"):
            rk.convert_kit(
                src,
                self.tmp / "dst_bad_arch",
                arch="sm_110a",
                ptxas_path=str(ptxas),
                nvcc_path=str(nvcc),
                exl3_src=exl3_src,
                prebuilt_kit=bad_prebuilt,
                jobs=1,
            )

        with self.assertRaisesRegex(ValueError, "must not overlap"):
            rk.convert_kit(src, src, arch="sm_110a", ptxas_path=str(ptxas))


if __name__ == "__main__":
    unittest.main()
