# Porting this recipe to 2× Jetson AGX Thor (`sm_110a` + `/dev/mgbe_zc`)

**Status:** feasibility study, 2026-11-25, branch `jthor_zc`. No repository code changed yet.
**Verdict:** portable. The two things that looked blocking — the sm_121a-only kernel kit and the missing
RoCE verbs — are both solved: the kit converts to `sm_110a` (verified 368/371 cubins), and a
published Thor port of this same recipe lineage exists with a ready-made sm_110a kit, a working
two-node ring, and a launcher.

---

## 1. Test environment

| | |
| --- | --- |
| Nodes | `thor` 192.168.100.38 (rank 0, head), `thor2` 192.168.100.39 (rank 1) |
| GPU | NVIDIA Thor, compute capability **11.0** (`sm_110`), 122 GiB unified memory each |
| Kernel | `6.8.12-1021-tegra`, aarch64 |
| CUDA / driver | CUDA **13.2** (V13.2.86), driver 595.78 |
| Rails | `mgbe0_0..mgbe3_0`, MTU **9000**, 192.168.100.38/24 … 192.168.103.38/24 (`.39` on `thor2`), all UP |
| `/dev/mgbe_zc` | present on **both** nodes (`mgbe_zc` module loaded, 2 refs) |
| `/dev/infiniband` | **absent** — no verbs devices, so no RoCE and no `ibv_*` path |
| Memory in use | 115/122 GiB on both nodes: the GLM-5.3 Thor service is running and must be stopped |

The current recipe assumes DGX Spark GB10 (`sm_121`, ConnectX-7 RoCE, `/dev/infiniband`, `--gpus all`).
Nothing in `scripts/nodes.sh`, `scripts/prepare.sh` or `start.sh` handles a Thor node today: `detect_link`
requires a RoCE device and a RoCE v2 GID and dies without one.

---

## 2. The engine's own CUDA kernels compile for `sm_110`

The engine build takes `-Dsm=<arch>` (`zig/build/cuda.zig`), which becomes
`-gencode=arch=compute_<sm>,code=sm_<sm>` per kernel translation unit (plus `a` for `arch_specific` kernels).

Test: every kernel TU in the engine's list (`gdn`, `probe`, `qmm_group`, `qmm_prefill`, `experts`,
`experts_prefill`, `experts_pack`, `prefill_attention`, `scan_rows`, `nemotron_ops`, `lane_gemv`,
`sample`, `torch_*`, `dsv41_ops`, `dsv41_torch`, `dsv41_rdma` — 21 TUs) compiled with the engine's
own flag set:

```
nvcc -fatbin -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ \
  -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ \
  --expt-relaxed-constexpr -std=c++20 -O3 -gencode=arch=compute_110,code=sm_110 …
```

**Result: 0 failures.** The kernels are gated on `__CUDA_ARCH__ >= 900` (`mma.sync` paths), no sm_121-only
intrinsics, so nothing has to be rewritten to boot. `nvcc --list-gpu-arch` on the node offers
`compute_110` (and `110a`), so `SM=110a` is a legal build target.

---

## 3. The kernel kit is the real arch dependency

The dsv41 family loads most of its work from the pinned kit (`TF_DS_KIT`, `TENSORFOLD_CUDA_KERNELS`),
not from the engine's fatbins:

| Kit path | Contents | Loaded by |
| --- | --- | --- |
| `kit/aot/` | `aot.json` + **366 Triton AOT cubins** (`sm_121a`) | `cuda/aot.Set.load` (`cuModuleLoadData`) |
| `kit/cubins/` | `linear.cubin`, `linear_grouped.cubin`, `experts.cubin`, `experts_cb.cubin` (`sm_121`) | `exl3_prefill`, `exl3_linear`, `exl3_experts` |
| `kit/vision/` | `torch_fmha_sm120.cubin` (`sm_120`-class SASS) | vision tower |

Kit cubins are `cuModuleLoadData`-ed by hash; an sm_121a cubin on an sm_110 device fails with
`CUDA_ERROR_NO_BINARY_FOR_GPU`. The Triton sources (`kernels.py`) are **not** in the public engine
repo or the public kit repo.

### 3.1 PTX retarget experiment (the kit is portable without sources)

Each Triton cubin still carries its full PTX in the ELF section `.nv_debug_ptx_txt`. Experiment:

1. Extract that section, NULs → newlines (valid PTX text, `.version 9.3`, `.target sm_121a`).
2. Rewrite `.target sm_121a` → `.target sm_110a`, and clamp `.version 9.3` → `9.2` (the node's driver
   is CUDA 13.2; 9.3 is rejected as `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`).
3. Assemble with `ptxas -arch=sm_110a`.

Results on the pinned kit:

| | |
| --- | --- |
| Converted | **368 of 371** cubins, 2 m 08 s total |
| No PTX (need source or a borrowed binary) | `cubins/linear.cubin`, `cubins/linear_grouped.cubin`, `vision/torch_fmha_sm120.cubin` |
| Runtime load check | a converted cubin `cuModuleLoadData` + `cuModuleGetFunction` **succeeds** on the Thor GPU |

Because the conversion preserves the kit's launch metadata (`aot.json` is untouched: grid, constexprs,
`shared`, `num_warps`, `num_ctas`), the engine keeps loading the same variant set — no engine change is
needed for this path.

---

## 4. Transport: no verbs on Thor

The dsv41 family has two transports:

* `comm.zig` — NCCL all-gather for every model reduction (`ncclAllGather`, `ncclSend/Recv`).
* `fabric/rdma.zig` — the family's own one-hop RDMA ring for decode-size gathers (host proxy +
  `rdma_gather.cu` kernel pair), enabled only when `TF_RDMA_DEVICES` is set. Unset ⇒ NCCL.

On Thor there is no `/dev/infiniband`, so:

* NCCL must run on its **socket** transport across the four MGBE rails
  (`NCCL_SOCKET_IFNAME=mgbe0_0,mgbe1_0,mgbe2_0,mgbe3_0`, `NCCL_IB_DISABLE=1`), and
* the RDMA ring cannot open, so `TF_RDMA_DEVICES` stays unset.

`mgbe_zc` (`/dev/mgbe_zc`, 4×25 GbE zero-copy VDMA) is a **third** option: it exists in this repo's
sibling GLM-5.3 recipe as engine patch `0018` (`zig/src/cuda/mgbe_zc.zig`, ~1 770 lines, plus
`roce_gather.cu`'s BF16 gather), but it is written against GLM's `tp_group.Group`, not dsv41's
`rdma.Ring`.

**Note:** the published Thor port of this recipe (section 5) does **not** use `mgbe_zc`; it uses a
small TCP ring behind the same `Ring` interface. `mgbe_zc` would be an optimisation on top of a
working baseline, not a prerequisite.

---

## 5. A published Thor port of this recipe lineage exists

`manateelazycat/DeepSeek-V4.1-Flash-1017-TensorFold-Zig-Thor-TP2` (Hugging Face, **public**, 392.7 GB
stored) is a deployed two-Thor package built from this recipe lineage. Local provenance trees also exist
under `/home/test/work/glm53/tmp/extracted/`.

### 5.1 What it contains

| Item | Detail |
| --- | --- |
| Runtime images | `tensorfold-zig-runtime-0.1.{0,24,25,26}-arm64.tar.gz` (794–828 MB, OCI layout) |
| Model | EXL3 **2.5 bpw** target (41 shards) + Engram shards + `kit/` (RoPE tables, `engram.json`, `gate_bias_vl`) |
| Rank files | `rank0/`, `rank1/` (292 MB each) + `*-recipe.json` |
| Source | `source/…-statistics-source-0.1.{3,4}.tar.gz` |
| Provenance (local) | `provenance/engine` (engine source freeze), `provenance/recipe`, `thor-patches/thor-runtime.patch`, `selected-kernels/`, `t5000-sm110a-kernels/`, `production-runtime-0.1.24/`, `tmr-nsight-0.1.25/`, `evidence/` |

### 5.2 The runtime image ships a complete sm_110a kit

Unpacked 0.1.26 image (`/kit`, `/opt/deepseek1017-zig-tp2`):

| Path | Contents | Arch |
| --- | --- | --- |
| `kit/aot/cubins/` | 345 cubins (350 variants) | **`arch=110`** |
| `kit/cubins/` | `linear`, `linear_grouped`, `experts`, `experts_cb` | **`arch=110`** |
| `kit/vision/torch_fmha_sm120.cubin` | PyTorch fmha | **`arch=110`** |
| `opt/…/kernels/` | `decode8-gu-b5`, `decode8-dn-b5`, `shard-head`, `prefill-shared2` | **`arch=110`** |
| `opt/…/bin/` | `tensorfold-native`, `tf-dsv41-lanes`, `run.py`, `healthcheck.py` | — |

Its `aot.json` generator note reads *"Thor rebuild of recipe v0.7.3"*: the same **40 kernel names** as our
v0.7.5 kit, **350 variants vs our 366**.

Variant coverage against our kit, matched on launch signature (name, warps, ctas, `shared`, params,
constexprs):

* **336 / 366 of our variants are covered** by the published sm_110 kit.
* Missing: 16 `_rowmm_parts` (our v0.7.5 additions), 12 `_gemm`, 2 `_bias` — the latter two differ only
  in Triton's `shared` staging size (98 304 vs 49 168 B), i.e. arch-specific scheduling, not a source
  difference. All of them have PTX in our kit and convert with the method in §3.1.
* The three files with **no PTX** in our kit (`linear`, `linear_grouped`, `vision/torch_fmha_sm120`) have
  **identical kernel symbol sets** in the published kit (110/110, 56/56, 22/22 FUNC symbols), so the
  same source version — they can simply be taken from there.

### 5.3 The published transport patch

`provenance/thor-patches/thor-runtime.patch` — 204 lines, 6 files:

* `zig/src/families/dsv41/rdma.zig` (+124): `Ring.createTcp`, `Ring.gatherChunked`, `tcpMaxBytes()`
* `zig/src/families/dsv41/model.zig`: `TF_THOR_TCP_RING` opens the TCP ring instead of the verbs ring
* `zig/src/families/dsv41/prompt.zig`: `gatherF32` falls back to `gatherChunked` for oversized slices
* `zig/build/cuda.zig`: two new tools, `tf-dsv41-export`, `tf-dsv41-tcp-check`

The deployed binaries contain `TF_THOR_TCP_RING` and no `mgbe_zc` symbol: the working two-node transport
on Thor is TCP over the rails.

### 5.4 The published launcher

`/opt/deepseek1017-zig-tp2/run.py` (100 lines) is the Thor equivalent of `start.sh`'s rank commands:

* reads four local and four peer rail IPv4 addresses, resolves them to interface names, sets
  `NCCL_SOCKET_IFNAME==mgbe0_0,…` and writes an NCCL graph XML (`nchannels=4`, 4 channels);
* exports `TF_DS_KIT=/kit`, `TENSORFOLD_CUDA_KERNELS=/kit/aot`, `TF_DS_ENGRAM`, `TF_DS_TOKEN_MAP`,
  `TF_DS_POOL`, `TF_DS_WINDOW`;
* rank 0: `tensorfold-native serve /model … --context … --parallel …`;
  rank 1: `tf-dsv41-lanes /model /cache 1 2 <master> <port> /kit …`;
* refuses to start under 112 GiB available and kills the tree under 2.5 GiB (4 s) / 1 GiB (2 samples).

### 5.5 sm_110a kernel optimisation work (later phase)

`t5000-sm110a-kernels/` + `selected-kernels/` hold the pair-optimised kernels and their build commands:

* `build_sm110a_cubins_r4.cu`, `experts-cpt8-r4.cuh` — `tf_exl3x::grouped_cp_kernel` variants
  (`cp.async`, K2=4/6 32-bit funnelshift, 8-value pair decode), selected per template specialisation;
* `decode8-gu-r4.cu`, `decode8-dn-r4.cu` (6 blocks/SM), `shard-head-opt.cu` (`float4` `__ldcv`),
  `prefill-shared2.cu` (direct-SMEM prefill);
* release notes claim, measured on two Thors: **+7.6 % single-stream decode, +8.1 % 4K prefill,
  +7.3 % 8K prefill, 2.7× argmax microbenchmark**.

These target the deployed engine's kit layout (it loads `decode8-*`, `shard-head`, `prefill-shared2`
directly), not our engine's, so they are Phase 3 material.

---

## 6. Which engine is newer

| | Our recipe pin | Deployment / provenance base |
| --- | --- | --- |
| Branch | `deepseek-v41-zig` | `deepseek-v41-zig-tp2` |
| Commit | `fbbc971` | `a57deb9d` (base label; branch tip `ea64a870`) |
| Date | 2026-10-25 04:47 UTC | 2026-10-24 03:00 UTC (tip 2026-11-22 16:41 UTC) |
| `CHANGELOG` head | **1.0.2** | **1.0.12** |
| Compare | diverged: **+129** commits ours alone | **+2 203** commits theirs alone |

Neither contains the other. The tp2 line carries upstream releases 1.0.3 … 1.0.12 plus the TP2 recipe
merges (2 203 commits); our pin carries 129 dsv41-recipe commits the tp2 line lacks (`TF_DS_ENGRAM_LEAD`,
`TF_RDMA_TIMEOUT`, `TF_FILL_CUT_IN`, the kv-shard pieces, …) while its `CHANGELOG` stops at 1.0.2.

The tp2 line is the **maintained** one, and its tip added exactly what this port needs:
`TF_DS_VISION=0` (skip the vision tower) and general SM121 AOT records for non-div-16 tile counts.

---

## 7. Gaps and how each is closed

| Gap | Status | How |
| --- | --- | --- |
| Engine CUDA kernels on sm_110 | **solved** | `-Dsm=110`; 21/21 TUs compile |
| Triton AOT kit (366 cubins) | **solved** | PTX retarget 368/371 verified; 336/366 variants also available prebuilt |
| `linear`, `linear_grouped`, vision fmha cubins | **solved** | take the `arch=110` files from the published kit (identical symbol sets) |
| Two-node transport on Thor | **solved (baseline)** | `thor-runtime.patch` TCP ring (rebased onto our base) |
| `/dev/mgbe_zc` zero-copy DMA | **open** | adapt GLM `mgbe_zc.zig` to dsv41's `Ring` contract; optional, Phase 2 |
| Launcher (arch, docker runtime, rails) | **open, small** | `SM` 11.0 → `110a`; `--runtime=nvidia` + relaxed `NVIDIA_REQUIRE_CUDA` (image wants 13.3, driver is 13.2); MGBE rail discovery instead of RoCE GIDs; kit conversion step |
| Weights | **not started** | 196 GiB checkpoint + ~203 GB Engram on the head (~505 GB), ~106 GB rank file on `thor2`; rank files are generated CPU-only (`tf-dsv41-rank-cache`) |
| sm_110a kernel tuning | **optional** | borrow the `t5000`/`cpt8` kernels (Phase 3) |

---

## 8. Options

**Path A — stay on our pin, convert the kit (recommended first step).**
Keep `TF_REF=fbbc971` and the 53 recipe patches as they are. Build the sm_110a kit as
*336 variants from the published kit + 30 PTX-retargeted from ours + 3 borrowed no-PTX files*. Rebase
`thor-runtime.patch` (204 lines) for the TCP ring, and add a Thor branch to `scripts/nodes.sh` /
`scripts/prepare.sh` / `start.sh` using `run.py`'s rail setup. Lowest risk; our engine behaviour is
unchanged on Spark.

**Path B — move to the deployment base.**
Pin `TF_REF` to `a57deb9d` (or the tp2 tip `ea64a870`) + its Thor patch set + the published kit: the
exact combination that is running today, plus newer upstream and `TF_DS_VISION=0`. Cost: rebase our 53
recipe patches onto a diverged 2 203-commit line.

**Path C — run the published image as-is.**
The 0.1.26 OCI image plus its 2.5 bpw pack would serve on this pair without any work, but it is a
different checkpoint, model id and engine build; useful only as a reference measurement.

**Recommendation:** Path A, then borrow the tp2 line's `TF_DS_VISION=0` and kit-variant fixes as
individual patches; revisit Path B only if the rebase of the 129 recipe commits turns out to be small.
`mgbe_zc` comes after the pair boots and serves, with the TCP ring as the A/B baseline.

---

## 9. Phases

1. **Kit:** conversion script (`sm_121 kit → sm_110a kit`, PTX retarget + `ptxas`), 3 borrowed files,
   MANIFEST exception; verify every cubin loads on the device.
2. **Build:** `SM=110a` image; `--runtime=nvidia`, relaxed CUDA floor; kit dir wired to the converted set.
3. **Launcher:** MGBE rail detection (4 point-to-point subnets, MTU 9000), `NCCL_SOCKET_IFNAME`,
   `NCCL_IB_DISABLE=1`, `RDMA=0`, memory floor (112 GiB) — the pair currently runs the GLM service and
   must be stopped.
4. **First boot:** rank files from the 2.9 bpw EAQ v1 pack, NCCL socket + TCP ring, smoke test.
5. **`mgbe_zc`:** adapt the GLM `mgbe_zc.zig` transport to `rdma.Ring` (`create/connect/start/gather/
   gatherChunked`), `/dev/mgbe_zc` device + group wiring, A/B against the TCP ring.
6. **Tuning:** sm_110a kernels (`cpt8` experts, `decode8`, `shard-head`, `prefill-shared2`), expect the
   published deltas (+7.6 % decode, +8.1 % 4K prefill).

## 10. Risks / open questions

* Kit v0.7.3 vs v0.7.5: 30 variants differ (16 additions, 14 `shared`-size differences). PTX retarget
  keeps our exact variant set and metadata, so this is a convenience question, not a blocker — but a
  Triton rebuild from the family runtime's `kernels.py` (v0.6.6 locally) would need signature checks.
* EXL3 cubin equality is inferred from identical symbol sets; if the published `linear*.cubin` were
  built from a different source revision the mangled names would still match (anonymous namespace) —
  worth one bit-exactness test on the pair.
* `experts_cb.cubin` in the published kit has 16 FUNC symbols against our 90 (their pack uses only the
  `mul1` codebook). Our kit has PTX for it, so this is covered by conversion.
* 4×25 GbE (100 Gb/s aggregate) vs ConnectX-7 200 Gb/s: expect lower decode and prefill than the Spark
  numbers in the README even at equal kernels.
* The engine version comparison is version-string based (`CHANGELOG` head 1.0.2 vs 1.0.12) plus the
  GitHub compare API (`ahead 2203 / behind 129`); the two branches are not linearly ordered.

---

## Appendix A — reproduction commands

```bash
# platform
uname -a; nvidia-smi --query-gpu=name,compute_cap --format=csv
ip -br link show | grep mgbe; ls -l /dev/mgbe_zc /dev/infiniband 2>&1
ssh test@192.168.100.39 'ls -l /dev/mgbe_zc; nvidia-smi --query-gpu=compute_cap --format=csv'

# engine kernels for sm_110 (0 failures)
nvcc -fatbin -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ \
  -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ \
  --expt-relaxed-constexpr -std=c++20 -O3 -gencode=arch=compute_110,code=sm_110 \
  -o /tmp/<kernel>.fatbin zig/kernels/cuda/<kernel>.cu

# kit conversion (per cubin): extract .nv_debug_ptx_txt, NUL->newline,
# .target sm_121a -> sm_110a, .version 9.3 -> 9.2, then
ptxas -arch=sm_110a -o out.cubin in.ptx

# published kit + runtime (public)
hf download manateelazycat/DeepSeek-V4.1-Flash-1017-TensorFold-Zig-Thor-TP2 \
  tensorfold-zig-runtime-0.1.26-arm64.tar.gz --local-dir /tmp/thorrt
tar xzf tensorfold-zig-runtime-0.1.26-arm64.tar.gz && tar xzf blobs/sha256/03b26d7e… -C rootfs
```

## Appendix B — raw numbers

| Measurement | Value |
| --- | --- |
| Engine kernel TUs compiled for `sm_110` | 21 / 21 |
| Kit cubins converted by PTX retarget | 368 / 371 (2 m 08 s) |
| Kit cubins without PTX | 3 (`linear`, `linear_grouped`, `vision/torch_fmha_sm120`) |
| Our kit | 366 variants, 40 kernel names, recipe v0.7.5 |
| Published kit | 350 variants (345 cubins), 40 kernel names, "Thor rebuild of recipe v0.7.3", all `arch=110` |
| Variant coverage of our kit | 336 / 366 |
| Missing variants | 16 `_rowmm_parts`, 12 `_gemm`, 2 `_bias` |
| `thor-runtime.patch` | 204 lines, 6 files |
| Engine compare | diverged, ahead 2 203 / behind 129 |
| Kit repo pin | `KIT_REF=abe07c7`, repo tip `fdde328` (2026-11-19), v0.7.5 maintenance mode |

## Appendix C — paths used

| Path | What |
| --- | --- |
| `/tmp/tf-dsv41` | engine clone at `TF_REF=fbbc971` |
| `/tmp/kit-dsv41` | kit clone (`KIT_REF=abe07c7`) |
| `/tmp/thorrt/rootfs` | unpacked 0.1.26 Thor runtime image (sm_110a kit, binaries, `run.py`) |
| `/home/test/work/glm53/tmp/extracted/provenance/engine` | engine source freeze (base `a57deb9d` + Thor patches) |
| `/home/test/work/glm53/tmp/extracted/provenance/thor-patches/thor-runtime.patch` | TCP ring transport |
| `/home/test/work/glm53/tmp/extracted/provenance/t5000-sm110a-kernels`, `selected-kernels` | sm_110a kernel sources and build commands |
| `/home/test/work/glm53/tmp/extracted/provenance/production-runtime-0.1.24` | host-side runtime patches |
| `/home/test/work/glm53/ds41/runtime/app/tensorfold` | family Python runtime (Triton `kernels.py`, `cuda/exl3/*.cu`) v0.6.6 |
| `/home/test/work/GLM-5.3-Flash-DGX-Sparks-1104` | sibling Thor recipe (GLM-5.3): `mgbe_zc` engine patch `0018`, sm_110a kernel patches `0019`–`0021`, `docs/thor.md` |
