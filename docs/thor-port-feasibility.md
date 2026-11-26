# Porting this recipe to 2× Jetson AGX Thor (`sm_110a` + `/dev/mgbe_zc`)

**Status:** Phases 1–5 implemented and live-validated on `thor` ↔ `thor2` (2026-11-25, branch `jthor_zc`); Phase 6 (`sm_110a` tuning) next.
**Verdict:** proven live. The pinned kit (`KIT_REF=abe07c7d`) converts cleanly to `sm_110a` via `scripts/retarget_kit.py` (preserving all 147 PDL variants), the engine builds natively for `-Dsm=110a` (`0069`), boots on both NCCL Socket TCP (`MGBE_ZC=off`) and 4-rail `/dev/mgbe_zc` (`0070`, `MGBE_ZC=on`, bit-identical), passes `tools/smoke.py` and 6/6 needle checks, and delivers **+83% to +208%** decode speedup (**38.1–39.6 tok/s** 1-stream, **74.8 tok/s** 4-stream aggregate) and **+50% to +67%** prefill speedup (**1,051 tok/s** at 32k) over NCCL Socket TCP with 0 `mmc_rx_fifo_overflow`.

---

## 0. Live Validation Results (Phases 1–5, 2026-11-25)

* **Microbenchmark (`tf-dsv41-rdma-bench ... mgbe_zc`, 4 rails, 50-gather PDL CUDA Graphs):** `bad=0`, `bad_into=0`, 0 `mmc_rx_fifo_overflow` delta:
  * 1 row (10 KiB, eager): **12.94 µs**; 6 rows (60 KiB, eager): **19.36 µs**; 16 rows (160 KiB, DDP): **42.96 µs**; 512 rows (5 MiB, DDP): **550.43 µs** (~76 Gbps uni); 2048 rows (20 MiB, DDP): **2.36 ms**.
* **Boot & memory:** Warm load **26.3 s** (API ready in **32 s**); `dsv41: mgbe_zc ring active (2 ranks, 4 rail(s), slots up to 20480 KiB)`; post-load headroom **10.1 GiB** (`thor`) / **12.8 GiB** (`thor2`).
* **Correctness:** `tools/smoke.py` 100% PASS; `tools/bench.py` needle 8k & 32k 6/6 CORRECT; greedy SHA-256 (`MGBE_ZC=on` vs `MGBE_ZC=off`, `short_math`, `short_think`, `prefill_2k`, `prefill_6k`): **4/4 IDENTICAL**.
* **A/B benchmark (`tools/bench.py --salt thor-ab-1`, 0 errors):**

| Metric | `MGBE_ZC=off` (4-rail NCCL TCP) | `MGBE_ZC=on` (4-rail `/dev/mgbe_zc`) | Gain |
| --- | ---: | ---: | ---: |
| Decode (1 stream, 256 tok) | 20.88 tok/s (TTFT 0.38 s) | **38.13 tok/s** (TTFT 0.26 s) | **+82.6%** |
| Multi ×1 | 18.42 tok/s | **39.60 tok/s** | **+114.9%** |
| Multi ×2 aggregate (per-stream) | 18.42 (9.53) tok/s | **56.66 (28.96) tok/s** | **+207.6%** |
| Multi ×4 aggregate (per-stream) | 28.78 (8.02) tok/s | **74.81 (19.71) tok/s** | **+159.9%** |
| Prefill 2k (`1,930` tok) | 556.56 tok/s (3.47 s) | **832.37 tok/s** (2.32 s) | **+49.6%** |
| Prefill 8k (`7,683` tok) | 623.49 tok/s (12.32 s) | **1,005.74 tok/s** (7.64 s) | **+61.3%** |
| Prefill 32k (`30,680` tok) | 629.79 tok/s (48.73 s) | **1,051.28 tok/s** (29.18 s) | **+66.9%** |

* **Operational notes found during bring-up:**
  * Thor's kernel NFS daemon serves NFSv3 (`+3`) only → `NFS_OPTS='vers=3,ro,rsize=1048576,wsize=1048576,hard,timeo=600'` in `scripts/local.sh`.
  * `hf_xet` derives download buffers from total RAM (~7.8 GiB on 122 GiB Thor), which OOMs if another server is resident; capping via `HF_XET_DISABLE_MEMORY_DERIVED_DOWNLOAD_BUFFERS=1` + `HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT=268435456` keeps RSS at ~322 MiB.

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
requires a RoCE device and a RoCE v2 GID and dies without one, and `start.sh` unconditionally passes
`--device /dev/infiniband`.

---

## 2. The engine's own CUDA kernels compile for `sm_110`

The engine build takes `-Dsm=<arch>` (`zig/build/cuda.zig`), which becomes
`-gencode=arch=compute_<sm>,code=sm_<sm>` per kernel translation unit (plus `a` for `arch_specific` kernels).

Test: every base kernel TU in `/tmp/tf-dsv41` (`gdn`, `probe`, `qmm_group`, `qmm_prefill`, `experts`,
`experts_prefill`, `experts_pack`, `prefill_attention`, `scan_rows`, `nemotron_ops`, `lane_gemv`,
`sample`, `torch_*`, `dsv41_ops`, `dsv41_torch`, `dsv41_rdma` — 21 TUs) compiled with the engine's
own flag set:

```
nvcc -fatbin -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ \
  -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ \
  --expt-relaxed-constexpr -std=c++20 -O3 -gencode=arch=compute_110,code=sm_110 …
```

**Result: 0 failures.** The kernels are gated on `__CUDA_ARCH__ >= 900` (`mma.sync` and `griddepcontrol`
paths), with no sm_121-only intrinsics, so nothing has to be rewritten to boot. `nvcc --list-gpu-arch` on
the node offers `compute_110` (and `110a`).

Two caveats for the actual image build:

* **22nd TU from our patch stack:** patches `0015` + `0022` add `zig/kernels/cuda/dsv41_shard.cu` (and
  `0017` adds opt-in `dsv41_port.cu`). They use the same `__CUDA_ARCH__ >= 900` / `__ldcv` / system-scope
  PTX primitives, but were not part of the unpatched 21-TU test and must be verified once the 53 patches are
  applied.
* **`-Dsm=110a` double-`a` footgun in `zig/build/cuda.zig`:** `fatbin()` appends `"a"` whenever
  `k.arch_specific` is true (`compute_{s}{a},code=sm_{s}{a}`). Passing `-Dsm=110a` works today only because
  no kernel sets `arch_specific = true`; once any Phase 6 `tcgen05` kernel sets it, `-Dsm=110a` would emit
  `compute_110aa` and fail. Either pass `-Dsm=110` for baseline or guard `cuda.zig` not to append `"a"`
  when `sm` already ends with `a`.

---

## 3. The kernel kit is the real arch dependency

The dsv41 family loads most of its work from the pinned kit (`TF_DS_KIT`, `TENSORFOLD_CUDA_KERNELS`),
not from the engine's fatbins:

| Kit path | Contents | Loaded by |
| --- | --- | --- |
| `kit/aot/` | `aot.json` + **366 Triton AOT cubins** (`sm_121a`, **233 with `pdl: true`**) | `cuda/aot.Set.load` (`cuModuleLoadData`) |
| `kit/cubins/` | `linear.cubin`, `linear_grouped.cubin`, `experts.cubin`, `experts_cb.cubin` (`sm_121`) | `exl3_prefill`, `exl3_linear`, `exl3_experts` |
| `kit/vision/` | `torch_fmha_sm120.cubin` (`sm_120`-class SASS) | vision tower |

Kit cubins are `cuModuleLoadData`-ed by hash; an sm_121a cubin on an sm_110 device fails with
`CUDA_ERROR_NO_BINARY_FOR_GPU`. The Triton sources (`kernels.py`) are **not** in the public engine
repo or the public kit repo.

### 3.1 PTX retarget experiment (the kit is portable without sources)

Each Triton cubin (plus `experts.cubin` and `experts_cb.cubin`) still carries its full PTX in the ELF
section `.nv_debug_ptx_txt`. Experiment:

1. Extract that section, NULs → newlines (valid PTX text, `.version 9.3`, `.target sm_121a`).
2. Rewrite `.target sm_121a` → `.target sm_110a`, and (when assembling with host CUDA 13.2 `ptxas`) clamp
   `.version 9.3` → `9.2` (CUDA 13.2 `ptxas` rejects 9.3; note that running `ptxas` inside the CUDA 13.3
   build container accepts 9.3 natively, and CUDA 13.3 `sm_110a` SASS cubins load on driver 595.78 — worth
   a one-cubin check to see if the version clamp can be dropped).
3. Assemble with `ptxas -arch=sm_110a`.

Results on the pinned kit:

| | |
| --- | --- |
| Converted | **368 of 371** cubins, 2 m 08 s total |
| No PTX (3 files) | `cubins/linear.cubin`, `cubins/linear_grouped.cubin`, `vision/torch_fmha_sm120.cubin` |
| Runtime load check | a converted cubin `cuModuleLoadData` + `cuModuleGetFunction` **succeeds** on the Thor GPU |

Because the conversion preserves the kit's launch metadata (`aot.json` is untouched: hashes, grid,
constexprs, `shared`, `num_warps`, `num_ctas`, and **all 233 `pdl: true` flags**), the engine keeps loading
the exact pinned v0.7.5 variant set with no engine change.

### 3.2 Handling the 3 cubins without PTX

1. **`cubins/linear.cubin` and `cubins/linear_grouped.cubin`:** the full CUDA sources (`linear.cu`,
   `linear_grouped.cu`, `linear_common.cuh`, `decode.cuh`) exist locally under
   `/home/test/work/glm53/ds41/runtime/app/tensorfold/cuda/exl3/`. Compiling them directly with
   `nvcc -arch=sm_110` is clean, auditable, eliminates the anonymous-namespace version-mismatch risk, and
   avoids pulling an 828 MB third-party OCI tarball. (Borrowing the prebuilt `arch=110` binaries from
   `/tmp/thorrt/rootfs/kit` remains a fast Phase 1 bootstrap/cross-check.)
2. **`vision/torch_fmha_sm120.cubin`:** `tools/dsv41-zig/build_kit.sh` step 2 extracts this directly from
   `libtorch_cuda.so` via `cuobjdump`; alternatively, borrow the single `arch=110` file from the published
   kit or skip loading the vision tower altogether once `TF_DS_VISION=0` is cherry-picked.

---

## 4. Transport: no verbs on Thor

The dsv41 family has two built-in transports:

* `comm.zig` — NCCL all-gather for every model reduction (`ncclAllGather`, `ncclSend/Recv`).
* `dsv41/rdma.zig` — the family's one-hop RDMA ring for decode-size gathers (host proxy +
  `dsv41/rdma_gather.cu`, plus `0022`'s `dsv41_shard.cu` for `KV_SHARD=1`), enabled only when
  `TF_RDMA_DEVICES` is set. Unset (`RDMA=0`) ⇒ **NCCL fallback everywhere with zero engine patches**.

On Thor there is no `/dev/infiniband`, giving three transport choices:

1. **NCCL Socket TCP across the 4 MGBE rails (recommended Phase 4 baseline):**
   `NCCL_SOCKET_IFNAME=mgbe0_0,mgbe1_0,mgbe2_0,mgbe3_0`, `NCCL_IB_DISABLE=1`, `RDMA=0`, plus
   `NCCL_MIN_NCHANNELS=4` / `NCCL_MAX_NCHANNELS=4` (or the 4-channel `NCCL_GRAPH_FILE` XML from `run.py` if
   NCCL does not stripe all 4 rails automatically). Boots immediately with **zero** engine transport changes.
2. **`/dev/mgbe_zc` zero-copy VDMA behind `rdma.Ring` (recommended Phase 5 primary transport):**
   Already proven in the sibling GLM-5.3 recipe as patch `0018` (`zig/src/cuda/mgbe_zc.zig`, ~1 770 lines):
   on this exact Thor pair, GLM-5.3 measured **13–23 µs** decode gathers vs **193–452 µs** over NCCL Socket
   (**15–23× faster**, **1.75×** live 1-stream decode) and **63–65 Gbps/dir** prefill vs **11–12 Gbps/dir**
   over NCCL Socket (**5–6× faster**).
   * **Clean fit with `dsv41`:** both `dsv41/rdma_gather.cu` and `0022`'s `dsv41_shard.cu` talk to the host
     proxy exclusively through the same `send` / `recv` / `meta` (`doorbell`, `sizes`, `flags`, `sent`,
     `abort_word`) ring buffers. Implementing `Ring.createMgbeZc` means replacing `Pinned.alloc`'s
     `mmap + cuMemHostRegister` with `CU_MEM_LOCATION_TYPE_HOST_NUMA` VMM + `MGBE_ZC_IOC_REG_MR` (for
     contiguous 48-bit SMMU IOVA) and plugging in `0018`'s 4-rail eager + hardware split-header DDP worker.
   * **Covers everything at once:** CUDA-Graph decode gathers (`gather` / `gatherInto`), prefill gathers
     (`gatherChunked` + optional BF16-on-the-wire partials like GLM's `roce_gather_bf16` to halve 4×25 GbE
     prefill payload), **and** `0022`'s asymmetric `KV_SHARD=1` `sendRows`/`recvRows` (since `0018`'s
     `WireHdr` already carries per-direction `nbytes`).
   * **Reliability requirement:** port the **latest (2026-11-25)** `0018` from GLM-5.3 — it includes the
     `flag_retrans`, `tx_sent == win_pkts`, bounded `peer_max_seq`, 50 ms total-loss `cts_gen` recovery, and
     `ctrl_q` drain fixes for the `mmc_rx_fifo_overflow` hangs documented in
     `GLM-5.3/docs/mgbe-zc-rails-4-vs-3-2026-11-23.md`, plus `MGBE_ZC_CPUS` and host IRQ pinning off CPU 0.
3. **Published `thor-runtime.patch` TCP ring (not recommended):**
   Rebasing it onto our 53-patch stack is throwaway work on the way to `/dev/mgbe_zc` and has concrete
   conflicts with our patches:
   * **Breaks `0022` (`KV_SHARD=1`):** `tcpLoop` checks `incoming[1] != n` (requiring identical byte counts
     in both directions), whereas `0022`'s `sendRows`/`recvRows` sends asymmetric per-rank row lists.
   * **Regresses `0010` and `0068`:** spawns 4 busy-spin threads (100% CPU across 4 cores even when idle)
     without idle-nap or CPU pinning, and allocates 480 MiB of pinned staging buffers by default (vs ~200 MiB
     for `mgbe_zc`).

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

### 5.2 The runtime image's sm_110a kit (and why pure PTX retargeting beats borrowing it)

Unpacked 0.1.26 image (`/kit`, `/opt/deepseek1017-zig-tp2`):

| Path | Contents | Arch |
| --- | --- | --- |
| `kit/aot/cubins/` | 345 cubins (350 variants, **0 with `pdl: true`**) | **`arch=110`** |
| `kit/cubins/` | `linear`, `linear_grouped`, `experts`, `experts_cb` | **`arch=110`** |
| `kit/vision/torch_fmha_sm120.cubin` | PyTorch fmha | **`arch=110`** |
| `opt/…/kernels/` | `decode8-gu-b5`, `decode8-dn-b5`, `shard-head`, `prefill-shared2` | **`arch=110`** |
| `opt/…/bin/` | `tensorfold-native`, `tf-dsv41-lanes`, `run.py`, `healthcheck.py` | — |

Its `aot.json` generator note reads *"Thor rebuild of recipe v0.7.3"*: the same **40 kernel names** as our
v0.7.5 kit, **350 variants vs our 366**.

Variant coverage against our kit:

* Ignoring `pdl`: **336 / 366** match `(name, warps, ctas, shared, params, constexprs)`; missing 16
  `_rowmm_parts` (our v0.7.5 additions), 12 `_gemm` (`shared` 98 304 vs 49 168 B), and 2 `_bias`
  (`shared` 32 768 vs 41 984 B).
* **Including `pdl`: only 121 / 366 match.** Our kit enables `pdl: true` on **233 / 366** variants, while
  the published Thor kit has `pdl: true` on **0 / 350** — borrowing their 336 cubins would strip PDL from
  **215** decode/attention/MoE kernels and force `aot.json` hash remapping.
* Therefore, **retargeting all 368 PTX-bearing cubins from our own `KIT_REF`** is strictly better than a
  336+30 hybrid: it keeps `aot.json` untouched, preserves PDL on all 233 kernels, and avoids depending on
  the 828 MB third-party OCI tarball.
* The three files with **no PTX** in our kit (`linear`, `linear_grouped`, `vision/torch_fmha_sm120`) have
  **identical kernel symbol sets** in the published kit (110/110, 56/56, 22/22 FUNC symbols), so they can
  be borrowed as a quick Phase 1 bootstrap and replaced cleanly with local-source / `libtorch_cuda.so`
  builds (§3.2).

### 5.3 The published transport patch (reference only)

`provenance/thor-patches/thor-runtime.patch` — 204 lines, 6 files (`rdma.zig`, `model.zig`, `prompt.zig`,
`cuda.zig`, `export_main.zig`, `tcp_check_main.zig`). Useful as a reference for `gatherChunked()` in
`rdma.zig` and `prompt.zig` (which routes prefill/Engram/head gathers through `gatherF32`), but **not**
recommended to rebase directly as our transport (see §4).

### 5.4 Launcher reference: prefer GLM-5.3's Thor helpers over `run.py`

`/opt/deepseek1017-zig-tp2/run.py` (100 lines) is a useful reference for the 4-channel `NCCL_GRAPH_FILE`
XML (`nchannels=4`), but expects manually configured IP lists. The sibling GLM-5.3 recipe
(`/home/test/work/GLM-5.3-Flash-DGX-Sparks-1104/scripts/{nodes.sh,prepare.sh}`, `start.sh`) already solves
every Thor launcher requirement automatically and is tested on **this exact `thor` / `thor2` pair**:

* `detect_sm` (`11.0 → 110a` / `110`);
* `gpu_mode` / `gpu_args` (`--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all` vs `--gpus all`) and
  `cuda_compat_args` (auto-relaxes `NVIDIA_REQUIRE_CUDA` when host driver CUDA < image CUDA);
* `rdma_present` + `fabric_pairs` (discovers all point-to-point rails ≥ `FABRIC_MIN_SPEED_MBPS=10000`,
  excluding 1G LAN, probes jumbo MTU bidirectionally), `rail_tcp_buffers` check (`rmem_max`/`wmem_max`);
* conditional `--device /dev/infiniband`, and `/dev/mgbe_zc` + per-host `mgbe_zc` GID discovery
  (`--device /dev/mgbe_zc --group-add <gid>`).

### 5.5 sm_110a kernel optimisation work (Phase 6)

Two distinct optimisation sources exist:

1. **`t5000-sm110a-kernels/` + `selected-kernels/` (decode-focused):**
   * `build_sm110a_cubins_r4.cu`, `experts-cpt8-r4.cuh`, `decode8-gu-r4.cu`, `decode8-dn-r4.cu` (6 blocks/SM),
     `shard-head-opt.cu` (`float4` `__ldcv`), `prefill-shared2.cu`.
   * **Important finding:** inspecting `build_sm110a_cubins_r4.cu` shows **zero `tcgen05` / TMEM usage** —
     it still runs `mma16816` (`mma.sync`), which is why its claimed prefill gain is only **+7.3–8.1%**
     (alongside **+7.6% single-stream decode** and **2.7× argmax**).
2. **GLM-5.3 `0019-exl3-cuda-sm110a-tcgen05-prompt-kernel.patch` (prefill-focused):**
   * True `sm_110a` **`tcgen05` + TMEM** EXL3 grouped MoE prompt kernel (`tmain_tcgen05`, `TPM=128`, 256 TMEM
     cols, 0 register spills), measured at **3.17× faster EXL3 MoE prompt** on this Thor pair
     (`~205 ms → 64.61 ms` per layer).
   * Applying `0019`'s `tcgen05` approach to `dsv41`'s `grouped_mma_kernel` (`exl3_experts.zig`) — paired
     with `t5000`'s `decode8`/`shard-head` kernels for decode and BF16 prefill all-gathers over `mgbe_zc` —
     targets both decode and the much larger prefill bottleneck.

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
`TF_RDMA_TIMEOUT`, `TF_FILL_CUT_IN`, the kv-shard pieces, …), plus our **53 recipe patches** (spill v2,
KV sharding + shard ring, mixed-pack BF16 EAQ v1.1, abliteration overlay, CUDA graph cap, prefill
interleave, Responses/Anthropic/DSML fixes).

The tp2 tip added two small things worth cherry-picking individually onto Path A: `TF_DS_VISION=0` (skip
the vision tower) and general SM121 AOT records for non-div-16 tile counts.

---

## 7. Gaps and how each is closed

| Gap | Status | How |
| --- | --- | --- |
| Engine CUDA kernels on sm_110 | **mostly solved** | 21/21 base TUs compile; verify 22nd TU (`dsv41_shard.cu` from `0015`/`0022`) after patches; use `-Dsm=110` or guard `cuda.zig` against `compute_110aa` |
| Triton AOT + EXL3 expert cubins (368 files) | **solved** | PTX-retarget all **368/371** from pinned `KIT_REF` (preserves all 233 PDL variants, `aot.json` untouched); store under arch-scoped `KIT_SUB` with regenerated `MANIFEST` |
| 3 no-PTX cubins (`linear`, `linear_grouped`, `vision/torch_fmha_sm120`) | **solved** | Build `linear`/`linear_grouped` for `sm_110` from local `exl3/*.cu` sources (or borrow 3 prebuilt files for Phase 1 bootstrap); extract fmha from `libtorch_cuda.so` or skip via `TF_DS_VISION=0` |
| Two-node transport baseline | **solved (zero patches)** | **NCCL Socket TCP** across 4 MGBE rails (`RDMA=0`, `NCCL_IB_DISABLE=1`, optional 4-channel graph XML) |
| `/dev/mgbe_zc` zero-copy DMA | **open (Phase 5)** | Port latest (Nov 25) GLM `0018` directly behind `dsv41/rdma.Ring` (`HOST_NUMA` VMM, eager + SPH=1 DDP, `gatherChunked`, optional BF16 prefill, asymmetric `0022` `sendRows`/`recvRows`) |
| Docker build & image identity | **open, small** | Add `SM` to `image_hash()` & `tf.sm` label, pass `--build-arg SM` (prevents GHCR/local `sm_121` collision); drop `-Dnvcc` from `INPUTS_REF` CPU-only build; include `sm=$SM` in `prepared_state()` |
| Launcher (GPU runtime, rails, NFS) | **open, small** | Port GLM-5.3's `detect_sm`, `gpu_args`, `cuda_compat_args`, `fabric_pairs`, `rail_tcp_buffers`; gate `--device /dev/infiniband` on existence; wire `/dev/mgbe_zc` + GID; make `NFS_OPTS` configurable (verify NFSv4.2 vs `vers=3` on Tegra 6.8.12) |
| Weights & disk | **not started** | Head: ~505 GB (196 GiB checkpoint + ~203 GB Engram + ~106 GB rank0). **`thor2`: ~310 GB** (~203 GB `WORKER_ENGRAM_DIR` + ~106 GB `WORKER_CACHE_DIR`); reuse existing 1017 Engram shards via `HEAD_ENGRAM_DIR`/`WORKER_ENGRAM_DIR` if present |
| sm_110a kernel tuning | **optional (Phase 6)** | `t5000` decode kernels (`decode8`, `shard-head`) + GLM `0019` `tcgen05` (`sm_110a`) EXL3 MoE prefill |

---

## 8. Options

**Path A — stay on our pin, pure PTX-retargeted kit, NCCL baseline → `/dev/mgbe_zc` (recommended).**
Keep `TF_REF=fbbc971` and all 53 recipe patches (spill v2, KV sharding + shard ring, mixed-pack BF16,
abliteration, graph cap, prefill interleave, API/DSML fixes). Convert all **368** PTX-bearing cubins from
our pinned `KIT_REF` (preserving all 233 PDL variants and leaving `aot.json` untouched), build/supply the
3 no-PTX cubins, port GLM-5.3's Thor launcher helpers, boot Phase 4 on NCCL Socket TCP (`RDMA=0`, zero
transport patches), then land GLM-5.3's Nov 25 `/dev/mgbe_zc` backend directly behind `rdma.Ring`. Lowest
risk; Spark behaviour is unchanged.

**Path B — move to the deployment base.**
Pin `TF_REF` to `a57deb9d` (or tp2 tip `ea64a870`) + its Thor patch set: newer upstream and `TF_DS_VISION=0`,
but requires rebasing 53 recipe patches across 2 203 diverged commits and still loses PDL unless the kit is
rebuilt.

**Path C — run the published image as-is.**
The 0.1.26 OCI image plus its 2.5 bpw pack serves on this pair without work, but is a different checkpoint,
quant, model id, and engine build; useful only as a reference measurement.

**Recommendation:** Path A, cherry-picking `TF_DS_VISION=0` and non-div-16 AOT fixes as small patches.
Skip `thor-runtime.patch`'s TCP ring (go straight from zero-patch NCCL Socket baseline to `/dev/mgbe_zc`).

---

## 9. Phases

1. **Kit:** conversion script (`sm_121 kit → sm_110a kit`, PTX-retargeting all **368** cubins from
   `KIT_REF` with `aot.json` and all 233 PDL flags intact), build `linear.cubin` + `linear_grouped.cubin`
   from local EXL3 sources (or borrow the 3 no-PTX files temporarily), write to arch-scoped `KIT_SUB`
   (`kit-<ref>-sm110`) with a regenerated `MANIFEST`; verify every cubin loads on the Thor GPU.
2. **Build:** `SM=auto` (`11.0 → 110` initially, or `110a` with the `cuda.zig` double-`a` guard), hash `SM`
   in `image_hash()` and pass `--build-arg SM`, drop `-Dnvcc` on the `INPUTS_REF` build, verify all 22
   patched CUDA TUs compile cleanly.
3. **Launcher:** port GLM-5.3's `gpu_args`, `cuda_compat_args`, `fabric_pairs` (4× MGBE rails, MTU 9000,
   `NCCL_SOCKET_IFNAME`, `NCCL_IB_DISABLE=1`), `rail_tcp_buffers`, conditional `--device /dev/infiniband`,
   configurable `NFS_OPTS`, `RDMA=0` auto-default without verbs, `sm=$SM` in `prepared_state()`, and
   memory floor (112 GiB — stop the GLM-5.3 service on both Thors first).
4. **First boot (baseline):** prepare rank files from the 2.9 bpw EAQ v1 pack (reusing local 1017 Engram
   shards if available), boot on **NCCL Socket TCP** (zero transport patches), run `smoke.py`, `exact.py`,
   and `bench.py` as the A/B baseline.
5. **`/dev/mgbe_zc`:** port GLM-5.3's latest (Nov 25) `0018` wire engine + `HOST_NUMA` VMM + `gatherChunked`
   (+ optional BF16 prefill gather) directly behind `dsv41/rdma.Ring` — covering decode, prefill, and `0022`
   asymmetric `KV_SHARD=1` `sendRows`/`recvRows`; wire `--device /dev/mgbe_zc --group-add <gid>` +
   `MGBE_ZC_CPUS` + IRQ pinning off CPU 0; A/B against the Phase 4 NCCL baseline.
6. **Tuning:** port `t5000` decode kernels (`decode8`, `shard-head`) for decode/argmax gains **and** port
   GLM `0019`'s `tcgen05` (`sm_110a`) approach to `dsv41`'s `grouped_mma_kernel` for EXL3 MoE prefill.

---

## 10. Risks / open questions

* **Container `ptxas` 13.3 vs host `ptxas` 13.2:** verify whether an `sm_110a` cubin assembled by CUDA 13.3
  `ptxas` (native `.version 9.3`) loads on driver 595.78 / CUDA 13.2; if not, keep the `.version 9.3 → 9.2`
  clamp with host `ptxas` 13.2.
* **EXL3 `linear*.cubin` equivalence:** if borrowing the published `linear*.cubin` during Phase 1 bootstrap,
  verify bit-exactness against local-source builds (`exl3/linear.cu`, `linear_grouped.cu`) since mangled
  anonymous-namespace names would still match across source revisions.
* **Image pull/sync footprint:** `zig/Dockerfile` uses `nvcr.io/nvidia/pytorch:26.07-py3` (~20+ GB, not
  currently cached locally on `thor`), making `docker save | docker load` across nodes slow on rebuilds.
  Fine for initial bring-up; consider moving later to GLM-5.3's ~6 GB `nvidia/cuda:13.3.1` multi-stage
  build once serving is proven.
* **NFSv4.2 vs NFSv3 on Tegra 6.8.12:** `scripts/nodes.sh` hardcodes `nfsvers=4.2,ro,nconnect=8`, while
  GLM-5.3's Thor setup used `vers=3`. Verify NFSv4.2 + `nconnect=8` on `thor`/`thor2` and make `NFS_OPTS`
  configurable.
* **Memory headroom at 2.9 bpw:** EAQ v1 uses ~98.9 GiB weights + 12.2 GiB reserve = ~111.1 GiB per rank,
  leaving ~10–11 GiB headroom on 122 GiB Thor (comparable to 121 GiB Spark; `manateelazycat`'s port used
  2.5 bpw for extra margin). `mgbe_zc` adds ~200 MiB per rank.
* **`mgbe_zc` RX FIFO overflow resilience:** ensure all Nov 25 `0018` fixes (`flag_retrans`, `tx_sent ==
  win_pkts`, bounded `peer_max_seq`, 50 ms total-loss `cts_gen` bump, `ctrl_q` drain) plus MGBE IRQ
  pinning off CPU 0 are carried over; verify `mmc_rx_fifo_overflow` stays 0 under 16K–128K prefill stress.
* **Bandwidth ceiling:** 4×25 GbE (100 Gb/s aggregate; ~63–65 Gbps/dir measured on `mgbe_zc`) vs ConnectX-7
  200 Gb/s on Spark — expect lower decode and prefill ceilings than the Spark README numbers at equal
  kernels, partially offset on prefill by `tcgen05` and BF16 wire partials.
* **Disk on `thor2`:** verify ≥ 350 GiB free on `thor2` for `WORKER_ENGRAM_DIR` (~203 GB) +
  `WORKER_CACHE_DIR` (~106 GB) + Docker image layers, or point `WORKER_ENGRAM_DIR` at existing 1017 shards.

---

## Appendix A — reproduction commands

```bash
# platform
uname -a; nvidia-smi --query-gpu=name,compute_cap --format=csv
ip -br link show | grep mgbe; ls -l /dev/mgbe_zc /dev/infiniband 2>&1
ssh test@192.168.100.39 'ls -l /dev/mgbe_zc; nvidia-smi --query-gpu=compute_cap --format=csv'

# engine kernels for sm_110 (0 failures on 21 base TUs; also test dsv41_shard.cu after patches)
nvcc -fatbin -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ \
  -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ \
  --expt-relaxed-constexpr -std=c++20 -O3 -gencode=arch=compute_110,code=sm_110 \
  -o /tmp/<kernel>.fatbin zig/kernels/cuda/<kernel>.cu

# kit conversion (per cubin): extract .nv_debug_ptx_txt, NUL->newline,
# .target sm_121a -> sm_110a, (.version 9.3 -> 9.2 for CUDA 13.2 ptxas), then
ptxas -arch=sm_110a -o out.cubin in.ptx

# published kit + runtime (public, reference / optional 3-file Phase 1 bootstrap)
hf download manateelazycat/DeepSeek-V4.1-Flash-1017-TensorFold-Zig-Thor-TP2 \
  tensorfold-zig-runtime-0.1.26-arm64.tar.gz --local-dir /tmp/thorrt
tar xzf tensorfold-zig-runtime-0.1.26-arm64.tar.gz && tar xzf blobs/sha256/03b26d7e… -C rootfs
```

## Appendix B — raw numbers

| Measurement | Value |
| --- | --- |
| Engine base kernel TUs compiled for `sm_110` | 21 / 21 (22nd TU `dsv41_shard.cu` added by patches `0015`/`0022`) |
| Kit cubins converted by PTX retarget | 368 / 371 (2 m 08 s) |
| Kit cubins without PTX | 3 (`linear`, `linear_grouped`, `vision/torch_fmha_sm120`) |
| Our kit (`KIT_REF=abe07c7`) | 366 variants, 40 kernel names, recipe v0.7.5, **233 with `pdl: true`** |
| Published Thor kit | 350 variants (345 cubins), 40 kernel names, v0.7.3, `arch=110`, **0 with `pdl: true`** |
| Published kit coverage of ours | **121 / 366** exact (with `pdl`); 336 / 366 ignoring `pdl` (215 lose PDL) |
| Non-PDL missing variants in published kit | 16 `_rowmm_parts`, 12 `_gemm` (`shared` 98 304 vs 49 168), 2 `_bias` (`shared` 32 768 vs 41 984) |
| `thor-runtime.patch` | 204 lines, 6 files (incompatible with `0022` asymmetric `sendRows`/`recvRows`) |
| GLM-5.3 `/dev/mgbe_zc` measured on this Thor pair | 13–23 µs decode (15–23× vs NCCL TCP), 63–65 Gbps/dir prefill (5–6× vs NCCL TCP) |
| GLM-5.3 `tcgen05` EXL3 prompt measured on this Thor pair | 64.61 ms vs ~205 ms `mma.sync` (**3.17× faster**) |
| Disk required | Head ~505 GB; `thor2` ~310 GB (~203 GB Engram + ~106 GB rank1) |
| Engine compare | diverged, ahead 2 203 / behind 129 (+ our 53 recipe patches) |
| Kit repo pin | `KIT_REF=abe07c7`, repo tip `fdde328` (2026-11-19), v0.7.5 maintenance mode |

## Appendix C — paths used

| Path | What |
| --- | --- |
| `/tmp/tf-dsv41` | engine clone at `TF_REF=fbbc971` |
| `/tmp/kit-dsv41` | kit clone (`KIT_REF=abe07c7`) |
| `/tmp/thorrt/rootfs` | unpacked 0.1.26 Thor runtime image (sm_110a kit, binaries, `run.py`) |
| `/home/test/work/glm53/tmp/extracted/provenance/engine` | engine source freeze (base `a57deb9d` + Thor patches) |
| `/home/test/work/glm53/tmp/extracted/provenance/thor-patches/thor-runtime.patch` | reference TCP ring + `gatherChunked` |
| `/home/test/work/glm53/tmp/extracted/provenance/t5000-sm110a-kernels`, `selected-kernels` | sm_110 decode kernel sources (`mma.sync`) and build commands |
| `/home/test/work/glm53/tmp/extracted/provenance/production-runtime-0.1.24` | host-side runtime patches |
| `/home/test/work/glm53/ds41/runtime/app/tensorfold` | family Python runtime (Triton `kernels.py`, `cuda/exl3/{linear,linear_grouped}.cu`) v0.6.6 |
| `/home/test/work/GLM-5.3-Flash-DGX-Sparks-1104` | sibling Thor recipe (GLM-5.3): `mgbe_zc` engine patch `0018` (Nov 25), `tcgen05` patches `0019`–`0021`, `scripts/{nodes.sh,prepare.sh}`, `start.sh`, `docs/thor.md`, `docs/mgbe-zc-rails-4-vs-3-2026-11-23.md` |
