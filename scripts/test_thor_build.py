#!/usr/bin/env python3
"""Offline unit tests for Phase 2 multi-arch build wiring (SM=auto, image_hash, KIT_SUB, Dockerfile, 0069 patch).

Run: python3 -m unittest -v scripts/test_thor_build.py
"""
from __future__ import annotations

import os
import shutil
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ThorBuildTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="tf-thor-build-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.bin = self.tmp / "bin"
        self.bin.mkdir()

    def set_fake_nvidia_smi(self, cc: str | None) -> None:
        p = self.bin / "nvidia-smi"
        if cc is None:
            p.write_text("#!/bin/sh\nexit 127\n", encoding="utf-8")
        else:
            p.write_text(f"#!/bin/sh\necho '{cc}'\n", encoding="utf-8")
        p.chmod(p.stat().st_mode | stat.S_IEXEC)

    def bash_config(self, body: str, extra_env: dict[str, str] | None = None) -> tuple[int, str]:
        env = {
            "PATH": f"{self.bin}:/usr/bin:/bin",
            "HOME": str(self.tmp / "home"),
            "LANG": "C",
            **(extra_env or {}),
        }
        script = f"set -euo pipefail\ncd {ROOT}\nsource ./scripts/config.sh\n{body}\n"
        r = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, check=False)
        return r.returncode, r.stdout + r.stderr

    def test_detect_sm_mapping_and_fallback(self):
        for cc, want in [("11.0", "110a"), ("12.1", "121"), ("12.0", "120"), (None, "121")]:
            with self.subTest(cc=cc):
                self.set_fake_nvidia_smi(cc)
                code, out = self.bash_config('echo "$SM|$TAG_BASE|$KIT_SUB"', {"SM": "auto"})
                self.assertEqual(code, 0, out)
                sm, tag_base, kit_sub = out.strip().split("|")
                self.assertEqual(sm, want)
                if want == "121":
                    self.assertEqual(tag_base, "exl3-tp2")
                    self.assertFalse(kit_sub.endswith("-sm121"))
                else:
                    self.assertEqual(tag_base, f"exl3-tp2-sm{want}")
                    self.assertTrue(kit_sub.endswith(f"-sm{want}"), kit_sub)

    def test_image_hash_and_prepared_state_depend_on_sm(self):
        self.set_fake_nvidia_smi("12.1")
        c1, o1 = self.bash_config(
            'worker_host() { echo w1; }; wval() { echo ""; }; image_ident() { echo img; }\n'
            'echo "$(image_hash)|$(prebuilt_image)|$(prepared_state)"',
            {"SM": "121"},
        )
        self.assertEqual(c1, 0, o1)
        h121, pb121, ps121 = o1.strip().split("|")

        c2, o2 = self.bash_config(
            'worker_host() { echo w1; }; wval() { echo ""; }; image_ident() { echo img; }\n'
            'echo "$(image_hash)|$(prebuilt_image)|$(prepared_state)"',
            {"SM": "110a"},
        )
        self.assertEqual(c2, 0, o2)
        h110, pb110, ps110 = o2.strip().split("|")

        self.assertNotEqual(h121, h110)
        self.assertIn(":exl3-tp2-", pb121)
        self.assertIn(":exl3-tp2-sm110a-", pb110)
        self.assertNotEqual(ps121, ps110)
        self.assertIn("sm=121", ps121)
        self.assertIn("sm=110a", ps110)

    def test_invalid_sm_rejected(self):
        self.set_fake_nvidia_smi("11.0")
        for bad in ("sm_110a", "110aa", "0", "110;rm"):
            with self.subTest(bad=bad):
                code, out = self.bash_config('echo "$SM"', {"SM": bad})
                self.assertNotEqual(code, 0, out)
                self.assertIn("SM must be", out)

    def test_dockerfile_and_prepare_wiring(self):
        df = (ROOT / "zig" / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("ARG SM=121", df)
        self.assertIn("LABEL tf.patches=${PATCHES_HASH} tf.sm=${SM}", df)
        # Serving build compiles kernels with -Dnvcc and -Dsm; CPU-only INPUTS_REF build omits -Dnvcc
        self.assertIn("/opt/zig/zig build -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=${SM}", df)
        inputs_part = df.split("ARG INPUTS_REF", 1)[1]
        self.assertNotIn("-Dnvcc", inputs_part)

        prep = (ROOT / "scripts" / "prepare.sh").read_text(encoding="utf-8")
        self.assertIn('--build-arg SM="$SM"', prep)
        self.assertIn("scripts/retarget_kit.py", prep)

        patch = (ROOT / "zig" / "patches" / "0069-cuda-sm-arch-suffix.patch").read_text(encoding="utf-8")
        self.assertIn('!std.mem.endsWith(u8, sm, "a")', patch)

    def test_gpu_args_and_cuda_compat_relaxation(self):
        # Simulate JetPack/Thor: docker info lists "nvidia" runtime, image wants cuda>=13.3, host driver is 13.2
        fake_docker = self.bin / "docker"
        fake_docker.write_text(
            '#!/bin/sh\n'
            'case "$1" in\n'
            '  info) echo \'{"nvidia":{"path":"nvidia-container-runtime"},"runc":{"path":"runc"}}\' ;;\n'
            '  image) echo \'["NVIDIA_REQUIRE_CUDA=cuda>=13.3 brand=unknown"]\' ;;\n'
            'esac\n',
            encoding="utf-8",
        )
        fake_docker.chmod(0o755)
        fake_smi = self.bin / "nvidia-smi"
        fake_smi.write_text(
            '#!/bin/sh\n'
            'case "$*" in\n'
            '  *compute_cap*) echo "11.0" ;;\n'
            '  *) echo "| NVIDIA-SMI 595.78  Driver Version: 595.78  CUDA Version: 13.2 |" ;;\n'
            'esac\n',
            encoding="utf-8",
        )
        fake_smi.chmod(0o755)

        code, out = self.bash_config('source ./scripts/nodes.sh\ngpu_args ""')
        self.assertEqual(code, 0, out)
        lines = out.strip().splitlines()
        self.assertEqual(lines[:3], ["--runtime=nvidia", "-e", "NVIDIA_VISIBLE_DEVICES=all"])
        self.assertIn("NVIDIA_REQUIRE_CUDA=cuda>=13.2 brand=unknown", lines)

        # When driver CUDA >= image requirement, NVIDIA_REQUIRE_CUDA is not overridden
        fake_smi.write_text(
            '#!/bin/sh\n'
            'case "$*" in\n'
            '  *compute_cap*) echo "12.1" ;;\n'
            '  *) echo "| NVIDIA-SMI 600.00  Driver Version: 600.00  CUDA Version: 13.3 |" ;;\n'
            'esac\n',
            encoding="utf-8",
        )
        fake_docker.write_text(
            '#!/bin/sh\n'
            'case "$1" in\n'
            '  info) echo \'{"runc":{"path":"runc"}}\' ;;\n'
            '  image) echo \'["NVIDIA_REQUIRE_CUDA=cuda>=13.3"]\' ;;\n'
            'esac\n',
            encoding="utf-8",
        )
        code, out = self.bash_config('source ./scripts/nodes.sh\ngpu_args ""')
        self.assertEqual(code, 0, out)
        self.assertEqual(out.strip().splitlines(), ["--gpus", "all"])

    def test_tcp_link_discovery_nccl_env_and_nfs_opts(self):
        self.set_fake_nvidia_smi("11.0")
        body = r'''
source ./scripts/nodes.sh
rdma_present() { return 1; }
fabric_pairs() {
  printf '%s\n' \
    'mgbe0_0 192.168.100.38 mgbe0_0 192.168.100.39 9000 9000/9000' \
    'mgbe1_0 192.168.101.38 mgbe1_0 192.168.101.39 9000 9000/9000' \
    'mgbe2_0 192.168.102.38 mgbe2_0 192.168.102.39 9000 9000/9000' \
    'mgbe3_0 192.168.103.38 mgbe3_0 192.168.103.39 9000 9000/9000'
}
rail_ping() { return 0; }
rail_tcp_buffers() { return 0; }
worker() {
  shift
  case "$*" in
    *rail_ping*) return 0 ;;
    *) echo "WORKER_CMD:$*" ;;
  esac
}
WORKER=test@192.168.100.39
detect_links
echo "KIND=$LINK_KIND RDMA=$RDMA MASTER=$MASTER_ADDR ZC=$LINK_ZC_PORTS/$WORKER_ZC_PORTS"
echo "NCCL0=$(rank_nccl_env 0)"
echo "IB0=$(infiniband_args '' | tr '\n' ' ')"
NFS_OPTS='vers=3,ro,rsize=1048576'
worker() { shift; [[ "$*" == *"volume inspect"* ]] && return 1; echo "VOL:$*" >&2; }
ensure_nfs_volume 1
'''
        code, out = self.bash_config(body)
        self.assertEqual(code, 0, out)
        self.assertIn("KIND=tcp RDMA=0 MASTER=192.168.100.38 ZC=0,1,2,3/0,1,2,3", out)
        self.assertIn("NCCL_SOCKET_IFNAME=mgbe0_0,mgbe1_0,mgbe2_0,mgbe3_0", out)
        self.assertIn("NCCL_IB_DISABLE=1", out)
        self.assertIn("NCCL_MIN_NCHANNELS=4", out)
        self.assertIn("NCCL_MAX_NCHANNELS=4", out)
        self.assertNotIn("NCCL_IB_HCA", out)
        self.assertIn("IB0=", out)
        self.assertNotIn("/dev/infiniband", out)
        self.assertIn("addr=192.168.100.38,vers=3,ro,rsize=1048576", out)

    def test_mgbe_zc_detection_args_and_ib_isolation(self):
        self.set_fake_nvidia_smi("11.0")
        # 1. LINK_KIND=ib leaves MGBE_ZC=auto resolved to off and emits no mgbe_zc_args
        body_ib = r'''
source ./scripts/nodes.sh
LINK_KIND=ib
detect_mgbe_zc
echo "MGBE_ZC=$MGBE_ZC|ARGS=$(mgbe_zc_args '' | tr '\n' ' ')"
'''
        code, out = self.bash_config(body_ib)
        self.assertEqual(code, 0, out)
        self.assertIn("MGBE_ZC=off|ARGS=", out)

        # 2. LINK_KIND=tcp with MGBE_ZC=off disables mgbe_zc even if device exists
        body_off = r'''
source ./scripts/nodes.sh
LINK_KIND=tcp; LINK_ZC_PORTS=0,1,2,3
detect_mgbe_zc
echo "MGBE_ZC=$MGBE_ZC|ARGS=$(mgbe_zc_args '' | tr '\n' ' ')"
'''
        code, out = self.bash_config(body_off, {"MGBE_ZC": "off"})
        self.assertEqual(code, 0, out)
        self.assertIn("MGBE_ZC=off|ARGS=", out)

        # 3. mgbe_zc_args emits --device /dev/mgbe_zc and per-node --group-add when MGBE_ZC=on
        body_on = r'''
source ./scripts/nodes.sh
MGBE_ZC=on; MGBE_ZC_GID_LOCAL=987; MGBE_ZC_GID_WORKER=988
echo "H=$(mgbe_zc_args '' | tr '\n' ' ')"
echo "W=$(mgbe_zc_args worker | tr '\n' ' ')"
'''
        code, out = self.bash_config(body_on)
        self.assertEqual(code, 0, out)
        self.assertIn("H=--device /dev/mgbe_zc --group-add 987", out)
        self.assertIn("W=--device /dev/mgbe_zc --group-add 988", out)

        # 4. start.sh wiring includes TF_MGBE_ZC, TF_MGBE_ZC_MAX_KB, TF_MGBE_ZC_PORTS, and mgbe_zc_args
        start = (ROOT / "start.sh").read_text(encoding="utf-8")
        self.assertIn("detect_mgbe_zc", start)
        self.assertIn("TF_MGBE_ZC=", start)
        self.assertIn("TF_MGBE_ZC_MAX_KB=", start)
        self.assertIn("TF_MGBE_ZC_PORTS=", start)
        self.assertIn("mgbe_zc_args", start)


if __name__ == "__main__":
    unittest.main()
