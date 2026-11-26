# The two Sparks: this machine (rank 0) and WORKER (rank 1, over ssh). The link between them is found from the routes and
# subnets between them (detect_link): one direct CX7 cable, its PCIe twin as a second rail when it reaches the peer.
# Sourced by start.sh, stop.sh and prepare.sh after config.sh.
# From the node helpers of MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold (Mia's AI Lab), cut to two Sparks.

# wvar <NAME> <i>: the name of worker i's own setting: NAME for worker 1 (WORKER, FABRIC_PEER, NFS_SERVER). The helpers
# keep the GLM scripts' per-worker form; this recipe has the one worker.
wvar() { if (( $2 == 1 )); then echo "$1"; else echo "$1$2"; fi; }
wval() { local v; v=$(wvar "$1" "$2"); echo "${!v:-}"; }
worker_host() { wval WORKER "$1"; }
# The workers this start uses (1 .. TP-1), and every worker configured at all (stop.sh stops them all).
worker_ids() { seq 1 $((TP - 1)); }
configured_workers() { local i; for i in 1 2 3; do [[ -z "$(worker_host "$i")" ]] || echo "$i"; done; }

# check_workers: this recipe serves two Sparks (the dsv41 family's TP=2), and WORKER is set.
check_workers() {
  [[ "$TP" == 2 ]] || die "TP is 2 (two Sparks): the dsv41 family runs tensor-parallel over two nodes here, not $TP"
  [[ -n "$(worker_host 1)" ]] || die "WORKER is not set: put WORKER=user@<address of the second Spark> in scripts/local.sh (see scripts/local.sh.example)"
}

# DRY_RUN=1 goes on without workers it cannot reach (a fictitious WORKER): its values print as <WORKER:...>.
declare -A WORKER_DOWN=()
# worker <i> <command...>: run a command on worker i (rank i). Key-based ssh only, no prompts.
worker() {
  local i=$1; shift
  [[ -z "${WORKER_DOWN[$i]:-}" ]] || return 255
  ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 "$(worker_host "$i")" "$@"
}

# save_log <dir> <rank> <container> <keep>: the container's log (stdout and stderr, with timestamps) gzipped into
# <dir>/<date>-<time>-rank<N>.log.gz before the container is removed (docker rm deletes it), keeping the newest <keep>
# of that rank; prints the file. Runs on the worker too (its definition is sent over ssh): no config.sh there.
save_log() {
  local dir=$1 rank=$2 c=$3 keep=$4 stamp f i=0
  [[ "$keep" =~ ^[0-9]+$ ]] && (( keep > 0 )) || return 0
  docker inspect "$c" >/dev/null 2>&1 || return 0
  mkdir -p "$dir" || return 1
  stamp=$(date +%Y%m%d-%H%M%S); f="$dir/$stamp-rank$rank.log.gz"
  while [[ -e "$f" ]]; do f="$dir/$stamp.$((++i))-rank$rank.log.gz"; done   # same second: still sorts as newer
  if ! docker logs --timestamps "$c" 2>&1 | gzip > "$f"; then rm -f "$f"; return 1; fi
  ls -1 "$dir"/*-rank"$rank".log.gz 2>/dev/null | { grep -vxF "$f" || true; } | LC_ALL=C sort -r | tail -n +"$keep" | xargs -r rm -f --
  echo "$f"
}
# worker_save_log <i>: the same, for rank i on worker i (into ~/.cache/dsv41-tensorfold/logs there)
worker_save_log() {
  worker "$1" "$(declare -f save_log); save_log \"\$HOME/.cache/dsv41-tensorfold/logs\" $1 '$CONTAINER_NAME' '$LOG_KEEP'"
}

need_worker() {
  local i=$1 v h
  v=$(wvar WORKER "$i"); h=$(worker_host "$i")
  [[ -n "$h" ]] || die "$v is not set: put $v=user@<worker address> in scripts/local.sh (see scripts/local.sh.example)"
  worker "$i" true 2>/dev/null && return 0
  if [[ "${DRY_RUN:-0}" == 1 ]]; then
    warn "DRY_RUN: cannot ssh to $h ($v): its values print as <$v:...>"
    WORKER_DOWN[$i]=1
    return 0
  fi
  die "cannot ssh to $h without a password: set up key-based ssh (ssh-copy-id $h)"
}
need_workers() { local i; for i in $(worker_ids); do need_worker "$i"; done; }

# Worker i's Hugging Face cache: its own HF_HOME, else ~/.cache/huggingface there. prepare.sh copies the checkpoint
# into it and start.sh mounts it into rank i. WORKER_HF_CACHE (worker 1) / WORKER_HF_CACHE<i> wins: a worker whose
# cache is not its HF_HOME (e.g. a shared models disk).
worker_hf_cache() {
  local c; c=$(wval WORKER_HF_CACHE "$1")
  if [[ -n "$c" ]]; then echo "$c"; return; fi
  [[ -z "${WORKER_DOWN[$1]:-}" ]] || { echo "<$(wvar WORKER "$1"):hf-cache>"; return 0; }
  worker "$1" 'echo "${HF_HOME:-$HOME/.cache/huggingface}"'
}

# Weights over NFS: rank i mounts the head's HF_CACHE read-only through the docker volume NFS_VOLUME on worker i, from
# nfs_server i: NFS_SERVER (worker 1) / NFS_SERVER<i>, else the head's address on that worker's link (detect_links).
# ensure_nfs_volume i creates it (or checks the one there names the same export); worker_nfs i <cmd...> runs a command
# in a throwaway container on worker i with the volume at /hf.
nfs_server() { local s; s=$(wval NFS_SERVER "$1"); echo "${s:-${LINK_HEAD_ADDR[$1]:-}}"; }
ensure_nfs_volume() {
  local i=$1 server have h
  server=$(nfs_server "$i"); h=$(worker_host "$i")
  [[ -n "$server" ]] || die "no NFS server address for $h: set $(wvar NFS_SERVER "$i") to the head's address on that worker's link"
  have=$(worker "$i" "docker volume inspect -f '{{index .Options \"device\"}} {{index .Options \"o\"}}' '$NFS_VOLUME'" 2>/dev/null || true)
  if [[ -z "$have" ]]; then
    worker "$i" "docker volume create --driver local --opt type=nfs --opt device=':$NFS_PATH' \
      --opt o='addr=$server,$NFS_OPTS' '$NFS_VOLUME'" >/dev/null ||
      die "could not create the docker volume $NFS_VOLUME on $h"
  elif [[ "${have%% *}" != ":$NFS_PATH" || "$have" != *"addr=$server,"* ]]; then
    die "$h's docker volume $NFS_VOLUME is ${have%% *} from ${have#* }, not :$NFS_PATH from $server: remove it there (docker volume rm $NFS_VOLUME) or set NFS_VOLUME / NFS_PATH / $(wvar NFS_SERVER "$i")"
  fi
}
worker_nfs() { local i=$1; shift; worker "$i" "docker run --rm --entrypoint '$1' -v '$NFS_VOLUME:/hf:ro' '$IMAGE' $(printf '%q ' "${@:2}")"; }
# Check the worker's actual file view, not just a successful mount or a prepared marker.
# Also works with the documented PREPARE=0 / host-path NFS_VOLUME setup: do not create a volume here.
worker_files_check() {
  local i=$1 out
  if out=$(worker_nfs "$i" sh -ec '
    for f in "$@"; do
      if [ ! -f "$f" ] || [ ! -r "$f" ] || [ ! -s "$f" ]; then
        echo "Missing, empty or unreadable on worker: $f" >&2
        exit 1
      fi
    done
  ' sh "$(model_in_container)/config.json" "/hf/$KIT_SUB/kit/MANIFEST" "/hf/$DATA_SUB/dsv41_token_map.json" 2>&1); then
    return 0
  fi
  printf '%s\n' "$out" >&2
  die "$(worker_host "$i") cannot read the model/kit through $NFS_VOLUME (NFS :$NFS_PATH from $(nfs_server "$i")). Check the actual NFSv4 root, stale file handles and competing host/container NFS servers (README: NFS); for a host-path mount check that worker's local files. No ranks were launched by this check"
}

# link_info <peer address>: this node's side of the link to <peer>: "<address> <netdev> <rdma device> <gid index>".
# The netdev and source address come from the route to the peer; the RDMA device from sysfs; the GID index is the
# RoCE v2 entry that holds this node's IPv4 address (an all-zero or v1 entry makes NCCL fail about a minute in).
link_info() {
  local peer=$1 route dev src hca idx type gid want
  route=$(ip -o -4 route get "$peer" 2>/dev/null) || return 1
  dev=$(sed -n 's/.* dev \([^ ]*\).*/\1/p' <<<"$route")
  src=$(sed -n 's/.* src \([^ ]*\).*/\1/p' <<<"$route")
  hca=$(ls /sys/class/net/"$dev"/device/infiniband 2>/dev/null | head -1)
  idx=""
  if [[ -n "$hca" ]]; then
    want=$(printf '0000:0000:0000:0000:0000:ffff:%02x%02x:%02x%02x' $(tr '.' ' ' <<<"$src"))
    for f in /sys/class/infiniband/"$hca"/ports/1/gids/*; do
      gid=$(cat "$f" 2>/dev/null) || continue
      [[ "$gid" == "$want" ]] || continue
      type=$(cat /sys/class/infiniband/"$hca"/ports/1/gid_attrs/types/"${f##*/}" 2>/dev/null)
      [[ "$type" == *"v2"* ]] && { idx=${f##*/}; break; }
    done
  fi
  echo "$src $dev ${hca:--} ${idx:--}"
}

# rails <netdev> <gid index>: the RoCE devices this node can run the link on - the link's own device first, then every
# other RoCE device whose RoCE v2 IPv4 GID sits at the same index: one on the link's subnet (a second CX7 port, when it
# is up and addressed there), and the second PCIe link of the same QSFP port (a DGX Spark's "twin", which lives in its
# own subnet, so the subnet scan below cannot see it).
rails() {
  local dev=$1 gid=$2 net cidr hcas other ip idx
  cidr=$(ip -o -4 addr show dev "$dev" | awk '{print $4}' | head -1)
  net=$(python3 -c "import ipaddress,sys; print(ipaddress.ip_interface(sys.argv[1]).network)" "$cidr")
  hcas=$(ls /sys/class/net/"$dev"/device/infiniband | head -1)
  for n in /sys/class/net/*; do
    other=${n##*/}
    [[ "$other" == "$dev" || ! -d $n/device/infiniband || "$(cat $n/operstate 2>/dev/null)" != up ]] && continue
    ip=$(ip -o -4 addr show dev "$other" | awk '{print $4}' | head -1)
    [[ -n "$ip" ]] || continue
    python3 -c "import ipaddress,sys; sys.exit(ipaddress.ip_interface(sys.argv[1]) not in ipaddress.ip_network(sys.argv[2]) and ipaddress.ip_interface(sys.argv[1]).ip not in ipaddress.ip_network(sys.argv[2]))" "$ip" "$net" || continue
    idx=$(ls $n/device/infiniband | head -1)
    [[ "$(cat /sys/class/infiniband/$idx/ports/1/gid_attrs/types/$gid 2>/dev/null)" == *v2* ]] || continue
    hcas+=",$idx"
  done
  # A DGX Spark's QSFP port reaches the GB10 over two independent PCIe Gen5 x4 links, so one cabled port shows up as
  # two netdevs and two RoCE devices (rocep1s0f0 and roceP2p1s0f0), and NVIDIA's own two-Spark playbook gives the two
  # twins different subnets - so the scan above never sees the twin and NCCL gets one x4: ~112 Gb/s of the port's 200.
  # Pair the twin by name instead of by subnet: same fN/npM tail, different PCIe prefix. Same conditions as above
  # (link up, a RoCE v2 GID at $gid), and never add a device twice.
  # (only for Spark-style names, <prefix>f<N>np<M>; the check below compares whole comma-separated names, so a twin the
  # subnet scan already added, as when both twins share the link's subnet, is not listed twice)
  local tail=${dev##*f} other2 ib2
  [[ "$dev" =~ f[0-9]+np[0-9]+$ ]] || { echo "$hcas"; return 0; }
  for n in /sys/class/net/*; do
    other2=${n##*/}
    [[ "$other2" == "$dev" || "$other2" != *f"$tail" ]] && continue
    [[ -d $n/device/infiniband && "$(cat $n/operstate 2>/dev/null)" == up ]] || continue
    ib2=$(ls $n/device/infiniband | head -1)
    [[ -n "$ib2" && ",$hcas," != *",$ib2,"* ]] || continue
    [[ "$(cat /sys/class/infiniband/$ib2/ports/1/gid_attrs/types/$gid 2>/dev/null)" == *v2* ]] || continue
    hcas+=",$ib2"
  done
  echo "$hcas"
}

# rail_info <rdma device>: "<netdev> <IPv4 address> <MTU>" of the port a RoCE device sits on ("- - -" when it has none).
# Self-contained: it runs on the workers too (its definition is sent over ssh).
rail_info() {
  local n
  for n in /sys/class/infiniband/"$1"/device/net/*; do
    [[ -d $n ]] || continue
    echo "${n##*/} $(ip -o -4 addr show dev "${n##*/}" | awk '{print $4}' | head -1 | cut -d/ -f1) $(cat "$n/mtu" 2>/dev/null || echo 1500)"
    return 0
  done
  echo "- - -"
}

# rail_ping <local netdev> <peer address> <MTU>: one rail's traffic reaches the peer's twin - a full-size, unfragmentable
# ping bound to that rail's own port (-I <netdev>: an address alone would be routed out of whichever port of the same
# subnet the kernel picks, usually the cabled one), so an unaddressed, differently-subnetted or smaller-MTU twin fails
# here instead of as vendor_err 0x81 (transport retry exceeded) in the first all-gather (issue #66). Self-contained.
rail_ping() { ping -c2 -W2 -I "$1" -M do -s $(( $3 - 28 )) "$2" >/dev/null 2>&1; }

# The same function on worker i (its definition is sent over ssh).
worker_link_info() { worker "$1" "$(declare -f link_info); link_info $2"; }
# An image's identity by content (its layers' diffIDs and runtime config), the same under Docker's overlay2 and
# containerd image stores: .Id is the config digest under one and the manifest digest under the other, so it never
# matches across a mixed pair (issue #8). The template holds no spaces: worker() passes it through ssh, which
# re-splits arguments. A missing image is "missing". worker_image_ident <i> <image>: on worker i.
IMAGE_IDENT='{{.RootFS.Layers}}{{.Config.Env}}{{.Config.Entrypoint}}{{.Config.Cmd}}{{.Config.WorkingDir}}'
image_ident() { local s; s=$(docker image inspect -f "$IMAGE_IDENT" "$1" 2>/dev/null) && sha256sum <<<"$s" | cut -c1-64 || echo missing; }
worker_image_ident() {
  local s; s=$(worker "$1" docker image inspect -f "$IMAGE_IDENT" "$2" 2>/dev/null) && sha256sum <<<"$s" | cut -c1-64 || echo missing
}
# cx7_peer <i>: worker i's address on a CX7 port that one of this node's CX7 ports reaches directly (same subnet, no
# gateway), for a worker given by its LAN address (issue #9). Empty when there is none.
cx7_peer() {
  local a route dev
  for a in $(worker "$1" 'for d in /sys/class/net/*; do [ -d "$d/device/infiniband" ] && ip -o -4 addr show dev "${d##*/}"; done' 2>/dev/null |
             awk '{print $4}' | cut -d/ -f1); do
    route=$(ip -o -4 route get "$a" 2>/dev/null) || continue
    [[ "$route" == *" via "* ]] && continue
    dev=$(sed -n 's/.* dev \([^ ]*\).*/\1/p' <<<"$route")
    [[ -d /sys/class/net/$dev/device/infiniband ]] && { echo "$a"; return 0; }
  done
  return 1
}

# probe_rails: before a second rail (the PCIe twin of the cabled port, or a second cabled port) is used, check that it
# reaches the peer. The RDMA ring pairs the two nodes' devices by list position, so a twin that is up and has a GID
# but no path to the peer's twin (no address, another subnet, another MTU, a cable on a different port) ends in vendor_err
# 0x81. Rail k of the head is probed against rail k of the worker, in both directions; a rail that fails is dropped from both
# lists with a warning (the first device, the link itself, is never probed: the launcher reached the worker over it).
probe_rails() {
  local -a h w keep_h=() keep_w=()
  local k hdev hip hmtu wdev wip wmtu mtu=0 ok
  IFS=, read -ra h <<<"$HEAD_HCAS"; IFS=, read -ra w <<<"$WORKER_HCAS"
  (( ${#h[@]} > 1 && ${#h[@]} == ${#w[@]} )) || return 0
  [[ -z "${WORKER_DOWN[1]:-}" ]] || { warn "DRY_RUN: the second rail ($HEAD_HCAS / $WORKER_HCAS) is not probed: the worker cannot be reached"; return 0; }
  keep_h=("${h[0]}"); keep_w=("${w[0]}")
  for (( k = 1; k < ${#h[@]}; k++ )); do
    read -r hdev hip hmtu <<<"$(rail_info "${h[k]}")"
    read -r wdev wip wmtu <<<"$(worker 1 "$(declare -f rail_info); rail_info ${w[k]}")"
    ok=0
    if [[ "$hip" =~ ^[0-9.]+$ && "$wip" =~ ^[0-9.]+$ && "$hmtu" =~ ^[0-9]+$ && "$wmtu" =~ ^[0-9]+$ ]]; then
      mtu=$(( hmtu < wmtu ? hmtu : wmtu ))
      if rail_ping "$hdev" "$wip" "$mtu" && worker 1 "$(declare -f rail_ping); rail_ping $wdev $hip $mtu"; then ok=1; fi
    fi
    if (( ok )); then
      keep_h+=("${h[k]}"); keep_w+=("${w[k]}")
      log "Rail ${h[k]} <-> ${w[k]}: $hip <-> $wip reaches the peer (MTU $mtu): used"
    else
      warn "second rail ${h[k]} (${hip:--}, MTU ${hmtu:--}) <-> ${w[k]} (${wip:--}, MTU ${wmtu:--}) does not reach the peer: NOT used, running on fewer rails. Give both twins an IPv4 address on one subnet per rail with the same MTU on both Sparks (README), or set NCCL_RAILS=1 to silence this"
    fi
  done
  HEAD_HCAS=$(IFS=,; echo "${keep_h[*]}"); WORKER_HCAS=$(IFS=,; echo "${keep_w[*]}")
}

# Non-RDMA Ethernet fabric helpers (Jetson AGX Thor 4x25GbE MGBE), adapted from GLM-5.3-Flash Thor launcher.
rdma_present() { compgen -G '/sys/class/infiniband/*' >/dev/null 2>&1; }

# node_addrs <min speed Mb/s>: this node's IPv4 addresses on UP physical netdevs at or above that speed. Self-contained.
node_addrs() {
  local min=${1:-0} d n a mtu spd
  for d in /sys/class/net/*; do
    n=${d##*/}
    [[ -e "$d/device" ]] || continue
    [[ "$(cat "$d/operstate" 2>/dev/null)" == up ]] || continue
    spd=$(cat "$d/speed" 2>/dev/null || echo 0)
    case "$spd" in ''|*[!0-9]*) spd=0 ;; esac
    (( spd >= min )) || continue
    mtu=$(cat "$d/mtu" 2>/dev/null || echo 1500)
    while read -r _ _ _ a _; do
      [[ -n "$a" ]] || continue
      printf '%s %s %s %s %s\n' "$n" "${a%%/*}" "${a##*/}" "$mtu" "$spd"
    done < <(ip -o -4 addr show dev "$n" 2>/dev/null)
  done
}

# fabric_pairs: direct subnet rail pairs between this node and worker 1 at or above FABRIC_MIN_SPEED_MBPS.
fabric_pairs() {
  local min=${FABRIC_MIN_SPEED_MBPS:-10000}
  { node_addrs "$min"; printf -- '--\n'
    worker 1 "$(declare -f node_addrs); node_addrs $min" 2>/dev/null || true
  } | python3 -c '
import ipaddress, sys
mine, theirs, side = [], [], 0
for line in sys.stdin:
    line = line.strip()
    if line == "--":
        side = 1; continue
    p = line.split()
    if len(p) != 5: continue
    (mine if side == 0 else theirs).append(p)
def net(a, pfx):
    try: return ipaddress.ip_network(f"{a}/{pfx}", strict=False)
    except ValueError: return None
def addr(a):
    try: return ipaddress.ip_address(a)
    except ValueError: return None
for dev, ip, pfx, mtu, _ in sorted(mine, key=lambda x: x[0]):
    ln, la = net(ip, pfx), addr(ip)
    if ln is None or la is None: continue
    for wdev, wip, wpfx, wmtu, _ in sorted(theirs, key=lambda x: x[0]):
        wn, wa = net(wip, wpfx), addr(wip)
        if wn is None or wa is None or la not in wn or wa not in ln: continue
        print(dev, ip, wdev, wip, min(int(mtu), int(wmtu)), f"{mtu}/{wmtu}")
'
}

tcp_buffer_ceiling() {
  sysctl -n net.core.rmem_max net.core.wmem_max 2>/dev/null | sort -n | head -1
}

rail_tcp_buffers() {
  local want=${RAIL_TCP_BUFFER_BYTES:-33554432} here there
  [[ -z "${WORKER_DOWN[1]:-}" ]] || return 0
  here=$(tcp_buffer_ceiling || true); here=${here:-0}
  there=$(worker 1 "$(declare -f tcp_buffer_ceiling); tcp_buffer_ceiling" 2>/dev/null || true); there=${there:-0}
  (( here >= want && there >= want )) && return 0
  warn "socket buffer ceiling is $here bytes here and $there on $(worker_host 1); high-speed Ethernet rails want $want on both (sudo sysctl -w net.core.rmem_max=$want net.core.wmem_max=$want)"
}

# GPU passthrough & CUDA floor compatibility (Jetson AGX Thor / JetPack uses --runtime=nvidia and may report an older
# driver CUDA minor than the PyTorch base image's NVIDIA_REQUIRE_CUDA).
gpu_mode() {
  local where=${1:-} runtimes
  if [[ "$where" == worker ]]; then
    [[ -z "${WORKER_DOWN[1]:-}" ]] && runtimes=$(worker 1 "docker info --format '{{json .Runtimes}}'" 2>/dev/null || true) || runtimes=""
  else
    runtimes=$(docker info --format '{{json .Runtimes}}' 2>/dev/null || true)
  fi
  case "${GPU_RUNTIME:-auto}" in
    runtime) echo runtime ;;
    gpus) echo gpus ;;
    *) [[ "$runtimes" == *'"nvidia"'* ]] && echo runtime || echo gpus ;;
  esac
}

driver_cuda() {
  local s
  if [[ "$1" == worker ]]; then
    [[ -z "${WORKER_DOWN[1]:-}" ]] && s=$(worker 1 nvidia-smi 2>/dev/null || true) || s=""
  else
    s=$(nvidia-smi 2>/dev/null || true)
  fi
  sed -n 's/.*CUDA Version: *\([0-9][0-9.]*\).*/\1/p' <<<"$s" | head -1
}

image_cuda_require() {
  local env
  if [[ "$1" == worker ]]; then
    [[ -z "${WORKER_DOWN[1]:-}" ]] && env=$(worker 1 "docker image inspect -f '{{json .Config.Env}}' '$IMAGE'" 2>/dev/null || true) || env=""
  else
    env=$(docker image inspect -f '{{json .Config.Env}}' "$IMAGE" 2>/dev/null || true)
  fi
  python3 -c 'import json, sys
try: env = json.loads(sys.argv[1]) or []
except Exception: sys.exit(0)
print(next((e.split("=", 1)[1] for e in env if e.startswith("NVIDIA_REQUIRE_CUDA=")), ""))' "$env"
}

cuda_compat_args() {
  local req drv out
  req=$(image_cuda_require "$1"); drv=$(driver_cuda "$1")
  [[ -n "$req" && -n "$drv" ]] || return 0
  out=$(python3 - "$req" "$drv" <<'PY'
import re, sys
req, drv = sys.argv[1], sys.argv[2]
m = re.search(r"cuda>=([0-9]+(?:\.[0-9]+)?)", req)
if not m: sys.exit(0)
key = lambda v: tuple(int(x) for x in v.split("."))
if key(drv) >= key(m.group(1)): sys.exit(0)
print(re.sub(r"cuda>=[0-9.]+", "cuda>=" + drv, req, count=1))
PY
  ) || true
  [[ -z "$out" ]] || printf '%s\n' -e "NVIDIA_REQUIRE_CUDA=$out"
}

gpu_args() {
  case "$(gpu_mode "$1")" in
    runtime) printf '%s\n' --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all ;;
    *) printf '%s\n' --gpus all ;;
  esac
  cuda_compat_args "$1"
}

infiniband_args() {
  [[ "${LINK_KIND:-ib}" != tcp ]] || return 0
  if [[ "$1" == worker ]]; then
    [[ -n "${WORKER_DOWN[1]:-}" ]] || worker 1 'test -e /dev/infiniband' 2>/dev/null || return 0
  else
    [[ -e /dev/infiniband ]] || return 0
  fi
  printf '%s\n' --device /dev/infiniband
}

MGBE_ZC_GID_LOCAL=""
MGBE_ZC_GID_WORKER=""

# detect_mgbe_zc: resolve MGBE_ZC (auto|on|off) after detect_links and discover the /dev/mgbe_zc udev group GID on both nodes.
detect_mgbe_zc() {
  local here there ports
  MGBE_ZC_GID_LOCAL=""; MGBE_ZC_GID_WORKER=""
  if [[ "${LINK_KIND:-ib}" != tcp ]]; then
    [[ "$MGBE_ZC" != on ]] || die "MGBE_ZC=on requires an Ethernet/MGBE link (LINK_KIND=tcp), not ${LINK_KIND:-ib}"
    MGBE_ZC=off
    return 0
  fi
  here=$([[ -c /dev/mgbe_zc ]] && echo yes || echo no)
  if [[ -n "${WORKER_DOWN[1]:-}" ]]; then
    there=$here
  else
    there=$(worker 1 'test -c /dev/mgbe_zc' >/dev/null 2>&1 && echo yes || echo no)
  fi
  ports="${MGBE_ZC_PORTS:-$LINK_ZC_PORTS}"
  if [[ "$MGBE_ZC" == auto ]]; then
    if [[ "$here" == yes && "$there" == yes && -n "$ports" ]]; then MGBE_ZC=on; else MGBE_ZC=off; fi
  fi
  if [[ "$MGBE_ZC" == on ]]; then
    [[ "$here" == yes && "$there" == yes ]] || die "MGBE_ZC=on requires /dev/mgbe_zc on both nodes (here=$here, worker=$there). Load mgbe_zc.ko on both hosts or set MGBE_ZC=off"
    [[ -n "$ports" ]] || die "MGBE_ZC=on requires at least one MTU>=8256 MGBE rail (set MGBE_ZC_PORTS or raise MTU to 9000)"
    MGBE_ZC_GID_LOCAL=$(getent group "$MGBE_ZC_GROUP" 2>/dev/null | cut -d: -f3 || true)
    if [[ -n "${WORKER_DOWN[1]:-}" ]]; then
      MGBE_ZC_GID_WORKER=$MGBE_ZC_GID_LOCAL
    else
      MGBE_ZC_GID_WORKER=$(worker 1 "getent group $(printf '%q' "$MGBE_ZC_GROUP") 2>/dev/null | cut -d: -f3" 2>/dev/null || true)
    fi
  fi
}

mgbe_zc_args() {
  [[ "${MGBE_ZC:-off}" == on ]] || return 0
  local gid
  if [[ "$1" == worker ]]; then gid=${MGBE_ZC_GID_WORKER:-}; else gid=${MGBE_ZC_GID_LOCAL:-}; fi
  printf '%s\n' --device /dev/mgbe_zc
  [[ -z "$gid" ]] || printf '%s\n' --group-add "$gid"
}

LINK_KIND=""
LINK_ZC_PORTS=""
WORKER_ZC_PORTS=""
HEAD_TCP_RAILS=0

detect_link_tcp() {
  local -a pairs=() keep=() zc_local=() zc_worker=()
  local rail dev ip wdev wip mtu mtus ok first=1 filter_peer=${FABRIC_PEER:-}
  if [[ -n "$filter_peer" && ! "$filter_peer" =~ ^[0-9]+(\.[0-9]+){3}$ ]]; then
    filter_peer=$(getent ahostsv4 "$filter_peer" | awk 'NR == 1 {print $1}') || true
    [[ -n "$filter_peer" ]] || die "cannot resolve FABRIC_PEER=$FABRIC_PEER to an IPv4 address"
  fi
  if [[ -n "${WORKER_DOWN[1]:-}" ]]; then
    LINK_KIND=tcp
    HEAD_ADDR=${MASTER_ADDR:-"<head:tcp-addr>"}; WORKER_ADDR="<WORKER:tcp-addr>"
    HEAD_DEV=${SOCKET_IFNAME:-"mgbe0_0,mgbe1_0,mgbe2_0,mgbe3_0"}; WORKER_DEV="$HEAD_DEV"
    HEAD_HCAS=""; WORKER_HCAS=""; HEAD_GID=0; WORKER_GID=0; HEAD_TCP_RAILS=4
    LINK_ZC_PORTS="0,1,2,3"; WORKER_ZC_PORTS="0,1,2,3"
    NODE_DEV=("$HEAD_DEV" "$WORKER_DEV"); NODE_HCAS=("" ""); NODE_GID=(0 0)
    LINK_HEAD_ADDR[1]=$HEAD_ADDR; LINK_WORKER_ADDR[1]=$WORKER_ADDR
    MASTER_ADDR=${MASTER_ADDR:-$HEAD_ADDR}
    [[ "$RDMA" != 1 ]] || warn "RDMA=1 but no RDMA verbs device is present: using RDMA=0 (NCCL Socket TCP)"
    RDMA=0
    return 0
  fi
  mapfile -t pairs < <(fabric_pairs)
  (( ${#pairs[@]} )) || die "no RDMA verbs device and no direct Ethernet rail >= ${FABRIC_MIN_SPEED_MBPS:-10000} Mb/s found to $(worker_host 1). Cable the nodes on direct point-to-point subnets or lower FABRIC_MIN_SPEED_MBPS"
  for rail in "${pairs[@]}"; do
    read -r dev ip wdev wip mtu mtus <<<"$rail"
    [[ -z "$filter_peer" || "$wip" == "$filter_peer" ]] || continue
    [[ -z "$mtus" || "${mtus%%/*}" == "${mtus##*/}" ]] ||
      warn "rail $dev ($ip) is MTU ${mtus%%/*} and $wdev ($wip) on $(worker_host 1) is MTU ${mtus##*/}: probing at $mtu"
    ok=1
    rail_ping "$dev" "$wip" "$mtu" || ok=0
    (( ok == 0 )) || worker 1 "$(declare -f rail_ping); rail_ping $(printf '%q' "$wdev") $(printf '%q' "$ip") $mtu" || ok=0
    if (( ok )); then
      keep+=("$dev $ip $wdev $wip $mtu")
      if (( first )); then
        HEAD_ADDR=$ip; WORKER_ADDR=$wip; first=0
      fi
      if [[ "$dev" =~ ^mgbe([0-3])_0$ && "$wdev" =~ ^mgbe([0-3])_0$ && "$mtu" -ge 8256 ]]; then
        zc_local+=("${BASH_REMATCH[1]}")
        zc_worker+=("${wdev#mgbe}"); zc_worker[-1]="${zc_worker[-1]%_0}"
      fi
    else
      warn "rail $dev ($ip) <-> $wdev ($wip) on $(worker_host 1) does not carry a $mtu-byte ping: leaving it out"
    fi
  done
  (( ${#keep[@]} )) || die "no Ethernet rail between this node and $(worker_host 1) carries a full-size ping. Check cables, addresses and MTU"
  LINK_KIND=tcp
  HEAD_TCP_RAILS=${#keep[@]}
  HEAD_HCAS=""; WORKER_HCAS=""; HEAD_GID=0; WORKER_GID=0
  LINK_ZC_PORTS=$(IFS=,; echo "${zc_local[*]}")
  WORKER_ZC_PORTS=$(IFS=,; echo "${zc_worker[*]}")
  if [[ -n "${SOCKET_IFNAME:-}" ]]; then
    HEAD_DEV=$SOCKET_IFNAME
    WORKER_DEV=$(printf '%s\n' "${keep[@]}" | awk -v d="$SOCKET_IFNAME" '$1 == d { print $3 }' | paste -sd, -)
    [[ -n "$WORKER_DEV" ]] || WORKER_DEV=$(printf '%s\n' "${keep[@]}" | awk '{ print $3 }' | paste -sd, -)
  else
    HEAD_DEV=$(printf '%s\n' "${keep[@]}" | awk '{ print $1 }' | paste -sd, -)
    WORKER_DEV=$(printf '%s\n' "${keep[@]}" | awk '{ print $3 }' | paste -sd, -)
  fi
  [[ "$RDMA" != 1 ]] || warn "RDMA=1 but there is no RDMA verbs device on both nodes: using RDMA=0 (NCCL Socket TCP)"
  RDMA=0
  NODE_DEV=("$HEAD_DEV" "$WORKER_DEV"); NODE_HCAS=("" ""); NODE_GID=(0 0)
  LINK_HEAD_ADDR[1]=$HEAD_ADDR; LINK_WORKER_ADDR[1]=$WORKER_ADDR
  MASTER_ADDR=${MASTER_ADDR:-$HEAD_ADDR}
  rail_tcp_buffers
}

# TP=2: HEAD_ADDR / HEAD_DEV / HEAD_HCA / HEAD_GID and WORKER_ADDR / WORKER_DEV / WORKER_HCA / WORKER_GID: the link the
# two ranks talk over (NCCL and the rendezvous). FABRIC_PEER overrides the worker's link address when WORKER is reached
# over another network. On nodes without RDMA verbs (Jetson AGX Thor), falls back to multi-rail Ethernet discovery.
detect_link() {
  if ! rdma_present || { [[ -z "${WORKER_DOWN[1]:-}" ]] && ! worker 1 'ls -d /sys/class/infiniband/* >/dev/null 2>&1'; }; then
    detect_link_tcp
    return 0
  fi
  LINK_KIND=ib
  local peer=${FABRIC_PEER:-${WORKER#*@}} cx7
  # a host name (an /etc/hosts alias of the CX7 address, say) is resolved first: ``ip route get`` takes addresses only
  if [[ ! "$peer" =~ ^[0-9]+(\.[0-9]+){3}$ ]]; then
    peer=$(getent ahostsv4 "$peer" | awk 'NR == 1 {print $1}') || true
    [[ -n "$peer" ]] || die "cannot resolve ${FABRIC_PEER:-${WORKER#*@}} to an IPv4 address"
  fi
  read -r HEAD_ADDR HEAD_DEV HEAD_HCA HEAD_GID <<<"$(link_info "$peer")" || true
  [[ -n "${HEAD_ADDR:-}" ]] || die "no route from this node to $peer"
  # WORKER given by a LAN address (the route goes out a port without RoCE): use the worker's CX7 address instead
  if [[ -z "${FABRIC_PEER:-}" && "${HEAD_HCA:--}" == "-" ]] && cx7=$(cx7_peer 1); then
    log "$peer is reached over $HEAD_DEV, not a CX7 port: using the worker's CX7 address $cx7 for the link (FABRIC_PEER)"
    peer=$cx7
    read -r HEAD_ADDR HEAD_DEV HEAD_HCA HEAD_GID <<<"$(link_info "$peer")" || true
  fi
  read -r WORKER_ADDR WORKER_DEV WORKER_HCA WORKER_GID <<<"$(worker_link_info 1 "$HEAD_ADDR")" || true
  [[ -n "${WORKER_ADDR:-}" ]] || die "the worker has no route back to $HEAD_ADDR"
  # every rail (both PCIe twins of the cabled port, a second cabled port): a chunk's all-gather is ~1.8x faster on two
  HEAD_HCAS=$(rails "$HEAD_DEV" "$HEAD_GID")
  WORKER_HCAS=$(worker 1 "$(declare -f rails); rails $WORKER_DEV $WORKER_GID")
  [[ "${NCCL_RAILS:-2}" == 1 ]] && { HEAD_HCAS=$HEAD_HCA; WORKER_HCAS=$WORKER_HCA; }
  [[ "${NCCL_RAILS:-2}" == 1 ]] || probe_rails
  [[ "$(tr ',' '\n' <<<"$HEAD_HCAS" | wc -l)" == "$(tr ',' '\n' <<<"$WORKER_HCAS" | wc -l)" ]] ||
    { HEAD_HCAS=$HEAD_HCA; WORKER_HCAS=$WORKER_HCA; }
  for v in HEAD_HCA HEAD_GID WORKER_HCA WORKER_GID; do
    [[ "${!v}" != "-" ]] || die "no RoCE device or RoCE v2 GID for the link ($v): is $HEAD_DEV / $WORKER_DEV the CX7 port between the Sparks? Set FABRIC_PEER to the worker's CX7 address"
  done
  [[ "$RDMA" != auto ]] || RDMA=1
  NODE_DEV=("${SOCKET_IFNAME:-$HEAD_DEV}" "${SOCKET_IFNAME:-$WORKER_DEV}"); NODE_HCAS=("$HEAD_HCAS" "$WORKER_HCAS"); NODE_GID=("$HEAD_GID" "$WORKER_GID")
  LINK_HEAD_ADDR[1]=$HEAD_ADDR; LINK_WORKER_ADDR[1]=$WORKER_ADDR
  MASTER_ADDR=${MASTER_ADDR:-$HEAD_ADDR}
}

# The NCCL settings of rank r's container (docker -e arguments): the RoCE link's netdev, RoCE devices (both rails) and GID
# index, as the dsv41 family's two-node README runs them (NCCL_SOCKET_IFNAME, NCCL_IB_HCA, NCCL_IB_GID_INDEX).
# Every other exported NCCL_* variable goes to both ranks unchanged (an A/B of NCCL settings: NCCL_NET_PLUGIN=none,
# NCCL_MAX_NCHANNELS=4, NCCL_IB_QPS_PER_CONNECTION=2, NCCL_DEBUG_SUBSYS=INIT,NET ...); none set: no change. The names and
# values are checked here, in the main shell, before anything is stopped: a value is [A-Za-z0-9_.,:=/-] only (no spaces,
# quotes, $, ;, newlines), because it ends up in a command that the worker's shell runs. NCCL_SOCKET_IFNAME, NCCL_IB_HCA
# and NCCL_IB_GID_INDEX are this script's own (from the link it finds); NCCL_RAILS is its setting, not NCCL's.
NCCL_EXTRA_ENV=()
NCCL_ARGS=()
nccl_extra_collect() {
  local name val
  NCCL_EXTRA_ENV=()
  while IFS= read -r name; do
    [[ "$name" =~ ^NCCL_[A-Za-z0-9_]+$ ]] || continue
    val=${!name}
    case "$name" in
      NCCL_RAILS) continue ;;
      NCCL_SOCKET_IFNAME|NCCL_IB_HCA|NCCL_IB_GID_INDEX) warn "$name is set by this script (from the link it finds): your value is ignored"; continue ;;
    esac
    if [[ -z "$val" ]]; then warn "$name is empty: not passed on"; continue; fi
    [[ "$val" =~ ^[A-Za-z0-9_.,:=/-]+$ ]] || die "$name has a value with characters outside A-Z a-z 0-9 _ . , : = / - (it goes into a shell command on the worker): not passed on, nothing started"
    [[ "$name" == NCCL_DEBUG ]] || NCCL_EXTRA_ENV+=("$name=$val")
  done < <(compgen -e | LC_ALL=C sort | grep -E '^NCCL_' || true)
}

nccl_env() {
  local dev=$1 hcas=$2 gid=$3
  if [[ "${LINK_KIND:-ib}" == tcp ]]; then
    local nch=${NCCL_CHANNELS:-${HEAD_TCP_RAILS:-4}} has_min=0 has_max=0 x
    NCCL_ARGS=(-e "NCCL_SOCKET_IFNAME=$dev" -e "NCCL_IB_DISABLE=1")
    for x in "${NCCL_EXTRA_ENV[@]}"; do
      [[ "$x" != NCCL_MIN_NCHANNELS=* ]] || has_min=1
      [[ "$x" != NCCL_MAX_NCHANNELS=* ]] || has_max=1
    done
    (( has_min )) || NCCL_ARGS+=(-e "NCCL_MIN_NCHANNELS=$nch")
    (( has_max )) || NCCL_ARGS+=(-e "NCCL_MAX_NCHANNELS=$nch")
  else
    NCCL_ARGS=(-e "NCCL_SOCKET_IFNAME=$dev" -e "NCCL_IB_HCA=$hcas" -e "NCCL_IB_GID_INDEX=$gid")
  fi
  [[ -z "${NCCL_DEBUG:-}" ]] || NCCL_ARGS+=(-e "NCCL_DEBUG=$NCCL_DEBUG")
  local x; for x in "${NCCL_EXTRA_ENV[@]}"; do NCCL_ARGS+=(-e "$x"); done
  # the same arguments as one line for a remote shell: any word with a character outside [A-Za-z0-9_.,:=/@%+-] is
  # shell-quoted (printf %q), so a value can never be taken for shell syntax there (the values nccl_extra_collect lets
  # through need no quoting, which keeps the line as it always was)
  local out="" w
  for w in "${NCCL_ARGS[@]}"; do
    if [[ "$w" =~ ^[A-Za-z0-9_.,:=/@%+-]+$ ]]; then out+="$w "; else out+="$(printf '%q' "$w") "; fi
  done
  echo "${out% }"
}
rank_nccl_env() { nccl_env "${NODE_DEV[$1]}" "${NODE_HCAS[$1]}" "${NODE_GID[$1]}"; }
# rank_nccl_args <r>: the same as the array NCCL_ARGS (docker arguments), for a command run here without a shell between
rank_nccl_args() { nccl_env "${NODE_DEV[$1]}" "${NODE_HCAS[$1]}" "${NODE_GID[$1]}" >/dev/null; }

# detect_links: the one link of two Sparks (start.sh and prepare.sh call it; the GLM scripts' name for it at any TP).
detect_links() { detect_link; }

# Rank 0's checkpoint volume (HEAD_NFS_SERVER): like ensure_nfs_volume, for this Spark. Prints what rank 0 mounts at /hf.
head_hf_mount() {
  [[ -n "$HEAD_NFS_SERVER" ]] || { echo "$HF_CACHE"; return 0; }
  local have
  have=$(docker volume inspect -f '{{index .Options "device"}} {{index .Options "o"}}' "$HEAD_NFS_VOLUME" 2>/dev/null || true)
  if [[ -z "$have" ]]; then
    docker volume create --driver local --opt type=nfs --opt device=":$NFS_PATH" \
      --opt o="addr=$HEAD_NFS_SERVER,$NFS_OPTS" "$HEAD_NFS_VOLUME" >/dev/null ||
      die "could not create the docker volume $HEAD_NFS_VOLUME here"
  elif [[ "${have%% *}" != ":$NFS_PATH" || "$have" != *"addr=$HEAD_NFS_SERVER,"* ]]; then
    die "the docker volume $HEAD_NFS_VOLUME here is ${have%% *} from ${have#* }, not :$NFS_PATH from $HEAD_NFS_SERVER: remove it (docker volume rm $HEAD_NFS_VOLUME) or set HEAD_NFS_VOLUME / HEAD_NFS_SERVER"
  fi
  echo "$HEAD_NFS_VOLUME"
}

# mem_available_gib: MemAvailable here, in GiB (one decimal)
# mem_available_gib: what start.sh's MIN_AVAIL_GIB check counts here, MemAvailable plus the pages parked on the per-CPU
# free lists (mem_counting below; plain MemAvailable if /proc/zoneinfo cannot be read)
mem_available_gib() { local m; m=$(mem_counting 2>/dev/null) && echo "${m%% *}" || awk '/^MemAvailable:/ {printf "%.1f", $2 / 1048576}' /proc/meminfo; }
# mem_counting: MemAvailable plus the pages parked on the per-CPU free lists, as the engine's startup budget counts it
# (scripts/memavail.sh; start.sh's MIN_AVAIL_GIB check)
source "${_cfg_root:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/scripts/memavail.sh"

# await_memory: sets HERE_GB and WORKER_GB (GiB) to the memory available on this Spark and on worker 1, once both reach
# MIN_AVAIL_GIB or after MIN_AVAIL_WAIT_S seconds at the latest (returns 1 then; the caller words the refusal). A build or a large
# copy that has just ended leaves page cache that MemAvailable counts only as it is written back (111.6 GiB was read a few
# seconds after a build, 119 GiB once it settled), so one reading right after it refuses a start the Spark can take.
await_memory() {
  local waited=0
  while :; do
    HERE_GB=$(mem_available_gib)
    WORKER_GB=$(worker 1 "$(declare -f mem_counting mem_available_gib); mem_available_gib" 2>/dev/null || echo 0)
    awk -v a="$HERE_GB" -v b="$WORKER_GB" -v m="$MIN_AVAIL_GIB" 'BEGIN {exit !(a >= m && b >= m)}' && return 0
    (( waited < MIN_AVAIL_WAIT_S )) || return 1
    sleep 2
    waited=$((waited + 2))
  done
}

# The memory guard (scripts/memguard.sh) on this Spark and on the worker, which gets a copy of the script.
MEMGUARD_REMOTE_DIR='$HOME/.cache/dsv41-tensorfold'
memguard_start() {
  awk -v g="$MEMGUARD_GIB" 'BEGIN {exit !(g > 0)}' || return 0
  "$_cfg_root/scripts/memguard.sh" start "$CONTAINER_NAME" "$MEMGUARD_GIB" "$LOG_DIR/memguard.log" "$STATE_DIR/memguard.pid" "$MEMGUARD_HOLD_S" "$MEMGUARD_HARD_GIB" ||
    die "the memory guard did not start here"
  local i
  for i in $(worker_ids); do
    worker "$i" "mkdir -p $MEMGUARD_REMOTE_DIR/logs && cat > $MEMGUARD_REMOTE_DIR/memguard.sh && chmod +x $MEMGUARD_REMOTE_DIR/memguard.sh" \
      < "$_cfg_root/scripts/memguard.sh" || die "could not copy the memory guard to $(worker_host "$i")"
    worker "$i" "$MEMGUARD_REMOTE_DIR/memguard.sh start '$CONTAINER_NAME' $MEMGUARD_GIB $MEMGUARD_REMOTE_DIR/logs/memguard.log $MEMGUARD_REMOTE_DIR/memguard.pid $MEMGUARD_HOLD_S $MEMGUARD_HARD_GIB" ||
      die "the memory guard did not start on $(worker_host "$i")"
  done
  log "Memory guard on: each rank's container is killed when the memory available stays under $MEMGUARD_GIB GiB for $MEMGUARD_HOLD_S s, or falls under $MEMGUARD_HARD_GIB GiB (log: memguard.log)"
}
memguard_stop() {
  local i
  "$_cfg_root/scripts/memguard.sh" stop "$STATE_DIR/memguard.pid" 2>/dev/null || true
  for i in $(configured_workers); do
    worker "$i" "[ -x $MEMGUARD_REMOTE_DIR/memguard.sh ] && $MEMGUARD_REMOTE_DIR/memguard.sh stop $MEMGUARD_REMOTE_DIR/memguard.pid; true" 2>/dev/null || true
  done
}
