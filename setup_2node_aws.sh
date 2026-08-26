#!/bin/bash
###############################################################################
# setup_2node_aws.sh — turn two fresh EC2 GPU instances into a working X-MoE-4D
#                      multi-node cluster (non-SLURM / torchrun path).
#
# Companion to setup_env_cuda.sh, which builds the software env on ONE node.
# This script wires TWO nodes together: ssh, a shared filesystem, env.sh,
# hostfile, and a 16-rank NCCL smoke test.
#
#   RUN IT ON NODE 1 ONLY.  Node 2 needs nothing but sshd and a matching AMI.
#
# USAGE
#   ./setup_2node_aws.sh preflight  <NODE2_IP>   # check before changing anything
#   ./setup_2node_aws.sh wire       <NODE2_IP>   # preflight+sshkey+nfs+hostfile.
#                                                # Needs NO conda env -- run this FIRST,
#                                                # before setup_env_cuda.sh, so the env
#                                                # you build lands on the shared mount.
#   ./setup_2node_aws.sh sshkey     <NODE2_IP>   # keygen + self-auth, print pubkey
#   ./setup_2node_aws.sh nfs        <NODE2_IP>   # export ~/xmoe-4d, mount on node 2
#   ./setup_2node_aws.sh envfile    <NODE2_IP>   # write scripts/env.sh (auto-detects iface + EFA)
#   ./setup_2node_aws.sh hostfile   <NODE2_IP>   # write scripts/hostfile
#   ./setup_2node_aws.sh smoke      <NODE2_IP>   # 16-rank NCCL all-reduce
#   ./setup_2node_aws.sh all        <NODE2_IP>   # every stage above, in order
#
# RECOMMENDED ORDER ON A FRESH PAIR OF INSTANCES
#   1. security-group rule (console -- see below)
#   2. ./setup_2node_aws.sh wire <NODE2_IP>      <- no env needed; ~2 min
#   3. ./setup_env_cuda.sh                       <- ~40 min, lands on the shared mount
#   4. bash .../examples_xmoe_4d/data/prepare_data_ae.sh
#   5. ./setup_2node_aws.sh all <NODE2_IP>       <- env.sh + smoke test (rest is idempotent)
#
#   NODE2_IP must be the PRIVATE ip. If you only have the public DNS name, resolve
#   it FROM INSIDE the VPC and you get the private address back:
#       getent hosts ec2-1-2-3-4.us-east-2.compute.amazonaws.com
#
# WHAT YOU MUST DO BY HAND FIRST (needs the AWS console; this script cannot):
#   Security group: add an inbound rule  All TCP, ports 0-65535, source = THE SAME
#   security group. torchrun rendezvous uses 29500 but NCCL grabs ephemeral ports,
#   so per-port rules do not work. `preflight` verifies this and tells you if it is
#   missing. If you later enable EFA, widen it to "All traffic" — EFA OS-bypass
#   requires all protocols to/from the SG itself, not just TCP.
#
# ASSUMPTIONS
#   * Both nodes: same AMI, same GPU driver, same user, same instance type.
#   * setup_env_cuda.sh has already been run ON NODE 1 (conda env exists).
#   * Node 2 needs NO install — the NFS mount gives it the identical env.
#
# Every stage is idempotent: re-running is safe.
###############################################################################
set -uo pipefail

C_HEAD=$'\033[1;36m'; C_OK=$'\033[0;32m'; C_WARN=$'\033[0;33m'; C_ERR=$'\033[0;31m'; C_OFF=$'\033[0m'
banner() { printf "\n${C_HEAD}==== [%s] %s ====${C_OFF}\n" "$(date '+%H:%M:%S')" "$*"; }
ok()   { printf "${C_OK}[ok]${C_OFF} %s\n" "$*"; }
info() { printf "     %s\n" "$*"; }
warn() { printf "${C_WARN}[warn]${C_OFF} %s\n" "$*" >&2; }
die()  { printf "${C_ERR}[FAILED]${C_OFF} %s\n" "$*" >&2; exit 1; }

STAGE="${1:-}"; NODE2="${2:-}"
[ -z "$STAGE" ] && { sed -n '2,45p' "$0" | sed 's/^#//; s/^ //'; exit 1; }
case "$STAGE" in -h|--help|help) sed -n '2,45p' "$0" | sed 's/^#//; s/^ //'; exit 0 ;; esac
[ -z "$NODE2" ] && die "need node 2's PRIVATE ip:  ./setup_2node_aws.sh $STAGE <NODE2_IP>"

# --- paths -------------------------------------------------------------------
XMOE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XMOE4D_ROOT="${XMOE4D_ROOT:-$(cd "$XMOE_ROOT/.." && pwd)}"   # the dir we NFS-export
SCRIPTS="$XMOE_ROOT/Megatron-DeepSpeed-X-MoE/examples_xmoe_4d/scripts"
CONDA_ROOT="${CONDA_ROOT:-$XMOE4D_ROOT/miniforge3}"
ENV_PREFIX="${ENV_PREFIX:-$XMOE4D_ROOT/XMoE4D_envs/X-MoE-4D-CUDA12.8_repro}"
SSH_OPTS="-o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10"

NODE1="$(hostname -I | awk '{print $1}')"

# The interface NCCL should use for coordination. Pick the first UP, non-loopback,
# non-docker, non-virtual device -- hardcoding "ens32" breaks on other instance
# families (eth0, ens5, enp...), and env.sh is SHARED, so a wrong name silently
# breaks NCCL on whichever node disagrees.
detect_iface() {
    ip -br addr 2>/dev/null | awk '$2=="UP"{print $1}' \
        | grep -vE '^(lo|docker|veth|br-|virbr)' | head -1
}

# --------------------------------------------------------------------- stages
stage_preflight() {
    banner "preflight: node1=$NODE1  node2=$NODE2"
    local fail=0

    [ -d "$ENV_PREFIX" ] || { warn "conda env missing at $ENV_PREFIX — run ./setup_env_cuda.sh first"; fail=1; }
    if [ -d "$ENV_PREFIX" ] && [ ! -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then
        warn "env exists but there is no conda at $CONDA_ROOT."
        warn "  Its base is probably another tree (setup_env_cuda.sh reuses any conda on PATH)."
        warn "  'envfile' will detect and use the real one; just be aware env.sh will point"
        warn "  outside \$XMOE4D_ROOT, so that path must exist on EVERY node too."
    fi
    # A fresh `git clone` without --recursive leaves the submodule empty, so $SCRIPTS
    # does not exist. Saying only "scripts dir not found" sends people off to run the
    # 40-minute setup_env_cuda.sh (which does happen to init it, at stage_xmoe) when a
    # 10-second submodule init is all that is needed. Name the actual cause.
    # A plain `git clone` (no --recursive) leaves the submodule empty, so $SCRIPTS does
    # not exist. Do NOT make --recursive a hard requirement: the published AE notes use a
    # plain clone, and setup_env_cuda.sh already initializes this submodule itself at
    # stage_xmoe. Doing it here too is consistent, and saves the reviewer from an error
    # whose obvious-looking fix ("run setup_env_cuda.sh") costs 40 minutes.
    #
    # ONLY this submodule: primus_turbo is AMD/Composable-Kernel-only and must stay
    # uninitialized on NVIDIA.
    if [ ! -d "$SCRIPTS" ]; then
        if [ -d "$XMOE_ROOT/.git" ] && [ -z "$(ls -A "$XMOE_ROOT/Megatron-DeepSpeed-X-MoE" 2>/dev/null)" ]; then
            info "Megatron-DeepSpeed-X-MoE submodule is empty (plain clone) — auto-initializing it now."
            if ( cd "$XMOE_ROOT" && git submodule update --init --recursive Megatron-DeepSpeed-X-MoE ) >/dev/null 2>&1; then
                ok "submodule initialized ($(cd "$XMOE_ROOT" && git rev-parse --short HEAD:Megatron-DeepSpeed-X-MoE 2>/dev/null))"
            else
                warn "automatic 'git submodule update --init' failed (no network? no credentials?)."
                warn "Run it by hand, then re-run this stage:"
                warn "    cd $XMOE_ROOT && git submodule update --init --recursive Megatron-DeepSpeed-X-MoE"
                die "submodule not initialized"
            fi
        fi
        [ -d "$SCRIPTS" ] || die "scripts dir not found: $SCRIPTS"
    fi

    local n1g; n1g=$(nvidia-smi -L 2>/dev/null | wc -l)
    [ "$n1g" -gt 0 ] && ok "node1 GPUs: $n1g" || { warn "node1: nvidia-smi found no GPUs"; fail=1; }

    local ifc; ifc=$(detect_iface)
    [ -n "$ifc" ] && ok "node1 interface: $ifc" || { warn "could not detect a network interface"; fail=1; }

    # Port 22 reachable? (distinguish "blocked" from "refused" -- a TIMEOUT means the
    # security group is dropping packets; "refused" means it arrived and nothing listened.)
    if timeout 5 bash -c "cat </dev/null >/dev/tcp/$NODE2/22" 2>/dev/null; then
        ok "node2 port 22 reachable"
    else
        warn "node2 port 22 NOT reachable — check the instance is running and the SG allows ssh"; fail=1
    fi

    # 29500 should be REFUSED (nothing listening yet), not time out. A timeout is the
    # signature of the missing all-TCP security-group rule, which is THE most common
    # multi-node failure: torchrun then hangs at rendezvous with no useful error.
    local t0 t1 dt
    t0=$(date +%s)
    timeout 8 bash -c "cat </dev/null >/dev/tcp/$NODE2/29500" 2>/dev/null
    t1=$(date +%s); dt=$((t1-t0))
    if [ "$dt" -ge 7 ]; then
        warn "port 29500 TIMED OUT after ${dt}s — the security group is almost certainly"
        warn "  missing the self-referencing All-TCP rule. torchrun WILL hang at rendezvous."
        warn "  Fix: EC2 > Security Groups > <your sg> > Inbound > Add rule:"
        warn "       Type=All TCP, Port=0-65535, Source=<that same security group id>"
        fail=1
    else
        ok "port 29500 answered in ${dt}s (connection refused = SG allows it, nothing listening yet)"
    fi

    if ssh $SSH_OPTS "$NODE2" true 2>/dev/null; then
        ok "passwordless ssh to node2 works"
        local n2g; n2g=$(ssh $SSH_OPTS "$NODE2" 'nvidia-smi -L 2>/dev/null | wc -l')
        [ "$n2g" = "$n1g" ] && ok "node2 GPUs: $n2g (matches node1)" || { warn "GPU count mismatch: node1=$n1g node2=$n2g"; fail=1; }
        local d1 d2
        d1=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
        d2=$(ssh $SSH_OPTS "$NODE2" 'nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1')
        [ "$d1" = "$d2" ] && ok "driver match: $d1" || { warn "DRIVER MISMATCH node1=$d1 node2=$d2 — NCCL will fail"; fail=1; }
        local i2; i2=$(ssh $SSH_OPTS "$NODE2" "ip -br addr | awk '\$2==\"UP\"{print \$1}' | grep -vE '^(lo|docker|veth|br-|virbr)' | head -1")
        [ "$ifc" = "$i2" ] && ok "interface match: $ifc" || warn "interface differs: node1=$ifc node2=$i2 (env.sh is shared — see envfile stage)"
        local u1 u2; u1=$(id -u); u2=$(ssh $SSH_OPTS "$NODE2" 'id -u')
        [ "$u1" = "$u2" ] && ok "uid match: $u1 (NFS permissions will work)" || warn "UID mismatch node1=$u1 node2=$u2 — NFS ownership will be wrong"
    else
        info "passwordless ssh to node2 not set up yet — run the 'sshkey' stage"
    fi

    if /opt/amazon/efa/bin/fi_info -p efa >/dev/null 2>&1; then
        ok "EFA present — env.sh will NOT force NCCL_NET=Socket"
    else
        info "no EFA on this instance; multi-node NCCL will run over TCP (correct, just slow)."
        info "  EFA cannot be attached to a RUNNING instance, but CAN be attached to a"
        info "  STOPPED one — see x-moe-docs/aws-multinode-guide.md, Appendix A."
    fi

    [ "$fail" -eq 0 ] && ok "preflight clean" || warn "preflight found problems (above) — fix them before 'all'"
    return 0
}

stage_sshkey() {
    banner "ssh keys"
    # The launcher ssh's to EVERY host in the hostfile INCLUDING ITSELF, so node1
    # must be able to ssh to node1. This is easy to miss and fails only at launch.
    [ -f ~/.ssh/id_ed25519 ] || ssh-keygen -t ed25519 -N '' -C "xmoe-4d-$(hostname -s)" -f ~/.ssh/id_ed25519 >/dev/null
    mkdir -p ~/.ssh && chmod 700 ~/.ssh
    grep -qF "$(cut -d' ' -f2 ~/.ssh/id_ed25519.pub)" ~/.ssh/authorized_keys 2>/dev/null \
        || cat ~/.ssh/id_ed25519.pub >> ~/.ssh/authorized_keys
    chmod 600 ~/.ssh/authorized_keys
    # sshd's StrictModes refuses public-key auth when $HOME or ~/.ssh is group- or
    # world-writable, with only "Permission denied (publickey)" to go on -- which looks
    # like a missing key, not a permissions problem. Some AMIs ship $HOME as drwxrwxr-x.
    # This bites node1->node1 specifically: the launcher ssh's to EVERY host in the
    # hostfile including itself, so the run dies at launch even though node2 is fine.
    if [ -n "$(find "$HOME" -maxdepth 0 -perm -g+w -o -maxdepth 0 -perm -o+w 2>/dev/null)" ]; then
        warn "\$HOME is group/world-writable ($(stat -c '%A' "$HOME")); sshd will refuse key auth."
        chmod go-w "$HOME" && ok "fixed: $HOME is now $(stat -c '%A' "$HOME")"
    fi
    chmod go-w "$HOME/.ssh" 2>/dev/null
    if ssh $SSH_OPTS "$NODE1" true 2>/dev/null; then
        ok "node1 -> node1 ok"
    else
        die "node1 -> node1 ssh FAILED. The launcher ssh's to every host in the hostfile,
     itself included, so this must work. Check: chmod go-w ~ ~/.ssh, and that
     ~/.ssh/authorized_keys contains ~/.ssh/id_ed25519.pub."
    fi

    if ssh $SSH_OPTS "$NODE2" true 2>/dev/null; then
        ok "node1 -> node2 already works"
    else
        printf "\n${C_HEAD}--- ACTION NEEDED: install this key on node 2 ---${C_OFF}\n"
        cat <<EOF

From a shell that can already reach node 2 (your laptop with the .pem, or the EC2
console's "Connect > EC2 Instance Connect" browser terminal), run:

  mkdir -p ~/.ssh && chmod 700 ~/.ssh
  echo '$(cat ~/.ssh/id_ed25519.pub)' >> ~/.ssh/authorized_keys
  chmod 600 ~/.ssh/authorized_keys

Then re-run:  ./setup_2node_aws.sh sshkey $NODE2
EOF
        return 1
    fi
}

stage_nfs() {
    banner "shared filesystem (NFS export of $XMOE4D_ROOT)"
    # A shared FS is NOT optional: the launcher writes node_launch.sh on node 1 and
    # then tells node 2 to run it AT THAT SAME ABSOLUTE PATH, and all ranks write
    # rank_*.log into one JOB_DIR. Exporting the whole XMOE4D_ROOT also gives node 2
    # the conda env, py-spy, the editable installs and the dataset for free.
    command -v exportfs >/dev/null || sudo dnf install -y nfs-utils >/dev/null 2>&1
    if ! grep -qF "$XMOE4D_ROOT $NODE2" /etc/exports 2>/dev/null; then
        echo "$XMOE4D_ROOT ${NODE2}(rw,sync,no_subtree_check,no_root_squash)" | sudo tee -a /etc/exports >/dev/null
    fi
    sudo systemctl enable --now nfs-server >/dev/null 2>&1
    sudo exportfs -ra
    sudo exportfs -v | grep -qF "$XMOE4D_ROOT" && ok "exported $XMOE4D_ROOT -> $NODE2" || die "export failed"

    ssh $SSH_OPTS "$NODE2" "command -v mount.nfs >/dev/null || sudo dnf install -y nfs-utils >/dev/null 2>&1"
    if ssh $SSH_OPTS "$NODE2" "mountpoint -q '$XMOE4D_ROOT'"; then
        ok "node2 already has $XMOE4D_ROOT mounted"
    else
        ssh $SSH_OPTS "$NODE2" "sudo mkdir -p '$XMOE4D_ROOT' && sudo mount -t nfs ${NODE1}:${XMOE4D_ROOT} '$XMOE4D_ROOT'" \
            || die "mount failed on node2 (SG must allow NFS/2049 — the all-TCP rule covers it)"
        ok "mounted on node2"
    fi
    # survive reboots
    ssh $SSH_OPTS "$NODE2" "grep -qF '$XMOE4D_ROOT' /etc/fstab || echo '${NODE1}:${XMOE4D_ROOT} ${XMOE4D_ROOT} nfs defaults,_netdev 0 0' | sudo tee -a /etc/fstab >/dev/null"

    ssh $SSH_OPTS "$NODE2" "touch '$XMOE4D_ROOT/.w' 2>/dev/null" && rm -f "$XMOE4D_ROOT/.w" \
        && ok "node2 can write to the share (needed for the shared JOB_DIR)" \
        || die "node2 cannot WRITE to the share — check uid match and no_root_squash"
}

# ---------------------------------------------------------------------------
# resolve_conda_root — where conda ACTUALLY is, not where we assumed it would be.
#
# setup_env_cuda.sh only installs Miniforge when there is no conda anywhere:
#     if [ ! -d "$CONDA_ROOT" ] && ! command -v conda >/dev/null; then install
# So if the operator had ANY conda active while building (very likely when the same
# person sets up a second tree, e.g. ~/xmoe-4d_test then ~/xmoe-4d), the new tree gets an
# env at $ENV_PREFIX but NO miniforge3 of its own. Writing the assumed path into
# env.sh then breaks every rank with "no such file or directory", on every node.
resolve_conda_root() {
    if [ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then
        echo "$CONDA_ROOT"; return
    fi
    local base
    base="$(conda info --base 2>/dev/null)"
    if [ -n "$base" ] && [ -f "$base/etc/profile.d/conda.sh" ]; then
        warn "no conda at $CONDA_ROOT; using the one actually on PATH: $base"
        warn "  (setup_env_cuda.sh reuses an existing conda instead of installing a"
        warn "   second copy -- expected when building more than one tree on a box)"
        echo "$base"; return
    fi
    # Nothing yet: this is the pre-build case (envfile running inside 'wire'). The
    # predicted path is not a guess -- setup_env_cuda.sh installs Miniforge to exactly
    # $XMOE4D_ROOT/miniforge3 -- so write it now and let 'all' re-verify afterwards.
    echo "$CONDA_ROOT"
}

stage_envfile() {
    banner "env.sh"
    CONDA_ROOT="$(resolve_conda_root)"
    local ifc; ifc=$(detect_iface)
    [ -n "$ifc" ] || die "could not detect a network interface"
    local i2; i2=$(ssh $SSH_OPTS "$NODE2" "ip -br addr | awk '\$2==\"UP\"{print \$1}' | grep -vE '^(lo|docker|veth|br-|virbr)' | head -1" 2>/dev/null)
    if [ -n "$i2" ] && [ "$i2" != "$ifc" ]; then
        warn "node1 iface=$ifc but node2 iface=$i2. env.sh is SHARED, so one value must"
        warn "  work on both. Using '$ifc'; if NCCL fails on node2, set NCCL_SOCKET_IFNAME"
        warn "  to a comma list, e.g. NCCL_SOCKET_IFNAME=$ifc,$i2"
    fi

    local net_line="export NCCL_NET=Socket        # no EFA on these instances; delete this when EFA is attached"
    if /opt/amazon/efa/bin/fi_info -p efa >/dev/null 2>&1; then
        net_line="# EFA detected: letting NCCL auto-select the aws-ofi-nccl plugin (do NOT force NCCL_NET)."
    fi

    XMOE4D_ENV_TMP="$(mktemp)"
    cat > "$XMOE4D_ENV_TMP" <<EOF
#!/bin/bash
# Generated by setup_2node_aws.sh. Sourced by the DRIVER and by every node
# (ssh carries no environment, so this file is how remote ranks get a usable shell).

# conda activate dies under \`set -u\` if the cuda-nvcc hook sees this unset.
export NVCC_PREPEND_FLAGS="\${NVCC_PREPEND_FLAGS:-}"
source $CONDA_ROOT/etc/profile.d/conda.sh
conda activate $ENV_PREFIX

# CUDA_HOME / CPATH come from the conda activation hook that setup_env_cuda.sh
# installs. If a fused-kernel build ever dies on "cuda_fp16.h: No such file or
# directory", that hook is missing: ./setup_env_cuda.sh envhook

export NCCL_SOCKET_IFNAME=$ifc
$net_line
export OMP_NUM_THREADS=1

# NEVER set RANK / LOCAL_RANK / WORLD_SIZE / SLURM_PROCID / SLURM_LOCALID here.
# torchrun sets the first three per worker, and pretrain_gpt_deepspeed.py reads
# SLURM_PROCID BEFORE RANK -- exporting it makes every rank think it is rank 0.
EOF
    if [ -f "$SCRIPTS/env.sh" ] && ! cmp -s "$XMOE4D_ENV_TMP" "$SCRIPTS/env.sh"; then
        cp "$SCRIPTS/env.sh" "$SCRIPTS/env.sh.bak.$(date +%s)"
        info "existing env.sh differed — backed up"
    fi
    mv "$XMOE4D_ENV_TMP" "$SCRIPTS/env.sh"; chmod 644 "$SCRIPTS/env.sh"
    ok "wrote $SCRIPTS/env.sh (iface=$ifc)"
    if [ -d "$ENV_PREFIX" ] && [ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then
        env -i HOME="$HOME" PATH=/usr/bin:/bin bash --noprofile --norc -c \
            "source '$SCRIPTS/env.sh' && python -c 'import torch,six,deepspeed,megatron; print(\"  verify: gpus\", torch.cuda.device_count())'" \
            2>/dev/null | tail -1 || warn "env.sh sourced but imports failed — check ./setup_env_cuda.sh verify"
    else
        info "env not built yet — env.sh written with the paths setup_env_cuda.sh will create."
        info "It is re-generated and verified by the 'all' stage once the env exists."
    fi
}

stage_hostfile() {
    banner "hostfile"
    printf '%s\n%s\n' "$NODE1" "$NODE2" > "$SCRIPTS/hostfile"
    ok "wrote $SCRIPTS/hostfile (line 1 = $NODE1 becomes MASTER_ADDR)"
    cat "$SCRIPTS/hostfile" | sed 's/^/     /'
}

stage_smoke() {
    banner "NCCL smoke test (16 ranks)"
    local NPROC; NPROC=$(nvidia-smi -L | wc -l)
    cat > "$XMOE4D_ROOT/ar_smoke.py" <<'EOF'
import os, torch, torch.distributed as dist
lr = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(lr)
dist.init_process_group("nccl", device_id=torch.device(f"cuda:{lr}"))
t = torch.ones(1 << 24, device="cuda")
dist.all_reduce(t)                       # sum of ones across N ranks == N
if dist.get_rank() == 0:
    print(f"SMOKE_RESULT world={dist.get_world_size()} sum={t[0].item():.1f}", flush=True)
dist.destroy_process_group()
EOF
    local OUT; OUT=$(mktemp)
    for h in "$NODE1" "$NODE2"; do
        ssh $SSH_OPTS "$h" "source '$SCRIPTS/env.sh' && torchrun --nnodes 2 --nproc_per_node $NPROC \
            --rdzv_backend c10d --rdzv_endpoint ${NODE1}:29500 --rdzv_id smoke \
            '$XMOE4D_ROOT/ar_smoke.py'" >>"$OUT" 2>&1 &
    done
    wait
    local EXPECT=$(( 2 * NPROC ))
    local R; R=$(grep -o "SMOKE_RESULT world=[0-9]* sum=[0-9.]*" "$OUT" | head -1)
    if [ -z "$R" ]; then
        warn "no result line — ranks never completed the all-reduce."
        grep -m5 -iE "error|refused|timeout|unreachable" "$OUT" | sed 's/^/     /'
        rm -f "$OUT"; return 1
    fi
    local W S; W=$(sed 's/.*world=\([0-9]*\).*/\1/' <<<"$R"); S=$(sed 's/.*sum=\([0-9.]*\)/\1/' <<<"$R")
    if [ "$W" = "$EXPECT" ] && [ "${S%.*}" = "$EXPECT" ]; then
        ok "PASS: world=$W sum=$S (== world, so every rank contributed)"
        # NOTE: torchrun may still print a RendezvousConnectionError traceback AFTER
        # this line. It is benign -- the agent hosting the c10d store exits first and
        # tears it down while the other agent is still doing shutdown bookkeeping.
        rm -f "$OUT"; return 0
    fi
    warn "FAIL: expected world=$EXPECT sum=$EXPECT, got world=$W sum=$S"
    rm -f "$OUT"; return 1
}

# ---------------------------------------------------------------------------
# stage_wire — everything that does NOT need the conda env, so it can run FIRST.
#
# Doing this before ./setup_env_cuda.sh is the better order: once ~/xmoe-4d is
# NFS-exported, the env you build afterwards lands on the shared filesystem
# automatically and node 2 sees it appear -- no "copy it over" step to forget, and
# no chance of the two nodes drifting. It also surfaces security-group and ssh
# problems in the first minute rather than 40 minutes into a build.
#
# env.sh generation is deliberately left to 'all', so its verification step runs
# against a real env instead of warning about a missing one.
stage_wire() {
    stage_preflight
    stage_sshkey || die "install the key on node 2 (instructions above), then re-run 'wire'"
    stage_nfs
    stage_envfile
    stage_hostfile
    banner "wiring done"
    info "Node 2 now mounts $XMOE4D_ROOT, so everything you install below is shared."
    info "Next:  ./setup_env_cuda.sh                      (build the env, ~40 min)"
    info "       cd $XMOE_ROOT/Megatron-DeepSpeed-X-MoE/examples_xmoe_4d/data && bash prepare_data_ae.sh"
    info "       ./setup_2node_aws.sh all $NODE2          (writes env.sh + NCCL smoke test)"
}

case "$STAGE" in
    preflight) stage_preflight ;;
    wire)      stage_wire ;;
    sshkey)    stage_sshkey ;;
    nfs)       stage_nfs ;;
    envfile)   stage_envfile ;;
    hostfile)  stage_hostfile ;;
    smoke)     stage_smoke ;;
    all)
        stage_preflight
        stage_sshkey || die "install the key on node 2 (instructions above), then re-run 'all'"
        stage_nfs
        stage_envfile
        stage_hostfile
        # smoke is the only stage that needs torch. Rather than dying with a confusing
        # error when 'all' is run before the env is built, do every other stage and say
        # plainly that the cluster is wired but NOT yet proven. Silently skipping would
        # be worse than failing: "all succeeded" must never imply an unvalidated cluster.
        if [ ! -d "$ENV_PREFIX" ]; then
            banner "WIRING COMPLETE, NOT YET VALIDATED"
            warn "Skipped the NCCL smoke test: no conda env at $ENV_PREFIX yet."
            warn "The cluster is wired (ssh + NFS + env.sh + hostfile) but UNPROVEN."
            info "Build the env, then re-run this to validate:"
            info "    ./setup_env_cuda.sh"
            info "    ./setup_2node_aws.sh all $NODE2"
            exit 0
        fi
        stage_smoke || die "smoke test failed — do not start training until this passes"
        banner "2-node cluster ready"
        info "Next:  cd $SCRIPTS && bash autorun.sh"
        ;;
    *) die "unknown stage '$STAGE' (preflight|wire|sshkey|nfs|envfile|hostfile|smoke|all)" ;;
esac
