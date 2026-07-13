#!/bin/bash
###############################################################################
# setup_env_cuda.sh — ELMoE reviewer reproduction environment (NVIDIA / AWS)
#
# NVIDIA counterpart of setup_env_rocm.sh. Builds the full ELMoE software stack
# on an AWS EC2 GPU instance running the Amazon Linux 2023 Deep Learning AMI.
#
# ---------------------------------------------------------------------------
# WHERE THINGS GO — one knob: ELMOE_ROOT
#
#   ELMOE_ROOT=/scratch/elmoe ./setup_env_cuda.sh
#
# That single variable relocates EVERYTHING this script writes:
#
#   $ELMOE_ROOT/ELMoE_envs/ELMoE-CUDA12.8_repro   conda environment
#   $ELMOE_ROOT/ELMoE_deps/                       apex + flash-attn sources
#   $ELMOE_ROOT/ELMoE_cache/                      pip + conda caches (these get big)
#
# Defaults to one level ABOVE the X-MoE checkout. Needs >= 25 GB free; the
# script checks this UP FRONT and fails in seconds if short. Relocate with:
#   local NVMe   -> sudo ./mount_scratch.sh /dev/nvme0n1 /scratch
#                   ELMOE_ROOT=/scratch/elmoe ./setup_env_cuda.sh
#   multi-node   -> ELMOE_ROOT=/fsx/elmoe   (a SHARED fs every node mounts)
#
# The X-MoE checkout itself is found from this script's own location, so the
# repo can live anywhere -- including outside ELMOE_ROOT.
#
# ---------------------------------------------------------------------------
# USAGE
#   ./setup_env_cuda.sh            # run all stages
#   ./setup_env_cuda.sh <stage>    # run a single stage (re-run one on failure)
#
#   stages:  conda  cuda  torch  apex  mpi4py  flashattn  xmoe  verify
#
#   Optional check, for multi-node runs only:
#     ./setup_env_cuda.sh efa      # verify NCCL will use EFA, not TCP.
#                                  # Nothing to build: the DLAMI ships the
#                                  # aws-ofi-nccl plugin preinstalled.
#
#   There is NO primus stage on NVIDIA. primus_turbo is AMD-only (Composable
#   Kernel, MI250+). ELMoE's portable Triton grouped-GEMM backend
#   (FusedExperts_Triton) is the NVIDIA equivalent and needs no extra build --
#   Triton ships inside the PyTorch wheel. Select it with use_triton=True.
#
# ---------------------------------------------------------------------------
# DIFFERENCES FROM THE ROCm SCRIPT (and why)
#   module stack   -> none. AWS has no Lmod; the DLAMI provides the driver and
#                     an EFA-aware OpenMPI at /opt/amazon/openmpi.
#   miniforge3     -> not a module. Installed by the conda stage if absent.
#   nvcc           -> installed INTO the conda env (cuda stage) so its version
#                     matches the torch wheel's CUDA. The DLAMI's system nvcc is
#                     often a different major (13.x) and apex hard-fails on a
#                     torch/nvcc mismatch.
#   MPICC          -> /opt/amazon/openmpi/bin/mpicc (not Cray's `cc -shared`).
#   flash-attn     -> plain `pip install`. On CUDA the upstream setup.py fetches
#                     an official prebuilt wheel, so unlike ROCm there is no
#                     source build. See stage_flashattn for the version caveat.
#   apex           -> NVIDIA/apex (not ROCm/apex), pinned by TAG. It reports
#                     version 0.1, not 1.11.0; that is expected, not drift.
#   aws-ofi-rccl   -> nothing to build. The DLAMI preinstalls aws-ofi-nccl.
#
# OVERRIDES: ELMOE_ROOT, ENV_PREFIX, DEPS_DIR, MAX_JOBS, TORCH_VERSION,
#            CUDA_VERSION, TORCH_CUDA_ARCH_LIST, APEX_TAG, FLASH_ATTN_VERSION
###############################################################################

set -euo pipefail

# --- resolve paths ---------------------------------------------------------
XMOE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- the one knob ----------------------------------------------------------
# Default: the directory ONE LEVEL ABOVE the X-MoE checkout. So a repo at
# ~/X-MoE puts the env, deps, and caches alongside it as siblings:
#
#   ~/X-MoE/            <- the repo (this script lives here)
#   ~/ELMoE_envs/       <- conda environment
#   ~/ELMoE_deps/       <- apex + flash-attn sources
#   ~/ELMoE_cache/      <- pip + conda caches
#
# This keeps the checkout clean (nothing large is written inside the repo) and
# mirrors how the ROCm script places its sibling ELMoE_deps/ folder.
# Override to relocate everything at once, e.g. ELMOE_ROOT=/fsx/elmoe on a
# multi-node cluster, or ELMOE_ROOT=/scratch/elmoe on a mounted local NVMe.
ELMOE_ROOT="${ELMOE_ROOT:-$(cd "$XMOE_ROOT/.." && pwd)}"

# Everything below derives from ELMOE_ROOT, but each is independently overridable.
# The ELMoE_* prefix matches the sibling ELMoE_deps/ folder the ROCm script creates.
ENV_PREFIX="${ENV_PREFIX:-$ELMOE_ROOT/ELMoE_envs/ELMoE-CUDA12.8_repro}"
DEPS_DIR="${DEPS_DIR:-$ELMOE_ROOT/ELMoE_deps}"
CACHE_DIR="${CACHE_DIR:-$ELMOE_ROOT/ELMoE_cache}"
# Python 3.12, NOT the 3.11 of the ROCm env. This is deliberate: it is the only
# Python for which upstream publishes a prebuilt flash-attn wheel against torch
# 2.9 (see stage_flashattn). Trading the Python minor -- the least load-bearing
# pin in the stack -- buys exact torch + flash-attn parity with the AMD build.
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

# Keep the fat caches off the (small) root volume too.
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$CACHE_DIR/pip}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$CACHE_DIR/conda}"
export TMPDIR="${TMPDIR:-$CACHE_DIR/tmp}"

# --- pinned versions (reproduction targets) --------------------------------
# torch 2.9.1 matches the ROCm reproduction environment (torch 2.9.1+rocm6.4)
# exactly. Paired with python 3.12 it also has a prebuilt flash-attn wheel, so
# nothing is compiled. Do not "simplify" this to python 3.11 -- see stage_flashattn.
TORCH_VERSION="${TORCH_VERSION:-2.9.1}"
CUDA_VERSION="${CUDA_VERSION:-12.8.1}"        # conda nvcc; major must match the wheel
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"

# flash-attn 2.8.3 — the same version as the ROCm build. NOT 2.8.3.post1: the
# .post1 release dropped the torch-2.9 wheels. See stage_flashattn.
FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.8.3}"

# NVIDIA/apex has no semantic version (it reports 0.1 forever), so the only
# meaningful pin is a git tag. Upstream tags CalVer monthly (25.01 ... 25.09).
# Pinning a tag is what makes the apex build reproducible for reviewers.
APEX_TAG="${APEX_TAG:-25.09}"

# Target GPU archs. 8.0 = A100 (p4d), 8.6 = A10G (g5, this dev box), 9.0 = H100
# (p5). Build for every arch you will RUN on: kernels compiled only for sm_86 on
# a g5 box will fail at launch on an A100. 10.0 = B200 (p6), add if needed.
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;9.0}"
export TORCH_CUDA_ARCH_LIST

# Parallel compile jobs. nvcc uses ~2-4 GB RAM per job: the Frontier value of 64
# OOMs an 8-vCPU/32 GB box.
MAX_JOBS="${MAX_JOBS:-$(( $(nproc) > 8 ? 8 : $(nproc) ))}"

# --- DLAMI-provided paths --------------------------------------------------
MPI_PATH="${MPI_PATH:-/opt/amazon/openmpi}"
EFA_PATH="${EFA_PATH:-/opt/amazon/efa}"
# aws-ofi-nccl, preinstalled by the DLAMI. NOTE: the path is /opt/amazon/ofi-nccl
# on current AMIs. The older /opt/aws-ofi-nccl/ that most blog posts cite is dead.
OFI_NCCL_PATH="${OFI_NCCL_PATH:-/opt/amazon/ofi-nccl}"

MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
CONDA_ROOT="${CONDA_ROOT:-$ELMOE_ROOT/miniforge3}"

# Free space (GB) required under ELMOE_ROOT, checked BEFORE any stage runs so a
# too-small volume fails in seconds rather than 25 minutes into the apex compile.
# Measured on a completed build: 18 GB final (14 GB env — the in-env CUDA toolkit
# is 2.7 GB of that — plus 3.5 GB caches), peaking ~22 GB during apex. 25 GB
# leaves a little headroom. The ROCm script uses the same ~22 GB figure.
MIN_FREE_GB="${MIN_FREE_GB:-25}"

# ---------------------------------------------------------------------------
# pretty logging
# ---------------------------------------------------------------------------
if [ -t 1 ]; then C_HEAD='\033[1;36m'; C_OK='\033[1;32m'; C_WARN='\033[1;33m'; C_ERR='\033[1;31m'; C_OFF='\033[0m'
else C_HEAD=''; C_OK=''; C_WARN=''; C_ERR=''; C_OFF=''; fi

CURRENT_STAGE="init"
banner()  { printf "\n${C_HEAD}==== [%s] %s ====${C_OFF}\n" "$(date +%H:%M:%S)" "$1"; }
ok()      { printf "${C_OK}[ok]${C_OFF} %s\n" "$1"; }
warn()    { printf "${C_WARN}[warn]${C_OFF} %s\n" "$1"; }
info()    { printf "     %s\n" "$1"; }

trap 'rc=$?; if [ $rc -ne 0 ]; then
  printf "\n${C_ERR}[FAILED]${C_OFF} stage \"%s\" exited %d.\n" "$CURRENT_STAGE" "$rc";
  printf "         Fix the error above, then re-run just this stage:\n";
  printf "             ELMOE_ROOT=%s ./setup_env_cuda.sh %s\n" "$ELMOE_ROOT" "$CURRENT_STAGE";
fi' EXIT

# ---------------------------------------------------------------------------
# shared setup (replaces the ROCm script's load_modules)
# ---------------------------------------------------------------------------
check_space() {
    local parent="$ELMOE_ROOT" avail
    while [ ! -d "$parent" ]; do parent="$(dirname "$parent")"; done
    avail="$(df -BG --output=avail "$parent" | tail -1 | tr -dc '0-9')"
    if [ "${avail:-0}" -lt "$MIN_FREE_GB" ]; then
        printf "${C_ERR}[FAILED]${C_OFF} only %s GB free on the filesystem holding ELMOE_ROOT=%s (need >= %s GB).\n" \
               "$avail" "$ELMOE_ROOT" "$MIN_FREE_GB" >&2
        cat >&2 <<EOF
         The full stack needs ~22 GB (14 GB conda env incl. the in-env CUDA
         toolkit, ~4 GB caches, plus apex build objects at peak).
         Point ELMOE_ROOT at a bigger volume. On a fresh AWS GPU box the local
         NVMe is usually unformatted and unmounted:

             sudo ./mount_scratch.sh /dev/nvme0n1 /scratch
             ELMOE_ROOT=/scratch/elmoe ./setup_env_cuda.sh

         (or grow the EBS root volume; or on multi-node, use a shared FSx/EFS path)
EOF
        exit 1
    fi
    info "free space at $parent: ${avail} GB (need >= ${MIN_FREE_GB} GB)"
}

# Conda's own shell hooks are not `set -u` safe. In particular the cuda-nvcc
# package ships an activate.d script that appends to $NVCC_PREPEND_FLAGS without
# initializing it, so `conda activate` dies with "NVCC_PREPEND_FLAGS: unbound
# variable" under `set -u`. Run every conda command with nounset off, then restore.
conda_safe() {
    local had_u=0
    [[ $- == *u* ]] && had_u=1
    set +u
    "$@"
    local rc=$?
    [ "$had_u" -eq 1 ] && set -u
    return $rc
}

conda_hook() {
    local base
    if [ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then
        base="$CONDA_ROOT"
    elif command -v conda >/dev/null 2>&1; then
        base="$(conda info --base)"
    else
        echo "ERROR: conda not found. Run './setup_env_cuda.sh conda' first." >&2; exit 1
    fi
    conda_safe source "$base/etc/profile.d/conda.sh"
}

activate_env() {
    conda_hook
    [ -d "$ENV_PREFIX" ] || { echo "ERROR: env '$ENV_PREFIX' not found. Run the 'conda' stage first." >&2; exit 1; }
    conda_safe conda activate "$ENV_PREFIX"
    # Point every downstream build at the conda CUDA, not the system one.
    export CUDA_HOME="$CONDA_PREFIX"
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
    info "root:   $ELMOE_ROOT"
    info "python: $(which python)  ($(python --version 2>&1))"
    command -v nvcc >/dev/null 2>&1 && \
        info "nvcc:   $(nvcc --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')  ($(which nvcc))"
    info "archs:  $TORCH_CUDA_ARCH_LIST | MAX_JOBS=$MAX_JOBS"
}

# The classic NVIDIA failure: torch wheel CUDA != nvcc CUDA. apex refuses to
# build (or builds broken kernels) on a mismatch. Check before every source build.
assert_cuda_match() {
    python - <<'PY'
import re, subprocess, sys, torch
wheel = torch.version.cuda
out = subprocess.run(["nvcc", "--version"], capture_output=True, text=True).stdout
m = re.search(r"release (\d+\.\d+)", out)
if not m:
    sys.exit("ERROR: nvcc not on PATH. Run './setup_env_cuda.sh cuda'.")
nvcc = m.group(1)
if wheel.split(".")[0] != nvcc.split(".")[0]:
    sys.exit(f"ERROR: torch is built for CUDA {wheel} but nvcc is {nvcc} (major mismatch).\n"
             f"       Source builds will fail. Re-run './setup_env_cuda.sh cuda' and make\n"
             f"       sure CUDA_HOME points at the conda env, not /usr/local/cuda.")
print(f"  [ok] torch CUDA {wheel} vs nvcc {nvcc}"
      + ("" if wheel == nvcc else "  (same major — builds OK)"))
PY
}

# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------
stage_conda() {
    CURRENT_STAGE="conda"; banner "conda env ($ENV_PREFIX, python $PYTHON_VERSION)"
    check_space
    mkdir -p "$ELMOE_ROOT" "$DEPS_DIR" "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" "$TMPDIR"
    if [ ! -d "$CONDA_ROOT" ] && ! command -v conda >/dev/null 2>&1; then
        info "no conda on this box — installing Miniforge3 to $CONDA_ROOT"
        curl -fsSL "$MINIFORGE_URL" -o "$TMPDIR/miniforge.sh"
        bash "$TMPDIR/miniforge.sh" -b -p "$CONDA_ROOT"
        rm -f "$TMPDIR/miniforge.sh"
        ok "Miniforge installed."
    else
        info "conda already present — reusing."
    fi
    conda_hook
    if [ -d "$ENV_PREFIX" ]; then
        warn "env already exists at $ENV_PREFIX — reusing it (delete it to recreate)."
    else
        conda_safe conda create -y -p "$ENV_PREFIX" "python=$PYTHON_VERSION"
    fi
    conda_safe conda activate "$ENV_PREFIX"
    # Build prerequisites that apex / flash-attn's setup.py expect to import.
    pip install -q packaging ninja wheel psutil
    ok "conda env ready and active."
}

stage_cuda() {
    CURRENT_STAGE="cuda"; banner "CUDA toolkit $CUDA_VERSION (inside the conda env)"
    activate_env
    # The DLAMI's system nvcc is often a different CUDA MAJOR than the torch wheel
    # (e.g. 13.2 vs cu128). Installing the toolkit into the env pins them together
    # and needs no root.
    conda_safe conda install -y -p "$ENV_PREFIX" -c nvidia "cuda-toolkit=$CUDA_VERSION"
    export CUDA_HOME="$ENV_PREFIX"; export PATH="$CUDA_HOME/bin:$PATH"
    nvcc --version | tail -2
    ok "nvcc $CUDA_VERSION installed in env."
}

stage_torch() {
    CURRENT_STAGE="torch"; banner "PyTorch $TORCH_VERSION (CUDA ${CUDA_VERSION%.*})"
    activate_env
    pip install "torch==$TORCH_VERSION" torchvision --index-url "$TORCH_INDEX"
    python -c "import torch; print('  torch', torch.__version__, '| cuda', torch.version.cuda)"
    assert_cuda_match
    ok "torch installed."
}

stage_apex() {
    CURRENT_STAGE="apex"; banner "apex (NVIDIA/apex @ $APEX_TAG, fused kernels)"
    activate_env; assert_cuda_match
    mkdir -p "$DEPS_DIR"; cd "$DEPS_DIR"
    if [ ! -d apex/.git ]; then
        git clone https://github.com/NVIDIA/apex.git    # NVIDIA fork, NOT ROCm/apex
    else
        info "apex/ already cloned — reusing."
    fi
    cd apex
    git fetch --tags --quiet
    # apex reports version "0.1" no matter what, so the git tag IS the version.
    # Pin it, or reviewers silently get whatever master happens to be that day.
    git checkout --quiet "$APEX_TAG"
    info "apex source pinned to tag $APEX_TAG ($(git rev-parse --short HEAD))"
    pip install -r requirements.txt
    rm -rf build
    # `python setup.py install` (the ROCm script's form) is deprecated upstream
    # and breaks on modern setuptools. This is the supported invocation.
    MAX_JOBS="$MAX_JOBS" pip install -v --no-cache-dir --no-build-isolation \
        --config-settings "--build-option=--cpp_ext" \
        --config-settings "--build-option=--cuda_ext" ./
    # Verify from OUTSIDE the source tree: a `python -c` run inside the apex clone
    # imports the local ./apex/ directory instead of the installed package.
    # Also import torch FIRST — the apex .so files link against libc10.so, so
    # importing amp_C standalone dies with "libc10.so: cannot open shared object".
    # And do NOT check `from apex import amp`: apex.amp was REMOVED upstream (it is
    # gone as of tag 25.09). Megatron-DeepSpeed does not use it; it needs the fused
    # kernels below. Checking for amp would fail on a perfectly good build.
    ( cd / && python - <<'PY'
import torch                      # must precede the apex extensions (loads libc10.so)
import amp_C, fused_layer_norm_cuda
from apex.normalization import FusedLayerNorm
from apex.optimizers import FusedAdam
from apex.multi_tensor_apply import multi_tensor_applier
print("  apex fused kernels importable: amp_C, fused_layer_norm_cuda")
if torch.cuda.is_available():     # exercise a real kernel, don't just import it
    ln = FusedLayerNorm(8).cuda()
    x = torch.randn(4, 8, device="cuda", requires_grad=True)
    ln(x).sum().backward()
    FusedAdam(ln.parameters(), lr=1e-3).step()
    print(f"  FusedLayerNorm fwd+bwd + FusedAdam.step() ran on {torch.cuda.get_device_name(0)}")
PY
    )
    ok "apex installed (tag $APEX_TAG; reports version 0.1 — expected on NVIDIA)."
}

stage_mpi4py() {
    CURRENT_STAGE="mpi4py"; banner "mpi4py (AWS OpenMPI, EFA-aware)"
    activate_env
    [ -x "$MPI_PATH/bin/mpicc" ] || { echo "ERROR: $MPI_PATH/bin/mpicc not found — is this a DLAMI?" >&2; exit 1; }
    MPICC="$MPI_PATH/bin/mpicc" pip install --no-cache-dir --force-reinstall --no-binary=mpi4py mpi4py
    python -c "import mpi4py; print('  mpi4py', mpi4py.__version__)"
    ok "mpi4py installed."
}

stage_flashattn() {
    CURRENT_STAGE="flashattn"; banner "flash-attention $FLASH_ATTN_VERSION"
    activate_env; assert_cuda_match
    # WHY A PLAIN pip install IS ENOUGH HERE (unlike ROCm):
    #   flash-attn publishes NO binary wheels on PyPI — only an sdist. But on
    #   CUDA its setup.py (CachedWheelsCommand) first tries to DOWNLOAD an
    #   official prebuilt wheel from GitHub Releases, named
    #       flash_attn-<ver>+cu12torch<X.Y>cxx11abi<TRUE|FALSE>-cp3XX-...whl
    #   under git tag v<ver>. Only if that 404s does it compile from source.
    #   On ROCm no such wheels are published — that is why you must build there.
    #
    # WHY python 3.12 AND flash-attn 2.8.3 (not .post1). The prebuilt matrix:
    #       cp311 -> torch 2.4 .. 2.8          (NO torch 2.9)
    #       cp312 -> torch 2.4 .. 2.8, AND 2.9
    #   and the torch-2.9 wheels exist ONLY under tag v2.8.3 — the later
    #   v2.8.3.post1 release dropped them (it stops at torch 2.8).
    #   So torch 2.9.1 + prebuilt wheel has exactly ONE solution:
    #       python 3.12 + flash-attn 2.8.3
    #   Change either pin and this stage silently falls back to a 45-90 minute
    #   source build. The preflight below makes that visible instead of silent.
    #
    # --no-build-isolation is required either way: setup.py imports torch.
    local wheel_url
    export FLASH_ATTN_VERSION          # the heredoc below reads it from the environment
    wheel_url="$(python - <<'PY'
import sys, torch
tv = ".".join(torch.__version__.split("+")[0].split(".")[:2])
abi = "TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE"
py = f"cp{sys.version_info.major}{sys.version_info.minor}"
import os
v = os.environ["FLASH_ATTN_VERSION"]
print(f"https://github.com/Dao-AILab/flash-attention/releases/download/v{v}/"
      f"flash_attn-{v}+cu12torch{tv}cxx11abi{abi}-{py}-{py}-linux_x86_64.whl")
PY
)"
    info "expected prebuilt wheel:"
    info "  $(basename "$wheel_url")"
    if curl -sIL -o /dev/null -w '%{http_code}' "$wheel_url" | grep -q 200; then
        ok "prebuilt wheel is available — this will be a download, not a compile."
    else
        warn "NO prebuilt wheel for this (python, torch, flash-attn) combination."
        warn "pip will fall back to a SOURCE BUILD (45-90 min on $(nproc) cores)."
        warn "That still works, but if you did not intend it, check the pins:"
        warn "  torch 2.9 needs python 3.12 + flash-attn 2.8.3 (not .post1)."
    fi
    pip install "flash-attn==$FLASH_ATTN_VERSION" --no-build-isolation
    python -c "import flash_attn; print('  flash_attn', flash_attn.__version__)"
    ok "flash-attention ready."
    info "A source-build fallback is visible in the log above as ninja/nvcc compile lines;"
    info "a prebuilt wheel shows a single download instead."
    info "To force a source build anyway: FLASH_ATTENTION_FORCE_BUILD=TRUE ./setup_env_cuda.sh flashattn"
}

stage_xmoe() {
    CURRENT_STAGE="xmoe"; banner "X-MoE + Megatron-DeepSpeed-X-MoE (editable)"
    activate_env
    cd "$XMOE_ROOT"
    # Deliberately NOT --recursive over all submodules: primus_turbo is AMD/CK-only
    # and must not be initialized on NVIDIA.
    git submodule update --init --recursive Megatron-DeepSpeed-X-MoE
    pip install -e .
    cd "$XMOE_ROOT/Megatron-DeepSpeed-X-MoE"
    pip install -e .
    ok "X-MoE stack installed (editable)."
}

stage_efa() {
    CURRENT_STAGE="efa"; banner "EFA / NCCL check (multi-node interconnect)"
    # NOTHING TO BUILD. This stage only verifies, and explains what to set.
    #
    # WHY THIS STILL MATTERS OFF FRONTIER:
    #   NCCL cannot talk to EFA natively — it has only two built-in transports,
    #   ibverbs and TCP sockets, and EFA is neither. The aws-ofi-nccl plugin is
    #   the bridge (NCCL -> libfabric -> EFA), exactly as aws-ofi-RCCL bridged to
    #   Slingshot on Frontier. With no plugin loaded, NCCL SILENTLY falls back to
    #   TCP over the ENA interface: no error, just a large fraction of the
    #   interconnect's bandwidth left on the floor (p4d: 400 Gbps; p5: 3200 Gbps).
    #
    # The good news: the DLAMI preinstalls the plugin AND puts it on the default
    # LD_LIBRARY_PATH, so on a stock DLAMI there is genuinely nothing to do.
    local libdir=""
    for d in "$OFI_NCCL_PATH/lib64" "$OFI_NCCL_PATH/lib"; do
        [ -f "$d/libnccl-net.so" ] && { libdir="$d"; break; }
    done
    if [ -z "$libdir" ]; then
        warn "aws-ofi-nccl NOT found under $OFI_NCCL_PATH."
        warn "Without it, multi-node NCCL runs over TCP instead of EFA."
        warn "Install the AWS EFA installer, or use a DLAMI that bundles it:"
        warn "  https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-start-nccl.html"
        return 0
    fi
    local ver; ver="$(strings "$libdir/libnccl-net.so" 2>/dev/null | grep -oE 'aws-ofi-nccl [0-9]+\.[0-9]+\.[0-9]+' | head -1)"
    ok "aws-ofi-nccl found: $libdir  (${ver:-version unknown})"

    if echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -qx "$libdir"; then
        ok "already on LD_LIBRARY_PATH — NCCL will discover it automatically."
    else
        warn "not on LD_LIBRARY_PATH. Add it in your job script:"
        warn "  export LD_LIBRARY_PATH=$libdir:$EFA_PATH/lib64:\$LD_LIBRARY_PATH"
    fi

    # Is there an actual EFA device on THIS instance?
    if fi_info -p efa >/dev/null 2>&1; then
        local nics; nics="$(fi_info -p efa 2>/dev/null | grep -c 'provider: efa' || true)"
        ok "EFA device(s) present on this instance (${nics} endpoint(s) reported)."
    else
        warn "No EFA device on this instance — expected on g5/g6 (they have none)."
        warn "Multi-node ELMoE needs p4d/p5/p6 with EFA enabled on every network card."
    fi

    printf "\n${C_HEAD}============ what to actually set for multi-node ============${C_OFF}\n"
    cat <<EOF
  NOTHING, on a stock DLAMI. The plugin is preinstalled and auto-discovered.
  Just make sure LD_LIBRARY_PATH survives into the job (mpirun -x LD_LIBRARY_PATH,
  or srun --export=ALL) — losing it is the usual way a run silently drops to TCP.

  VERIFY, once, on a real 2-node run:
      NCCL_DEBUG=INFO <launcher> ...   # look for:  NET/OFI Selected Provider is efa
  If you instead see "NET/Socket" or "No plugin found", you are on TCP.

  DO NOT set these — they are widely copy-pasted but obsolete with a modern
  plugin (this DLAMI ships ${ver:-1.18.x}, and they only applied to <= 1.5.0):
      FI_PROVIDER=efa            # the plugin filters to efa itself now
      FI_EFA_USE_DEVICE_RDMA=1   # upstream says do NOT set on libfabric >= 1.18
      NCCL_PROTO=simple          # actively HURTS latency now (disables LL/LL128)
      FI_EFA_FORK_SAFE=1         # set automatically

  EC2-side requirements (these are the ones that actually bite):
    - EFA enabled on EVERY network card (p4d = 4, p5 = 32). One ENI => a fraction
      of the bandwidth.
    - Security group must allow ALL inbound+outbound traffic TO AND FROM ITSELF.
    - All nodes in the same subnet/AZ; a cluster placement group is recommended.
      EFA traffic is not routable and cannot cross AZs.
EOF
    printf "${C_HEAD}============================================================${C_OFF}\n"
}

stage_verify() {
    CURRENT_STAGE="verify"; banner "verify imports"
    activate_env
    python - <<'PY'
import importlib, importlib.metadata as md
def show(mod, dist=None):
    try:
        m = importlib.import_module(mod)
        v = getattr(m, "__version__", None) or (md.version(dist) if dist else "?")
        print(f"  [ok] {mod:<22} {v}")
    except Exception as e:
        print(f"  [--] {mod:<22} NOT importable: {e}")
import torch
print(f"  [ok] {'torch':<22} {torch.__version__} | cuda {torch.version.cuda}")
print(f"  [ok] {'gpus visible':<22} {torch.cuda.device_count()} "
      f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'})")
show("torchvision")
show("apex", "apex")          # NVIDIA apex always reports 0.1 — the git tag is the real version
# The real test that apex built: the compiled .so files. (torch is already imported
# above, which they need — they link libc10.so. Note apex.amp no longer exists.)
try:
    import amp_C, fused_layer_norm_cuda           # noqa: F401
    print(f"  [ok] {'apex fused kernels':<22} amp_C, fused_layer_norm_cuda")
except ImportError as e:
    print(f"  [--] {'apex fused kernels':<22} NOT importable: {e}")
show("flash_attn")
show("mpi4py")
show("triton")                # backs FusedExperts_Triton, the primus_turbo stand-in on NVIDIA
show("deepspeed")             # X-MoE installs as the 'deepspeed' package
try:
    from deepspeed.moe.experts import FusedExperts_Triton   # noqa: F401
    print("  [ok] FusedExperts_Triton   importable (NVIDIA grouped-GEMM backend)")
except Exception as e:
    print(f"  [--] FusedExperts_Triton   NOT importable: {e}")
PY
    ok "verification complete."
    info "primus_turbo is intentionally absent on NVIDIA — use the Triton backend (use_triton=True)."
    info "For multi-node, also run:  ./setup_env_cuda.sh efa"
}

# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------
run_all() {
    banner "FULL SETUP"
    # Echo every resolved path and pin, so a reviewer can see exactly what their
    # overrides produced before a 30-minute build starts.
    info "repo:       $XMOE_ROOT"
    info "ELMOE_ROOT: $ELMOE_ROOT"
    info "env:        $ENV_PREFIX"
    info "deps:       $DEPS_DIR"
    info "cache:      $CACHE_DIR"
    info "versions:   python $PYTHON_VERSION | torch $TORCH_VERSION | cuda $CUDA_VERSION | apex $APEX_TAG | flash-attn $FLASH_ATTN_VERSION"
    info "build:      archs $TORCH_CUDA_ARCH_LIST | MAX_JOBS=$MAX_JOBS | need ${MIN_FREE_GB}GB free"
    # Fail fast: check disk BEFORE the first stage, not 25 min into apex.
    check_space
    local stages=(conda cuda torch apex mpi4py flashattn xmoe verify)
    local n=${#stages[@]} i=0
    for s in "${stages[@]}"; do
        i=$((i+1))
        printf "\n${C_HEAD}########## STAGE %d/%d : %s ##########${C_OFF}\n" "$i" "$n" "$s"
        "stage_$s"
    done
    printf "\n${C_OK}All stages complete.${C_OFF}\n"
    info "Multi-node runs: check the interconnect with  ./setup_env_cuda.sh efa"
}

STAGE="${1:-all}"
case "$STAGE" in
    all)                    run_all ;;
    conda|env)              stage_conda ;;
    cuda|toolkit)           stage_cuda ;;
    torch)                  stage_torch ;;
    apex)                   stage_apex ;;
    mpi4py)                 stage_mpi4py ;;
    flashattn|flash-attn)   stage_flashattn ;;
    xmoe|x-moe)             stage_xmoe ;;
    efa|nccl|aws-ofi-nccl)  stage_efa ;;
    verify)                 stage_verify ;;
    primus)
        trap - EXIT
        echo "primus_turbo is AMD-only (Composable Kernel, MI250+); there is no primus stage on" >&2
        echo "NVIDIA. Use ELMoE's Triton grouped-GEMM backend instead: use_triton=True." >&2
        exit 1 ;;
    -h|--help|help)
        sed -n '2,60p' "$0" | sed 's/^#//; s/^ //'
        ;;
    *)
        trap - EXIT
        echo "ERROR: unknown stage '$STAGE'." >&2
        echo "Valid: conda cuda torch apex mpi4py flashattn xmoe verify   (no arg = all)" >&2
        echo "Extra: efa (multi-node interconnect check)" >&2
        exit 1 ;;
esac
