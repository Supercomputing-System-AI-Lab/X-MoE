#!/bin/bash
###############################################################################
# setup_env_rocm.sh — X-MoE-4D reviewer reproduction environment (AMD / Frontier)
#
# Builds the full X-MoE-4D software stack on Frontier (ROCm 6.4.1, gfx90a).
# AMD counterpart of setup_env_cuda.sh — same layout, same one knob, same stages.
#
# ---------------------------------------------------------------------------
# WHERE THINGS GO — one knob: XMOE4D_ROOT
#
#   XMOE4D_ROOT=/lustre/orion/<proj>/proj-shared/$USER/xmoe-4d ./setup_env_rocm.sh
#
# That single variable relocates EVERYTHING this script writes:
#
#   $XMOE4D_ROOT/XMoE4D_envs/X-MoE-4D-ROCM6.4.1_repro   conda environment (~20 GB)
#   $XMOE4D_ROOT/XMoE4D_deps/                           apex + flash-attn (+ aws-ofi-rccl)
#   $XMOE4D_ROOT/XMoE4D_cache/                          pip + conda caches (these get big)
#
# Defaults to one level ABOVE the X-MoE checkout, so a repo at <dir>/X-MoE puts
# them alongside it as siblings — nothing large is ever written inside the repo.
#
# ON OLCF, PUT THIS ON LUSTRE, NOT $HOME:
#   * /ccs/home has a 50 GB quota that is typically almost full — a 20 GB env
#     will not fit, and NFS is the wrong filesystem for a conda env anyway
#     (thousands of ranks importing Python hammer its metadata server).
#   * /lustre/orion/<proj>/scratch IS PURGED (files untouched for a while are
#     deleted). Use proj-shared or world-shared, which are not purged:
#         /lustre/orion/<proj>/proj-shared/$USER/xmoe-4d
#   The script checks free space (and your quota) UP FRONT and refuses to start
#   if there is not room, rather than dying 20 GB into the torch install.
#
# ---------------------------------------------------------------------------
# USAGE
#   ./setup_env_rocm.sh            # run ALL stages EXCEPT the two below
#   ./setup_env_rocm.sh <stage>    # run a single stage (re-run one on failure)
#
#   stages:  conda  torch  apex  mpi4py  flashattn  xmoe  verify
#
#   Two stages are NOT part of the default run (Frontier-/calibration-specific).
#   Run them explicitly, only if you need them:
#     ./setup_env_rocm.sh aws-ofi-rccl   # Slingshot RCCL plugin -> XMoE4D_deps/.
#                                        # Frontier-only; skip on non-Frontier systems.
#                                        # Prints the exports you must set afterward.
#                                        # (NVIDIA equivalent: the DLAMI preinstalls
#                                        #  aws-ofi-nccl — see setup_env_cuda.sh efa)
#     ./setup_env_rocm.sh primus         # calibrated primus_turbo backend.
#                                        # AMD-only (Composable Kernel); on NVIDIA use
#                                        # X-MoE-4D's Triton grouped-GEMM instead.
#
# PREREQUISITE (do this once, by hand — it is how you got this script):
#   module reset
#   module load cpe/24.11 PrgEnv-gnu/8.6.0 rocm/6.4.1 cray-mpich/9.1.0 \
#               craype-accel-amd-gfx90a miniforge3/23.11.0-0 ninja/1.12.1.lua
#   cd /lustre/orion/<proj>/proj-shared/$USER            # NOT ~ — see above
#   git clone -b X-MoE-4D --single-branch https://github.com/Supercomputing-System-AI-Lab/X-MoE.git
#   cd X-MoE && ./setup_env_rocm.sh
#
# OVERRIDES: XMOE4D_ROOT, ENV_PREFIX, DEPS_DIR, CACHE_DIR, MAX_JOBS, APEX_GIT_REF,
#            LIBFABRIC_PATH, GCC_NATIVE_BIN, RUNTIME_ENV_FILE, MIN_FREE_GB
###############################################################################

set -euo pipefail

# --- resolve paths ---------------------------------------------------------
XMOE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- the one knob ----------------------------------------------------------
# Default: the directory ONE LEVEL ABOVE the X-MoE checkout, so a repo at
# <dir>/X-MoE puts env, deps and caches alongside it as siblings:
#
#   <dir>/X-MoE/          <- the repo (this script lives here)
#   <dir>/XMoE4D_envs/     <- conda environment
#   <dir>/XMoE4D_deps/     <- apex + flash-attn (+ aws-ofi-rccl) sources
#   <dir>/XMoE4D_cache/    <- pip + conda caches
#
# Identical layout to setup_env_cuda.sh. Override to relocate everything at once.
XMOE4D_ROOT="${XMOE4D_ROOT:-$(cd "$XMOE_ROOT/.." && pwd)}"

# Each derives from XMOE4D_ROOT but stays independently overridable.
ENV_PREFIX="${ENV_PREFIX:-$XMOE4D_ROOT/XMoE4D_envs/X-MoE-4D-ROCM6.4.1_repro}"
DEPS_DIR="${DEPS_DIR:-$XMOE4D_ROOT/XMoE4D_deps}"
CACHE_DIR="${CACHE_DIR:-$XMOE4D_ROOT/XMoE4D_cache}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

# Keep the fat caches off $HOME. This is NOT cosmetic on OLCF: pip caches the
# ~2.5 GB torch wheel and conda unpacks GBs of packages, and both default to
# $HOME (~/.cache/pip, ~/.conda/pkgs) — where the 50 GB quota is typically
# almost exhausted. Without these three lines the build blows the home quota
# even when ENV_PREFIX points at Lustre.
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$CACHE_DIR/pip}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$CACHE_DIR/conda}"
export TMPDIR="${TMPDIR:-$CACHE_DIR/tmp}"

# Free space required under XMOE4D_ROOT, checked BEFORE any stage runs. Measured:
# conda env ~20 GB + caches ~4 GB, peaking higher during the apex/flash-attn build.
MIN_FREE_GB="${MIN_FREE_GB:-25}"

# Pinned versions (reproduction targets).
APEX_VERSION="1.11.0"          # verified after build; warns on drift
APEX_GIT_REF="${APEX_GIT_REF:-}"   # optional exact commit/tag that yields 1.11.0
FLASH_ATTN_TAG="v2.8.3"        # built from source (ROCm needs a source build)
TORCH_INDEX="https://download.pytorch.org/whl/rocm6.4"   # validated: torch 2.9.1+rocm6.4
MAX_JOBS="${MAX_JOBS:-64}"

# AWS-OFI-RCCL plugin (Slingshot networking). These paths are environment-specific;
# verify against your loaded modules:  module show libfabric   |   echo "$MPICH_DIR"
# libfabric default matches the validated build (Cray libfabric 1.22.0).
LIBFABRIC_PATH="${LIBFABRIC_PATH:-/opt/cray/libfabric/1.22.0}"
# Native GCC 13 bin dir — runtime CC/CXX so on-the-fly C++/HIP extension rebuilds
# match the Stage-2 toolchain.
GCC_NATIVE_BIN="${GCC_NATIVE_BIN:-/opt/cray/pe/gcc-native/13/bin}"
# Runtime env file the rccl stage writes (source it inside your SLURM job).
RUNTIME_ENV_FILE="${RUNTIME_ENV_FILE:-$DEPS_DIR/xmoe_4d_runtime_env.sh}"

# Exact Frontier module stack (kept identical to the manual prerequisite).
MODULE_STACK="cpe/24.11 PrgEnv-gnu/8.6.0 rocm/6.4.1 cray-mpich/9.1.0 craype-accel-amd-gfx90a miniforge3/23.11.0-0 ninja/1.12.1.lua"

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
  printf "             ./setup_env_rocm.sh %s\n" "$CURRENT_STAGE";
fi' EXIT

# ---------------------------------------------------------------------------
# shared setup: modules + conda activation (run at the start of every stage so
# a single stage can be re-run in a fresh shell)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# preflight_space — fail loudly BEFORE building, not 20 GB into a torch install.
# Checks free space on the filesystem that will hold the target dir (walking up to
# the nearest existing parent, since the dir itself may not exist yet).
# ---------------------------------------------------------------------------
# Remaining USER QUOTA in GiB on the filesystem holding $HOME, or empty if there is
# no quota. On a quota'd filesystem (e.g. OLCF /ccs/home, 50 GB) `df` reports the
# whole filesystem's free space — 7.7 TB — which is meaningless to the user, so we
# must consult the quota instead or the check silently passes.
home_quota_free_gib() {
    command -v quota >/dev/null 2>&1 || return 0
    quota -s 2>/dev/null | awk '
        # rows: <used> <soft> <hard> ...  e.g. "46816M  51200M  51200M"
        function to_gib(v,   u, n) {
            u = substr(v, length(v)); n = substr(v, 1, length(v)-1) + 0
            if (u == "K") return n/1048576; if (u == "M") return n/1024
            if (u == "G") return n;         if (u == "T") return n*1024
            return (v + 0)/1073741824       # plain bytes
        }
        $1 ~ /^[0-9]+[KMGT]?$/ && $2 ~ /^[0-9]+[KMGT]?$/ {
            used = to_gib($1); lim = to_gib($2)
            if (lim > 0) { printf "%d\n", (lim - used); exit }
        }'
}

# Free space in GiB for a path that may not exist yet (walk up to the nearest
# existing parent). If the path lives on the same filesystem as $HOME, take the
# MIN of df-free and the user's remaining quota.
free_gib() {
    local d="$1"
    while [ ! -d "$d" ] && [ "$d" != "/" ]; do d="$(dirname "$d")"; done
    local dffree
    dffree="$(df -BG --output=avail "$d" 2>/dev/null | tail -1 | tr -dc '0-9')"

    if [ "$(stat -c %d "$d" 2>/dev/null)" = "$(stat -c %d "$HOME" 2>/dev/null)" ]; then
        local q; q="$(home_quota_free_gib)"
        if [ -n "$q" ] && { [ -z "$dffree" ] || [ "$q" -lt "$dffree" ]; }; then
            echo "$q"; return 0
        fi
    fi
    echo "$dffree"
}

check_space() {
    local avail; avail="$(free_gib "$XMOE4D_ROOT")"
    info "free space at XMOE4D_ROOT=${XMOE4D_ROOT}: ${avail:-?} GB (need >= ${MIN_FREE_GB} GB)"

    case "$XMOE4D_ROOT" in
        "$HOME"/*|"$HOME"|/ccs/home/*)
            warn "XMOE4D_ROOT is under your NFS home. It is small (50 GB quota on OLCF) and"
            warn "is the wrong filesystem for a conda env — thousands of ranks importing"
            warn "Python hammer its metadata server. Use a parallel filesystem." ;;
    esac

    if [ -n "$avail" ] && [ "$avail" -lt "$MIN_FREE_GB" ]; then
        printf "\n${C_ERR}[FAILED]${C_OFF} only %s GB free (or left in quota) on the filesystem holding\n" "$avail" >&2
        printf "         XMOE4D_ROOT=%s  (need >= %s GB).\n" "$XMOE4D_ROOT" "$MIN_FREE_GB" >&2
        cat >&2 <<EOF

         The full stack needs ~25 GB (a ~20 GB conda env plus pip/conda caches).
         Point XMOE4D_ROOT at a bigger filesystem. On OLCF use a NON-PURGED
         Lustre area (scratch IS purged; /ccs/home is quota-limited):

             XMOE4D_ROOT=/lustre/orion/<proj>/proj-shared/\$USER/xmoe-4d ./setup_env_rocm.sh

         Aborting before any build. Nothing was installed.
EOF
        exit 1
    fi
    ok "check_space: enough room."
}

ensure_module_cmd() {
    if ! type module >/dev/null 2>&1; then
        source /etc/profile.d/lmod.sh 2>/dev/null \
            || source /usr/share/lmod/lmod/init/bash 2>/dev/null \
            || { echo "ERROR: 'module' command unavailable and lmod init not found." >&2; exit 1; }
    fi
}

load_modules() {
    ensure_module_cmd
    module reset
    # shellcheck disable=SC2086
    module load $MODULE_STACK
    info "modules loaded: $MODULE_STACK"
}

conda_hook() {
    # miniforge3 module puts conda on PATH; source its shell hook so `activate` works.
    local base; base="$(conda info --base)"
    source "$base/etc/profile.d/conda.sh"
}

activate_env() {
    conda_hook
    if [ ! -d "$ENV_PREFIX" ]; then
        echo "ERROR: conda env '$ENV_PREFIX' not found. Run './setup_env_rocm.sh conda' first." >&2
        exit 1
    fi
    conda activate "$ENV_PREFIX"
    info "python: $(which python)  ($(python --version 2>&1))"
}

# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------
stage_conda() {
    CURRENT_STAGE="conda"; banner "conda env ($ENV_PREFIX, python $PYTHON_VERSION)"
    check_space
    mkdir -p "$XMOE4D_ROOT" "$DEPS_DIR" "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" "$TMPDIR"
    load_modules
    conda_hook
    if [ -d "$ENV_PREFIX" ]; then
        warn "env already exists at $ENV_PREFIX — reusing it (delete to recreate)."
    else
        conda create -y -p "$ENV_PREFIX" "python=$PYTHON_VERSION"
    fi
    conda activate "$ENV_PREFIX"
    ok "conda env ready and active."
}

stage_torch() {
    CURRENT_STAGE="torch"; banner "PyTorch (ROCm 6.4)"
    load_modules; activate_env
    pip3 install torch torchvision --index-url "$TORCH_INDEX"
    python -c "import torch; print('  torch', torch.__version__, '| hip', torch.version.hip)"
    ok "torch installed."
}

stage_apex() {
    CURRENT_STAGE="apex"; banner "apex (target $APEX_VERSION)"
    load_modules; activate_env
    mkdir -p "$DEPS_DIR"; cd "$DEPS_DIR"
    if [ ! -d apex/.git ]; then
        git clone https://github.com/ROCm/apex.git
    else
        info "apex/ already cloned — reusing."
    fi
    cd apex
    [ -n "$APEX_GIT_REF" ] && { info "checkout $APEX_GIT_REF"; git checkout "$APEX_GIT_REF"; }
    pip install -r requirements.txt
    rm -rf build
    python setup.py install --cpp_ext --cuda_ext
    local got; got="$(python -c 'import apex, importlib.metadata as m; print(m.version("apex"))' 2>/dev/null || echo unknown)"
    if [ "$got" = "$APEX_VERSION" ]; then
        ok "apex $got installed (matches pin)."
    else
        warn "apex installed version is '$got', expected $APEX_VERSION."
        warn "If reproduction requires exactly $APEX_VERSION, set APEX_GIT_REF to the"
        warn "commit/tag that yields it and re-run: APEX_GIT_REF=<ref> ./setup_env_rocm.sh apex"
    fi
}

stage_mpi4py() {
    CURRENT_STAGE="mpi4py"; banner "mpi4py (Cray MPICH)"
    load_modules; activate_env
    MPICC="cc -shared" pip install --no-cache-dir --force-reinstall --no-binary=mpi4py mpi4py
    python -c "import mpi4py; print('  mpi4py', mpi4py.__version__)"
    ok "mpi4py installed."
}

stage_flashattn() {
    CURRENT_STAGE="flashattn"; banner "flash-attention ($FLASH_ATTN_TAG, source build — this is slow)"
    load_modules; activate_env
    mkdir -p "$DEPS_DIR"; cd "$DEPS_DIR"
    if [ ! -d flash-attention/.git ]; then
        git clone https://github.com/Dao-AILab/flash-attention.git
    else
        info "flash-attention/ already cloned — reusing."
    fi
    cd flash-attention
    git checkout "$FLASH_ATTN_TAG"     # repo HEAD is FA4 beta otherwise
    MAX_JOBS="$MAX_JOBS" pip install . --no-build-isolation
    python -c "import flash_attn; print('  flash_attn', flash_attn.__version__)"
    ok "flash-attention $FLASH_ATTN_TAG installed."
}

stage_xmoe() {
    CURRENT_STAGE="xmoe"; banner "X-MoE + Megatron-DeepSpeed-X-MoE (editable)"
    load_modules; activate_env
    cd "$XMOE_ROOT"
    git submodule update --init --recursive
    pip install -e .
    cd "$XMOE_ROOT/Megatron-DeepSpeed-X-MoE"
    pip install -e .
    ok "X-MoE stack installed (editable)."
}

write_runtime_env() {
    # A script cannot persist exports into your shell/SLURM job, so emit a file to
    # `source`. Build-time values expand now; runtime vars ($LD_LIBRARY_PATH,
    # $PYTHONPATH) stay literal so they append at source time.
    local plugin_lib="$DEPS_DIR/aws-ofi-rccl/lib"
    cat > "$RUNTIME_ENV_FILE" <<EOF
# X-MoE-4D runtime environment — SOURCE THIS in your SLURM job before srun.
# Generated by setup_env_rocm.sh (rccl stage); paths resolved at build time.
export NCCL_NET_PLUGIN="$plugin_lib/librccl-net.so"
export LD_LIBRARY_PATH="$plugin_lib:\$LD_LIBRARY_PATH"
export PYTHONPATH="$XMOE_ROOT:$XMOE_ROOT/primus_turbo:\$PYTHONPATH"
# Pin CC/CXX to native GCC 13 so on-the-fly C++/HIP extension rebuilds match Stage 2.
export CC="$GCC_NATIVE_BIN/gcc"
export CXX="$GCC_NATIVE_BIN/g++"
EOF
    ok "wrote runtime env: $RUNTIME_ENV_FILE"
    info "add to your SLURM script:  source $RUNTIME_ENV_FILE"
}

stage_rccl() {
    CURRENT_STAGE="aws-ofi-rccl"; banner "AWS-OFI-RCCL plugin (Slingshot / libfabric)"
    load_modules
    : "${MPICH_DIR:?MPICH_DIR unset — is cray-mpich loaded?}"
    local rocm="${ROCM_PATH:-/opt/rocm-6.4.1}"
    info "ROCm=$rocm | libfabric=$LIBFABRIC_PATH | MPICH_DIR=$MPICH_DIR"
    [ -d "$LIBFABRIC_PATH" ] || warn "libfabric path '$LIBFABRIC_PATH' not found — check 'module show libfabric' and set LIBFABRIC_PATH."
    mkdir -p "$DEPS_DIR"; cd "$DEPS_DIR"
    if [ ! -d aws-ofi-rccl/.git ]; then
        git clone --recursive https://github.com/ROCmSoftwarePlatform/aws-ofi-rccl
    else
        info "aws-ofi-rccl/ already cloned — reusing."
    fi
    cd aws-ofi-rccl
    ./autogen.sh
    CC=hipcc CFLAGS="-I$rocm/include" ./configure \
        --with-libfabric="$LIBFABRIC_PATH" \
        --with-rccl="$rocm" \
        --with-hip="$rocm" \
        --with-mpi="$MPICH_DIR" \
        --prefix="$PWD"
    make -j"$MAX_JOBS"
    make install
    # Some RCCL versions look for libnccl-net.so instead of librccl-net.so.
    if [ -f lib/librccl-net.so ] && [ ! -e lib/libnccl-net.so ]; then
        ln -s librccl-net.so lib/libnccl-net.so
        info "symlinked lib/libnccl-net.so -> librccl-net.so"
    fi
    local plugin_lib="$PWD/lib"
    write_runtime_env
    ok "aws-ofi-rccl plugin built at $plugin_lib"

    # ---- clear, required "what to do next" instructions --------------------
    printf "\n${C_HEAD}============ NEXT STEPS — required to USE the plugin ============${C_OFF}\n"
    printf "The plugin is built but does NOTHING until these are set in the\n"
    printf "environment of your training job. Add ONE of the following to your\n"
    printf "SLURM script, BEFORE srun:\n\n"
    printf "  ${C_OK}(A)${C_OFF} source the generated env file (also sets PYTHONPATH + CC/CXX):\n"
    printf "        source %s\n\n" "$RUNTIME_ENV_FILE"
    printf "  ${C_OK}(B)${C_OFF} or export the two plugin lines directly:\n"
    printf "        export NCCL_NET_PLUGIN=%s/librccl-net.so\n" "$plugin_lib"
    printf "        export LD_LIBRARY_PATH=%s:\$LD_LIBRARY_PATH\n" "$plugin_lib"
    printf "${C_HEAD}================================================================${C_OFF}\n"
    printf "  (single-node jobs won't initialize the plugin — that is expected on Frontier)\n"
}

stage_primus() {
    CURRENT_STAGE="primus"; banner "primus_turbo (calibrated build — explicit stage)"
    load_modules; activate_env
    cd "$XMOE_ROOT"
    git submodule update --init --recursive primus_turbo
    cd "$XMOE_ROOT/primus_turbo"
    bash frontier_setup.sh
    ok "primus_turbo installed."
}

stage_verify() {
    CURRENT_STAGE="verify"; banner "verify imports"
    load_modules; activate_env
    python - <<'PY'
import importlib, importlib.metadata as md
def show(mod, dist=None):
    try:
        m = importlib.import_module(mod)
        v = getattr(m, "__version__", None) or (md.version(dist) if dist else "?")
        print(f"  [ok] {mod:<20} {v}")
    except Exception as e:
        print(f"  [--] {mod:<20} NOT importable: {e}")
import torch
print(f"  [ok] torch                {torch.__version__} | hip {torch.version.hip}")
show("torchvision")
show("apex", "apex")
show("flash_attn")
show("mpi4py")
show("deepspeed")          # X-MoE installs as the 'deepspeed' package
show("primus_turbo.pytorch")   # only present after the primus stage
PY
    if [ -f "$DEPS_DIR/aws-ofi-rccl/lib/librccl-net.so" ]; then
        echo "  [ok] aws-ofi-rccl         $DEPS_DIR/aws-ofi-rccl/lib/librccl-net.so"
        [ -f "$RUNTIME_ENV_FILE" ] && echo "  [ok] runtime env          $RUNTIME_ENV_FILE (source in SLURM)"
    else
        echo "  [--] aws-ofi-rccl         not built (run: ./setup_env_rocm.sh aws-ofi-rccl)"
    fi
    ok "verification complete (missing primus_turbo is expected until you run the primus stage)."
}

# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------
run_all() {
    banner "FULL SETUP (all stages except aws-ofi-rccl / primus)"
    # Echo every resolved path and pin, so a reviewer sees exactly what their
    # overrides produced before a long build starts.
    info "repo:       $XMOE_ROOT"
    info "XMOE4D_ROOT: $XMOE4D_ROOT"
    info "env:        $ENV_PREFIX"
    info "deps:       $DEPS_DIR"
    info "cache:      $CACHE_DIR"
    info "versions:   python $PYTHON_VERSION | torch (rocm6.4 index) | apex $APEX_VERSION | flash-attn $FLASH_ATTN_TAG"
    info "build:      MAX_JOBS=$MAX_JOBS | need ${MIN_FREE_GB}GB free"
    # Fail fast: check disk BEFORE the first stage, not 20 GB into the torch install.
    check_space
    local stages=(conda torch apex mpi4py flashattn xmoe verify)
    local n=${#stages[@]} i=0
    for s in "${stages[@]}"; do
        i=$((i+1))
        printf "\n${C_HEAD}########## STAGE %d/%d : %s ##########${C_OFF}\n" "$i" "$n" "$s"
        "stage_$s"
    done
    printf "\n${C_OK}Core stages complete.${C_OFF}\n"
    info "Explicit extra stages (run only if you need them):"
    info "    ./setup_env_rocm.sh aws-ofi-rccl   # Frontier/Slingshot RCCL plugin (multi-node)"
    info "    ./setup_env_rocm.sh primus         # calibrated primus_turbo backend"
}

STAGE="${1:-all}"
case "$STAGE" in
    all)                    run_all ;;
    conda|env)              stage_conda ;;
    torch)                  stage_torch ;;
    apex)                   stage_apex ;;
    mpi4py)                 stage_mpi4py ;;
    flashattn|flash-attn)   stage_flashattn ;;
    xmoe|x-moe)             stage_xmoe ;;
    aws-ofi-rccl|rccl)      stage_rccl ;;
    primus)                 stage_primus ;;
    verify)                 stage_verify ;;
    -h|--help|help)
        sed -n '3,59p' "$0" | sed 's/^#//; s/^ //'
        ;;
    *)
        trap - EXIT   # not a stage failure — suppress the resume hint
        echo "ERROR: unknown stage '$STAGE'." >&2
        echo "Valid: conda torch apex mpi4py flashattn xmoe verify   (no arg = these)" >&2
        echo "Explicit extras: aws-ofi-rccl (Frontier-only), primus" >&2
        exit 1 ;;
esac
