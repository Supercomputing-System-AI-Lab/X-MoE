#!/bin/bash
###############################################################################
# setup_env_rocm.sh — ELMoE reviewer reproduction environment (AMD / Frontier)
#
# Builds the full ELMoE software stack on Frontier (ROCm 6.4.1, gfx90a).
# Run from inside the X-MoE repo. Third-party sources are cloned into a sibling
# ELMoE_deps/ folder (NOT inside the repo, to keep the checkout clean).
#
#   X-MoE/            <- this repo (contains this script)
#   ELMoE_deps/       <- apex + flash-attention sources (created here)
#
# ---------------------------------------------------------------------------
# USAGE
#   ./setup_env_rocm.sh            # run ALL stages EXCEPT primus (see below)
#   ./setup_env_rocm.sh <stage>    # run a single stage (re-run one on failure)
#
#   stages:  conda  torch  apex  mpi4py  flashattn  xmoe  verify
#
#   Two stages are NOT part of the default run (Frontier-/calibration-specific).
#   Run them explicitly, only if you need them:
#     ./setup_env_rocm.sh aws-ofi-rccl   # Slingshot RCCL plugin -> ELMoE_deps/.
#                                        # Frontier-only; skip on non-Frontier systems.
#                                        # Prints the exports you must set afterward.
#     ./setup_env_rocm.sh primus         # calibrated primus_turbo backend
#
# PREREQUISITE (do this once, by hand — it is how you got this script):
#   module reset
#   module load cpe/24.11 PrgEnv-gnu/8.6.0 rocm/6.4.1 cray-mpich/9.1.0 \
#               craype-accel-amd-gfx90a miniforge3/23.11.0-0 ninja/1.12.1.lua
#   cd ~ && git clone https://github.com/Supercomputing-System-AI-Lab/X-MoE.git
#   # (for the ELMoE branch: git clone -b ELMoE https://github.com/.../X-MoE.git)
#   cd X-MoE && ./setup_env_rocm.sh
#
# OVERRIDES (env vars): ENV_PREFIX, DEPS_DIR, MAX_JOBS, APEX_GIT_REF,
#                       LIBFABRIC_PATH, GCC_NATIVE_BIN, RUNTIME_ENV_FILE
###############################################################################

set -euo pipefail

# --- resolve paths ---------------------------------------------------------
XMOE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- user-tunable configuration --------------------------------------------
# Conda env location. Reviewers: override with `ENV_PREFIX=/your/path ./setup_env_rocm.sh`.
ENV_PREFIX="${ENV_PREFIX:-$HOME/envs/ELMoE-ROCM6.4.1_repro}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

# Third-party sources live OUTSIDE the repo, in a sibling ELMoE_deps/ folder.
DEPS_DIR="${DEPS_DIR:-$(cd "$XMOE_ROOT/.." && pwd)/ELMoE_deps}"

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
RUNTIME_ENV_FILE="${RUNTIME_ENV_FILE:-$DEPS_DIR/elmoe_runtime_env.sh}"

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
# ELMoE runtime environment — SOURCE THIS in your SLURM job before srun.
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
    banner "FULL SETUP (all stages except primus)"
    info "env:   $ENV_PREFIX"
    info "deps:  $DEPS_DIR"
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
        sed -n '2,40p' "$0" | sed 's/^#//; s/^ //'
        ;;
    *)
        trap - EXIT   # not a stage failure — suppress the resume hint
        echo "ERROR: unknown stage '$STAGE'." >&2
        echo "Valid: conda torch apex mpi4py flashattn xmoe verify   (no arg = these)" >&2
        echo "Explicit extras: aws-ofi-rccl (Frontier-only), primus" >&2
        exit 1 ;;
esac
