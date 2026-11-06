# X-MoE branch `ft-sys-v2`: failure detection and warm recovery on Frontier

ft-sys-v2 adds two things to X-MoE training on Frontier. A heartbeat finds a failure within seconds: a killed rank, a hang, a
lost node, a hardware fault, out of memory, an exception. A warm recovery then lets a standby node take over the lost node's ranks
inside the same job, with the optimizer state restored from a partner's copy in memory. Nothing runs unless `XMOE_FT_V2=1`
(heartbeat) and `FTV2_RECOVERY=1` (recovery) are set.
The code and its own README are in the Megatron-DeepSpeed submodule:
[`Megatron-DeepSpeed-X-MoE/examples_elmoe/scripts-frontier/ft-sys-v2/README.md`](Megatron-DeepSpeed-X-MoE/examples_elmoe/scripts-frontier/ft-sys-v2/README.md).

## 1. Clone

```
git clone -b ft-sys-v2 --recurse-submodules https://github.com/Supercomputing-System-AI-Lab/X-MoE.git
cd X-MoE
git submodule update --init Megatron-DeepSpeed-X-MoE        # the pinned commit of Megatron-DeepSpeed branch ft-sys-v2
git -C Megatron-DeepSpeed-X-MoE checkout ft-sys-v2          # on the branch instead of a detached HEAD
# later commits of the Megatron side:  git submodule update --remote Megatron-DeepSpeed-X-MoE   (.gitmodules: branch = ft-sys-v2)
```

- The two repositories:
  - This branch is X-MoE (the DeepSpeed fork). It holds the DeepSpeed files the ft-sys-v2 runs imported (section 3).
  - `Megatron-DeepSpeed-X-MoE` is pinned to a commit of branch `ft-sys-v2` of
    https://github.com/Supercomputing-System-AI-Lab/Megatron-DeepSpeed. That branch holds ft-sys-v2 itself, its run files and
    the launcher.
- `--recurse-submodules` also clones `primus_turbo` (https://github.com/HenryJlh/Primus-Turbo) and its `composable_kernel`.
  Both are large, and these runs do not use them (X-MoE never sets `USE_AITER_GROUPGEMM`). To skip them, clone without
  `--recurse-submodules` and run only the `git submodule update --init Megatron-DeepSpeed-X-MoE` line.

## 2. Environment and every dependency outside the two repositories

The runs use the "new stack": PyTorch 2.13 with the ROCm 7.14.1 runtime for gfx90a (MI250X). Every path is a variable of
`ft-sys-v2/env/ftv2_paths.sh`. A variable you set in your shell wins. The defaults are the folders the runs used on 2026-11-06.

- Folders that every member of gen150 can read: `/lustre/orion/world-shared/gen150` and `/opt`.
- Folders that only the owner can read: `/lustre/orion/gen150/scratch/zixianw4` (`drwx------`). Use a copy of your own for
  each of these.

| What | Variable | Default (readable by gen150?) | Where to get it |
|---|---|---|---|
| Frontier modules: cpe/24.11, PrgEnv-gnu/8.6.0, rocm/6.4.1 then rocm/7.1.1, cray-mpich/9.1.0, craype-accel-amd-gfx90a, miniforge3/23.11.0-0 | - | site (yes) | loaded with explicit versions by `scripts-frontier/eval/submit_env.sh` and `ft-sys-v2/env/newstack_env.sh` |
| Python venv: torch 2.13.0+rocm7.1, apex 1.16.0a0, flash-attn 2.8.3 (CK build, gfx90a, head dim 128), mpi4py 4.1.2 (built on Cray MPICH), transformers 4.36.2, numpy 2.4.6 | `NEWSTACK_V` | `/lustre/orion/world-shared/gen150/zixianw4/envs/TORCH2.13-ROCM7.14-XMOE_venv` (yes); its base interpreter is the conda env `.../envs/TORCH-ROCM6.4.1_env_groupgemm_v5` (yes) | use it as it is (read only). To build your own: torch from `https://download.pytorch.org/whl/rocm7.1`, apex from `https://github.com/ROCm/apex`, flash-attention v2.8.3 from `https://github.com/Dao-AILab/flash-attention` (`OPT_DIM=128 GPU_ARCHS=gfx90a`), every C++/HIP build with ROCm 7.1.1 |
| this branch's DeepSpeed | `NEWSTACK_DEEPSPEED` | the user's X-MoE checkout (no) | your clone's root (section 4) |
| ROCm 7.14.1 for gfx90a (HIP, HSA, RCCL 2.30.4). The site's rocm/7.14.1 RCCL has no gfx90a code. | `NEWSTACK_R714` | `/lustre/orion/gen150/scratch/zixianw4/therock-7.14.1-gfx90a/rocm` (no) | `https://repo.amd.com/rocm/tarball-multi-arch/therock-dist-linux-gfx90a-7.14.1.tar.gz` (2,045,529,743 bytes, md5 `3e90c49b60f2029e7a6353a11332988a`): `mkdir -p <dir>/rocm && tar -xzf <tarball> -C <dir>/rocm` |
| aws-ofi-nccl 1.22.0 (RCCL's network plugin) | `NEWSTACK_PLUGIN` | `/lustre/orion/gen150/scratch/zixianw4/aws-ofi-nccl-v1.22.0/install/lib` (no) | `https://github.com/aws/aws-ofi-nccl`, tag v1.22.0: `autoreconf -ivf && ./configure --prefix=<dir>/install --with-rocm=/opt/rocm-7.14.1 --with-libfabric=/opt/cray/libfabric/2.3.1 --disable-tests --enable-werror=no CC=gcc CXX=g++ && make -j && make install` |
| libfabric 2.3.1 | `NEWSTACK_LIBFABRIC` | `/opt/cray/libfabric/2.3.1/lib64` (yes) | site |
| JIT build caches (torch extensions, Megatron fused kernels, Triton); the job writes here | `NEWSTACK_JIT` | a folder in the user's scratch (no) | any folder you can write on Lustre |
| Triton loaded with RTLD_DEEPBIND (`triton_deepbind/sitecustomize.py`) | `NEWSTACK_OVERLAY` | a folder in the user's scratch (no) | `Megatron-DeepSpeed-X-MoE/examples_elmoe/scripts-frontier/ft-sys-v2/env/overlay` (a copy in the repository) |
| etcd 3.5.17 (the heartbeat's key-value store) | `FTV2_ETCD_BIN` | `/lustre/orion/world-shared/gen150/zixianw4/tools/etcd-v3.5.17/etcd-v3.5.17-linux-amd64/etcd` (yes) | or `bash ft-sys-v2/heartbeat-sys/get_etcd.sh <dir>` (GitHub release, SHA-256 checked) |
| py-spy (the stacks of a hang) | PATH, else `~/.local/bin/py-spy` | each user's own | `pip install --user py-spy` (0.4.2 tested) |
| training data (Megatron .bin + .idx, DeepSeek tokenizer) | `DATA_PATH` (the job template reads it) | `.../examples_elmoe/data/longctx/L8-dolma3-deepseek_text_document` in the user's scratch, 6.8 GB (no) | a copy from the user, or any dataset made with `examples_elmoe/data/README.md` ("For deepseek-moe-16B") |
| tokenizer | `FTV2_VOCAB_FILE` | `.../examples_elmoe/data/deepseek_tokenizer/tokenizer.json` in the user's scratch (no) | `tokenizer.json` of `deepseek-ai/deepseek-moe-16b-base` on Hugging Face (100,015 tokens; Megatron pads to 102,400) |
| Weights & Biases | `WANDB_MODE` | online, the user's entity | `export WANDB_MODE=disabled`, or `wandb login` once |
| Slurm | - | account gen150, QOS debug, 30 min | - |

`bash Megatron-DeepSpeed-X-MoE/examples_elmoe/scripts-frontier/ft-sys-v2/env/ftv2_check_env.sh` checks every row above with your
variables. It prints each path as OK, MISSING, NOT READABLE or NOT WRITABLE, together with the variable that sets it. It also
prints where `deepspeed` and `megatron` will be imported from, and it exits 1 on any problem.

## 3. What this branch changes in DeepSpeed

The branch starts at `35b4eb47` of `feat/deepseek-v3`. That commit is not on any remote branch, so pushing `ft-sys-v2` also
pushes its history. Two commits follow with the six DeepSpeed files exactly as every ft-sys-v2 run imported them. These are the
ft-sys-v1 resilience hooks, and each one does nothing unless its flag is set:

| File | Change | Active only with |
|---|---|---|
| `deepspeed/comm/torch.py` | the timeout of the groups DeepSpeed creates (the expert group among them) | `XMOE_PG_TIMEOUT_MIN` |
| `deepspeed/moe/moe_v2.py` | heartbeat marks around the dispatch-splits, dispatch and combine all-to-alls | `XMOE_HB=1` (ft-sys-v1) |
| `deepspeed/sequence/layer.py` | heartbeat marks around the Ulysses all-to-all | `XMOE_HB=1` |
| `deepspeed/runtime/pipe/p2p.py` | heartbeat marks around pipeline send/recv | `XMOE_HB=1` |
| `deepspeed/runtime/pipe/engine.py` | heartbeat phase forward / backward / step | `XMOE_HB=1` |
| `deepspeed/runtime/checkpoint_engine/async_pinned_checkpoint_engine.py` | node-local checkpoint copy in `/dev/shm`, wait for a published save, `stop_for_exit`, skip of a repeated tag, save-time record | the async engine (`DS_ASYNC_PINNED_CHECKPOINT=1`); the copy also needs `AUTORUN_LOCAL_COPIES` |

`csrc/adam/multi_tensor_apply.cuh` and `csrc/lion/multi_tensor_apply.cuh` are kept. They were deleted in the user's working copy,
but nothing explains the deletion. `multi_tensor_adam.cu` and `multi_tensor_lion.cu` include them, and there is no other copy.

## 4. This DeepSpeed in the Python env

- **Shared venv (`NEWSTACK_V`, read only).** No install is needed. Set `NEWSTACK_DEEPSPEED=<your X-MoE clone>` and the run puts
  your clone first on PYTHONPATH. `ftv2_run_common.sh` stops at render time if `deepspeed` would come from anywhere else. The
  venv's own editable `deepspeed` points at the user's checkout, which you cannot read; it is used only when nothing comes
  first on PYTHONPATH.
- **Your own env.** Install this DeepSpeed and the Megatron side into it:
  ```
  cd X-MoE && pip install -e .                               # DeepSpeed ops are built just in time; no build flags needed
  cd Megatron-DeepSpeed-X-MoE && pip install -e .
  ```
  Then set `NEWSTACK_V=<your venv>`. Never install into the shared venv: you cannot write there, and it serves the user's runs.

## 5. Run the 10B example

```
cd X-MoE/Megatron-DeepSpeed-X-MoE
FT=$PWD/examples_elmoe/scripts-frontier/ft-sys-v2
export NEWSTACK_DEEPSPEED=$(cd .. && pwd) NEWSTACK_OVERLAY=$FT/env/overlay
export NEWSTACK_JIT=/lustre/orion/<project>/scratch/$USER/ftv2_jit
export NEWSTACK_R714=<dir>/rocm NEWSTACK_PLUGIN=<dir>/install/lib
export DATA_PATH=<data prefix> FTV2_VOCAB_FILE=<tokenizer.json> WANDB_MODE=disabled
bash $FT/env/ftv2_check_env.sh                 # 0 problems before going on
cd $FT/runs && RENDER_ONLY=1 bash ftv2_10B.sh  # writes temp_slurm/*.slurm, submits nothing
bash ftv2_10B.sh                               # X-MoE 10B, 3 nodes + 1 standby, EP 8, debug QOS, 30 min
```

- Where the job writes: `examples_elmoe/scripts-frontier/logs/job_<id>_.../`. The heartbeat's records go to `ft/` inside it.
- How to inject failures: see `FTV2_ARMS` and `FTV2_INJECT` in the ft-sys-v2 README.
- The other run files are `ftv2_63B.sh`, `ftv2_DeepSeek_V3_671B_L2.sh` (2 layers), `ftv2_DeepSeek_V3_671B_L8D1.sh` (8 layers),
  `ftv2_DeepSeek_V3_671B.sh` (full) and `ftv2_DeepSeek_16B.sh`.
- The OOM path that rebuilds the job with a different world size is described but not built. It plugs in at "A new recovery
  plan: where it plugs in", item 4, in the ft-sys-v2 README.

## 6. What is tested

| Part | Tested how | Record |
|---|---|---|
| Heartbeat: the failure list for a killed rank, an aborted rank, a hang, a lost node (also the coordinator's node), out of memory, an exception, every rank stuck; a pause or a slow rank gives no list | GPU, X-MoE 10B, 3 nodes, 2026-11-05 (11 jobs, HB2d to HB12) | the user's record (`2026-11-05-ftv2_heartbeat/jobs.tsv`, not in the repository) |
| Milestone 1: warm recovery at the step boundary (standby takes the lost node, Keep restores the optimizer state) | GPU, X-MoE 10B, 2026-11-06. Failure list to ready in about 8-9 s; first step after it about 23 s (a cold restart in the same allocation takes about 106 s); Keep costs 3-6% of the step; bit-identical to an uninterrupted run in the determinism mode | ft-sys-v2 README, "What the recovery covers today"; the cold restart: the user's record (C1) |
| Milestone 2: resume inside a step (`FTV2_COMMIT=1`, ZeRO stage 2) | GPU, X-MoE 10B, 2026-11-06. A kill in the middle of step 10 resumes at micro-batch 3; four failure kinds resume inside the step; bit-exact in the determinism mode | ft-sys-v2 README (M2-A, E5d); the user's record (E6, E7) |
| DeepSeek-V3 2 layers with Keep | GPU, 2026-11-06, trained (through the record's own run file) | the user's record (V3-2L, not in the repository) |
| Real-fault handling: a decision when no stack can be taken (process in D state, dead GPU), every process group's timeout at least 1800 s (Megatron `b2635e0f`, `de93e355`) | CPU and login-node tests only, until its GPU run reports | ft-sys-v2 heartbeat-sys and recovery tests |
| The run files of 16B, V3 2 layers, V3 8 layers, V3 full, 63B | rendered only (no GPU run through these files) | ft-sys-v2 README run table |
| Run from a fresh clone (paths as variables, `ftv2_check_env.sh`) | a clone of Megatron `ft-sys-v2` rendered all six run files with the default variables (2026-11-06). The 10B job script is the same as the one rendered in the user's tree, apart from the tree's path. | no GPU job |
| OOM ladder with a new world size | not built (`MegatronTrainer.set_world_size` is written but not tested) | ft-sys-v2 README |
