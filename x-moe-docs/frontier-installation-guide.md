# X-MoE-4D Installation Guide for OLCF Frontier

This guide installs and runs X-MoE-4D on **AMD GPUs**, with the OLCF Frontier supercomputer (AMD MI250X,
ROCm 6.4.1) as the example. The same steps are a reference for other AMD systems.

There are two ways to build the environment: `setup_env_rocm.sh` runs every step with the versions that were
validated, and the manual steps below run the same commands one by one.

## 1. Clone X-MoE-4D

```bash
cd /lustre/orion/<project>/proj-shared/$USER        # Lustre, not $HOME (see below)
mkdir -p xmoe-4d && cd xmoe-4d
git clone -b X-MoE-4D --recurse-submodules https://github.com/Supercomputing-System-AI-Lab/X-MoE
cd X-MoE
```

Keep the checkout and the environment on Lustre: `/ccs/home` has a 50 GB quota and the environment needs about
20 GB, and `/lustre/orion/<project>/scratch` is purged, so `proj-shared` is the right place. Do not pass
`--remote` to `git submodule update`; it moves the submodules away from the pinned commits.

## 2. Build the environment with setup_env_rocm.sh (recommended)

```bash
./setup_env_rocm.sh                  # conda, torch, apex, mpi4py, flash-attention, X-MoE, verify
./setup_env_rocm.sh aws-ofi-rccl     # the Slingshot plugin for RCCL; Frontier only
./setup_env_rocm.sh primus           # optional: the Primus-Turbo grouped GEMM
```

`XMOE4D_ROOT` (by default the directory above the checkout, `xmoe-4d/` above) holds the conda environment
`XMoE4D_envs/X-MoE-4D-ROCM6.4.1_repro`, the dependency sources and the caches. The script checks the free space
first, and `./setup_env_rocm.sh <stage>` reruns one stage after a failure. The `aws-ofi-rccl` stage prints the
variables to export in the training environment.

## 3. Manual installation (the same steps)

### 3.1 Modules and compilers

```bash
module reset
module load cpe/24.11 PrgEnv-gnu/8.6.0 rocm/6.4.1 cray-mpich/9.1.0 craype-accel-amd-gfx90a miniforge3/23.11.0-0 ninja
export CXX=/opt/cray/pe/gcc-native/13/bin/g++
export CC=/opt/cray/pe/gcc-native/13/bin/gcc
```

The GCC 13 compilers avoid C++ ABI mismatches when PyTorch extensions and kernels are built. Export them in every
new shell before training too.

### 3.2 Conda environment and PyTorch

```bash
conda create -y -p $PWD/../XMoE4D_envs/X-MoE-4D-ROCM6.4.1_repro python=3.11
conda activate $PWD/../XMoE4D_envs/X-MoE-4D-ROCM6.4.1_repro
pip3 install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/rocm6.4
```

### 3.3 Apex (ROCm fork)

```bash
git clone https://github.com/ROCm/apex.git && cd apex
git checkout af25af4e26f0546d897c3e2ff31502ef50530ecc    # the validated commit (apex 1.11.0)
pip install -r requirements.txt
python setup.py install --cpp_ext --cuda_ext
cd ..
```

The validated build reports apex 1.11.0.

### 3.4 mpi4py with Cray MPICH

```bash
MPICC="cc -shared" pip install --no-cache-dir --force-reinstall --no-binary=mpi4py mpi4py
```

### 3.5 FlashAttention (required)

The example templates pass `--use-flash-attn-v2`, so FlashAttention is required. On AMD it is built from source:

```bash
git clone https://github.com/Dao-AILab/flash-attention.git && cd flash-attention
git checkout v2.8.3
MAX_JOBS=64 pip install . --no-build-isolation
cd ..
```

### 3.6 X-MoE and Megatron-DeepSpeed-X-MoE

```bash
pip install -e .
cd Megatron-DeepSpeed-X-MoE && pip install -e . && cd ..
pip install six nltk pandas openpyxl matplotlib tensorboard py-spy
```

## 4. Running X-MoE-4D on Frontier

`run_exp_training.sh` submits Slurm jobs itself, so run it from a login node; it needs your project account.
The jobs run in the environment of the shell that submits them (the Slurm template loads no modules and activates
no environment), so load the modules, activate the environment and source the runtime file first.

```bash
module reset
module load cpe/24.11 PrgEnv-gnu/8.6.0 rocm/6.4.1 cray-mpich/9.1.0 craype-accel-amd-gfx90a miniforge3/23.11.0-0 ninja
conda activate <XMOE4D_ROOT>/XMoE4D_envs/X-MoE-4D-ROCM6.4.1_repro
source <XMOE4D_ROOT>/XMoE4D_deps/xmoe_4d_runtime_env.sh    # from the aws-ofi-rccl stage: plugin, CC/CXX
export SBATCH_ACCOUNT=<your project>
cd Megatron-DeepSpeed-X-MoE/examples_xmoe_4d/data && bash prepare_data_ae.sh    # once
cd ../scripts-frontier
./run_exp_training.sh env_validate
```

`env_validate` trains X-MoE 10B for 10 iterations on one node (8 GCDs, pipeline size 2, expert-parallel size 4).
The first run compiles kernels, which can take about ten minutes more. The logs of each job are in
`scripts-frontier/logs/job_<id>_.../`: the Slurm output `.o` and `.e`, one `rank_<i>.log` per rank and
`full_run.log`. Every iteration line reports the throughput as `TFLOPs:` per GCD. With pipeline stages, the
iteration lines are written by the last stage, so read the log of the last rank.

The other modes (training, the planner profiles, the paper's main results, the convergence check) are described in
the examples README, `Megatron-DeepSpeed-X-MoE/examples_xmoe_4d/README.md`.

## 5. Troubleshooting

**JIT compile errors of the fused kernels.** Clear the build directory and check the compilers:

```bash
rm -rf Megatron-DeepSpeed-X-MoE/megatron/fused_kernels/build
export CXX=/opt/cray/pe/gcc-native/13/bin/g++
export CC=/opt/cray/pe/gcc-native/13/bin/gcc
```

**A job waits before its first step and never starts training.** A job killed while it compiled an extension leaves
a `lock` file in `scripts-frontier/deepspeed/<extension>/` or in `megatron/fused_kernels/build/`. The templates
delete locks older than 20 minutes at the start of every job; delete a newer one by hand if no other job is
compiling.

**sbatch rejects the job.** Set `SBATCH_ACCOUNT` to a project you are a member of.

## Support and Resources

- **OLCF Documentation:** [https://docs.olcf.ornl.gov/systems/frontier_user_guide.html](https://docs.olcf.ornl.gov/systems/frontier_user_guide.html)
