# X-MoE Installation Guide for OLCF Frontier


This guide provides step-by-step instructions for installing and running X-MoE on **AMD platforms**. We use OLCF Frontier supercomputer (equipped with MI250X) as an example here, and this guide can also provide general reference for installation on AMD platforms.

**For Frontier Users**: You can either build the environment yourself by following the guide below, or use our pre-built environment available on the Frontier system. To access the pre-built environment, simply run:
```
source activate /lustre/orion/world-shared/gen150/yueming/TORCH-ROCM6.3.1_env
```

## Environment Setup

### 1. Create Environment

Load the required modules for the Frontier system:

```bash
module reset
module load cpe/24.11
module load PrgEnv-gnu/8.6.0
module load rocm/6.3.1
module load craype-accel-amd-gfx90a
module load miniforge3/23.11.0-0
module load ninja/1.12.1.lua
```

### 2. Set Compiler Paths

```bash
export CXX=/opt/cray/pe/gcc-native/13/bin/g++
export CC=/opt/cray/pe/gcc-native/13/bin/gcc
```

These compiler paths are crucial for avoiding C++ ABI compatibility issues when building PyTorch extensions and CUDA kernels.

### 3. Create Conda Environment

Create a Python environment with Python 3.11:

```bash
conda create -p $PWD/TORCH-ROCM6.3.1_env python=3.11 -c conda-forge -y
source activate $PWD/TORCH-ROCM6.3.1_env
```
## Dependency Installation

### 1. Install PyTorch with ROCm Support

Install the nightly build of PyTorch with ROCm 6.3 support:

```bash
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.3
```


### 2. Install NVIDIA Apex (ROCm Fork)

Install the ROCm-compatible version of Apex for mixed precision training:

```bash
git clone https://github.com/ROCm/apex.git
cd apex
pip install -r requirements.txt
python setup.py install --cpp_ext --cuda_ext
```


### 3. Install MPI4Py

Install MPI4Py with proper Cray MPI integration:

```bash
MPICC="cc -shared" pip install --no-cache-dir --no-binary=mpi4py mpi4py
```

### 4. Install Flash-Attention (Optional)

Flash-Attention requires building from source on AMD platforms:

```bash
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
python setup.py install
```

**Alternative:** If Flash-Attention fails to compile, you can disable it in your training scripts by removing the `--use-flash-attn` and `--use-flash-attn-v2` flags.

## Installing X-MoE

```bash
cd ~
git clone https://github.com/Supercomputing-System-AI-Lab/X-MoE
cd X-MoE
git submodule update --init --recursive --remote
pip install -e .
cd Megatron-DeepSpeed-X-MoE && pip install -e .
```


## Running X-MoE

Here is an example of training a 10B model on **one computing node with 4 MI250X GPUs (8 GCDs)**. Here we assume you've requested for one computing node before running these commands.
### 1. Set Compiler Environment

Before running any training scripts, ensure the compiler paths are set:

```bash
export CXX=/opt/cray/pe/gcc-native/13/bin/g++
export CC=/opt/cray/pe/gcc-native/13/bin/gcc
```

**Important:** These exports must be done in every new shell session before running training scripts.

### 2. Execute Training Script



Navigate to the script directory and run the training:

```bash
cd X-MoE/Megatron-DeepSpeed-X-MoE/examples_xmoe/scripts-frontier

# Run with 8 GPUs on 1 node
./n8-Small-XMoE.slurm
```

**Note**: The first run may require an additional 10 minutes to compile kernels
Expectation:

**Expectation**:
After initialization, you can see the training logs during training progress in the terminal. The training logs will also be saved as `n8-Small-XMoE.log`.


This training script will launch the training with micro batch size 4, sequence length 2048, and train a model based on DeepSeek-MoE architecture from scratch. **Expected throughput is ~50-55 TFLOPs with X-MoE optimizations.**

You may now modify the script to train with your own model settings.


## Troubleshooting

### Common Issues and Solutions

#### 1. JIT Compiler Errors for Fused Kernels

**Problem:** CUDA/ROCm kernels fail to compile during runtime with JIT compilation errors.

**Solution:** Clear the build cache and ensure compiler paths are set:

```bash
cd X-MoE/Megatron-DeepSpeed-X-MoE/megatron/fused_kernels
rm -rf build
```

Then ensure compiler environment is properly set:

```bash
export CXX=/opt/cray/pe/gcc-native/13/bin/g++
export CC=/opt/cray/pe/gcc-native/13/bin/gcc
```

Re-run the training script:

```bash
./n8-Small-XMoE.slurm
```

## Support and Resources

- **OLCF Documentation:** [https://docs.olcf.ornl.gov/systems/frontier_user_guide.html](https://docs.olcf.ornl.gov/systems/frontier_user_guide.html)

