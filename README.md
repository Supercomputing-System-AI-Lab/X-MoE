
# X-MoE: Cross-Platform Training Framework for Expert-Specialized MoE

<p align="center">
  <img src="images/xmoe-overview.jpg" alt="X-MoE Overview" width="600"/>
</p>

## News
 2025-06-26: X-MoE has been accepted at SC 2025 and received Best Student Paper Nomination!

## About
X-MoE is an optimized cross-platform framework for training large-scale expert-specialized Mixture-of-Experts (MoE) models (e.g. DeepSeek-MoE style). It introduces system-level enhancements for improved throughput and memory efficiency. This project is built on top of DeepSpeed and integrates with Megatron-DeepSpeed for end-to-end MoE training.



For more details on the optimizations and experiments, refer to our paper and the project page: https://supercomputing-system-ai-lab.github.io/projects/x-moe/.


## Quick Start

**IMPORTANT for Artifact Evaluation: This is a general guide. For detailed installation & execution guide on Artifact Evaluation, please refer to the AE document.**



### Installing X-MoE
Clone repository:
```bash
cd ~
git clone https://github.com/Supercomputing-System-AI-Lab/X-MoE
cd X-MoE
git submodule update --init --recursive --remote
```
Install dependencies:
```bash
# For NVIDIA GPUs:
./scripts/install_dep_cuda.sh

# For AMD GPUs:
./scripts/install_dep_rocm.sh
```

Install X-MoE:
```bash
pip install -e .
cd Megatron-DeepSpeed-X-MoE && pip install -e .
```

### Preparing data

We provide a script to download a sample dataset and process it using Megatron's data pipeline.
```bash
cd ~/X-MoE/Megatron-DeepSpeed-X-MoE/examples_xmoe/data
./prepare_data_ae.sh
```

## Running Training with X-MoE

We provide tiny training examples in `Megatron-DeepSpeed-X-MoE/examples_xmoe/scripts` that launch 10.1 B DeepSeek-MoE-like model training task on one GPU node with multiple GPUs.

Running training example with X-MoE:
```bash
cd ~/X-MoE/Megatron-DeepSpeed-X-MoE/examples_xmoe/scripts
./X-MoE-Small-node-1.sh <NUM_GPUS> <MICRO_BATCH_SIZE>
```
------

## Evaluation
Our evaluation on the Frontier supercomputer demonstrates that X-MoE enables training of models up to 545B parameters on 1024 AMD GPUs—10× larger than existing solutions—while achieving up to 1.42× higher training throughput.
<p align="center">
  <img src="images/main-result.jpg" alt="X-MoE Overview" width="600"/>
</p>


## Citation
coming soon

