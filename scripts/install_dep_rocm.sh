#!/bin/bash

# Install PyTorch with ROCm support
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/rocm6.1

# Clone and install apex
git clone https://github.com/ROCm/apex
cd apex && git checkout release/1.4.0
pip install -v --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./

# Install flash-attn
pip install flash-attn --no-build-isolation