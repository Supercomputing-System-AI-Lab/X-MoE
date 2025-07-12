#!/bin/bash

# Clone the NVIDIA Apex repository
git clone https://github.com/NVIDIA/apex

# Change directory to apex
cd apex

# Install Apex with CUDA extensions
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./

# Install FlashAttention
pip install flash-attn --no-build-isolation
