# Third-party code in X-MoE

X-MoE is a fork of [DeepSpeed](https://github.com/deepspeedai/DeepSpeed) (Apache-2.0),
taken at upstream commit `9a2c209c` (2024-11-13). The files below carry code from
other projects, with the notices those projects require.

| Code | Where in this repository | Origin | License |
|---|---|---|---|
| Token scatter and gather operations and their Triton kernels, modified for padding-free MoE dispatch | `deepspeed/moe/v2opt/kernels.py`, `drop_kernel.py`, `gating.py`, `drop_gating.py`, `rbd_gating.py`, `rbd_kernels.py` | [MegaBlocks](https://github.com/databricks/megablocks), Copyright 2024 Databricks | Apache-2.0 |
| Fused bias and GeLU | `deepspeed/moe/fused_bias_gelu.py` | Megatron-LM `megatron/model/fused_bias_gelu.py`, Copyright 2022 NVIDIA Corporation | BSD-3-Clause (the license of Megatron-LM, in `Megatron-DeepSpeed-X-MoE/LICENSE`) |
| Megatron-DeepSpeed (git submodule `Megatron-DeepSpeed-X-MoE`) | `Megatron-DeepSpeed-X-MoE/` | fork of [Megatron-DeepSpeed](https://github.com/deepspeedai/Megatron-DeepSpeed) | `Megatron-DeepSpeed-X-MoE/LICENSE` |
| Primus-Turbo (git submodule `primus_turbo`), with MI250X (gfx90a) support added | `primus_turbo/` | fork of [AMD-AGI/Primus-Turbo](https://github.com/AMD-AGI/Primus-Turbo), Copyright 2025 Advanced Micro Devices | MIT |
