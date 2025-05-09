# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from deepspeed.utils import groups, log_dist
from .experts import Experts
from .sharded_moe import MOELayer, UnblancedMOELayer, TopKGate, see_memory_usage
from .moe_v2 import MOEv2Layer, TopKGatev2
from .moe_rbd import TopKGateRBD, MOEv2LayerRBD
from .v2opt.utils import print_rank

SEE_MEMORY = False

class MoE(nn.Module):
    """Initialize an MoE layer.

    Arguments:
        hidden_size (int): the hidden dimension of the model, importantly this is also the input and output dimension.
        expert (nn.Module): the torch module that defines the expert (e.g., MLP, torch.linear).
        num_experts (int, optional): default=1, the total number of experts per layer.
        ep_size (int, optional): default=1, number of ranks in the expert parallel world or group.
        k (int, optional): default=1, top-k gating value, only supports k=1 or k=2.
        capacity_factor (float, optional): default=1.0, the capacity of the expert at training time.
        eval_capacity_factor (float, optional): default=1.0, the capacity of the expert at eval time.
        min_capacity (int, optional): default=4, the minimum capacity per expert regardless of the capacity_factor.
        use_residual (bool, optional): default=False, make this MoE layer a Residual MoE (https://arxiv.org/abs/2201.05596) layer.
        noisy_gate_policy (str, optional): default=None, noisy gate policy, valid options are 'Jitter', 'RSample' or 'None'.
        drop_tokens (bool, optional): default=True, whether to drop tokens - (setting to False is equivalent to infinite capacity).
        use_rts (bool, optional): default=True, whether to use Random Token Selection.
        use_tutel (bool, optional): default=False, whether to use Tutel optimizations (if installed).
        enable_expert_tensor_parallelism (bool, optional): default=False, whether to use tensor parallelism for experts
        top2_2nd_expert_sampling (bool, optional): default=True, whether to perform sampling for 2nd expert
    """

    def __init__(self,
                 hidden_size: int,
                 expert: nn.Module,
                 num_experts: int = 1,
                 num_shared_experts: int = 0,
                 ep_size: int = 1,
                 k: int = 1,
                 capacity_factor: float = 1.0,
                 eval_capacity_factor: float = 1.0,
                 min_capacity: int = 4,
                 use_residual: bool = False,
                 noisy_gate_policy: Optional[str] = None,
                 drop_tokens: bool = True,
                 use_rts: bool = True,
                 use_tutel: bool = False,
                 enable_expert_tensor_parallelism: bool = False,
                 enable_expert_sequence_parallelism: bool = False,
                 top2_2nd_expert_sampling: bool = True,
                 use_uneven_all2all: bool = False,
                 use_pft: bool = False,
                 use_rbd: bool = False,
                 rbd_mesh_size: int = 8,
                 ) -> None:

        super(MoE, self).__init__()

        self.use_residual = use_residual
        self.enable_expert_tensor_parallelism = enable_expert_tensor_parallelism
        self.enable_expert_sequence_parallelism = enable_expert_sequence_parallelism
        assert num_experts % ep_size == 0, f"Number of experts ({num_experts}) should be divisible by expert parallel size ({ep_size})"
        self.ep_size = ep_size
        self.expert_group_name = f"ep_size_{self.ep_size}"
        self.num_experts = num_experts
        self.num_local_experts = num_experts // self.ep_size
        self.num_shared_experts = num_shared_experts

        self.use_rbd = use_rbd
        self.mesh_size = min(self.ep_size, rbd_mesh_size)
        self.rbd_local_group_name = f"local_size_{self.mesh_size}"

        log_dist(
            f'Creating MoE layer with num_experts: {num_experts} | num_local_experts: {self.num_local_experts} | expert_parallel_size: {self.ep_size} | num_shared_experts {self.num_shared_experts}',
            [0])

        assert noisy_gate_policy is None or noisy_gate_policy in ['None', 'Jitter', 'RSample'], \
            'Unsupported noisy_gate_policy: ' + noisy_gate_policy

        if use_rbd:
            assert use_pft and use_uneven_all2all, "RBD requires PFT and uneven all2all"
        if use_pft:
            assert use_uneven_all2all, "PFT requires uneven all2all"
        if use_tutel:
            assert not use_uneven_all2all, "Tutel is incompatible with uneven all2all"

        experts = Experts(expert, self.num_local_experts, self.expert_group_name, is_uneven_tokens=use_uneven_all2all)

        gate_params = {
            "model_dim": hidden_size,
            "num_experts": num_experts,
            "k": k,
            "capacity_factor": capacity_factor, 
            "eval_capacity_factor": eval_capacity_factor,
            "min_capacity": min_capacity,
            "noisy_gate_policy": noisy_gate_policy,
            "drop_tokens": drop_tokens,
            "use_rts": use_rts,
            "ep_group": None,
            "top2_2nd_expert_sampling": top2_2nd_expert_sampling
        }

        if use_rbd:
            gate = TopKGateRBD(**gate_params, use_rbd=True)
        elif use_uneven_all2all:
            gate = TopKGatev2(**gate_params)
        else:
            gate = TopKGate(**gate_params, use_uneven_all2all=use_uneven_all2all)

        moe_layer_params = {
            "gate": gate,
            "experts": experts,
            "ep_group_name": self.expert_group_name,
            "ep_size": self.ep_size,
            "num_local_experts": self.num_local_experts
        }

        if use_rbd:
            self.deepspeed_moe = MOEv2LayerRBD(
                **moe_layer_params,
                k=k,
                use_pft=use_pft,
                drop_tokens=drop_tokens
            )
        elif use_uneven_all2all:
            self.deepspeed_moe = MOEv2Layer(
                **moe_layer_params,
                k=k,
                use_pft=use_pft,
                drop_tokens=drop_tokens
            )
        else:
            self.deepspeed_moe = MOELayer(
                **moe_layer_params,
                use_tutel=use_tutel
            )

        if self.use_residual:
            self.mlp = expert
            # coefficient is used for weighted sum of the output of expert and mlp
            self.coefficient = nn.Linear(hidden_size, 2)

    def set_deepspeed_parallelism(self, use_data_before_expert_parallel_: bool = False) -> None:
        self._create_process_groups(use_data_before_expert_parallel_=use_data_before_expert_parallel_)

    def _create_process_groups(self, use_data_before_expert_parallel_: bool = False) -> None:
        # Create process group for a layer if needed
        if self.expert_group_name not in groups._get_expert_parallel_group_dict():
            print(f"No existing process group found, creating a new group named: {self.expert_group_name}")
            if (groups.mpu is None) or (not self.enable_expert_tensor_parallelism and not self.enable_expert_sequence_parallelism):
                # Condition 1 - no groups.mpu means no tensor parallelism
                # Condition 2 - disabling expert tensor parallelism on purpose
                groups._create_expert_and_data_parallel(
                    self.ep_size, use_data_before_expert_parallel_=use_data_before_expert_parallel_)
            elif self.enable_expert_sequence_parallelism:
                groups._create_expert_data_and_sequence_parallel(
                    self.ep_size, mpu=groups.mpu, use_data_before_expert_parallel_=use_data_before_expert_parallel_)
            else:
                # expert tensor parallelism is enabled
                groups._create_expert_data_and_model_parallel(
                    self.ep_size, mpu=groups.mpu, use_data_before_expert_parallel_=use_data_before_expert_parallel_)
            
            if self.use_rbd:
                groups._create_rbd_local_group(self.ep_size, self.mesh_size, mpu=groups.mpu, enable_expert_tensor_parallelism=self.enable_expert_tensor_parallelism)

        # Set the group handle for the MOELayer (deepspeed_moe) object
        self.deepspeed_moe._set_ep_group(groups._get_expert_parallel_group(self.expert_group_name))
        if self.use_rbd: 
            self.deepspeed_moe._set_local_group(groups._get_rbd_local_group(self.rbd_local_group_name))
            self.deepspeed_moe._set_rbd()


    def forward(self,
                hidden_states: torch.Tensor,
                used_token: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ MoE forward

        Arguments:
            hidden_states (Tensor): input to the layer
            used_token (Tensor, optional): default: None, mask only used tokens

        Returns:
            A tuple including output, gate loss, and expert count.

            * output (Tensor): output of the model

            * l_aux (Tensor): gate loss value

            * exp_counts (Tensor): expert count
        """
        
        # torch.cuda.memory._record_memory_history(
        #     max_entries=100000
        # )
        # see_memory_usage("before MoE", force=SEE_MEMORY) 

        output = self.deepspeed_moe(hidden_states, used_token)
        # see_memory_usage("after MoE", force=SEE_MEMORY) 
        # torch.cuda.memory._dump_snapshot("/lustre/orion/gen150/scratch/pinaster/moe-arch/system-benchmark/deepseek-style/profile/moe-layer.pickle")
        # torch.cuda.memory._record_memory_history(enabled=None)
        if self.use_residual:
            # Residual MoE
            output_mlp = self.mlp(hidden_states)
            if isinstance(output_mlp, tuple):
                output_mlp = output_mlp[0]  # Ignore the bias term for now
            coef = self.coefficient(hidden_states)
            coef = F.softmax(coef, dim=-1)
            output = output * coef[..., 0:1] + output_mlp * coef[..., 1:]
        return output, self.deepspeed_moe.l_aux, self.deepspeed_moe.exp_counts
