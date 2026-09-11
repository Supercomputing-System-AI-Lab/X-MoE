# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import torch

from deepspeed.utils import logger
from deepspeed.utils.tensor_fragment import map_to_flat_opt_states
from deepspeed.runtime.utils import bwc_tensor_model_parallel_rank
from deepspeed.checkpoint.utils import load_checkpoint_file


class DeepSpeedOptimizer(object):
    pass


class ZeROOptimizer(DeepSpeedOptimizer):

    def _hp_param_folder(self, checkpoint_dir, name, aliases=None):
        """Atom directory for one parameter, with an optional on-miss alias fallback.

        A universal checkpoint stores one directory per parameter, named exactly as the
        SAVING model named that parameter. A model whose own naming differs -- e.g. the
        same network built as a pipeline module, where layers are named by global spec
        index rather than by their module path -- can therefore not find atoms written
        by its counterpart, even though the tensors correspond one-to-one.

        A model may declare those correspondences via
        `universal_checkpoint_name_aliases()` (see DeepSpeedEngine); they are consulted
        ONLY when the directory for the model's own name is absent, so a checkpoint that
        matches the model is loaded exactly as before. Nothing on disk is ever renamed.

        On failure the PRIMARY path is returned, so the subsequent read raises naming the
        parameter this model actually asked for rather than the last alias tried.
        """
        folder = os.path.join(checkpoint_dir, name)
        if not aliases or os.path.isdir(folder):
            return folder
        for alt_name in aliases.get(name, ()):
            alt = os.path.join(checkpoint_dir, alt_name)
            if os.path.isdir(alt):
                logger.info(f'Universal checkpoint has no atom for {name}; using the '
                            f'alias {alt_name} declared by the model.')
                return alt
        return folder

    def load_hp_checkpoint_state_from_checkpoint_dir(self,
                                                     lp_groups_name: str,
                                                     checkpoint_dir: str,
                                                     param_name_aliases=None) -> None:
        checkpoint_dir = os.path.join(checkpoint_dir, "zero")
        optim_state_path = os.path.join(checkpoint_dir, "optimizer_state.pt")
        assert os.path.isfile(
            optim_state_path), f'{optim_state_path} containing optimizer global state is missing! Cannot proceed.'
        optim_sd = torch.load(optim_state_path)

        self._load_global_state(optim_sd)

        tp_rank = bwc_tensor_model_parallel_rank(mpu=self.mpu)
        if self.mpu is None:
            logger.warn("MPU is not provided, setting tp size to 1 in checkpoint loading.")
            tp_world_size = 1
        else:
            tp_world_size = self.mpu.get_slice_parallel_world_size() if hasattr(self.mpu, "get_slice_parallel_world_size") \
                else self.mpu.get_tensor_model_parallel_world_size()

        # How many parameter groups a rank builds is not a property of the model alone.
        # deepspeed.moe.utils chunks expert parameters into groups bounded by a maximum
        # flat-buffer size, and a rank holds num_experts/ep_size experts, so the number of
        # MoE groups is proportional to 1/ep_size: a model that builds 8 MoE groups with
        # ep_size=8 builds roughly 16 with ep_size=4.
        #
        # Pairing the saved groups with this run's by position is therefore only meaningful
        # while the counts agree. Truncating is not an acceptable fallback: zip() stops at
        # the shorter list, and a universal load deliberately skips load_module_state_dict
        # because the weights come from the fp32 optimizer state -- so every parameter in an
        # unpaired group would keep the value it was *initialised* with, with zeroed Adam
        # moments and a step count of 0, and nothing would say so.
        loaded_param_groups = optim_sd['param_groups']
        positional = len(loaded_param_groups) == len(self.optimizer.param_groups)
        if not positional:
            logger.info(f'Universal checkpoint holds {len(loaded_param_groups)} parameter groups and '
                        f'this run builds {len(self.optimizer.param_groups)}. Optimizer state is '
                        f'addressed per parameter and is unaffected; group hyperparameters are taken '
                        f'from this run rather than from the checkpoint.')

        restored_step = None
        for i, param_group in enumerate(self.optimizer.param_groups):
            # We have an assumption that all params in the same param_group have the same keys
            opt_keys = set()
            steps = []

            lp_groups = getattr(self, lp_groups_name)
            for lp in lp_groups[i]:
                if lp._hp_mapping is not None:
                    #print(f"Loading {self.param_names[lp]} {tp_rank=} {tp_world_size=}")
                    step = lp.load_hp_checkpoint_state(os.path.join(checkpoint_dir, self.param_names[lp]), tp_rank,
                                                       tp_world_size)
                    for key in lp._hp_mapping.get_optim_state_keys():
                        opt_keys.add(key)
                    steps.append(step)

            hp_param = param_group['params'][0]
            assert all(step == steps[0] for step in steps), f"Steps {steps} are not equal"
            if steps and steps[0] is not None:
                self.optimizer.state[hp_param]['step'] = steps[0]
                restored_step = steps[0]

            map_to_flat_opt_states(hp_param, lp_groups[i], self.optimizer.state, opt_keys)
            if timing:
                timing.row('group', key=str(i), count=len(steps), t0_ns=t_map0, t1_ns=time.perf_counter_ns())

            # 'name' identifies the group in THIS process -- for a parameter group tied to a
            # process group it is the key that group is registered under -- so it describes
            # the current run, not the run that wrote the checkpoint. Restoring it would
            # rename a group that already exists. Every case where the two agreed is a case
            # where copying it changed nothing, so it is never restored.
            if positional:
                for key, value in loaded_param_groups[i].items():
                    if key in ('params', 'name'):
                        continue
                    param_group[key] = value

        # Everything else in a parameter group -- lr, betas, eps, weight decay, the
        # multipliers -- is rebuilt from this run's own configuration when the optimizer is
        # constructed, and lr is set again by the scheduler before the first step. 'step' is
        # not: it drives the optimizer's bias correction and only the checkpoint has it.
        #
        # It must be set on EVERY group or on none. ZeRO's next save tests
        # param_groups[0] for 'step' and then reads it from all of them, so a partial restore
        # raises KeyError at the following checkpoint rather than here.
        if not positional and restored_step is not None:
            for param_group in self.optimizer.param_groups:
                param_group['step'] = restored_step

            for key, value in loaded_param_group.items():
                if key == 'params':
                    continue
                param_group[key] = value
