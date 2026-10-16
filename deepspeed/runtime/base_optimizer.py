# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import time

import torch

from deepspeed.utils import logger
from deepspeed.utils.tensor_fragment import map_to_flat_opt_states
from deepspeed.runtime.utils import bwc_tensor_model_parallel_rank
from deepspeed.checkpoint.utils import load_checkpoint_file
from deepspeed.checkpoint.constants import (FP32_WEIGHT_KEY, PARAM, CAT_DIM, VOCAB_TENSOR, PARAM_N_SUB_PARAMS, SUB_PARAM_SHAPE)
from deepspeed.checkpoint.universal_checkpoint import universal_load_timing


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

    def _universal_tp_rank_and_size(self):
        tp_rank = bwc_tensor_model_parallel_rank(mpu=self.mpu)
        if self.mpu is None:
            logger.warn("MPU is not provided, setting tp size to 1 in checkpoint loading.")
            tp_world_size = 1
        else:
            tp_world_size = self.mpu.get_slice_parallel_world_size() if hasattr(self.mpu, "get_slice_parallel_world_size") \
                else self.mpu.get_tensor_model_parallel_world_size()
        return tp_rank, tp_world_size

    def _load_hp_weights_only(self, lp_groups_name, zero_dir, param_name_aliases):
        """[X-MoE, 2026-09-12] The weights-only universal load (load_optimizer_states=False): every parameter's fp32.pt
        atom into its fp32 master fragment, nothing else read or restored (see load_hp_checkpoint_state_from_checkpoint_dir)."""
        tp_rank, tp_world_size = self._universal_tp_rank_and_size()
        loaded = 0
        for i, param_group in enumerate(self.optimizer.param_groups):
            for lp in getattr(self, lp_groups_name)[i]:
                if lp._hp_mapping is None:
                    continue
                folder = self._hp_param_folder(zero_dir, self.param_names[lp], param_name_aliases)
                lp.load_hp_checkpoint_state(folder, tp_rank, tp_world_size, keys=(FP32_WEIGHT_KEY,))
                loaded += 1
        logger.info(f'universal checkpoint {zero_dir}: loaded the fp32 master weights of {loaded} parameters and nothing '
                    f'of the optimizer (load_optimizer_states=False: the moments, the step and the parameter groups\' '
                    f'hyper-parameters stay as this run built them)')

    def _load_hp_weights_and_moments(self, lp_groups_name, zero_dir, param_name_aliases):
        """[X-MoE 2026-10-15, XMOE_FINETUNE_KEEP_OPTIM=1; assistant_deep_dive/2026-10-15-adam_warmstart] The HOT weights-only load.
        With Megatron's --finetune (load_optimizer_states=False) every parameter's fp32.pt AND its optimizer atoms (exp_avg.pt,
        exp_avg_sq.pt) are loaded into this rank's fragments exactly as a resume loads them (load_hp_checkpoint_state with every key,
        map_to_flat_opt_states), and the optimizer's step counter is set on EVERY parameter group: from a step.pt atom when the
        converter wrote one, else from zero/optimizer_state.pt param_groups[*]['step'] (apex FusedAdam keeps the step in the group,
        and ds_to_universal writes no step.pt for it; the lc128k-v2 step-1000 universal holds 13 groups each with step 1000). The
        step matters: with 1000-step moments and a fresh step of 1 the bias correction scales the first update by
        (1 - beta1) / sqrt(1 - beta2) = 0.1 / sqrt(0.05) = 2.24 at betas (0.9, 0.95).
        Nothing else of the checkpoint is restored, which is what separates this from a resume: the groups' lr, betas, eps, weight
        decay and multipliers stay as this run built them (the resume's positional group restore is NOT run, so iteration 1 does
        not take the source run's lr), the loss scaler and the clip stay, and Megatron's --finetune keeps the iteration, the LR
        schedule, the consumed samples and the data position at 0. Independent of the layout: the atoms are addressed per parameter
        and the step per group, so the moments follow the weights under any TP/EP/DP (the group counts need not agree).
        XMOE_KEEP_OPTIM_VERIFY="<name>,<name>,...": for the named parameters, every rank holding a fragment re-reads exp_avg_sq.pt and
        exp_avg.pt from disk, slices them as the loader does (TP chunk on the atom's cat dim, then this rank's fragment) and prints
        max |state - atom| against the flat optimizer state it just built (the probe's equality check)."""
        tp_rank, tp_world_size = self._universal_tp_rank_and_size()
        optim_state_path = os.path.join(zero_dir, "optimizer_state.pt")
        step_from_file, n_groups_file = None, None
        if os.path.isfile(optim_state_path):
            optim_sd = load_checkpoint_file(optim_state_path)
            pgs = optim_sd.get('param_groups') or []
            n_groups_file = len(pgs)
            steps_pg = [g['step'] for g in pgs if 'step' in g]
            if steps_pg:
                assert all(st == steps_pg[0] for st in steps_pg), \
                    f'{optim_state_path}: the parameter groups hold unequal steps {steps_pg}; refusing to guess'
                step_from_file = steps_pg[0]
        verify = [v.strip() for v in os.environ.get('XMOE_KEEP_OPTIM_VERIFY', '').split(',') if v.strip()]
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        loaded, all_keys, restored_step = 0, set(), None
        lp_groups = getattr(self, lp_groups_name)
        for i, param_group in enumerate(self.optimizer.param_groups):
            opt_keys, steps, to_verify = set(), [], []
            for lp in lp_groups[i]:
                if lp._hp_mapping is None:
                    continue
                name = self.param_names[lp]
                folder = self._hp_param_folder(zero_dir, name, param_name_aliases)
                step = lp.load_hp_checkpoint_state(folder, tp_rank, tp_world_size)   # every atom: fp32 + exp_avg + exp_avg_sq (+ step)
                for key in lp._hp_mapping.get_optim_state_keys():
                    opt_keys.add(key)
                steps.append(step)
                loaded += 1
                if name in verify:
                    to_verify.append((lp, name, folder))
            all_keys |= opt_keys
            hp_param = param_group['params'][0]
            assert all(st == steps[0] for st in steps), f"Steps {steps} are not equal"
            if steps and steps[0] is not None:
                self.optimizer.state[hp_param]['step'] = steps[0]
                restored_step = steps[0]
            map_to_flat_opt_states(hp_param, lp_groups[i], self.optimizer.state, opt_keys)
            for lp, name, folder in to_verify:
                self._xmoe_verify_fragment(lp, name, folder, hp_param, tp_rank, tp_world_size, rank)
        step = restored_step if restored_step is not None else step_from_file
        assert step is not None, (f'XMOE_FINETUNE_KEEP_OPTIM=1: no optimizer step found in {zero_dir} (no step.pt atom and no '
                                  f"param_groups[*]['step'] in optimizer_state.pt); refusing a hot start with a step of 0")
        for param_group in self.optimizer.param_groups:
            param_group['step'] = step
        src = 'the step.pt atoms' if restored_step is not None else f"optimizer_state.pt param_groups[*]['step'] ({n_groups_file} groups saved, {len(self.optimizer.param_groups)} built here)"
        msg = (f'[XMOE_FINETUNE_KEEP_OPTIM] universal checkpoint {zero_dir}: loaded the fp32 master weights AND the optimizer moments '
               f'{sorted(all_keys)} of {loaded} parameters on rank {rank}; the optimizer step counter is {step} on all '
               f'{len(self.optimizer.param_groups)} parameter groups (from {src}); the groups\' lr / betas / eps / weight decay, the '
               f'loss scaler and the clip stay as this run built them; the iteration, the LR schedule, the consumed samples and the '
               f'data position start at 0 (--finetune)')
        logger.info(msg)
        if rank == 0:
            print(msg, flush=True)

    def _xmoe_verify_fragment(self, lp, name, folder, hp_param, tp_rank, tp_world_size, rank):
        """Print max |flat optimizer state - atom on disk| for this rank's fragment of one parameter (exp_avg_sq and exp_avg)."""
        hp_addr = lp._hp_mapping.get_hp_fragment_address()
        lp_addr = lp._hp_mapping.lp_fragment_address
        for key in ('exp_avg_sq', 'exp_avg'):
            f = os.path.join(folder, f'{key}.pt')
            try:
                ck = load_checkpoint_file(f)
                full = ck[PARAM]
                special = ck.get(VOCAB_TENSOR, False) or ck.get(PARAM_N_SUB_PARAMS, 1) > 1 or ck.get(SUB_PARAM_SHAPE, None) is not None
                if special:
                    print(f'[keep-optim verify] rank {rank} {name} {key}: skipped (vocab / sub-parameter atom layout)', flush=True)
                    continue
                if full.shape != lp.shape:
                    full = full.chunk(tp_world_size, ck.get(CAT_DIM, 0))[tp_rank]
                atom = full.flatten().narrow(0, lp_addr.start, lp_addr.numel).float()
                state = self.optimizer.state[hp_param][key].narrow(0, hp_addr.start, hp_addr.numel).detach().float().cpu()
                diff = (state - atom).abs().max().item() if atom.numel() else 0.0
                print(f'[keep-optim verify] rank {rank} {name} {key}: fragment numel {atom.numel()} (of {full.numel()} in this TP slice), '
                      f'max|state - atom| = {diff:.6e}, |atom|_2 = {atom.double().norm().item():.6e}, atom[0:3] = {atom[:3].tolist()}',
                      flush=True)
            except Exception as e:
                print(f'[keep-optim verify] rank {rank} {name} {key}: FAILED ({type(e).__name__}: {e})', flush=True)

    def load_hp_checkpoint_state_from_checkpoint_dir(self,
                                                     lp_groups_name: str,
                                                     checkpoint_dir: str,
                                                     param_name_aliases=None,
                                                     load_optimizer_states: bool = True) -> None:
        """Load this rank's fragments of every parameter from the universal checkpoint's atoms (zero/<name>/).

        [X-MoE, 2026-09-12] load_optimizer_states=False (Megatron's --finetune: engine.load_checkpoint with
        load_module_only=True) is the weights-only load: each parameter's fp32.pt atom -- the parameter's value in the
        universal format, the fp32 master weight (Universal Checkpointing, Lian et al. 2024-06, section 5.2: model.pt,
        "a fp32 tensor represents parameter's weight"; the optimizer states are the other two atoms) -- is copied into the
        fp32 partition, so the masters are exact instead of rebuilt from the bf16 module weights; the moments (exp_avg,
        exp_avg_sq), the step and the parameter groups' hyper-parameters (lr, betas, weight decay, step) stay as this run
        built them, and zero/optimizer_state.pt is not required. Before this the universal path ignored the flag and
        restored all of it: arm B1s (2026-09-11, the source's own layout, so the positional group restore applied) took its
        first step at the source checkpoint's learning rate 1.03e-5 with its step count 61 instead of the warm-up's 0, and
        its fp32 masters moved 1.2e-3 relative in that one step; arm B2s at a changed layout (13 groups against the
        source's 8, the restore skipped) took the rate-0 step and kept them exactly."""
        # Per-rank timing, enabled by UCP_LOAD_TIMING_DIR (see deepspeed.checkpoint.universal_checkpoint).
        timing = universal_load_timing()
        if timing:
            timing.n_files = 0
            timing.n_bytes = 0
            t_bracket0 = time.perf_counter_ns()
        checkpoint_dir = os.path.join(checkpoint_dir, "zero")
        if not load_optimizer_states:
            if os.environ.get('XMOE_FINETUNE_KEEP_OPTIM') == '1':
                # [X-MoE 2026-10-15] the HOT weights-only load: weights + Adam moments + step; see the method
                self._load_hp_weights_and_moments(lp_groups_name, checkpoint_dir, param_name_aliases)
            else:
                self._load_hp_weights_only(lp_groups_name, checkpoint_dir, param_name_aliases)
            return
        optim_state_path = os.path.join(checkpoint_dir, "optimizer_state.pt")
        assert os.path.isfile(
            optim_state_path), f'{optim_state_path} containing optimizer global state is missing! Cannot proceed.'
        if timing:
            t_opt0 = time.perf_counter_ns()
        optim_sd = load_checkpoint_file(optim_state_path)
        if timing:
            t_opt1 = time.perf_counter_ns()
            timing.file_row('optstate', optim_state_path, t_opt0, t_opt1, t_opt1, key='optimizer_state')

        self._load_global_state(optim_sd)

        tp_rank, tp_world_size = self._universal_tp_rank_and_size()

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
                    if timing:
                        t_folder0 = time.perf_counter_ns()
                    folder = self._hp_param_folder(checkpoint_dir, self.param_names[lp], param_name_aliases)
                    if timing:
                        timing.isdir_ns = time.perf_counter_ns() - t_folder0
                    step = lp.load_hp_checkpoint_state(folder, tp_rank, tp_world_size)
                    for key in lp._hp_mapping.get_optim_state_keys():
                        opt_keys.add(key)
                    steps.append(step)

            hp_param = param_group['params'][0]
            assert all(step == steps[0] for step in steps), f"Steps {steps} are not equal"
            if steps and steps[0] is not None:
                self.optimizer.state[hp_param]['step'] = steps[0]
                restored_step = steps[0]

            if timing:
                t_map0 = time.perf_counter_ns()
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

        if timing:
            t_bracket1 = time.perf_counter_ns()
            timing.row('bracket',
                       path=checkpoint_dir,
                       tp_rank=tp_rank,
                       tp_world_size=tp_world_size,
                       nbytes=timing.n_bytes,
                       count=timing.n_files,
                       t0_ns=t_bracket0,
                       t1_ns=t_bracket1,
                       wall_time=time.time())
            logger.info(f'universal checkpoint load timing: rank {timing.rank} read {timing.n_files} files, '
                        f'{timing.n_bytes} bytes from {checkpoint_dir} in {(t_bracket1 - t_bracket0) * 1e-9:.3f} s '
                        f'(per-event csv: {timing.path})')
