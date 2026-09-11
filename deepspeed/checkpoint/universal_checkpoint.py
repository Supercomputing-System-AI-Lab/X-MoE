# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import re
import time
import torch
import types
from typing import List, Tuple, Union
from dataclasses import dataclass
from .constants import (FP32_WEIGHT_KEY, PARAM, VOCAB_TENSOR, CAT_DIM, PARAM_N_SUB_PARAMS, SUB_PARAM_SHAPE)
from .utils import load_checkpoint_file

# Per-rank timing of the universal checkpoint load, enabled by UCP_LOAD_TIMING_DIR.
#
# Off unless the variable is set: the environment is read once per process and the answer
# cached, after which the disabled path is one global lookup per call and adds no syscall,
# no import and no file. When set, every process writes its own
#     <UCP_LOAD_TIMING_DIR>/load_<hostname>_rank<rank>_<pid>.csv
# (rank from torch.distributed when initialised, else -1) with one row per event, flushed
# as written so a wall-killed job keeps what it measured. Every clock is the process's own:
# time.time() once in the 'proc' row and once at the end of the bracket, perf_counter_ns per
# event, so rows join to Darshan DXT records by pid and to other ranks by wall time.
#
# Events (dt_s = t1_ns - t0_ns in seconds; blank columns do not apply):
#   proc      once per file: hostname, pid, rank, wall_time, t0_ns; path = the csv itself
#   optstate  <tag>/zero/optimizer_state.pt: nbytes, t_read_s
#   folder    one per parameter directory: os.listdir (dt_s) and the os.path.isdir probe the
#             caller makes when name aliases are declared (t_isdir_s); count = .pt files;
#             tp_rank / tp_world_size as passed by the caller
#   key       one per <parameter>/<key>.pt: nbytes, t_read_s around torch.load, t_post_s for
#             the TP-slice / DP-fragment cut and copy, full_numel as loaded and frag_numel
#             kept by this rank, tp_rank / tp_world_size as used for the cut
#   param     one per parameter: the whole load_hp_checkpoint_state call
#   group     one per optimizer param group: map_to_flat_opt_states; count = params mapped
#   bracket   the whole load_hp_checkpoint_state_from_checkpoint_dir call: count = files read,
#             nbytes = bytes read (atoms + optimizer_state.pt), wall_time at the end
_LOAD_TIMING = None  # unresolved; resolved once to False (off) or a _LoadTimingWriter


class _LoadTimingWriter(object):
    FIELDS = ('event', 'hostname', 'pid', 'rank', 'tp_rank', 'tp_world_size', 'param', 'key', 'path', 'nbytes',
              'full_numel', 'frag_numel', 't0_ns', 't1_ns', 'dt_s', 't_read_s', 't_post_s', 't_isdir_s', 'count',
              'wall_time')

    def __init__(self, timing_dir):
        import csv  # only when enabled
        self.hostname = os.uname().nodename
        self.pid = os.getpid()
        self.rank = torch.distributed.get_rank() if torch.distributed.is_available() \
            and torch.distributed.is_initialized() else -1
        self.n_files = 0  # files read since the bracket opened
        self.n_bytes = 0
        self.isdir_ns = 0  # cost of the caller's directory probe, consumed by the next 'folder' row
        os.makedirs(timing_dir, exist_ok=True)
        self.path = os.path.join(timing_dir, f'load_{self.hostname}_rank{self.rank}_{self.pid}.csv')
        self._fh = open(self.path, 'w', newline='')
        self._csv = csv.writer(self._fh)
        self._csv.writerow(self.FIELDS)
        self.row('proc', path=self.path, t0_ns=time.perf_counter_ns(), wall_time=time.time())

    def row(self, event, **fields):
        fields['event'] = event
        fields.setdefault('hostname', self.hostname)
        fields.setdefault('pid', self.pid)
        fields.setdefault('rank', self.rank)
        if 't0_ns' in fields and 't1_ns' in fields:
            fields['dt_s'] = round((fields['t1_ns'] - fields['t0_ns']) * 1e-9, 9)
        self._csv.writerow([fields.get(name, '') for name in self.FIELDS])
        self._fh.flush()

    def file_row(self, event, path, t_read0, t_read1, t_post1, **fields):
        # os.path.getsize is issued after the timed windows so it never sits inside t_read_s.
        nbytes = os.path.getsize(path)
        self.n_files += 1
        self.n_bytes += nbytes
        self.row(event,
                 path=path,
                 nbytes=nbytes,
                 t0_ns=t_read0,
                 t1_ns=t_post1,
                 t_read_s=round((t_read1 - t_read0) * 1e-9, 9),
                 t_post_s=round((t_post1 - t_read1) * 1e-9, 9),
                 **fields)


def universal_load_timing():
    """The per-rank load timing writer when UCP_LOAD_TIMING_DIR is set, else False."""
    global _LOAD_TIMING
    if _LOAD_TIMING is None:
        timing_dir = os.environ.get('UCP_LOAD_TIMING_DIR')
        _LOAD_TIMING = _LoadTimingWriter(timing_dir) if timing_dir else False
    return _LOAD_TIMING


@dataclass
class SubparamShape:
    patterns: List[str]
    shape: Tuple[Union[Tuple[int], int]]
    partition_dim: int


def load_hp_checkpoint_state(self, folder, tp_rank, tp_world_size):
    hp_mapping = self._hp_mapping
    hp_mapping.optim_fragment = {}

    timing = universal_load_timing()
    if timing:
        param_name = os.path.basename(folder)
        t_call = time.perf_counter_ns()

    hp_keys = []
    for file in os.listdir(folder):
        # We expect files named something like "exp_avg.pt", "exp_avg_sq.pt", "fp32.pt"
        pattern = r'(.+).pt'
        match = re.search(pattern, file)
        if match:
            hp_keys.append(match.group(1))

    if timing:
        timing.row('folder',
                   param=param_name,
                   path=folder,
                   tp_rank=tp_rank,
                   tp_world_size=tp_world_size,
                   count=len(hp_keys),
                   t0_ns=t_call,
                   t1_ns=time.perf_counter_ns(),
                   t_isdir_s=round(timing.isdir_ns * 1e-9, 9))
        timing.isdir_ns = 0

    step = None
    for key in hp_keys:
        ckpt_file = os.path.join(folder, f"{key}.pt")
        if timing:
            t_read0 = time.perf_counter_ns()
        ckpt_dict = load_checkpoint_file(ckpt_file)
        if timing:
            t_read1 = time.perf_counter_ns()

        if key == "step":
            step = ckpt_dict
            if timing:
                timing.file_row('key',
                                ckpt_file,
                                t_read0,
                                t_read1,
                                t_read1,
                                param=param_name,
                                key=key,
                                tp_rank=tp_rank,
                                tp_world_size=tp_world_size)
            continue

        full_hp_param = ckpt_dict[PARAM]
        if timing:
            loaded_numel = full_hp_param.numel()

        # need to deal with slices that were averaged.
        # the opposite of averaging here becomes an exact copy of the first slice
        # I thought of 2 ways:
        # implementation a. find a way for a client to pass a dict with patterns
        # if any(re.search(pattern, folder) for pattern in WEIGHTS_TO_AVERAGE_PATTERNS):
        #     tp_rank = 0
        #     tp_world_size = 1
        # the other approach is to assume that the saved data is correct and if full_hp_param.shape ==
        # self.shape that means we automatically copy?
        # implementation b.
        # this version requires no additional data passed from the client
        # if the shapes already match it must be slices that were averaged - so we just hack around those
        if full_hp_param.shape == self.shape:
            tp_rank = 0
            tp_world_size = 1

        # special case for word_embeddings weights which get padded differently depending on TP degree.
        # the converter to universal currently strips the original padding completely so the saved
        # weight is padding-free and we just need to add new padding depending on the target TP
        # degree
        is_vocab_tensor = ckpt_dict.get(VOCAB_TENSOR, False)
        if is_vocab_tensor:
            # In the absence of data passed from the user wrt new padded vocab specific to tp degree
            # we can again derive that data by reverse engineering the target shapes like so:
            padded_target_vocab_size = self.shape[0] * tp_world_size
            assert padded_target_vocab_size >= full_hp_param.shape[0], \
                f'Vocab tensor padded size {padded_target_vocab_size} < loaded universal size {full_hp_param.shape[0]}'
            if padded_target_vocab_size > full_hp_param.shape[0]:
                padding_size = padded_target_vocab_size - full_hp_param.shape[0]
                full_hp_param = torch.nn.functional.pad(full_hp_param, (0, 0, 0, padding_size), "constant", 0)

        full_param_numel = full_hp_param.numel()
        tp_slice_numel = self.numel()
        #        if key == FP32_WEIGHT_KEY and 'word_embeddings.weight' in folder:
        #            print_rank_0(f'{full_hp_param[:10]=}', force=True)


        assert full_param_numel == tp_world_size * tp_slice_numel, \
            f'Loading {ckpt_file} full param numel {full_param_numel} != tensor slice numel {tp_slice_numel} * tp_world_size {tp_world_size}'

        #        print(f"{full_hp_param.shape=} {full_param_numel=} {folder=}")
        #        print(f"{dst_tensor.shape=} {dst_tensor.numel()=}{folder=}")

        sub_param_shape = ckpt_dict.get(SUB_PARAM_SHAPE, None)
        # since when we do many to 1 on tp we cat sometimes on dim=0 and other times on dim=1 we have to do exactly the same in reverse
        # special case is when a single parameter is effectively a container for multiple sub parameters
        # (more details at PARAM_N_SUB_PARAMS definition)
        chunk_dim = ckpt_dict.get(CAT_DIM, 0)
        n_sub_params = ckpt_dict.get(PARAM_N_SUB_PARAMS, 1)
        if sub_param_shape:
            partition_dim = sub_param_shape.partition_dim
            sub_dim_sizes = sub_param_shape.shape[partition_dim]
            if not isinstance(sub_dim_sizes, tuple):
                sub_dim_sizes = (sub_dim_sizes, )

            partition_shape = [sum(d) if isinstance(d, tuple) else d for d in sub_param_shape.shape]
            full_hp_param = full_hp_param.view(partition_shape)

            offset = 0
            merged_chunks = []
            for sub_dim_size in sub_dim_sizes:
                sub_params_tp_slice = full_hp_param.narrow(partition_dim,
                                                           offset, sub_dim_size).chunk(tp_world_size,
                                                                                       dim=partition_dim)[tp_rank]
                merged_chunks.append(sub_params_tp_slice)
                offset += sub_dim_size
            tp_hp_slice = torch.cat(merged_chunks, dim=partition_dim)

        elif n_sub_params > 1:
            sub_params = full_hp_param.chunk(n_sub_params, dim=chunk_dim)
            sub_params_tp_slice = [p.chunk(tp_world_size, dim=chunk_dim)[tp_rank] for p in sub_params]
            tp_hp_slice = torch.cat(sub_params_tp_slice, dim=chunk_dim)
        else:
            # this performs the opposite of cat when merging TP slices
            tp_hp_slice = full_hp_param.chunk(tp_world_size, chunk_dim)[tp_rank]

        tp_hp_slice = tp_hp_slice.flatten()

        lp_frag_address = hp_mapping.lp_fragment_address
        tp_hp_fragment = tp_hp_slice.narrow(0, lp_frag_address.start, lp_frag_address.numel)

        #        print(f"{key} SHAPE: {tp_hp_slice.shape=}")
        #        print(f"{key} SHAPE: {dst_tensor.shape=}")
        #        print(f"{key} SHAPE: {tp_hp_fragment.shape=}")

        if key == FP32_WEIGHT_KEY:
            dst_tensor = hp_mapping.get_hp_fragment()
            assert dst_tensor.numel() == lp_frag_address.numel, \
                f'Load checkpoint {key} dst numel {dst_tensor.numel()} != src numel {lp_frag_address.numel}'
            dst_tensor.data.copy_(tp_hp_fragment.data)
        else:
            assert tp_hp_fragment.numel() == lp_frag_address.numel, \
                f'Load checkpoint {key} dst numel {tp_hp_fragment.numel()} != src numel {lp_frag_address.numel}'

            hp_mapping.optim_fragment[key] = tp_hp_fragment.clone().detach()

        if timing:
            timing.file_row('key',
                            ckpt_file,
                            t_read0,
                            t_read1,
                            time.perf_counter_ns(),
                            param=param_name,
                            key=key,
                            tp_rank=tp_rank,
                            tp_world_size=tp_world_size,
                            full_numel=loaded_numel,
                            frag_numel=lp_frag_address.numel)

    if timing:
        timing.row('param',
                   param=param_name,
                   path=folder,
                   tp_rank=tp_rank,
                   tp_world_size=tp_world_size,
                   count=len(hp_keys),
                   t0_ns=t_call,
                   t1_ns=time.perf_counter_ns())

    return step


def enable_universal_checkpoint(param_list):
    for param in param_list:
        param.load_hp_checkpoint_state = types.MethodType(load_hp_checkpoint_state, param)
