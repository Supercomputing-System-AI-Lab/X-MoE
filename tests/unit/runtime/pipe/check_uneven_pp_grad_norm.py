"""test_uneven_pp_grad_norm.py (2026-11-01): the ZeRO-1 gradient norm must not depend on how the layers are split into
pipeline stages.

Ground facts the test rests on:
- gradient clipping scales the update by the norm of the WHOLE model's gradient; each stage holds only its own layers, so
  the stages must add their squared norms together;
- an all-reduce finishes only when every member calls it, the same number of times;
- ZeRO-1 keeps one entry per optimizer parameter group, and the stages of an uneven split have different numbers of groups.

The test builds the same 4-block model three ways, with the same weights (each block seeded by its index) and the same
batches (seeded by data-parallel rank and micro-batch): (1) no pipeline, plain PyTorch, the reference norm; (2) PP 2 with an
even split, 2 + 2 blocks; (3) PP 2 with the X-MoE fork's custom partition 3 + 1 and UNEVEN_PP=True, as ELM_PP_launch.py sets
it. Every block is its own optimizer group, so the stages hold 2 / 2 groups in (2) and 3 / 1 in (3). It passes when every
rank's ZeRO-1 norm (engine.get_global_grad_norm()) equals the reference to rtol 1e-5 (fp32; the only difference is the order
of the sums). A hang ends at the 60-s gloo timeout and fails. 4 processes: PP 2 x data parallel 2, CPU, gloo.

    DS_ACCELERATOR=cpu PYTHONPATH=<deepspeed tree> python -m torch.distributed.run --nproc_per_node 4 \
        test_uneven_pp_grad_norm.py --mode even|uneven
"""
import argparse
import datetime
import json
import os

import torch
import torch.distributed
import torch.nn as nn

H, BLOCKS, MBS, GAS = 16, 4, 2, 4


class Pre(nn.Module):
    """A parameter-free layer before the blocks: the fork's custom partition counts two of them on the first stage."""

    def forward(self, x):
        return x


class Post(nn.Module):
    """A parameter-free layer after the blocks: the fork's custom partition counts three of them on the last stage."""

    def forward(self, x):
        return x


class Block(nn.Module):

    def __init__(self, idx):
        super().__init__()
        g = torch.Generator().manual_seed(100 + idx)
        self.lin = nn.Linear(H, H)
        with torch.no_grad():
            self.lin.weight.copy_(torch.randn(H, H, generator=g) / H**0.5)
            self.lin.bias.copy_(torch.randn(H, generator=g) * 0.1)

    def forward(self, x):
        return torch.tanh(self.lin(x))


def batch(dp, mb):
    g = torch.Generator().manual_seed(1000 * dp + mb)
    return torch.randn(MBS, H, generator=g), torch.randn(MBS, H, generator=g)


def loss_fn(out, y):
    return ((out - y)**2).mean()


def reference_norm(dp_size):
    """Plain PyTorch: the gradient of the mean loss over every micro-batch of every data-parallel rank."""
    blocks = [Block(i) for i in range(BLOCKS)]
    total = 0.0
    for dp in range(dp_size):
        for mb in range(GAS):
            x, y = batch(dp, mb)
            for b in blocks:
                x = b(x)
            total = total + loss_fn(x, y)
    (total / (dp_size * GAS)).backward()
    return torch.sqrt(sum((p.grad.double()**2).sum() for b in blocks for p in b.parameters())).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['even', 'uneven'], required=True)
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    if a.mode == 'uneven':
        os.environ['UNEVEN_PP'] = 'True'
    else:
        os.environ.pop('UNEVEN_PP', None)
    import deepspeed
    import deepspeed.comm as dist
    from deepspeed.pipe import LayerSpec, PipelineModule
    dist.init_distributed(dist_backend='gloo', timeout=datetime.timedelta(seconds=60))
    specs = [LayerSpec(Pre), LayerSpec(Pre)] + [LayerSpec(Block, i) for i in range(BLOCKS)] + [LayerSpec(Post)] * 3
    kw = dict(layers=specs, num_stages=2, loss_fn=loss_fn, partition_method='type:Block')
    if a.mode == 'uneven':
        kw['custom_pipeline_partition'] = [3, 1]
    net = PipelineModule(**kw)
    groups = [{'params': list(m.parameters())} for m in net.forward_funcs if isinstance(m, Block)]
    opt = torch.optim.SGD(groups, lr=0.0)     # lr 0: the step changes nothing; only the norm is read
    cfg = {"train_micro_batch_size_per_gpu": MBS, "gradient_accumulation_steps": GAS, "zero_optimization": {"stage": 1},
           "zero_allow_untested_optimizer": True, "gradient_clipping": 1.0, "steps_per_print": 10**9}
    engine, _, _, _ = deepspeed.initialize(model=net, optimizer=opt, config=cfg)
    dp, dp_size = engine.grid.get_data_parallel_id(), engine.grid.get_data_parallel_world_size()
    engine.train_batch(data_iter=iter([batch(dp, mb) for mb in range(GAS)]))
    norm = float(engine.get_global_grad_norm())
    rows = [None] * dist.get_world_size()
    torch.distributed.all_gather_object(rows, {'rank': dist.get_rank(), 'stage': engine.stage_id, 'groups': len(groups), 'norm': norm})
    if dist.get_rank() == 0:
        ref = reference_norm(dp_size)
        ok = all(abs(r['norm'] - ref) <= 1e-5 * ref for r in rows)
        res = {'mode': a.mode, 'reference': ref, 'ranks': rows, 'max_rel_err': max(abs(r['norm'] - ref) / ref for r in rows),
               'pass': ok}
        print('RESULT ' + json.dumps(res))
        if a.out:
            json.dump(res, open(a.out, 'w'), indent=1)
    dist.barrier()


if __name__ == '__main__':
    main()
