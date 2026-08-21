# ELMoE on AWS — handover / resume notes

Written 2026-08-21. State of the non-SLURM (torchrun) multi-node path after a full
bring-up on two p4d.24xlarge nodes, and what is left to do.

---

## Cluster as it stands

| | |
|---|---|
| node 1 | `172.31.15.205` (public 3.136.19.132) — NFS **server**, exports `~/elmoe` |
| node 2 | `172.31.10.93` (public 3.149.27.22) — mounts it; **nothing installed locally** |
| GPUs | 8x A100-SXM4-**40GB** per node, driver 595.91.07 |
| SG | `sg-0e8795bb2beb47e0b` — All TCP self-referencing rule added |
| EFA | **not attached** — multi-node NCCL runs over TCP |

Use **private** IPs. Public DNS resolves to the private IP from inside the VPC:
`getent hosts ec2-3-149-27-22.us-east-2.compute.amazonaws.com`

---

## Bring-up from scratch

```bash
# 0. (console) SG inbound: All TCP, ports 0-65535, source = that same SG
./setup_2node_aws.sh wire <NODE2_IP>     # ssh + NFS + hostfile   (~2 min, no env needed)
./setup_env_cuda.sh                       # conda env             (~40 min, lands on the share)
cd Megatron-DeepSpeed-X-MoE/examples_elmoe/data && bash prepare_data_ae.sh   # (~25 min)
cd ~/elmoe/X-MoE && ./setup_2node_aws.sh all <NODE2_IP>   # env.sh + NCCL smoke test
```

`wire` first is deliberate: once `~/elmoe` is exported, everything built afterwards is
shared automatically, and SG/ssh problems surface in minute one instead of minute forty.

Activate the env with **two lines and nothing else** — no `CPATH`, no `PYTHONPATH`:

```bash
source ~/elmoe/miniforge3/etc/profile.d/conda.sh
conda activate ~/elmoe/ELMoE_envs/ELMoE-CUDA12.8_repro
```

(`./setup_env_cuda.sh activate` reprints this.)

---

## Verified working

| Run | Result |
|---|---|
| 10B, 1 node, PP2-EP4, mbs1+ckpt | 15/15 steps, 21.0 GB peak |
| 21B, 2 nodes, PP2-EP8, GBS=1024 | 15/15, 65.2 TFLOPs (75.1 with planner/uneven) |
| X-MoE / DS-MoE baselines, 2 nodes | 15/15 each |
| `loss_validate`, 1 node, 100 steps | ELMoE 7.8124 vs X-MoE 7.8058 — curves converge |

---

## Still open

1. **Planner cache is Frontier's.** `planner_profiling_cache_backup/` holds MI250X
   timings. Any `yes-planner` run (including `main_results`) plans against the wrong
   machine. Generate real ones — two sweeps, single node, no EFA needed:
   ```bash
   bash run_exp_training.sh profiling_cache ELMoE                    # d5120: 21B/25B/50B/63B
   bash run_exp_training.sh profiling_cache ELMoE --model-size 10B   # d2048: 10B
   ```
   The key ignores depth, so one sweep serves a whole family.
2. **`autorun.sh` committed state** still carries an experiment scratchpad with
   `yes-planner` active — which now skips, since the live cache dir was moved back.
3. **EFA.** Attachable to a **stopped** instance (no relaunch needed), but stopping p4d
   risks capacity on restart. Take an AMI first. SG must then allow **all traffic**, not
   just all TCP.

---

## Traps worth remembering

- **40 GB vs 64 GB.** Every upstream config is tuned for MI250X. On A100-40GB use
  `mbs=1` + activation checkpointing. See guide Appendix C.
- **`NCCL Error 1: unhandled cuda error`** in the MoE all-to-all is a *masked CUDA OOM*.
  Check `HBM%` in `a-monitor.txt`; 85%+ means memory, not network.
- **`analyze_memory.py` under-reports by ~4.5 GB/rank** — torch's `max_reserved` excludes
  the CUDA context, NCCL buffers and cuBLAS workspaces. `PLANNER_MEMORY_HEADROOM_GB`
  defaults to 1.5, ~3x too small here.
- **Fewer nodes uses MORE memory per GPU** — halving nodes halves EP, doubling experts held.
- **Read `rank_<LAST>.log`, not `rank_0.log`.** With PP>1 Megatron logs
  `iteration N/M ... lm loss` only on the final pipeline stage.
- **torchrun's `ChildFailedError` is never the real error** — that's in the rank log.
- **A `RendezvousConnectionError` traceback AFTER a successful result is benign** — the
  agent hosting the c10d store exits first and tears it down.
- **Runs must be sequential.** Two concurrent runs collide on `MASTER_PORT 29500`.
- **VS Code sets `GIT_ASKPASS`**, which makes `git push` hang with no prompt over SSH.
  `unset GIT_ASKPASS` first.

---

## Never commit

`scripts/env.sh`, `scripts/hostfile` (private IPs), `scripts/planner_profiling_cache/`
(hardware-specific), and `scripts/{logs,output,temp_sh,temp_ds_json,results}/`.
Templates are `env.sh.example` / `hostfile.example`.
