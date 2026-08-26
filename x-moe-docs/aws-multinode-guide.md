# X-MoE-4D multi-node on AWS EC2 (torchrun / non-SLURM)

Step-by-step for bringing up `examples_xmoe_4d/scripts/autorun.sh` across two or more
EC2 GPU instances, with no scheduler. The SLURM path
(`scripts-frontier/autorun_frontier.sh`) is unaffected by anything here.

**Reference environment for this guide**

| | |
|---|---|
| Instance type | `p4d.24xlarge` (8x A100-SXM4-40GB per node) |
| AMI | AWS Deep Learning AMI (Amazon Linux 2023) |
| Node count | 2 (16 GPUs total) |
| Conda env | `~/xmoe-4d/XMoE4D_envs/X-MoE-4D-CUDA12.8_repro` (torch 2.9.1+cu128, deepspeed 0.15.5) |
| Repo root | `~/xmoe-4d/X-MoE` |
| Interconnect | ENA/TCP -- **EFA not attached**, see [Appendix A](#appendix-a-efa) |

Substitute your own values as you go. Two placeholders appear throughout:

- `<NODE1_IP>` -- private IP of the node you launch from (also the rendezvous master)
- `<NODE2_IP>` -- private IP of the second node

---

## How the launcher works

Worth understanding before you debug anything, because it explains every
prerequisite below.

`autorun.sh` fills placeholders in `xmoe_4d.sh.template` and runs the result. When
`NODES > 1`, that rendered script (`xmoe_4d.sh.template:505-540`):

1. Generates two helper scripts inside the job directory:
   - `logs/job_<id>/rank_cmd.sh` -- the per-rank command (`python ELM_PP_launch.py ...`)
   - `logs/job_<id>/node_launch.sh` -- the per-node command (`source env.sh; torchrun ...`)
2. Reads `./hostfile`; **host on line 1 becomes `MASTER_ADDR`**.
3. `ssh -o BatchMode=yes <host> "bash '<abs-path>/node_launch.sh'"` for **every**
   host in the file -- including itself.
4. Each node's `torchrun` rendezvouses at `MASTER_ADDR:29500` (c10d backend).
5. Every rank writes `logs/job_<id>/rank_<N>.log`; `run_analysis()` then globs
   those files to build `full_run.log`.

Three consequences:

- **Passwordless SSH is required to node 1 as well as node 2** (step 3 does not
  special-case the local host).
- **A shared filesystem is required.** Node 2 is told to run a script at an
  absolute path that node 1 just wrote. On local-only disks that file does not exist.
- **`env.sh` is required.** SSH carries no environment, so conda activation and
  NCCL settings must live in a file sourced on every node.

---

## Phase A -- get one node working

Do not attempt multi-node until `bash autorun.sh` completes on a single node.
Debugging a missing dataset and a broken rendezvous simultaneously is misery.

### A1. Fill the dependency gaps

The env built by `setup_env_cuda.sh` is missing a package Megatron imports
unconditionally:

```bash
source ~/xmoe-4d/miniforge3/etc/profile.d/conda.sh
conda activate ~/xmoe-4d/XMoE4D_envs/X-MoE-4D-CUDA12.8_repro
pip install six nltk tensorboard py-spy
```

| Package | Why | Required? |
|---|---|---|
| `six` | `megatron/tokenizer/bert_tokenization.py:25`, imported unconditionally via `megatron/__init__.py` | **Yes** -- hard crash without it |
| `nltk` | `tools/preprocess_data.py:18` sentence splitting during data prep | Yes, for A2 |
| `tensorboard` | `megatron/global_vars.py:144`; the template passes `--tensorboard-dir` | No -- guarded by `try/except`, you just lose TB logs |
| `wandb` | `megatron/training.py:58` | No -- guarded, set to `None` when absent |
| `sentencepiece` | `megatron/tokenizer/tokenizer.py:343`, lazy | No -- only for the SP tokenizer, not GPT2BPE |
| `py-spy` | The run monitor -- see [Monitoring a run](#monitoring-a-run-py-spy) | No -- monitor self-disables without it |

The missing-`six` failure looks like this, and is easy to misread as a
distributed problem because torchrun reports it as `ChildFailedError`:

```
torch.distributed.elastic.multiprocessing.errors.ChildFailedError:
bash FAILED .../logs/job_<id>/a-xmoe-pp-<id>.o
```

The real message is always in `logs/job_<id>/rank_0.log`, never in torchrun's
traceback. Get in the habit of reading it first.

### A2. Build the dataset

`xmoe_4d.sh.template:119-121` expects three files that the repo does not ship:

```bash
export VOCAB_FILE=../data/gpt2-vocab.json
export MERGE_FILE=../data/gpt2-merges.txt
export DATA_PATH=../data/my-gpt2_text_document
```

Generate them (~20-30 min; downloads a 1 GB corpus, needs ~10 GB free):

```bash
cd ~/xmoe-4d/X-MoE/Megatron-DeepSpeed-X-MoE/examples_xmoe_4d/data
bash prepare_data_ae.sh
```

Verify:

```bash
ls -la gpt2-vocab.json gpt2-merges.txt my-gpt2_text_document.bin my-gpt2_text_document.idx
```

All four must exist. These paths are gitignored, so each machine builds its own
copy -- or, better, builds it once on the shared filesystem from Phase B.

### A3. Create `env.sh`

Sourced on every node before `torchrun` starts (`xmoe_4d.sh.template:58, 517-518`).
Copy `env.sh.example` or write it directly:

```bash
cd ~/xmoe-4d/X-MoE/Megatron-DeepSpeed-X-MoE/examples_xmoe_4d/scripts
cat > env.sh <<'EOF'
#!/bin/bash
# --- 1. Python environment ---
# Guard against `set -u` + the conda cuda-nvcc activate.d script.
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
source /home/ec2-user/xmoe-4d/miniforge3/etc/profile.d/conda.sh
conda activate /home/ec2-user/xmoe-4d/XMoE4D_envs/X-MoE-4D-CUDA12.8_repro

# CUDA_HOME and CPATH are NOT set here on purpose -- setup_env_cuda.sh installs a conda
# activation hook that provides both. See "CUDA headers" below if you hit cuda_fp16.h errors.

# --- 2. Network / collectives ---
export NCCL_SOCKET_IFNAME=ens32   # `ip -br addr` to confirm your interface name
export NCCL_NET=Socket            # DELETE THIS once EFA is attached -- see Appendix A
export OMP_NUM_THREADS=1
EOF
```

#### CUDA headers: `cuda_fp16.h: No such file or directory`

If the run dies compiling fused kernels, your env predates the activation hook. Fix it once:

```bash
./setup_env_cuda.sh envhook
conda deactivate && conda activate ~/xmoe-4d/XMoE4D_envs/X-MoE-4D-CUDA12.8_repro
./setup_env_cuda.sh verify      # look for the "activation hook:" line
```

Why it happens: Megatron JIT-builds fused kernels via `torch.utils.cpp_extension`, which
assembles its own flag list and adds exactly one CUDA include dir, `$CUDA_HOME/include`.
No conda hook sets `CUDA_HOME`, and conda's `cuda-toolkit` puts the headers in
`$CONDA_PREFIX/targets/x86_64-linux/include`. Conda's `~cuda-nvcc_activate.sh` masks this
for `make`/setuptools builds by exporting `-I` flags in `CFLAGS`/`CPPFLAGS` -- which is why
`megatron/data/helpers.cpp` compiles fine moments before the fused kernels fail.

`write_activation_hook()` in `setup_env_cuda.sh` writes
`$CONDA_PREFIX/etc/conda/activate.d/zzz-xmoe-4d-cuda.sh`, exporting `CUDA_HOME` and `CPATH`
(honoured by both gcc and nvcc), with a matching `deactivate.d` script that restores them.
Because it lives in the env, `conda activate` alone is sufficient -- no wrapper script, and
nothing to duplicate per launcher. `stage_cuda` writes it on every fresh build; `envhook`
is only for envs created before it existed.

Never set `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `SLURM_PROCID` or `SLURM_LOCALID`
here. torchrun sets the first three per worker, and `pretrain_gpt_deepspeed.py`
reads `SLURM_PROCID` *before* `RANK` -- exporting it makes every rank believe it
is the same rank.

Sanity check it in a scrubbed shell -- no SSH keys required, and it reproduces
exactly what `node_launch.sh` does (`source env.sh`, then run):

```bash
env -i HOME=$HOME PATH=/usr/bin:/bin bash --noprofile --norc -c '
  cd ~/xmoe-4d/X-MoE/Megatron-DeepSpeed-X-MoE/examples_xmoe_4d/scripts
  source ./env.sh
  echo "CUDA_HOME=$CUDA_HOME"
  echo "CPATH=$CPATH"
  python -c "import six, torch; print(\"gpus\", torch.cuda.device_count())"'
```

All three lines must be non-empty, with `gpus 8` last.

**Do not test with `bash env.sh`.** That runs the file in a *subshell*; its exports
die with it, so the parent shell sees nothing and you get a false negative. `env.sh`
only ever makes sense `source`d, which is how the launcher uses it.

### A4. Run single-node

`autorun.sh` ships with the single-node config already active:

```bash
PP_STRATEGY_MAP["1:8"]="2:4"                                             # PP2, EP4
PP_BATCH_MAP["1:8"]=" 4:20:15:X-MOE-4D:10B:no-ckpt:0:even:no-planner "   # mbs 4, 20 micro-batches, 15 iters
```

```bash
cd ~/xmoe-4d/X-MoE/Megatron-DeepSpeed-X-MoE/examples_xmoe_4d/scripts
bash autorun.sh 2>&1 | tee /tmp/1node.log
```

Success looks like iteration lines with loss and TFLOPs:

```bash
grep -E "iteration|elapsed time|TFLOPs" logs/job_*/rank_0.log | tail -20
```

**Checkpoint: do not continue until this works.**

---

## Phase B -- connect the two nodes

### B1. Security group

torchrun rendezvous uses port 29500, but NCCL's socket transport opens
**ephemeral** ports, so per-port rules will not work.

In the EC2 console, on the security group attached to both instances, add an
inbound rule:

| Type | Protocol | Port range | Source |
|---|---|---|---|
| All TCP | TCP | 0-65535 | *the same security group's ID* |

A self-referencing source keeps this private to the cluster. Add **NFS (2049/tcp)**
too if you use the NFS option in B3 and did not open all TCP.

Or via CLI:

```bash
aws ec2 authorize-security-group-ingress --region <REGION> \
    --group-id <SG_ID> --protocol tcp --port 0-65535 --source-group <SG_ID>
```

### B2. Passwordless SSH

The launcher ssh's to every host in the hostfile, node 1 included, so **node 1
must be able to ssh to itself.**

```bash
# on node 1
ssh-keygen -t ed25519 -N '' -f ~/.ssh/id_ed25519
cat ~/.ssh/id_ed25519.pub >> ~/.ssh/authorized_keys        # node1 -> node1
ssh-copy-id -o IdentityFile=~/<your-launch-key>.pem \
            -i ~/.ssh/id_ed25519.pub ec2-user@<NODE2_IP>   # node1 -> node2
```

Verify with the *same* flags the launcher uses -- `BatchMode=yes` fails rather
than prompting, which is what you want to catch now:

```bash
ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new <NODE1_IP> hostname
ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new <NODE2_IP> hostname
```

Both must print a hostname with no prompt. Run them once more afterward so the
host keys are cached -- an unknown host key will hang the launcher.

### B3. Shared filesystem

Required, not optional -- see "How the launcher works" above. Every node must see
the *same absolute path*.

#### Option 1: NFS export from node 1 (fastest to set up)

Exporting `~/xmoe-4d` covers the repo, the conda env, the deps and the dataset at once.

```bash
# --- on node 1 ---
sudo bash -c 'echo "/home/ec2-user/xmoe-4d <NODE2_IP>(rw,sync,no_subtree_check,no_root_squash)" >> /etc/exports'
sudo systemctl enable --now nfs-server
sudo exportfs -ra
sudo exportfs -v          # confirm the export is listed

# --- on node 2 ---
sudo dnf install -y nfs-utils
sudo mkdir -p /home/ec2-user/xmoe-4d
sudo mount -t nfs <NODE1_IP>:/home/ec2-user/xmoe-4d /home/ec2-user/xmoe-4d
```

Persist across reboots by appending to node 2's `/etc/fstab`:

```
<NODE1_IP>:/home/ec2-user/xmoe-4d  /home/ec2-user/xmoe-4d  nfs  defaults,_netdev  0 0
```

Exporting the whole of `~/xmoe-4d` is what makes the rest of Phase B trivial. In one
mount node 2 inherits, at byte-identical paths:

| | |
|---|---|
| `miniforge3/` + `XMoE4D_envs/` | the conda env, **including the CUDA activation hook** (`CUDA_HOME` / `CPATH`) |
| the env's `bin/` | `python`, `nvcc`, and `py-spy` (which the monitor needs on *every* node) |
| `X-MoE/` | repo, both editable installs, and the JIT-cached fused kernels |
| `.../scripts/env.sh` | the file the driver and every node source |
| `.../examples_xmoe_4d/data/` | the tokenized dataset |
| `.../scripts/logs/` | the shared `JOB_DIR` all 16 ranks write into |

So there is no "install everything twice" step, and no risk of the two nodes
drifting.

Trade-off: every Python import on node 2 crosses NFS, so rank startup is slower.
Correctness is unaffected. If node 2 has its own `~/xmoe-4d` install, this mount
shadows it -- which is what you want: identical paths, one shared job directory.

#### Option 2: FSx for Lustre (production)

Create an FSx for Lustre filesystem in the same VPC/subnet, mount at the same
path on every node, and place the repo, conda env and dataset on it. Higher
throughput and it scales past two nodes, but it is a separate provisioning step.

#### Verify either option

```bash
ssh -o BatchMode=yes <NODE2_IP> '
  source ~/xmoe-4d/X-MoE/Megatron-DeepSpeed-X-MoE/examples_xmoe_4d/scripts/env.sh
  echo "CUDA_HOME=$CUDA_HOME"
  echo "CPATH=$CPATH"
  echo "iface=$(ip -br addr | awk "\$2==\"UP\"{print \$1}" | head -1)"
  command -v py-spy
  python -c "import torch, six, deepspeed, megatron; print(\"gpus\", torch.cuda.device_count())"'
```

Every line must be non-empty, ending in `gpus 8`. This one command proves the mount,
the paths, the conda env, the activation hook, py-spy, and `env.sh` all work from
node 2. Check `iface` against the `NCCL_SOCKET_IFNAME` you set in `env.sh` -- both
nodes must agree on the interface name, and `env.sh` is shared, so a mismatch would
silently break NCCL on one node.

### B4. Create `hostfile`

One host per line, **line 1 becomes `MASTER_ADDR`**. Use IPs rather than
hostnames to avoid DNS resolution differences between nodes.

```bash
cd ~/xmoe-4d/X-MoE/Megatron-DeepSpeed-X-MoE/examples_xmoe_4d/scripts
printf '<NODE1_IP>\n<NODE2_IP>\n' > hostfile
```

The line count must equal the node count in the `autorun.sh` map key, or the
launcher aborts (`xmoe_4d.sh.template:373-376`).

### B5. NCCL smoke test

Validate the collective path before committing to a training launch. This uses the
same torchrun invocation the real run uses, and fails in seconds rather than 20
minutes into training.

```bash
cat > ~/xmoe-4d/ar_smoke.py <<'EOF'
import os, torch, torch.distributed as dist

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))

t = torch.ones(1 << 24, device="cuda")
dist.all_reduce(t)                      # sum of ones across N ranks == N
if dist.get_rank() == 0:
    print(f"SMOKE_RESULT world={dist.get_world_size()} sum={t[0].item():.1f}", flush=True)

dist.destroy_process_group()
EOF
```

```bash
cat > ~/xmoe-4d/smoke2node.sh <<'EOF'
#!/bin/bash
SCRIPTS=$HOME/xmoe-4d/X-MoE/Megatron-DeepSpeed-X-MoE/examples_xmoe_4d/scripts
MASTER=<NODE1_IP>
HOSTS=(<NODE1_IP> <NODE2_IP>)
NPROC=8
OUT=$(mktemp)

for h in "${HOSTS[@]}"; do
  ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$h" \
    "source $SCRIPTS/env.sh && torchrun --nnodes ${#HOSTS[@]} --nproc_per_node $NPROC \
     --rdzv_backend c10d --rdzv_endpoint ${MASTER}:29500 --rdzv_id smoke \
     $HOME/xmoe-4d/ar_smoke.py" 2>&1 | sed "s/^/[$h] /" >> "$OUT" &
done
wait
[ "$1" = "-v" ] && cat "$OUT"

EXPECT=$(( ${#HOSTS[@]} * NPROC ))
RESULT=$(grep -o "SMOKE_RESULT world=[0-9]* sum=[0-9.]*" "$OUT" | head -1)
echo "---------------------------------------------"
if [ -z "$RESULT" ]; then
    echo "FAIL: no result line -- ranks never completed the all-reduce."
    grep -m5 -iE "error|refused|timeout|unreachable" "$OUT" | head -5
    rm -f "$OUT"; exit 1
fi
W=$(sed 's/.*world=\([0-9]*\).*/\1/' <<<"$RESULT")
S=$(sed 's/.*sum=\([0-9.]*\)/\1/' <<<"$RESULT")
if [ "$W" = "$EXPECT" ] && [ "${S%.*}" = "$EXPECT" ]; then
    echo "PASS: world=$W, all-reduce sum=$S (== world, so every rank contributed)"
    rm -f "$OUT"; exit 0
fi
echo "FAIL: expected world=$EXPECT sum=$EXPECT, got world=$W sum=$S"
rm -f "$OUT"; exit 1
EOF
chmod +x ~/xmoe-4d/smoke2node.sh
bash ~/xmoe-4d/smoke2node.sh
```

Expected:

```
PASS: world=16, all-reduce sum=16.0 (== world, so every rank contributed)
```

`sum == world` is the real assertion: a tensor of ones all-reduced across N ranks
sums to N only if every rank actually took part. A `world=16` that summed to 8 would
mean half the ranks silently no-oped.

Both files live under `~/xmoe-4d` deliberately -- they must exist on every node, which
the shared mount guarantees.

#### The traceback you will see, and why it is not a failure

With `--rdzv_backend c10d` (which `xmoe_4d.sh.template` also uses), whichever agent
hosts the rendezvous store exits first and tears it down. Any agent still doing its
shutdown bookkeeping then logs a ~40-line stack ending in:

```
RendezvousConnectionError
[c10d] recvVector failed on SocketImpl(...): Failed to recv, got 0 bytes.
      Connection was likely closed. Did the remote server shutdown or crash?
```

It is emitted **after** the collective completed, and it is logged at `W` (warning),
not `E`. Nothing in the script can prevent it -- a `dist.barrier()` before
`destroy_process_group()` does not help, because the race is between the torchrun
*agents*, which outlive the Python processes. That is why the wrapper above hides it
and prints a verdict instead; use `-v` to see the raw output.

Also expect rank 0 to land on either node: torchrun assigns ranks by rendezvous join
order, not hostfile order. Only `MASTER_ADDR` is pinned to line 1 of the hostfile.

---

## Phase C -- the multi-node run

### C1. Point `autorun.sh` at two nodes

Comment out the single-node pair and add a two-node one:

```bash
# ---- single node (8 GPUs): PP2-EP4 ----
# PP_STRATEGY_MAP["1:8"]="2:4"
# PP_BATCH_MAP["1:8"]=" 1:2:15:X-MOE-4D:10B:ckpt:1:even:no-planner "

# ---- 2 nodes (16 GPUs): PP2-EP8 ----
PP_STRATEGY_MAP["2:16"]="2:8"
PP_BATCH_MAP["2:16"]=" 1:2:15:X-MOE-4D:10B:ckpt:1:even:no-planner "
```

Key format is `NODES:TOTAL_GPUS`; the strategy value is `PP_SIZE:EP_PARALLEL_SIZE`.

With `MP_SIZE=1` and `PP=2`, data-parallel width is `16 / 2 / 1 = 8`, so `EP=8`
divides evenly -- the natural scale-up of the PP2-EP4 config validated in A4.
Global batch is `NUM_BATCHES * BATCH_SIZE * (TOTAL_GPUS / PP_SIZE / MP_SIZE)`
= `2 * 1 * 8` = 16.

`BS=1` with `ckpt` is not arbitrary: it is the config measured at **21.0 GB peak of
39.5 GB** on this hardware in A4. The upstream `4:20:...:no-ckpt` values are tuned
for 64 GB MI250X and OOM on a 40 GB A100 -- see [Appendix C](#appendix-c-memory-tuning-on-40-gb-a100s).
Raise `BS` only after the 2-node path is green.

Keep `no-planner`. The planner needs a profiling cache generated on *this*
hardware; `autorun.sh:171-180` skips the run outright when it is missing. To
build one later, use `PROFILE_MAP` (it sweeps micro-batch sizes 1-8).

### C2. Launch

```bash
cd ~/xmoe-4d/X-MoE/Megatron-DeepSpeed-X-MoE/examples_xmoe_4d/scripts
bash autorun.sh 2>&1 | tee /tmp/2node.log
```

Override the rendezvous port if 29500 is taken:

```bash
MASTER_PORT=29600 bash autorun.sh
```

### C3. Verify both nodes actually participated

```bash
grep -E "NODES=|MASTER_ADDR|HOSTS:" /tmp/2node.log
ls logs/job_*/rank_*.log | wc -l                            # expect 16, not 8
grep -m1 -E "NET/Socket|NET/OFI" logs/job_*/rank_0.log      # which transport
grep -E "iteration|TFLOPs" logs/job_*/rank_15.log | tail    # last rank is alive
```

16 rank logs is the proof that node 2 joined. 8 means only node 1 ran and the
rendezvous quietly timed out into a single-node world.

---

## Monitoring a run (py-spy)

`scripts/monitor_run.sh` is the torchrun port of `scripts-frontier/monitor_run.sh`. It is
wired into `xmoe_4d.sh.template` and starts automatically, backgrounded, alongside training.
Same CLI and same output contract as the Frontier version, so anything you know about
reading those files transfers directly.

Every interval it snapshots each rank with `py-spy dump` and writes:

| File | Contents |
|---|---|
| `<JOB_DIR>/a-monitor.txt` | **Read this first.** Per-interval state histogram across all ranks, `LIVE x/world` liveness, frozen/odd counts, culprit native stack, dmesg scrape on rank death |
| `<JOB_DIR>/monitor/node{i}.txt` | Per-node table: `STATUS RANK GPU PID WATTS HBM% RAM_GB TRACE` |
| `<JOB_DIR>/monitor/monitor.tsv` | One row per (interval, rank), for grep/plot |

Watch it live:

```bash
tail -f logs/job_<id>/a-monitor.txt
```

### What it tells you

```
##  LIVE 15/16 ranks   GONE: 172.31.x.y:r7
##    r7 on 172.31.x.y GONE -- last HBM% 98 (peak 99) | RAM 41.2G (peak 41.2G)
##  FROZENx>=2: 15/16 (93%)    ODD: 1    longest-frozen: 172.31.x.y:r3 (moe/expert)
##  STATE HISTOGRAM (ranks by phase):
##        12  p2p-recv
##         3  moe/expert
```

- **`LIVE x/world`** -- rank count vs expected. A rank absent one interval is `suspect`;
  absent twice it is `GONE`. Peak HBM%/RAM are retained *after* the process dies, which is
  usually the only surviving evidence of an OOM.
- **`FROZENx<n>`** -- consecutive intervals with an unchanged stack. Nearly all ranks frozen
  in `p2p-recv` while one sits in `moe/expert` means that one rank is the straggler and the
  rest are blocked waiting on it.
- **`ODD`** -- this rank's stack differs from its node's modal stack. Odd + frozen = culprit.
- **State histogram** -- distribution across `p2p-send/recv`, `moe/expert`, `barrier`,
  `data-load`, `backward`, `compute-fwd`, `sync/timer`.
- **`unsamplable`** -- py-spy could not read the process (dead, or ptrace denied).

### Requirements and knobs

- `py-spy` on `PATH` or at `~/.local/bin/py-spy`, **on every node**. Absent, the monitor
  logs one line and skips -- it is never fatal.
- `/proc/sys/kernel/yama/ptrace_scope` must be `0`. Check with
  `cat /proc/sys/kernel/yama/ptrace_scope`; set temporarily with
  `sudo sysctl -w kernel.yama.ptrace_scope=0`.
- Single-node runs need no SSH: the collector detects that a host is the local machine and
  runs directly. Multi-node reuses the Phase B keys.

```bash
XMOE4D_MONITOR=0 bash autorun.sh                 # disable entirely
MON_FAST=30 MON_SLOW=120 bash autorun.sh        # 30s for the first MON_WINDOW s, then 120s
MON_TRACE_FRAMES=30 bash autorun.sh             # deeper stacks
```

Defaults: `MON_FAST=60`, `MON_SLOW=180`, `MON_WINDOW=300`, `MON_TRACE_FRAMES=20`.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ChildFailedError` with no useful message | torchrun never surfaces the child's error | Read `logs/job_<id>/rank_0.log` -- always |
| `ModuleNotFoundError: No module named 'six'` | Env gap | A1 |
| `fatal error: cuda_fp16.h: No such file or directory` | Env predates the CUDA activation hook; `CUDA_HOME` unset and headers live in `targets/x86_64-linux/include` | `./setup_env_cuda.sh envhook`, then re-activate |
| `cuda_fp16.h` still missing *after* running `envhook` | Conda activation hooks run only **at activation time** -- a shell that activated earlier never got them, and with no `env.sh` the launcher inherits that stale shell | Create `env.sh` (A3), or `conda activate <prefix>` again -- re-activating the same prefix re-runs `activate.d` |
| `FileNotFoundError: ../data/my-gpt2_text_document` | Dataset not built | A2 |
| `FATAL: NODES=2 but no hostfile` | Missing `hostfile` | B4 |
| `FATAL: hostfile has N hosts but NODES=M` | Line count vs map key mismatch | B4 / C1 |
| `Permission denied (publickey)` during launch | SSH not passwordless, or node 1 cannot ssh to itself | B2 |
| `bash: .../node_launch.sh: No such file` on node 2 | No shared filesystem | B3 |
| Hangs at rendezvous, then times out | Security group blocking, or wrong `MASTER_ADDR` | B1, B4 |
| `conda: NVCC_PREPEND_FLAGS: unbound variable` | `set -u` + the cuda-nvcc `activate.d` hook | Keep the `NVCC_PREPEND_FLAGS` guard line in `env.sh` |
| `torch.OutOfMemoryError` in `moe/experts.py` `torch.cat` | 40 GB A100 vs configs tuned for 64 GB MI250X; `MBS` too high and/or `no-ckpt` | [Appendix C](#appendix-c-memory-tuning-on-40-gb-a100s) |
| Only 8 rank logs for a 2-node run | Node 2 never joined | Re-run B5 |
| Every rank reports rank 0 | `SLURM_PROCID` exported in `env.sh` | Remove it; `pretrain_gpt_deepspeed.py` reads it before `RANK` |
| Multi-node run behaves differently from single-node (earlier OOM, no `NCCL INFO`) | `ssh` carries no environment, so driver globals never reached remote nodes | Fixed: `xmoe_4d.sh.template` now emits `XMOE4D_PROPAGATE_VARS` into `node_launch.sh` |
| `RuntimeError: NCCL Error 1: unhandled cuda error` in `all_to_all_single` | Usually a masked CUDA OOM -- NCCL cannot allocate its buffers | Check `HBM%` in `a-monitor.txt`; if it is 85%+, it is memory. See [Appendix C](#appendix-c-memory-tuning-on-40-gb-a100s) |

---

## Appendix C: memory tuning on 40 GB A100s

The shipped configs in `autorun.sh` were tuned on Frontier's MI250X, which has **64 GB
per GCD**. A p4d's A100-SXM4 has **40 GB** -- 62% of that -- so the reference
`no-ckpt` + `MBS=4` settings do not transfer, and the 10B model OOMs in the first
forward pass, inside the MoE expert `torch.cat`.

### Reading the budget

Grep the param count, which Megatron prints per pipeline stage:

```bash
grep "number of parameters on" logs/job_<id>/rank_0.log
```

For 10B with `PP=2` that is ~1.41B params on stage 0. At bf16 with ZeRO-1 over `DP=4`:

| Component | Bytes/param | Per GPU |
|---|---|---|
| Params (bf16) | 2 | 2.8 GB |
| Gradients (bf16) | 2 | 2.8 GB |
| Optimizer, ZeRO-1 sharded over DP (fp32 master + Adam m/v = 12, / 4) | 3 | 4.2 GB |
| **Fixed** | | **~9.9 GB** |

Everything above ~10 GB is activations, and activations scale linearly with
micro-batch size. A run that allocated 29.8 GB was therefore spending ~20 GB on them.

### The two knobs

Batch config format in `autorun.sh`:

```
BS : NBS : TRAIN_ITERS : MOE_TYPE : MODEL_SIZE : CHECKPOINT : CKPT_LAYERS : PP_PARTITION : PLANNER
```

- **`BS`** -- micro-batch size. Activations scale linearly; 4 -> 1 cuts them ~4x.
- **`CHECKPOINT`** -- `no-ckpt` | `ckpt` | `dynamic-ckpt`. Recomputes activations in the
  backward pass instead of storing them. Costs roughly 30% throughput, saves far more
  memory than `BS` alone. Mapped to flags by `derive_checkpoint_flags()`.

A safe first green run on 40 GB:

```bash
PP_BATCH_MAP["1:8"]=" 1:2:15:X-MOE-4D:10B:ckpt:1:even:no-planner "
```

Once green, raise `BS` until it OOMs again, then back off one step. Dropping back to
`no-ckpt` at `BS=1` should also fit (~15 GB estimated) and is faster.

### 50B and larger on 16 GPUs

50B is not "10B but bigger" -- nearly every dimension changes, and sequence length
doubles:

| | 10B | 50B |
|---|---|---|
| hidden | 2048 | 5120 |
| experts | 64 | 128 |
| **seq len** | 2048 | **4096** |
| layers | 24 | 24 |

`run_exp_training.sh` requires **32 GPUs** for 50B (`min_gpus_for()`); `autorun.sh`
has no such guard and will happily launch and die. On 16x40 GB at PP2/EP8 the
monitor shows `HBM%` reaching 85-87% and the run fails inside the MoE all-to-all
with `NCCL Error 1: unhandled cuda error` -- a masked OOM, not a network fault.

Use `50B_1L` / `50B_2L` / `50B_4L` to keep 50B's shape (hidden 5120, 128 experts,
seq 4096) while cutting depth, or give it the 32 GPUs the launcher asks for.

### If it still will not fit

The registry ships reduced-layer variants for exactly this -- `10B_1L`, `10B_2L`,
`10B_4L` (see `utils/model_registry.py`). They keep the model's shape and expert count
but cut the layer count, which is enough to validate plumbing:

```bash
PP_BATCH_MAP["1:8"]=" 1:2:15:X-MOE-4D:10B_4L:ckpt:1:even:no-planner "
```

### Catching the OOM in the monitor

The default `MON_FAST=60` can step right over a fast OOM. When hunting memory, sample
faster so peak `HBM%` per rank is actually recorded:

```bash
MON_FAST=15 MON_SLOW=30 bash autorun.sh
```

The monitor retains last/peak `HBM%` and `RAM_GB` per rank *after* the process dies,
which is usually the only surviving evidence of which rank blew up first.

---

## Appendix A: EFA

NCCL has two built-in transports, ibverbs and TCP sockets. EFA is neither -- the
`aws-ofi-nccl` plugin bridges NCCL to libfabric to EFA. Without a device, NCCL
falls back to TCP over the ENA interface **silently**: no error, just a large
fraction of the interconnect left unused (p4d: 400 Gbps, p5: 3200 Gbps).

Check whether this instance has EFA:

```bash
ls /sys/class/infiniband/         # EFA devices appear here
/opt/amazon/efa/bin/fi_info -p efa
ip -br addr                       # EFA-enabled instances show more than one ENI
```

`fi_getinfo: -61 (No data available)` and an empty `/sys/class/infiniband` mean
no EFA device is attached.

**Network cards are fixed at instance launch time -- EFA cannot be added to a
running instance.** It requires relaunching with EFA enabled on every network
card (4 for p4d), ideally in a cluster placement group.

Until then, keep `NCCL_NET=Socket` in `env.sh`. The plugin is preinstalled on the
DLAMI and on the system `ld.so.conf`, so NCCL will try to load it, find zero
devices, and either abort confusingly or fall back anyway. Forcing Socket makes
behaviour deterministic.

Once EFA is present, delete that line. The DLAMI auto-discovers the plugin --
confirm with `NCCL_DEBUG=INFO` and look for:

```
NET/OFI Selected Provider is efa
```

Do **not** set `FI_PROVIDER=efa`, `FI_EFA_USE_DEVICE_RDMA=1`, `NCCL_PROTO=simple`
or `FI_EFA_FORK_SAFE=1`. They are widely copy-pasted but obsolete with modern
plugin versions, and `NCCL_PROTO=simple` actively hurts latency by disabling
LL/LL128.

This guide's numbers are functional-validation only. Anything published should
come from an EFA-enabled cluster.

---

## Appendix B: files you create, and what to commit

| File | Committed? |
|---|---|
| `scripts/env.sh` | **No** -- site-specific; `env.sh.example` is the template |
| `scripts/hostfile` | **No** -- site-specific; `hostfile.example` is the template |
| `scripts/logs/` | No -- already in `.gitignore` |
| `examples_xmoe_4d/data/*.{bin,idx,json,txt,jsonl}` | No -- already in `.gitignore` |
| `scripts/monitor_run.sh` | **Yes** -- part of the repo, like its Frontier counterpart |
| `scripts/temp_sh/` | No -- generated |
| `scripts/output/` | No -- generated |

`env.sh` and `hostfile` are **not** currently gitignored. Add them before
committing so nobody pushes their private IPs:

```bash
cd ~/xmoe-4d/X-MoE/Megatron-DeepSpeed-X-MoE
printf 'examples_xmoe_4d/scripts/env.sh\nexamples_xmoe_4d/scripts/hostfile\nexamples_xmoe_4d/scripts/temp_sh/\nexamples_xmoe_4d/scripts/output/\n' >> .gitignore
```
