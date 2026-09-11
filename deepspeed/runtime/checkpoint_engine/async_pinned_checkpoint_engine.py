# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
#
# [X-MoE, 2026-09-07] Proposal S2 of the UCP I/O campaign (examples_elmoe/ucp_io/round2/PROPOSALS_0906_2026.md):
# a checkpoint engine that blocks training only for the copy of the state to pinned host memory and writes the
# files from a background thread while training goes on. Measured problem (63B, 4 nodes, 2026-09-06): the save
# holds every GPU for 52-57 s, about 20 s of them copying the state to host memory piece by piece at 2.0 GB/s
# through torch.save's own path, then 22 s of writing one 26 GB shard as one stream, then a barrier.
#
# How it works, per rank:
#   save(state_dict, path)   snapshot: every GPU tensor is copied into a pinned host buffer on a side stream
#                            (buffers are taken from a pool keyed by dtype and shape and reused save after save);
#                            CPU tensors are cloned; other values are deep-copied. save() returns once the copies
#                            have landed, which is the only time training is blocked. The (host state, path) pair
#                            is queued for the writer thread, which runs torch.save exactly as the torch engine does.
#   create(tag)              the start of a new checkpoint: waits until the previous checkpoint's writes are done,
#                            so that at most one checkpoint is in flight and the pinned memory of a rank is bounded
#                            by one copy of its state.
#   commit(tag)              returns at once. A committer thread waits for this rank's writes of the tag to finish,
#                            leaves a marker <save_dir>/<tag>/.async_done_rank<R>; on rank 0 it then waits for the
#                            markers of every rank, writes <save_dir>/latest = tag, and removes the markers. The
#                            engine sets writes_latest = True so that DeepSpeedEngine.save_checkpoint does not write
#                            'latest' itself: 'latest' never names a checkpoint whose files are still being written.
#   load(path)               torch.load, as the torch engine.
# What does not change: the files, their names and their contents (torch.save of the same state, from host
# copies; storages are tagged 'cpu' instead of 'cuda:N' in the pickle, which every loader maps anyway).
# What to watch: pinned host memory of one full state per rank; a failure in the writer is raised by the next
# create()/save() of that rank and leaves <save_dir>/<tag>/.async_error_rank<R>; a job that ends while writes
# are in flight joins the writer at interpreter exit (non-daemon thread), so the last checkpoint completes.

import copy
import os
import queue
import threading
import time

import torch
from deepspeed.utils import logger, log_dist
from deepspeed.runtime.checkpoint_engine.checkpoint_engine import CheckpointEngine


class AsyncPinnedCheckpointEngine(CheckpointEngine):

    writes_latest = True   # DeepSpeedEngine.save_checkpoint leaves 'latest' to this engine

    def __init__(self, config_params=None):
        super().__init__(config_params)
        self._pool = {}                 # (dtype, shape) -> [free pinned tensors]
        self._pool_lock = threading.Lock()
        self._queue = queue.Queue()     # (host_state, path, buffers, tag)
        self._writer = None
        self._writes_done = threading.Condition()
        self._in_flight = 0             # files queued and not yet written
        self._stream = None
        self._errors = []
        self._current_tag = None
        self._save_dirs = {}            # tag -> save_dir (the parent of <save_dir>/<tag>/...)
        self._commit_timeout_s = float(os.environ.get('DS_ASYNC_CKPT_COMMIT_TIMEOUT_S', 7200))
        self._stats = {}                # tag -> dict of timings for the log
        self._pinned_bytes = 0

    # ------------------------------------------------------------------ helpers
    def _rank(self):
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                return dist.get_rank(), dist.get_world_size()
        except Exception:
            pass
        return 0, 1

    def _buffer(self, t):
        key = (t.dtype, tuple(t.shape))
        with self._pool_lock:
            free = self._pool.get(key)
            if free:
                return free.pop()
        buf = torch.empty(t.shape, dtype=t.dtype, device='cpu', pin_memory=True)
        self._pinned_bytes += buf.numel() * buf.element_size()
        return buf

    def _release(self, buffers):
        with self._pool_lock:
            for b in buffers:
                self._pool.setdefault((b.dtype, tuple(b.shape)), []).append(b)

    def _snapshot(self, obj, buffers):
        """The same object graph with every accelerator tensor replaced by a pinned host copy (copies are issued on
        the side stream and waited for by the caller), CPU tensors cloned, everything else deep-copied."""
        if torch.is_tensor(obj):
            if obj.is_cuda:
                buf = self._buffer(obj)
                buf.copy_(obj, non_blocking=True)
                buffers.append(buf)
                return buf
            return obj.detach().clone()
        if isinstance(obj, dict):
            return {k: self._snapshot(v, buffers) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._snapshot(v, buffers) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self._snapshot(v, buffers) for v in obj)
        try:
            return copy.deepcopy(obj)
        except Exception:
            return obj

    def _check_errors(self):
        if self._errors:
            raise RuntimeError(f'[AsyncPinned] a checkpoint write failed earlier on rank {self._rank()[0]}: {self._errors[0]!r}')

    def _start_writer(self):
        if self._writer is None:
            self._writer = threading.Thread(target=self._writer_loop, name='ds-async-ckpt-writer', daemon=False)
            self._writer.start()

    def _writer_loop(self):
        main = threading.main_thread()
        while True:
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                if not main.is_alive():
                    return
                continue
            host_state, path, buffers, tag = item
            t0 = time.perf_counter()
            try:
                torch.save(host_state, path)
                dt = time.perf_counter() - t0
                self._stats.setdefault(tag, {}).setdefault('write_s', []).append(round(dt, 3))
                logger.info(f'[AsyncPinned] wrote {path} in {dt:.1f} s (background)')
            except Exception as e:
                self._errors.append(e)
                logger.error(f'[AsyncPinned] write of {path} FAILED: {e!r}')
                try:
                    with open(os.path.join(os.path.dirname(path), f'.async_error_rank{self._rank()[0]}'), 'a') as fh:
                        fh.write(f'{path}: {e!r}\n')
                except Exception:
                    pass
            finally:
                del host_state, item
                self._release(buffers)
                with self._writes_done:
                    self._in_flight -= 1
                    self._writes_done.notify_all()
                self._queue.task_done()

    def wait_all(self, timeout=None):
        """Block until every queued write of this rank has finished (used by create(), commit's thread, and tests)."""
        with self._writes_done:
            self._writes_done.wait_for(lambda: self._in_flight == 0, timeout=timeout)
            return self._in_flight == 0

    # ------------------------------------------------------------------ the interface
    def create(self, tag):
        t0 = time.perf_counter()
        self.wait_all()                    # at most one checkpoint in flight: bounds the pinned memory
        self._check_errors()
        self._current_tag = tag
        self._stats[tag] = {'wait_previous_s': round(time.perf_counter() - t0, 3), 'snapshot_s': [], 'write_s': []}
        log_dist(f'[AsyncPinned] Checkpoint {tag} is about to be saved (previous writes waited for {time.perf_counter() - t0:.2f} s)', ranks=[0])

    def save(self, state_dict, path):
        self._check_errors()
        t0 = time.perf_counter()
        buffers = []
        if torch.cuda.is_available():
            if self._stream is None:
                self._stream = torch.cuda.Stream()
            self._stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self._stream):
                host_state = self._snapshot(state_dict, buffers)
                event = torch.cuda.Event()
                event.record(self._stream)
            event.synchronize()
        else:
            host_state = self._snapshot(state_dict, buffers)
        dt = time.perf_counter() - t0
        nbytes = sum(b.numel() * b.element_size() for b in buffers)
        tag = self._current_tag or os.path.basename(os.path.dirname(path))
        self._save_dirs[tag] = os.path.dirname(os.path.dirname(path))
        self._stats.setdefault(tag, {}).setdefault('snapshot_s', []).append(round(dt, 3))
        logger.info(f'[AsyncPinned] snapshot of {path}: {nbytes / 1e9:.2f} GB of device tensors to pinned memory in {dt:.2f} s '
                    f'({nbytes / 1e9 / dt if dt > 0 else 0:.1f} GB/s); the write continues in the background')
        with self._writes_done:
            self._in_flight += 1
        self._start_writer()
        self._queue.put((host_state, path, buffers, tag))
        return None

    def load(self, path, map_location=None):
        logger.info(f'[AsyncPinned] Loading checkpoint from {path}...')
        partition = torch.load(path, map_location=map_location, weights_only=False)
        logger.info(f'[AsyncPinned] Loaded checkpoint from {path}.')
        return partition

    def commit(self, tag):
        """Return at once; a committer thread publishes 'latest' when every rank's writes of the tag are done."""
        rank, world = self._rank()
        save_dir = self._save_dirs.get(tag)
        if save_dir is None:
            logger.warning(f'[AsyncPinned] commit({tag}) without a save on this rank; nothing to publish')
            return True
        th = threading.Thread(target=self._commit_loop, args=(tag, save_dir, rank, world), name='ds-async-ckpt-commit', daemon=False)
        th.start()
        return True

    def _commit_loop(self, tag, save_dir, rank, world):
        t0 = time.perf_counter()
        ok = self.wait_all(timeout=self._commit_timeout_s)
        tag_dir = os.path.join(save_dir, tag)
        if not ok or self._errors:
            logger.error(f'[AsyncPinned] rank {rank}: checkpoint {tag} NOT complete (timeout={not ok}, errors={len(self._errors)}); latest is not updated')
            return
        marker = os.path.join(tag_dir, f'.async_done_rank{rank}')
        with open(marker, 'w') as fh:
            fh.write(f'{time.time()}\n')
        st = self._stats.get(tag, {})
        logger.info(f'[AsyncPinned] rank {rank}: checkpoint {tag} written in the background: snapshots {st.get("snapshot_s")} s, '
                    f'writes {st.get("write_s")} s, {time.perf_counter() - t0:.1f} s after commit')
        if rank != 0:
            return
        deadline = time.time() + self._commit_timeout_s
        while time.time() < deadline:
            done = [r for r in range(world) if os.path.exists(os.path.join(tag_dir, f'.async_done_rank{r}'))]
            if len(done) == world:
                break
            if any(f.startswith('.async_error_rank') for f in os.listdir(tag_dir)):
                logger.error(f'[AsyncPinned] checkpoint {tag}: a rank reported a failed write; latest is not updated')
                return
            time.sleep(0.5)
        else:
            logger.error(f'[AsyncPinned] checkpoint {tag}: only {len(done)} of {world} ranks finished within {self._commit_timeout_s} s; latest is not updated')
            return
        tmp = os.path.join(save_dir, 'latest.async_tmp')
        with open(tmp, 'w') as fh:
            fh.write(tag)
        os.replace(tmp, os.path.join(save_dir, 'latest'))
        for r in range(world):
            try:
                os.remove(os.path.join(tag_dir, f'.async_done_rank{r}'))
            except OSError:
                pass
        logger.info(f'[AsyncPinned] checkpoint {tag} is complete on all {world} ranks: latest -> {tag} ({time.perf_counter() - t0:.1f} s after commit)')
