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
#
# [X-MoE resilience, 2026-10-09 evening] THE NODE-LOCAL CHECKPOINT COPY (design 1009_2026_RESILIENCE_DESIGN_TOPDOWN.md 3.5),
# flag AUTORUN_LOCAL_COPIES=0|1|2 (default 0 = the behaviour above, unchanged). Frontier compute nodes are diskless, so the
# "local" tier is node memory: the tmpfs /dev/shm, which survives the process (a restart inside the job is a NEW process on the
# same node) and dies with the node or the job.
#   save:   the writer thread serialises the host copy of each file to /dev/shm/xmoe_ckpt/<job>_<run>/<tag>/<file> (a torch.save
#           into memory at memcpy speed, no Lustre) BEFORE the Lustre write of the same file; the local copy is complete when the
#           Lustre copy starts. The newest k = AUTORUN_LOCAL_COPIES tags are kept: once 'latest' on Lustre names a tag (every rank's
#           writes of it are complete), a daemon thread removes the local tag folders older than the k newest (whole folders: no rank
#           writes an older tag once 'latest' has passed it, and leftovers of an earlier process on the node go the same way). The
#           write is skipped with a logged line when the copy would not fit (/dev/shm free minus a 2 GB reserve). A failure of the
#           local write never touches the Lustre write.
#   load:   load(path) maps the Lustre file to its local twin; when the twin exists and has the same size as the Lustre file (both
#           are torch.save of the same host state) it is read from /dev/shm, else from Lustre as today. The decision is per FILE,
#           so a rank that moved to another node, or a spare node that never saved, reads Lustre without a special case, and a file
#           written by another rank (the mp_rank model states, the expert files of the first replica) is read from Lustre by the
#           ranks that did not write it. 'latest' is read from Lustre only (DeepSpeedEngine.load_checkpoint).
#   record: <RES_DATA_DIR>/local_copy.tsv gets one row per local write and per file loaded (time attempt rank node tag kind file
#           bytes seconds shm_used_gb shm_free_gb); the rank log gets one line per file ("local copy: rank r loaded ... from
#           /dev/shm in X s" / "from Lustre in Y s") and a per-rank summary at the next create().
#   also:   a save whose tag equals the last tag this process committed is skipped (create/save/commit become no-ops): the
#           end-of-training save of a step that was also a periodic save (lessons.md 2026-10-03 15:26) rewrote complete files and
#           its repeated commit raced the first commit's marker cleanup, which hung the ranks until the commit timeout.

import copy
import os
import queue
import re
import shutil
import socket
import threading
import time

import torch
from deepspeed.utils import logger, log_dist


def _hb_save(where, background=False):
    """[X-MoE resilience 2026-10-09 late evening] heartbeat hook (XMOE_HB=1): tell the heartbeat that a save is in progress here.
    Found in the 260-node job 5645677 at 22:34: the last pipeline stage's ranks finished their checkpoint write 53 s after rank 0;
    the Python-level host copy of this engine holds the GIL, so their heartbeat thread wrote nothing for 49-59 s and their phase
    still read 'step' (the save hook of training.py covered save_checkpoint only), and the aggregator's stillness and minority rules
    raised a false alarm. The hook sets the phase to 'save' and writes one heartbeat line from the calling thread itself (the
    starved heartbeat thread cannot), and in the background writer and the commit wait it stamps the save activity so that the
    aggregator keeps the save window open. A no-op when the hb module is not loaded."""
    _xh = __import__('sys').modules.get('xmoe_hb')
    if _xh is None:
        return
    try:
        if background:
            _xh.save_activity(where)
        else:
            _xh.save_mark(where)
    except Exception:
        pass
from deepspeed.runtime.checkpoint_engine.checkpoint_engine import CheckpointEngine

_TAG_RX = re.compile(r'^global_step\d+$')


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
        # [resilience 2026-10-09] the node-local copy and the duplicate-tag skip
        try:
            self._local_k = int(os.environ.get('AUTORUN_LOCAL_COPIES', '0') or 0)
        except ValueError:
            self._local_k = 0
        self._local_base = os.environ.get('XMOE_LOCAL_CKPT_DIR', '/dev/shm/xmoe_ckpt')
        self._local_reserve = float(os.environ.get('XMOE_LOCAL_CKPT_RESERVE_GB', '2')) * 1e9
        self._local_lock = threading.Lock()
        self._local_files = {}          # tag -> [local files this rank wrote], insertion order = save order
        self._local_skipped = set()     # tags whose local write was skipped for space (logged once)
        self._load_log = []             # (src, bytes, seconds, file) per file loaded since the last summary
        self._last_committed = None
        self._skip_tag = None
        self._node = socket.gethostname().split('.')[0]
        # [X-MoE wall-aware exit 2026-10-10] what the trainer needs to exit without cutting a save: the commit threads (joined by
        # wait_published), the last save this rank finished (last_marker) and, on rank 0 only, the last save that is complete on
        # every rank with its time from the commit to 'latest' (last_complete; the 260-node job measured 68.6 s)
        self._commit_threads = []
        self.last_marker = None         # {'tag', 'seconds'}: this rank's writes of the tag were done, its marker written
        self.last_complete = None       # rank 0: {'tag', 'commit_to_latest_s', 'time'} after 'latest' named the tag
        self._stop_writer = False       # set by stop_for_exit: the writer loop returns at its next idle poll (0.5 s)

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
                if not main.is_alive() or self._stop_writer:
                    return
                continue
            host_state, path, buffers, tag = item
            if self._local_k > 0:
                self._local_write(host_state, path, tag)      # the node-local copy first (memory speed); never raises
            t0 = time.perf_counter()
            try:
                torch.save(host_state, path)
                dt = time.perf_counter() - t0
                self._stats.setdefault(tag, {}).setdefault('write_s', []).append(round(dt, 3))
                logger.info(f'[AsyncPinned] wrote {path} in {dt:.1f} s (background)')
                _hb_save(f'background write of {os.path.basename(path)} done in {dt:.1f} s', background=True)
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

    # ------------------------------------------------------------------ the node-local copy (AUTORUN_LOCAL_COPIES >= 1)
    def _local_path(self, path):
        """<local base>/<job>_<run>/<tag>/<file relative to the tag dir>, or None when the path holds no global_step<N> component."""
        p = os.path.normpath(path)
        parts = p.split(os.sep)
        idx = None
        for i in range(len(parts) - 1, -1, -1):
            if _TAG_RX.match(parts[i]):
                idx = i
                break
        if idx is None or idx == 0:
            return None
        run = parts[idx - 1] or 'run'
        job = os.environ.get('SLURM_JOB_ID', 'nojob')
        return os.path.join(self._local_base, f'{job}_{run}', *parts[idx:])

    @staticmethod
    def _tag_of(path):
        for part in reversed(os.path.normpath(path).split(os.sep)):
            if _TAG_RX.match(part):
                return part
        return '-'

    @staticmethod
    def _state_bytes(obj):
        if torch.is_tensor(obj):
            return obj.numel() * obj.element_size()
        if isinstance(obj, dict):
            return sum(AsyncPinnedCheckpointEngine._state_bytes(v) for v in obj.values())
        if isinstance(obj, (list, tuple)):
            return sum(AsyncPinnedCheckpointEngine._state_bytes(v) for v in obj)
        return 0

    def _shm_usage(self):
        """(used GB, free GB) of the filesystem that holds the local copies (the tmpfs /dev/shm on Frontier)."""
        try:
            st = os.statvfs(self._local_base if os.path.isdir(self._local_base) else os.path.dirname(self._local_base) or '/')
            total = st.f_blocks * st.f_frsize; free = st.f_bavail * st.f_frsize
            return (total - free) / 1e9, free / 1e9
        except OSError:
            return -1.0, -1.0

    def _record(self, tag, kind, path, nbytes, seconds):
        """One row of <RES_DATA_DIR>/local_copy.tsv: time attempt rank node tag kind file bytes seconds shm_used_gb shm_free_gb."""
        d = os.environ.get('RES_DATA_DIR')
        if not d:
            return
        try:
            used, free = self._shm_usage()
            with open(os.path.join(d, 'local_copy.tsv'), 'a') as fh:
                fh.write('\t'.join([time.strftime('%Y-%m-%dT%H:%M:%S'), os.environ.get('RES_ATTEMPT', '0'), str(self._rank()[0]), self._node,
                                    tag, kind, os.path.basename(path), str(int(nbytes)), f'{seconds:.3f}', f'{used:.2f}', f'{free:.2f}']) + '\n')
        except Exception:
            pass

    def _local_write(self, host_state, path, tag):
        rank = self._rank()[0]
        loc = self._local_path(path)
        if loc is None:
            return
        # the same basename in a per-rank temporary folder, then an atomic rename: torch.save names the zip archive inside the
        # file after the file's basename, so a copy written as <file>.tmp would differ from the Lustre file by a few bytes per
        # record and fail the size check of load()
        tmp = os.path.join(os.path.dirname(loc), f'.tmp_rank{rank}', os.path.basename(loc))
        try:
            need = self._state_bytes(host_state) * 1.02 + 64e6
            used, free = self._shm_usage()
            if free >= 0 and free * 1e9 - need < self._local_reserve:
                if tag not in self._local_skipped:
                    self._local_skipped.add(tag)
                    logger.warning(f'[local copy] rank {rank} on {self._node}: the local copy of {tag} is SKIPPED: {need / 1e9:.2f} GB needed for '
                                   f'{os.path.basename(path)}, {free:.2f} GB free in {self._local_base} (reserve {self._local_reserve / 1e9:.0f} GB); Lustre only')
                self._record(tag, 'shm_skip', path, need, 0.0)
                return
            os.makedirs(os.path.dirname(tmp), exist_ok=True)
            t0 = time.perf_counter()
            torch.save(host_state, tmp)
            os.replace(tmp, loc)
            dt = time.perf_counter() - t0
            try:
                os.rmdir(os.path.dirname(tmp))
            except OSError:
                pass
            size = os.path.getsize(loc)
            with self._local_lock:
                self._local_files.setdefault(tag, []).append(loc)
            used, free = self._shm_usage()
            logger.info(f'[local copy] rank {rank} on {self._node}: {os.path.basename(path)} of {tag} kept in {os.path.dirname(loc)} '
                        f'({size / 1e9:.2f} GB in {dt:.2f} s, {size / 1e9 / dt if dt > 0 else 0:.1f} GB/s); {self._local_base} used {used:.1f} GB, free {free:.1f} GB')
            self._record(tag, 'shm_write', path, size, dt)
        except Exception as e:
            logger.warning(f'[local copy] rank {rank} on {self._node}: local write of {os.path.basename(path)} ({tag}) failed, Lustre only: {e!r}')
            try:
                os.remove(tmp)
            except OSError:
                pass

    @staticmethod
    def _step(tag):
        m = re.search(r'(\d+)$', tag or '')
        return int(m.group(1)) if m else -1

    def _local_prune(self, latest_tag, save_dir):
        """Keep the newest k local tag folders whose step <= the step of 'latest'; remove the older ones (whole folders)."""
        if self._local_k <= 0:
            return
        probe = self._local_path(os.path.join(save_dir, latest_tag, 'x'))
        if probe is None:
            return
        run_dir = os.path.dirname(os.path.dirname(probe))
        try:
            tags = [d for d in os.listdir(run_dir) if _TAG_RX.match(d) and os.path.isdir(os.path.join(run_dir, d))]
        except OSError:
            return
        done = sorted([t for t in tags if self._step(t) <= self._step(latest_tag)], key=self._step)
        drop = done[:-self._local_k] if len(done) > self._local_k else []
        for t in drop:
            shutil.rmtree(os.path.join(run_dir, t), ignore_errors=True)
            with self._local_lock:
                self._local_files.pop(t, None)
        if drop:
            used, free = self._shm_usage()
            logger.info(f'[local copy] rank {self._rank()[0]} on {self._node}: released the local copies of {", ".join(drop)} (latest is {latest_tag}; '
                        f'keep {self._local_k} newest: {", ".join(done[-self._local_k:])}); {self._local_base} used {used:.1f} GB, free {free:.1f} GB')

    def _prune_when_published(self, tag, save_dir):
        """Daemon thread: wait until 'latest' on Lustre names this tag (or a later one), then prune; never blocks the exit."""
        deadline = time.time() + self._commit_timeout_s
        latest_p = os.path.join(save_dir, 'latest')
        while time.time() < deadline:
            try:
                cur = open(latest_p).read().strip()
            except OSError:
                cur = ''
            if cur and self._step(cur) >= self._step(tag) >= 0:
                try:
                    self._local_prune(cur, save_dir)
                except Exception as e:
                    logger.warning(f'[local copy] rank {self._rank()[0]}: pruning the older local copies failed: {e!r}')
                return
            time.sleep(0.5)

    def _load_summary(self):
        if not self._load_log:
            return
        rank = self._rank()[0]
        tag = self._load_log[0][4] if len(self._load_log[0]) > 4 else '-'
        for src, name in (('local', '/dev/shm'), ('lustre', 'Lustre')):
            sel = [x for x in self._load_log if x[0] == src]
            if sel:
                logger.info(f'[local copy] rank {rank} on {self._node}: loaded step {tag}: {len(sel)} files ({sum(x[1] for x in sel) / 1e9:.2f} GB) from {name} in {sum(x[2] for x in sel):.2f} s')
                self._record(tag, f'load_summary_{src}', f'{len(sel)}_files', sum(x[1] for x in sel), sum(x[2] for x in sel))
        self._load_log = []

    # ------------------------------------------------------------------ the interface
    def create(self, tag):
        t0 = time.perf_counter()
        _hb_save(f'save of {tag}: waiting for the previous checkpoint\'s background writes')   # heartbeat: part of the save window
        self.wait_all()                    # at most one checkpoint in flight: bounds the pinned memory
        self._check_errors()
        self._load_summary()
        if tag is not None and tag == self._last_committed:
            self._skip_tag = tag
            self._current_tag = tag
            log_dist(f'[AsyncPinned] checkpoint {tag} was already written and committed by this process: this repeated save of the same '
                     f'step (the end-of-training save after a periodic save) is skipped, the files are complete', ranks=[0])
            return
        self._skip_tag = None
        self._current_tag = tag
        self._stats[tag] = {'wait_previous_s': round(time.perf_counter() - t0, 3), 'snapshot_s': [], 'write_s': []}
        log_dist(f'[AsyncPinned] Checkpoint {tag} is about to be saved (previous writes waited for {time.perf_counter() - t0:.2f} s)', ranks=[0])

    def save(self, state_dict, path):
        if self._skip_tag is not None and self._current_tag == self._skip_tag:
            return None
        self._check_errors()
        t0 = time.perf_counter()
        buffers = []
        _hb_save(f'host copy of {os.path.basename(path)} to pinned memory starts')   # heartbeat: the save is in progress on this thread
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
        _hb_save(f'host copy of {os.path.basename(path)} done in {dt:.1f} s; the write continues in the background')
        with self._writes_done:
            self._in_flight += 1
        self._start_writer()
        self._queue.put((host_state, path, buffers, tag))
        return None

    def load(self, path, map_location=None):
        src, where = 'lustre', path
        if self._local_k > 0:
            loc = self._local_path(path)
            if loc and os.path.isfile(loc):
                try:
                    ls, rs = os.path.getsize(loc), os.path.getsize(path)
                    if ls == rs:
                        src, where = 'local', loc
                    else:
                        logger.warning(f'[local copy] rank {self._rank()[0]} on {self._node}: {loc} has {ls} bytes, the Lustre file {rs}: the local copy is not used')
                except OSError as e:
                    logger.warning(f'[local copy] rank {self._rank()[0]} on {self._node}: {loc} not used ({e!r})')
        logger.info(f'[AsyncPinned] Loading checkpoint from {where}...')
        t0 = time.perf_counter()
        partition = torch.load(where, map_location=map_location, weights_only=False)
        dt = time.perf_counter() - t0
        logger.info(f'[AsyncPinned] Loaded checkpoint from {where}.')
        if self._local_k > 0:
            try:
                size = os.path.getsize(where)
            except OSError:
                size = 0
            tag = self._tag_of(path)
            logger.info(f'[local copy] rank {self._rank()[0]} on {self._node}: loaded {os.path.basename(path)} of {tag} from '
                        f'{"/dev/shm" if src == "local" else "Lustre"} in {dt:.2f} s ({size / 1e9:.2f} GB)')
            self._load_log.append((src, size, dt, os.path.basename(path), tag))
            self._record(tag, f'load_{src}', path, size, dt)
        return partition

    def commit(self, tag):
        """Return at once; a committer thread publishes 'latest' when every rank's writes of the tag are done."""
        if self._skip_tag is not None and tag == self._skip_tag:
            return True
        rank, world = self._rank()
        save_dir = self._save_dirs.get(tag)
        if save_dir is None:
            logger.warning(f'[AsyncPinned] commit({tag}) without a save on this rank; nothing to publish')
            return True
        self._last_committed = tag
        th = threading.Thread(target=self._commit_loop, args=(tag, save_dir, rank, world), name='ds-async-ckpt-commit', daemon=False)
        th.start()
        self._commit_threads = [t for t in self._commit_threads if t.is_alive()] + [th]
        return True

    def wait_published(self, tag, timeout=None):
        """[X-MoE wall-aware exit 2026-10-10] Block until the save of `tag` is complete on EVERY rank: first this rank's commit
        thread (its own background writes done, its marker written), then <save_dir>/latest naming the tag, which rank 0 writes
        only after every rank's marker exists. Returns (complete, seconds waited, what 'latest' names). A tag that was skipped as
        a repeat of the committed save (create's _skip_tag) is complete at once when 'latest' already names it."""
        t0 = time.perf_counter()
        budget = float(timeout if timeout is not None else self._commit_timeout_s)
        for th in list(self._commit_threads):
            left = budget - (time.perf_counter() - t0)
            if left > 0:
                th.join(left)
        self._commit_threads = [t for t in self._commit_threads if t.is_alive()]
        save_dir = self._save_dirs.get(tag) or (list(self._save_dirs.values())[-1] if self._save_dirs else None)
        latest_p = os.path.join(save_dir, 'latest') if save_dir else None
        cur = None
        while True:
            if latest_p:
                try:
                    cur = open(latest_p).read().strip()
                except OSError:
                    cur = None
            if cur == tag or latest_p is None:
                break
            if time.perf_counter() - t0 > budget:
                break
            time.sleep(0.5)
        return (cur == tag), round(time.perf_counter() - t0, 1), cur

    def stop_for_exit(self, timeout=30.0):
        """[X-MoE wall-aware exit 2026-10-10 02:30] Called by the trainer right before os._exit(0): tell the writer thread to return
        at its next idle poll and join it for at most `timeout` seconds, so that the process ends with no thread still inside
        torch.save. Once wait_published has returned true, this rank's queue is empty (its marker was written only after
        wait_all), so the join is a formality that ends within 0.5 s; a writer still busy (a save that was never committed) is
        given the timeout and then left to die with the process, and the return value says which happened. Nothing here
        touches torch.distributed."""
        self._stop_writer = True
        th = self._writer
        if th is None or not th.is_alive():
            return True
        th.join(timeout)
        return not th.is_alive()

    def _commit_loop(self, tag, save_dir, rank, world):
        t0 = time.perf_counter()
        _hb_save(f'commit of {tag}: waiting for this rank\'s background writes', background=True)
        ok = self.wait_all(timeout=self._commit_timeout_s)
        _hb_save(f'commit of {tag}: this rank\'s writes are done', background=True)
        tag_dir = os.path.join(save_dir, tag)
        if not ok or self._errors:
            logger.error(f'[AsyncPinned] rank {rank}: checkpoint {tag} NOT complete (timeout={not ok}, errors={len(self._errors)}); latest is not updated')
            return
        if self._local_k > 0:
            threading.Thread(target=self._prune_when_published, args=(tag, save_dir), name='ds-local-copy-prune', daemon=True).start()
        marker = os.path.join(tag_dir, f'.async_done_rank{rank}')
        with open(marker, 'w') as fh:
            fh.write(f'{time.time()}\n')
        st = self._stats.get(tag, {})
        logger.info(f'[AsyncPinned] rank {rank}: checkpoint {tag} written in the background: snapshots {st.get("snapshot_s")} s, '
                    f'writes {st.get("write_s")} s, {time.perf_counter() - t0:.1f} s after commit')
        self.last_marker = dict(tag=tag, seconds=round(time.perf_counter() - t0, 1))
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
        dt = round(time.perf_counter() - t0, 1)
        logger.info(f'[AsyncPinned] checkpoint {tag} is complete on all {world} ranks: latest -> {tag} ({dt:.1f} s after commit)')
        # [X-MoE wall-aware exit 2026-10-10] the measured cost of the asynchronous part of a save, for the trainer's prediction of the
        # next save (last_complete, in this process) and for the next attempt of the same job (data/save_async_times.tsv in the job
        # folder, read by wall_exit.read_save_files before that attempt's first save)
        self.last_complete = dict(tag=tag, commit_to_latest_s=dt, time=time.time())
        dd = os.environ.get('RES_DATA_DIR')
        if dd:
            try:
                p = os.path.join(dd, 'save_async_times.tsv')
                new = not os.path.exists(p)
                with open(p, 'a') as fh:
                    if new:
                        fh.write('# save_async_times.tsv: one line per complete save (the async engine, rank 0): tag, seconds from the commit to latest, time\n')
                    fh.write(f'{tag}\t{dt:.1f}\t{time.strftime("%Y-%m-%dT%H:%M:%S")}\n')
            except OSError as e:
                logger.warning(f'[AsyncPinned] save_async_times.tsv not written: {e}')
