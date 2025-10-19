# profiling_utils/memory_profiler.py

import torch
import os
import csv
import atexit
from datetime import datetime
from pathlib import Path

# This is a dummy logger that does nothing if the real logger isn't initialized.
class _DummyLogger:
    def log(self, *args, **kwargs): pass
    def next_step(self, *args, **kwargs): pass

class MemoryLogger:
    _instance = None
    log_dir = None # This will be correctly populated on ALL ranks now.

    def __init__(self, rank, work_dir):
        """Initializes the logger for a specific rank."""
        if hasattr(self, 'initialized'):
            return

        self.rank = rank

        # --- NEW & CORRECTED LOGIC ---

        # We need a container to broadcast the directory path. A list is perfect.
        log_dir_container = [None]

        # Only Rank 0 determines the path and creates the directory.
        if self.rank == 0:
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            log_dir_path = os.path.join(work_dir, "memory_logs", timestamp)
            os.makedirs(log_dir_path, exist_ok=True)
            log_dir_container[0] = log_dir_path

        # Broadcast the container from rank 0 to all other ranks.
        # After this call, log_dir_container[0] will have the same path string on ALL ranks.
        if torch.distributed.is_initialized():
            torch.distributed.broadcast_object_list(log_dir_container, src=0)
        
        # Now that every rank has the path, assign it to the class variable.
        MemoryLogger.log_dir = log_dir_container[0]

        # --- End of new logic ---

        # This line will now work for all ranks because MemoryLogger.log_dir is no longer None.
        self.log_file_path = os.path.join(MemoryLogger.log_dir, f"memory_rank_{self.rank}.csv")
        
        try:
            self.file_handler = open(self.log_file_path, 'w', newline='')
            self.writer = csv.writer(self.file_handler)
            self.writer.writerow([
                "timestamp", "step", "event", "layer",
                "allocated_gb", "peak_allocated_gb",
                "reserved_gb", "peak_reserved_gb"
            ])
            self.file_handler.flush()
        except IOError as e:
            print(f"Rank {self.rank}: Error opening log file {self.log_file_path}: {e}")
            self.file_handler = None
            self.writer = None

        self.step = 0
        self.initialized = True
        atexit.register(self.close)

    @staticmethod
    def get_instance():
        """Retrieves the logger instance. Returns a dummy if not initialized."""
        if MemoryLogger._instance is None:
            return _DummyLogger()
        return MemoryLogger._instance

    # ... (the log, next_step, and close methods are unchanged) ...
    def log(self, event, layer="N/A"):
        if not self.writer: return
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / 1e9
        peak_allocated = torch.cuda.max_memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        peak_reserved = torch.cuda.max_memory_reserved() / 1e9
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        self.writer.writerow([
            timestamp, self.step, event, layer,
            f"{allocated:.4f}", f"{peak_allocated:.4f}",
            f"{reserved:.4f}", f"{peak_reserved:.4f}"
        ])
        self.file_handler.flush()

    def next_step(self):
        self.step += 1
        torch.cuda.reset_peak_memory_stats()
        self.log(event="start_of_step")

    def close(self):
        if self.file_handler:
            self.file_handler.close()
            self.file_handler = None


# --- GLOBAL FUNCTIONS ---

def init_memory_logger(work_dir=None):
    """
    Initializes the memory logger for the current rank.
    This MUST be called by all processes after torch.distributed.init_process_group.
    """
    if MemoryLogger._instance is None:
        try:
            rank = torch.distributed.get_rank()
            if work_dir is None:
                work_dir = Path.home()
            
            MemoryLogger._instance = MemoryLogger(rank, work_dir)
            
            if rank == 0:
                print(f"Memory logger initialized. Log directory: {MemoryLogger.log_dir}")

        except Exception as e:
            # Added more detail to the error print
            print(f"Rank {torch.distributed.get_rank()}: Failed to initialize MemoryLogger: {e}")
            import traceback
            traceback.print_exc()
            MemoryLogger._instance = _DummyLogger()


def log_memory(event, layer_num="N/A"):
    if layer_num == 'N/A' or (layer_num % 4 ==0): 
        MemoryLogger.get_instance().log(event=event, layer=layer_num)

def set_step(step_num):
    logger = MemoryLogger.get_instance()
    logger.step = step_num
    torch.cuda.reset_peak_memory_stats()
    logger.log(event="start_of_step")