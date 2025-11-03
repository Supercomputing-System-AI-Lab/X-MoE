# profiling_utils/memory_profiler.py

import torch
import os
import csv
import atexit
from datetime import datetime
from pathlib import Path


# --- NEW: Add imports needed for final analysis ---
import pandas as pd
import glob
import re

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
            # self.writer = csv.writer(self.file_handler)
            self.writer = csv.writer(self.file_handler, quoting=csv.QUOTE_ALL)
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

    # def close(self):
    #     if self.file_handler:
    #         self.file_handler.close()
    #         self.file_handler = None
    
# --- MODIFIED: The close method now triggers the summary printout ---
    def close(self):
        """Close the file handle and, on rank 0, print the max memory summary."""
        if self.file_handler:
            self.file_handler.close()
            self.file_handler = None

        # Synchronize all processes to ensure all log files are fully written
        # before rank 0 starts analyzing them.
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        # Only Rank 0 should perform the analysis and print the summary.
        if self.rank == 0:
            _calculate_and_print_max_memory(log_dir=MemoryLogger.log_dir)



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
    
# --- NEW: ANALYSIS FUNCTION CALLED AUTOMATICALLY AT EXIT ---

def _calculate_and_print_max_memory(log_dir):
    """
    Analyzes all memory CSVs to find the max 'reserved_gb' and 'allocated_gb'
    for each rank and prints a summary table.
    """
    print("\n" + "="*80)
    print("--- [Rank 0] Analyzing Peak Memory Usage from Logs ---")
    print(f"Log Directory: {log_dir}")
    print("="*80)

    if not log_dir or not os.path.isdir(log_dir):
        print("Error: Log directory not found. Cannot analyze memory usage.")
        return

    file_pattern = os.path.join(log_dir, 'memory_rank_*.csv')
    all_files = sorted(glob.glob(file_pattern), key=lambda p: int(re.search(r'rank_(\d+).csv', p).group(1)))

    if not all_files:
        print("Error: No memory log files found to analyze.")
        return

    results = []
    for file_path in all_files:
        try:
            rank_id = int(re.search(r'rank_(\d+).csv', file_path).group(1))
            df = pd.read_csv(file_path)
            if not df.empty:
                max_reserved = df['reserved_gb'].max()
                max_allocated = df['allocated_gb'].max()
                results.append({'rank': rank_id, 'max_reserved': max_reserved, 'max_allocated': max_allocated})
        except Exception as e:
            print(f"Warning: Could not process file {file_path}. Error: {e}")

    if results:
        print(f"{'Rank':<6} | {'Max Reserved (GB)':<20} | {'Max Allocated (GB)':<20}")
        print("-" * 55)
        for res in results:
            print(f"{res['rank']:<6} | {res['max_reserved']:<20.4f} | {res['max_allocated']:<20.4f}")
        
        overall_max_reserved = max(res['max_reserved'] for res in results)
        overall_max_allocated = max(res['max_allocated'] for res in results)
        print("-" * 55)
        print(f"Overall Peak Reserved:  {overall_max_reserved:.4f} GB")
        print(f"Overall Peak Allocated: {overall_max_allocated:.4f} GB")
    else:
        print("No valid data found in log files.")
    print("="*80 + "\n")