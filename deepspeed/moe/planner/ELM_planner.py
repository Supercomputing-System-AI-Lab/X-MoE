import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Literal, Dict
import copy
import argparse
import sys
import json
import os

# =========================================================
# PART 1: The Provided Memory Predictor
# =========================================================
from .memory_calculator import MemoryPredictor 

# =========================================================
# PART 2: Profile Loader & Logic
# =========================================================

class ProfileLoader:
    def __init__(self, profile_dir: str, model_config: Dict, total_gpus: int, num_nodes: int):
        self.profile_dir = profile_dir
        self.cfg = model_config
        self.total_gpus = total_gpus
        self.num_nodes = num_nodes
        self._cache = {}

    def _get_filename(self, mbs: int) -> str:
        return (
            f"N{self.num_nodes}_n{self.total_gpus}_"
            f"d{self.cfg['d_model']}_e{self.cfg['num_experts']}_"
            f"f{self.cfg['expert_dim']}_k{self.cfg['topk']}_"
            f"s{self.cfg['seqlen']}_b{mbs}_profile.json"
        )

    def get_layer_times(self, mbs: int) -> Tuple[float, float]:
        fwd_gemm, fwd_comm = self._load_raw_values(mbs)
        if fwd_gemm is None:
            base_gemm, base_comm = self._load_raw_values(1)
            if base_gemm is None:
                raise FileNotFoundError(
                    f"Could not find profile for MBS={mbs} OR MBS=1 in {self.profile_dir}. "
                    f"Expected filename like: {self._get_filename(1)}"
                )
            fwd_gemm = base_gemm * mbs
            fwd_comm = base_comm * mbs

        t_std = (3 * fwd_gemm) + (2 * fwd_comm)
        t_ckpt = (4 * fwd_gemm) + (3 * fwd_comm)
        return t_std, t_ckpt

    def _load_raw_values(self, mbs: int) -> Tuple[Optional[float], Optional[float]]:
        if mbs in self._cache: return self._cache[mbs]
        filename = self._get_filename(mbs)
        filepath = os.path.join(self.profile_dir, filename)
        if not os.path.exists(filepath): return None, None
            
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            gemm = data.get("fwd_gemm_time_per_layer", 0.0)
            comm = data.get("fwd_comm_time_per_layer", 0.0)
            self._cache[mbs] = (gemm, comm)
            return gemm, comm
        except Exception as e:
            print(f"Warning: Error reading {filepath}: {e}")
            return None, None

# =========================================================
# PART 3: The Planner Infrastructure
# =========================================================

@dataclass
class PlannerConfig:
    model_config: Dict 
    target_gbs: int   
    memory_limit_gb: float
    profile_loader: ProfileLoader
    
    # [SC-Level Analytical Model Parameters]
    optimizer_step_ratio: float = 0.10      # Flat TFLOP penalty for Adam update (once per GBS)
    mb_framework_overhead_ms: float = 55.0  # Constant overhead per microbatch (Kernel launch, MoE All2All Latency, PP Send/Recv)

@dataclass
class StagePlan:
    stage_id: int
    num_layers: int
    num_checkpoints: int
    time_ms: float
    memory_used_gb: float

@dataclass
class PartitionResult:
    micro_batch_size: int
    effective_global_batch_size: int
    num_batches: int
    bottleneck_time_ms: float
    total_step_time_ms: float
    throughput_tokens_per_sec: float
    bubble_overhead: float
    stage_plans: List[StagePlan]

# =========================================================
# PART 4: The Adapter (Modified for Loader)
# =========================================================

class PhysicalMemoryAdapter:
    def __init__(self, config: PlannerConfig):
        self.cfg = config
        self.predictor = MemoryPredictor(config.model_config)
        self.loader = config.profile_loader
        
    def get_stage_performance(self, stage_idx: int, num_layers: int, u_batch_size: int, force_strategy=None) -> Tuple[float, int, float]:
        if num_layers == 0: return 0.0, 0, 0.0
        t_std, t_ckpt = self.loader.get_layer_times(u_batch_size)
        
        if force_strategy == "all": range_to_search =[num_layers]
        elif force_strategy == "none": range_to_search = [0]
        else: range_to_search = range(0, num_layers + 1)
            
        best_c, valid_mem = -1, float('inf')
        for c in range_to_search:
            mem_gb = self.predictor.query_stage_memory(stage_idx, num_layers, c, u_batch_size)
            if mem_gb <= self.cfg.memory_limit_gb:
                best_c, valid_mem = c, mem_gb
                break 
        
        if best_c == -1: return float('inf'), 0, float('inf')
        time_cost = (num_layers - best_c) * t_std + best_c * t_ckpt
        return time_cost, best_c, valid_mem

    def query_specific_config(self, stage_idx: int, num_layers: int, num_ckpt: int, u_batch_size: int):
        t_std, t_ckpt = self.loader.get_layer_times(u_batch_size)
        mem_gb = self.predictor.query_stage_memory(stage_idx, num_layers, num_ckpt, u_batch_size)
        time_cost = (num_layers - num_ckpt) * t_std + num_ckpt * t_ckpt
        return time_cost, mem_gb

# =========================================================
# PART 5: The Optimizer 
# =========================================================

class DPPlanner:
    def __init__(self, config: PlannerConfig):
        self.cfg = config
        self.adapter = PhysicalMemoryAdapter(config)
        
    def solve_for_micro_batch(self, u_batch_size: int) -> Tuple[float, List[StagePlan]]:
        N, S = self.cfg.model_config['num_layers'], self.cfg.model_config['pp_stages']
        dp = [[float('inf')] * (N + 1) for _ in range(S + 1)]
        parent = [[-1] * (N + 1) for _ in range(S + 1)]
        dp[0][0] = 0
        
        for k in range(1, S + 1):
            stage_idx = k - 1
            for i in range(1, N + 1):
                for j in range(i + 1):
                    if dp[k-1][j] == float('inf'): continue
                    cost, _, _ = self.adapter.get_stage_performance(stage_idx, i - j, u_batch_size)
                    if cost == float('inf'): continue
                    bottleneck = max(dp[k-1][j], cost)
                    if bottleneck < dp[k][i]:
                        dp[k][i] = bottleneck
                        parent[k][i] = j
                        
        if dp[S][N] == float('inf'): return float('inf'),[]
        plans =[]
        curr_n = N
        for k in range(S, 0, -1):
            split = parent[k][curr_n]
            l = curr_n - split
            t, c, m = self.adapter.get_stage_performance(k-1, l, u_batch_size)
            plans.append(StagePlan(k-1, l, c, t, m))
            curr_n = split
        plans.reverse()
        return dp[S][N], plans
    
    
    def solve_memory_balanced(self, u_batch_size: int, full_ac: bool = True):
        """DP solver that minimizes max-stage MEMORY with full AC on every stage.
    
        Same recurrence as solve_for_micro_batch, but:
        - Objective: minimize max(stage_memory) instead of max(stage_time)
        - Constraint: force full checkpointing (num_ckpt = num_layers)
        - All stages must fit within memory_limit_gb
    
        Returns (max_memory_gb, list[StagePlan]) or (inf, []) if infeasible.
        """
        N = self.cfg.model_config['num_layers']
        S = self.cfg.model_config['pp_stages']
    
        # dp[k][i] = min possible max-stage-memory when placing first i layers across k stages
        dp = [[float('inf')] * (N + 1) for _ in range(S + 1)]
        parent = [[-1] * (N + 1) for _ in range(S + 1)]
        dp[0][0] = 0.0
    
        for k in range(1, S + 1):
            stage_idx = k - 1
            for i in range(1, N + 1):
                for j in range(i):  # j = last layer index of previous stage
                    if dp[k - 1][j] == float('inf'):
                        continue
    
                    num_layers_here = i - j
                    # Choose AC strategy based on the flag
                    num_ckpt = num_layers_here if full_ac else 0
    
                    mem_gb = self.adapter.predictor.query_stage_memory(
                        stage_idx, num_layers_here, num_ckpt, u_batch_size
                    )
    
                    if mem_gb > self.cfg.memory_limit_gb:
                        continue  # infeasible
    
                    bottleneck_mem = max(dp[k - 1][j], mem_gb)
    
                    if bottleneck_mem < dp[k][i]:
                        dp[k][i] = bottleneck_mem
                        parent[k][i] = j
    
        if dp[S][N] == float('inf'):
            return float('inf'), []
    
        # Backtrack to recover the partition
        plans = []
        curr_n = N
        for k in range(S, 0, -1):
            split = parent[k][curr_n]
            num_layers_here = curr_n - split
            num_ckpt = num_layers_here if full_ac else 0
    
            # Query time and memory for this stage
            time_ms, mem_gb = self.adapter.query_specific_config(
                k - 1, num_layers_here, num_ckpt, u_batch_size
            )
            plans.append(StagePlan(k - 1, num_layers_here, num_ckpt, time_ms, mem_gb))
            curr_n = split
    
        plans.reverse()
        return dp[S][N], plans

class PipelineOptimizer:
    def __init__(self, config: PlannerConfig):
        self.cfg = config
        self.planner = DPPlanner(config)
        self.adapter = self.planner.adapter 
        
    def find_optimal_config(self, candidate_micro_batches: List[int]):
        best_result = None
        max_throughput = 0.0
        
        for ub in candidate_micro_batches:
            if ub <= 0: continue
            
            dp_size = self.cfg.model_config['dp']
            batch_unit = ub * dp_size
            
            if self.cfg.target_gbs % batch_unit != 0:
                print(f"Skipping MBS={ub}: Target GBS ({self.cfg.target_gbs}) is not cleanly divisible by batch_unit ({batch_unit}).")
                continue
                
            num_batches = self.cfg.target_gbs // batch_unit
            effective_gbs = self.cfg.target_gbs
            
            try:
                raw_bottleneck, plans = self.planner.solve_for_micro_batch(ub)
            except FileNotFoundError as e:
                print(f"Skipping MBS={ub}: {e}")
                continue

            if raw_bottleneck == float('inf'): 
                continue 
            
            # --- SC Rigorous Analytical Formulation ---
            # 1. Add framework/latency overhead per micro-batch
            bottleneck_ms = raw_bottleneck + self.cfg.mb_framework_overhead_ms
            
            # [NEW] Decomposed execution time: warm-up + steady + cool-down
            t_sum = sum(p.time_ms + self.cfg.mb_framework_overhead_ms for p in plans)
            t_max = bottleneck_ms
            fwd_bwd_time_ms = t_sum + (num_batches - 1) * t_max
            
            # 2. Add fixed optimizer step
            optimizer_time_ms = fwd_bwd_time_ms * self.cfg.optimizer_step_ratio
            total_step_time_ms = fwd_bwd_time_ms + optimizer_time_ms
            
            throughput = (effective_gbs * self.cfg.model_config['seqlen']) / (total_step_time_ms / 1000.0)
            
            ideal_time = num_batches * t_max
            bubble = (fwd_bwd_time_ms - ideal_time) / fwd_bwd_time_ms if fwd_bwd_time_ms > 0 else 0.0
            
            if throughput > max_throughput:
                max_throughput = throughput
                best_result = PartitionResult(ub, effective_gbs, num_batches, bottleneck_ms, total_step_time_ms, throughput, bubble, plans)
        
        return best_result
    
    def find_max_mbs_membal_config(self, candidate_micro_batches: List[int], full_ac: bool):
        best_result = None
        
        # Sort ascending so the last successful one saved is the absolute maximum MBS
        for ub in sorted(candidate_micro_batches):
            if ub <= 0: continue
            
            dp_size = self.cfg.model_config['dp']
            batch_unit = ub * dp_size
            
            if self.cfg.target_gbs % batch_unit != 0:
                continue
                
            num_batches = self.cfg.target_gbs // batch_unit
            effective_gbs = self.cfg.target_gbs
            
            # Use the Memory-Balanced DP Solver
            max_mem, plans = self.planner.solve_memory_balanced(ub, full_ac=full_ac)
            
            if max_mem == float('inf') or not plans: 
                continue # OOM or infeasible
            
            # Calculate metrics for the report (throughput is NOT used for selection here)
            pp_stages = self.cfg.model_config['pp_stages']
            max_stage_time = max(p.time_ms for p in plans)
            bottleneck_ms = max_stage_time + self.cfg.mb_framework_overhead_ms
            fwd_bwd_time_ms = (num_batches + pp_stages - 1) * bottleneck_ms
            optimizer_time_ms = fwd_bwd_time_ms * self.cfg.optimizer_step_ratio
            total_step_time_ms = fwd_bwd_time_ms + optimizer_time_ms
            throughput = (effective_gbs * self.cfg.model_config['seqlen']) / (total_step_time_ms / 1000.0)
            ideal_time = num_batches * bottleneck_ms
            bubble = (fwd_bwd_time_ms - ideal_time) / fwd_bwd_time_ms if fwd_bwd_time_ms > 0 else 0.0
            
            # ALWAYS overwrite if it fits, because a higher MBS means higher memory utilization!
            best_result = PartitionResult(ub, effective_gbs, num_batches, bottleneck_ms, total_step_time_ms, throughput, bubble, plans)
            
        return best_result
    
    def evaluate_baseline(self, ub: int, strategy: Literal["all", "none"]):
        N = self.cfg.model_config['num_layers']
        S = self.cfg.model_config['pp_stages']
        dp_size = self.cfg.model_config['dp']
        
        batch_unit = ub * dp_size
        if self.cfg.target_gbs % batch_unit != 0: return None
        num_batches = self.cfg.target_gbs // batch_unit
        effective_gbs = self.cfg.target_gbs
        
        base = N // S
        rem = N % S
        plans =[]
        max_time = 0
        
        try:
            for s in range(S):
                count = base + (1 if s < rem else 0)
                t, c, m = self.adapter.get_stage_performance(s, count, ub, force_strategy=strategy)
                if t == float('inf'): return None
                max_time = max(max_time, t)
                plans.append(StagePlan(s, count, c, t, m))
        except FileNotFoundError:
            return None
        
        if max_time == 0.0: return None
        
        # Apply Overheads and Decomposed Runtime Model
        bottleneck_ms = max_time + self.cfg.mb_framework_overhead_ms
        t_sum = sum(p.time_ms + self.cfg.mb_framework_overhead_ms for p in plans)
        fwd_bwd_time_ms = t_sum + (num_batches - 1) * bottleneck_ms
        
        optimizer_time_ms = fwd_bwd_time_ms * self.cfg.optimizer_step_ratio
        total_step_time_ms = fwd_bwd_time_ms + optimizer_time_ms
        
        throughput = (effective_gbs * self.cfg.model_config['seqlen']) / (total_step_time_ms / 1000.0)
        return PartitionResult(ub, effective_gbs, num_batches, bottleneck_ms, total_step_time_ms, throughput, 0.0, plans)
    
    def run_ablation(self, mbs: int):
        """Run the 4-config ablation study and print comparison.
    
        Configs:
        1. Even PP + Full AC
        2. Even PP + No AC
        3. Memory-Balanced PP + Full AC  (NEW)
        4. Minimax (joint optimization)
    
        Args:
            mbs: micro-batch size to use for all configs
        """
        N = self.cfg.model_config['num_layers']
        S = self.cfg.model_config['pp_stages']
        dp_size = self.cfg.model_config['dp']
    
        batch_unit = mbs * dp_size
        if self.cfg.target_gbs % batch_unit != 0:
            print(f"Error: GBS ({self.cfg.target_gbs}) not divisible by mbs*dp ({batch_unit})")
            return
        num_batches = self.cfg.target_gbs // batch_unit
        seqlen = self.cfg.model_config['seqlen']
    
        results = {}
    
        # --- Config 1: Even PP + Full AC ---
        res1 = self.evaluate_fixed_strategy(mbs, "all")
        if res1:
            results["Even+FullAC"] = res1
        else:
            print("Config 1 (Even+FullAC): OOM or infeasible")
            results["Even+FullAC"] = None
    
        # --- Config 2: Even PP + No AC ---
        res2 = self.evaluate_fixed_strategy(mbs, "none")
        if res2:
            results["Even+NoAC"] = res2
        else:
            print("Config 2 (Even+NoAC): OOM or infeasible")
            results["Even+NoAC"] = None
    
        # --- Config 3: Memory-Balanced PP + Full AC ---
        max_mem, mem_plans = self.planner.solve_memory_balanced(mbs, full_ac=True)
        if max_mem < float('inf') and mem_plans:
            # Compute throughput for this partition
            max_stage_time = max(p.time_ms for p in mem_plans)
            total_time = (num_batches + S - 1) * max_stage_time
            throughput = (self.cfg.target_gbs * seqlen) / (total_time / 1000.0)
            ideal_time = num_batches * max_stage_time
            bubble = (total_time - ideal_time) / total_time if total_time > 0 else 0.0
    
            res3 = PartitionResult(
                mbs, self.cfg.target_gbs, num_batches,
                max_stage_time, total_time, throughput, bubble, mem_plans
            )
            results["MemBalanced+FullAC"] = res3
        else:
            print("Config 3 (MemBalanced+FullAC): OOM or infeasible")
            results["MemBalanced+FullAC"] = None
    
        # --- Config 4: Memory-Balanced PP + No AC ---
        max_mem_no, mem_plans_no = self.planner.solve_memory_balanced(mbs, full_ac=False)
        if max_mem_no < float('inf') and mem_plans_no:
            max_stage_time = max(p.time_ms for p in mem_plans_no)
            total_time = (num_batches + S - 1) * max_stage_time
            throughput = (self.cfg.target_gbs * seqlen) / (total_time / 1000.0)
            ideal_time = num_batches * max_stage_time
            bubble = (total_time - ideal_time) / total_time if total_time > 0 else 0.0
            res4 = PartitionResult(mbs, self.cfg.target_gbs, num_batches, max_stage_time, total_time, throughput, bubble, mem_plans_no)
            results["MemBalanced+NoAC"] = res4
        else:
            print("Config 4 (MemBalanced+NoAC): OOM or infeasible")
            results["MemBalanced+NoAC"] = None
    
        # --- Config 5: Minimax (joint optimization) ---
        minimax_bottleneck, minimax_plans = self.planner.solve_for_micro_batch(mbs)
        if minimax_bottleneck < float('inf') and minimax_plans:
            max_stage_time = max(p.time_ms for p in minimax_plans)
            t_sum = sum(p.time_ms for p in minimax_plans)
            total_time = t_sum + (num_batches - 1) * max_stage_time
            
            throughput = (self.cfg.target_gbs * seqlen) / (total_time / 1000.0)
            ideal_time = num_batches * max_stage_time
            bubble = (total_time - ideal_time) / total_time if total_time > 0 else 0.0
    
            res5 = PartitionResult(
                mbs, self.cfg.target_gbs, num_batches,
                max_stage_time, total_time, throughput, bubble, minimax_plans
            )
            results["Minimax"] = res5
        else:
            print("Config 5 (Minimax): infeasible")
            results["Minimax"] = None
    
        # --- Print comparison table ---
        print("\n" + "=" * 100)
        print(f"ABLATION STUDY: {N} layers, {S} stages, MBS={mbs}, GBS={self.cfg.target_gbs}")
        print("=" * 100)
        print(f"{'Config':<25} | {'Partition':<30} | {'Ckpt':<25} | {'Bottleneck(ms)':>14} | {'Throughput':>14} | {'Status'}")
        print("-" * 100)
    
        for name, res in results.items():
            if res is None:
                print(f"{name:<25} | {'---':<30} | {'---':<25} | {'OOM':>14} | {'---':>14} | FAILED")
            else:
                sorted_plans = sorted(res.stage_plans, key=lambda x: x.stage_id)
                part_str = " ".join(str(p.num_layers) for p in sorted_plans)
                ckpt_str = " ".join(str(p.num_checkpoints) for p in sorted_plans)
    
                # Truncate for display
                if len(part_str) > 28:
                    part_str = part_str[:25] + "..."
                if len(ckpt_str) > 23:
                    ckpt_str = ckpt_str[:20] + "..."
    
                print(f"{name:<25} | {part_str:<30} | {ckpt_str:<25} | "
                    f"{res.bottleneck_time_ms:>14.2f} | {res.throughput_tokens_per_sec:>14.0f} | OK")
    
        print("-" * 100)
    
        # Print detailed reports
        for name, res in results.items():
            if res is not None:
                self.print_detailed_report(res, f"ABLATION: {name}")
    
        # Print bash exports for the two uneven configs (3 and 4)
        print("\n" + "=" * 80)
        print("BASH EXPORTS FOR LAUNCH SCRIPTS")
        print("=" * 80)
    
        if results.get("MemBalanced+FullAC"):
            res3 = results["MemBalanced+FullAC"]
            sorted_plans = sorted(res3.stage_plans, key=lambda x: x.stage_id)
            layers_str = " ".join(str(p.num_layers) for p in sorted_plans)
            ckpts_str = " ".join(str(p.num_checkpoints) for p in sorted_plans)
            print(f"\n# Config 3: Memory-Balanced + Full AC")
            print(f"MEMBAL_FULLAC_MBS={res3.micro_batch_size}")
            print(f"MEMBAL_FULLAC_NUM_BATCHES={res3.num_batches}")
            print(f'MEMBAL_FULLAC_PARTITION="{layers_str}"')
            print(f'MEMBAL_FULLAC_CKPT="{ckpts_str}"')
            
        if results.get("MemBalanced+NoAC"):
            res = results["MemBalanced+NoAC"]
            print(f"\n# Config 4: Memory-Balanced + No AC")
            print(f"MEMBAL_NOAC_MBS={res.micro_batch_size}")
            print(f"MEMBAL_NOAC_NUM_BATCHES={res.num_batches}")
            print(f'MEMBAL_NOAC_PARTITION="{" ".join(str(p.num_layers) for p in sorted(res.stage_plans, key=lambda x: x.stage_id))}"')
            print(f'MEMBAL_NOAC_CKPT="{" ".join(str(p.num_checkpoints) for p in sorted(res.stage_plans, key=lambda x: x.stage_id))}"')
    
        if results.get("Minimax"):
            res5 = results["Minimax"]
            sorted_plans = sorted(res5.stage_plans, key=lambda x: x.stage_id)
            layers_str = " ".join(str(p.num_layers) for p in sorted_plans)
            ckpts_str = " ".join(str(p.num_checkpoints) for p in sorted_plans)
            print(f"\n# Config 5: Minimax")
            print(f"MINIMAX_MBS={res5.micro_batch_size}")
            print(f"MINIMAX_NUM_BATCHES={res5.num_batches}")
            print(f'MINIMAX_PARTITION="{layers_str}"')
            print(f'MINIMAX_CKPT="{ckpts_str}"')
    
        return results

    def print_detailed_report(self, result: PartitionResult, title: str):
        model_size_b = self.planner.adapter.predictor.calculate_model_size_in_billions()
        print("\n" + "="*80)
        print(f"|{title}")
        print("="*80)
        print(f"|Model Size:  {model_size_b:.2f} Billion Parameters")
        print(f"|Micro-Batch: {result.micro_batch_size}")
        print(f"|Num Batches: {result.num_batches} (Calculated from GBS)")
        print(f"|Global Batch:{result.effective_global_batch_size} (Fixed Target)")
        print(f"|Throughput:  {result.throughput_tokens_per_sec:.2f} tokens/s")
        print(f"|Step Time:   {result.total_step_time_ms:.2f} ms")
        print(f"|Overheads:   {self.cfg.mb_framework_overhead_ms}ms Kernel/Comm Latency per mb | {self.cfg.optimizer_step_ratio*100:.1f}% Opt. Update")
        print(f"|Bubble:      {result.bubble_overhead*100:.2f}%")
        print("-" * 80)
        print(f"|{'Stage':<6} | {'Layers':<7} | {'Ckpt':<5} | {'Mem (GB)':<10} | {'Raw Time(ms)':<10} | {'Status'}")
        print("-" * 80)
        for p in result.stage_plans:
            status = "OK" if p.memory_used_gb <= self.cfg.memory_limit_gb else "OOM (!)"
            print(f"|{p.stage_id:<6} | {p.num_layers:<7} | {p.num_checkpoints:<5} | {p.memory_used_gb:<10.2f} | {p.time_ms:<10.2f} | {status}")
        print("-" * 80)

    def print_bash_export(self, result: PartitionResult):
        sorted_plans = sorted(result.stage_plans, key=lambda x: x.stage_id)
        layers =[str(p.num_layers) for p in sorted_plans]
        ckpts =[str(p.num_checkpoints) for p in sorted_plans]
        print("\n# --- planner.py: EXPORT FOR BASH ---")
        print(f"OPTIMAL_MBS={result.micro_batch_size}")
        print(f"OPTIMAL_NUM_BATCHES={result.num_batches}") 
        print(f"OPTIMAL_PARTITION=\"{' '.join(layers)}\"")
        print(f"OPTIMAL_CKPT=\"{' '.join(ckpts)}\"")
    
    def compare_baselines(self, best_result: PartitionResult):
        print("\n\n|>>> COMPARING WITH BASELINES (Same MBS)")
        for name, strat in[("No Checkpointing (Speed)", "none"), ("Full Checkpointing (Memory)", "all")]:
            base_res = self.evaluate_baseline(best_result.micro_batch_size, strat)
            if base_res:
                diff = base_res.throughput_tokens_per_sec - best_result.throughput_tokens_per_sec
                pct = (diff / best_result.throughput_tokens_per_sec) * 100
                title = f"BASELINE: {name} | Diff: {pct:.2f}% Tput"
                self.print_detailed_report(base_res, title)
            else:
                print("\n" + "="*80)
                print(f"|BASELINE: {name}")
                print("="*80)
                print("|FAILED (OOM - Out of Memory)")
                
    def evaluate_throughput_for_mbs(self, ub: int):
        N = self.cfg.model_config['num_layers']
        S = self.cfg.model_config['pp_stages']
        dp_size = self.cfg.model_config['dp']
        
        batch_unit = ub * dp_size
        if self.cfg.target_gbs % batch_unit != 0:
            print(f"Skipping MBS={ub}: Target GBS not divisible by batch unit.")
            return None
            
        num_batches = self.cfg.target_gbs // batch_unit
        effective_gbs = self.cfg.target_gbs

        base_layers = N // S
        remainder = N % S
        plans =[]
        max_stage_time = 0.0

        try:
            for stage_idx in range(S):
                num_layers_in_stage = base_layers + (1 if stage_idx < remainder else 0)
                if num_layers_in_stage == 0:
                    plans.append(StagePlan(stage_idx, 0, 0, 0.0, 0.0))
                    continue

                time_ms, ckpts, mem_gb = self.adapter.get_stage_performance(stage_idx, num_layers_in_stage, ub)
                if time_ms == float('inf'):
                    print(f"Warning: MBS={ub} is infeasible (OOM) for a balanced partition.")
                    return None

                max_stage_time = max(max_stage_time, time_ms)
                plans.append(StagePlan(stage_idx, num_layers_in_stage, ckpts, time_ms, mem_gb))

        except FileNotFoundError as e:
            print(f"Skipping MBS={ub}: {e}")
            return None

        if max_stage_time == 0.0: return None
        
        bottleneck_ms = max_stage_time + self.cfg.mb_framework_overhead_ms
        t_sum = sum(p.time_ms + self.cfg.mb_framework_overhead_ms for p in plans)
        fwd_bwd_time_ms = t_sum + (num_batches - 1) * bottleneck_ms
        
        optimizer_time_ms = fwd_bwd_time_ms * self.cfg.optimizer_step_ratio
        total_step_time_ms = fwd_bwd_time_ms + optimizer_time_ms

        throughput = (effective_gbs * self.cfg.model_config['seqlen']) / (total_step_time_ms / 1000.0)
        ideal_time = num_batches * bottleneck_ms
        bubble = (fwd_bwd_time_ms - ideal_time) / fwd_bwd_time_ms if fwd_bwd_time_ms > 0 else 0.0

        return PartitionResult(
            micro_batch_size=ub, effective_global_batch_size=effective_gbs, num_batches=num_batches,
            bottleneck_time_ms=bottleneck_ms, total_step_time_ms=total_step_time_ms, throughput_tokens_per_sec=throughput,
            bubble_overhead=bubble, stage_plans=plans
        )
        
    def evaluate_fixed_strategy(self, ub: int, strategy: Literal["all", "none"]):
        N = self.cfg.model_config['num_layers']
        S = self.cfg.model_config['pp_stages']
        dp_size = self.cfg.model_config['dp']
        
        batch_unit = ub * dp_size
        if self.cfg.target_gbs % batch_unit != 0:
            print(f"Skipping MBS={ub}: Target GBS not divisible by batch unit.")
            return None
            
        num_batches = self.cfg.target_gbs // batch_unit
        effective_gbs = self.cfg.target_gbs
        
        base_layers = N // S
        remainder = N % S
        plans =[]
        max_stage_time = 0.0
        
        try:
            for stage_idx in range(S):
                num_layers_in_stage = base_layers + (1 if stage_idx < remainder else 0)
                if num_layers_in_stage == 0:
                    plans.append(StagePlan(stage_idx, 0, 0, 0.0, 0.0))
                    continue

                num_ckpt = num_layers_in_stage if strategy == "all" else 0
                time_ms, mem_gb = self.adapter.query_specific_config(stage_idx, num_layers_in_stage, num_ckpt, ub)

                if mem_gb > self.cfg.memory_limit_gb:
                    print(f"OOM Detected: For MBS={ub}, strategy='{strategy}', stage {stage_idx} requires {mem_gb:.2f} GB "
                          f"(Limit: {self.cfg.memory_limit_gb:.2f} GB).")
                    return None
                
                max_stage_time = max(max_stage_time, time_ms)
                plans.append(StagePlan(stage_idx, num_layers_in_stage, num_ckpt, time_ms, mem_gb))

        except FileNotFoundError as e:
            print(f"Skipping MBS={ub}: {e}")
            return None
        
        if max_stage_time == 0.0: return None
        
        bottleneck_ms = max_stage_time + self.cfg.mb_framework_overhead_ms
        t_sum = sum(p.time_ms + self.cfg.mb_framework_overhead_ms for p in plans)
        fwd_bwd_time_ms = t_sum + (num_batches - 1) * bottleneck_ms
        
        optimizer_time_ms = fwd_bwd_time_ms * self.cfg.optimizer_step_ratio
        total_step_time_ms = fwd_bwd_time_ms + optimizer_time_ms

        throughput = (effective_gbs * self.cfg.model_config['seqlen']) / (total_step_time_ms / 1000.0)
        ideal_time = num_batches * bottleneck_ms
        bubble = (fwd_bwd_time_ms - ideal_time) / fwd_bwd_time_ms if fwd_bwd_time_ms > 0 else 0.0
            
        return PartitionResult(ub, effective_gbs, num_batches, bottleneck_ms, total_step_time_ms, throughput, bubble, plans)

# =========================================================
# Execution
# =========================================================

# def get_args():
#     parser = argparse.ArgumentParser(description="LLM MoE Training Configuration")
#     model_group = parser.add_argument_group('Model Architecture')
#     model_group.add_argument('--d_model', type=int, required=True, help='Hidden dimension size')
#     model_group.add_argument('--seqlen', type=int, required=True, help='Sequence length')
#     model_group.add_argument('--num_layers', type=int, required=True, help='Total number of layers')
#     model_group.add_argument('--vocab_size', type=int, default=50257)
#     model_group.add_argument('--attention_heads', type=int, default=16)
#     model_group.add_argument('--tied_embedding', action='store_true')

#     parallel_group = parser.add_argument_group('Parallelism')
#     parallel_group.add_argument('--pp_stages', type=int, required=True)
#     parallel_group.add_argument('--dp', type=int, required=True)
#     parallel_group.add_argument('--ep', type=int, required=True)
#     parallel_group.add_argument('--num_nodes', type=int, default=1, help='Number of nodes')
#     parallel_group.add_argument('--gpus_per_node', type=int, default=8, help='GPUs per node')

#     moe_group = parser.add_argument_group('Mixture of Experts')
#     moe_group.add_argument('--num_experts', type=int, required=True)
#     moe_group.add_argument('--expert_dim', type=int, required=True)
#     moe_group.add_argument('--topk', type=int, required=True)
#     moe_group.add_argument('--moe_train_capacity_factor', type=float, default=1.25)
#     moe_group.add_argument('--use_groupgemm', type=int, default=0)

#     train_group = parser.add_argument_group('Training')
#     train_group.add_argument('--activation_checkpointing', type=str, default='selective',  choices=['none', 'full', 'selective'])
#     train_group.add_argument('--mbs', type=int, required=True)
#     train_group.add_argument('--profile_dir', type=str, default="./planner_profiling_cache")

#     planner_group = parser.add_argument_group('Planner Limits')
#     planner_group.add_argument('--fixed_num_batches', type=int, required=True)
#     planner_group.add_argument('--memory_limit_gb', type=float, default=60)
    
#     planner_group.add_argument('--mb_framework_overhead_ms', type=float, default=55.0, 
#                                help='Constant overhead per MB modeling CPU kernel launch bound and MoE network latency (alpha).')
#     planner_group.add_argument('--optimizer_step_ratio', type=float, default=0.10, 
#                                help='Flat overhead applied once per step to model the Adam update.')
    
#     exec_group = parser.add_argument_group('Execution Mode')
#     exec_group.add_argument('--eval-mbs-list', type=int, nargs='+', default=None)
#     exec_group.add_argument('--eval-strategy', type=str, choices=['none', 'all'], default=None)
#     # exec_group.add_argument('--mode', type=str, default='optimize', choices=['optimize', 'eval', 'ablation'])
#     exec_group.add_argument('--mode', type=str, default='optimize', choices=['optimize', 'eval', 'ablation', 'optimize-membal'])
#     exec_group.add_argument('--ablation-mbs', type=int, default=None, help='MBS to use for ablation study')
    
    
#     return parser.parse_args()


# if __name__ == "__main__":
#     args = get_args()
#     total_gpus = args.num_nodes * args.gpus_per_node
    
#     target_gbs = args.mbs * args.dp * args.fixed_num_batches 

#     model_config = {
#         'd_model': args.d_model, 
#         'seqlen': args.seqlen, 
#         'pp_stages': args.pp_stages, 
#         'num_layers': args.num_layers,
#         'dp': args.dp,
#         'ep': args.ep,
#         'num_experts': args.num_experts, 
#         'expert_dim': args.expert_dim, 
#         'topk': args.topk, 
#         'use_groupgemm': args.use_groupgemm, 
#         'vocab_size': args.vocab_size,
#         'attention_heads': args.attention_heads,  
#         'tied_embedding': args.tied_embedding, 
#         'moe-train-capacity-factor': args.moe_train_capacity_factor,
#         'activation_checkpointing': args.activation_checkpointing, 
#         'mbs': args.mbs, 
#         'gbs': target_gbs, 
#     }
    
#     profile_loader = ProfileLoader(
#         profile_dir=args.profile_dir,
#         model_config=model_config,
#         total_gpus=total_gpus,
#         num_nodes=args.num_nodes
#     )
    
#     planner_cfg = PlannerConfig(
#         model_config=model_config,
#         target_gbs=target_gbs, 
#         memory_limit_gb=args.memory_limit_gb,
#         profile_loader=profile_loader,
#         mb_framework_overhead_ms=args.mb_framework_overhead_ms,
#         optimizer_step_ratio=args.optimizer_step_ratio
#     )
    
#     optimizer = PipelineOptimizer(planner_cfg)

#     # =========================================================
#     # MAIN EXECUTION LOGIC
#     # =========================================================
#     if args.mode == 'ablation':
#         ablation_mbs = args.ablation_mbs if args.ablation_mbs else args.mbs
#         results = optimizer.run_ablation(ablation_mbs)
        
#     elif args.mode == 'optimize-membal':
#         print("="*120)
#         print(f"|>>> MAXIMIZING MEMORY: MEMORY-BALANCED (Fixing GBS={target_gbs})")
        
#         # Determine AC strategy from arguments
#         full_ac = (args.activation_checkpointing != 'none' and args.activation_checkpointing != 'false')
#         ac_label = "Full AC" if full_ac else "No AC"
#         prefix = "MEMBAL_FULLAC" if full_ac else "MEMBAL_NOAC"
        
#         # Search for the absolute largest MBS that fits
#         best = optimizer.find_max_mbs_membal_config([1, 2, 3, 4, 5, 6, 7, 8], full_ac=full_ac)
        
#         if best:
#             optimizer.print_detailed_report(best, f"MAX MEMORY PLAN ({ac_label})")
            
#             # Print Bash Exports exactly how your script expects them
#             print(f"\n# Config: Auto MemBal Max MBS")
#             print(f"{prefix}_MBS={best.micro_batch_size}")
#             print(f"{prefix}_NUM_BATCHES={best.num_batches}")
#             print(f'{prefix}_PARTITION="{" ".join(str(p.num_layers) for p in sorted(best.stage_plans, key=lambda x: x.stage_id))}"')
#             print(f'{prefix}_CKPT="{" ".join(str(p.num_checkpoints) for p in sorted(best.stage_plans, key=lambda x: x.stage_id))}"')
#         else:
#             print(f"{prefix}_MBS=FAILED")

 
#     elif args.mode == 'eval':
#         if not args.eval_mbs_list or not args.eval_strategy:
#             print("Error: --eval-mbs-list and --eval-strategy required for eval mode.")
#             sys.exit(1)
            
#         strategy_name = "No Checkpointing" if args.eval_strategy == 'none' else "Full Checkpointing"
#         print(f"|>>> EVALUATING MBS VALUES: {args.eval_mbs_list}")
#         print(f"|>>> Using fixed strategy: {strategy_name} | Target GBS: {target_gbs}")
        
#         for mbs in sorted(args.eval_mbs_list):
#             result = optimizer.evaluate_fixed_strategy(mbs, args.eval_strategy)
#             if result:
#                 optimizer.print_detailed_report(result, f"STRATEGY: '{args.eval_strategy.upper()}' | MBS={mbs}")
#             else:
#                 print(f"\n{'='*80}")
#                 print(f"| FAILED: MBS={mbs} with strategy '{args.eval_strategy}' is INFEASIBLE")
#                 print("=" * 80)
 
#     else:  # mode == 'optimize' (default)
#         print("="*120)
#         print(f"|>>> AUTOMATIC OPTIMIZATION (Fixing GBS={target_gbs})")
#         best = optimizer.find_optimal_config([1, 2, 3, 4, 5, 6, 7, 8])
#         if best:
#             optimizer.print_detailed_report(best, "OPTIMAL PLAN FOUND")
#             optimizer.compare_baselines(best)
#             optimizer.print_bash_export(best)
#         else:
#             print("OPTIMAL_MBS=FAILED")