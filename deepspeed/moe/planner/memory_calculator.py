import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Optional, List
import matplotlib.patches as mpatches
import math

class MemoryPredictor:
    """
    A class to predict and visualize GPU memory usage for training large-scale 
    Mixture-of-Experts (MoE) transformer models.
    """
    
        
        
        

    FP8_BYTES = 1
    FP16_BYTES = 2
    BF16_BYTES = 2
    FP32_BYTES = 4
    GIGA = 1024**3

    def __init__(self, config: Dict):
        self.config = config
        self._validate_config()

    def _validate_config(self):
        required_keys = [
            'd_model', 'seqlen', 'pp_stages', 'num_layers', 'dp', 'ep', 
            'num_experts', 'expert_dim', 'mbs', 'gbs', 'topk', 'vocab_size',
            'attention_heads', 'tied_embedding'
        ]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required key in config: '{key}'")
        
        gbs = self.config['gbs']
        dp = self.config['dp']
        b = self.config['mbs']
        if gbs % (dp * b) != 0:
            raise ValueError(f"Global batch size (gbs={gbs}) must be divisible by dp*mbs ({dp*b}).")

        if 'pp_partitioning' in self.config:
            parts = self.config['pp_partitioning']
            if not isinstance(parts, list):
                 raise ValueError("pp_partitioning must be a list of integers.")
            if len(parts) != self.config['pp_stages']:
                raise ValueError(f"pp_partitioning length ({len(parts)}) must match pp_stages ({self.config['pp_stages']})")
            if sum(parts) != self.config['num_layers']:
                raise ValueError(f"Sum of pp_partitioning ({sum(parts)}) must match num_layers ({self.config['num_layers']})")
        
        # [NEW] Validate Dynamic Checkpointing
        if 'dynamic_checkpointing' in self.config:
            dyn_ckpt = self.config['dynamic_checkpointing']
            if not isinstance(dyn_ckpt, list):
                raise ValueError("dynamic_checkpointing must be a list of integers.")
            if len(dyn_ckpt) != self.config['pp_stages']:
                 raise ValueError(f"dynamic_checkpointing length ({len(dyn_ckpt)}) must match pp_stages ({self.config['pp_stages']})")
            # We cannot easily validate max layers here without duplicating partitioning logic, 
            # so w

    # --- Calculation Methods ---

    def _model_param_attention_per_gpu_stage(self, num_layers_in_stage: float) -> float:
        h = self.config['d_model']
        tp = self.config.get('tp', 1)
        # Weights are sharded by TP
        return num_layers_in_stage * 4 * h**2 * self.FP16_BYTES / tp

    def _model_grad_attention_per_gpu_stage(self, num_layers_in_stage: float) -> float:
        return self._model_param_attention_per_gpu_stage(num_layers_in_stage)

    def _model_optim_attention_per_gpu_stage(self, num_layers_in_stage: float) -> float:
        h = self.config['d_model']
        # dp is assumed to be dp_non_moe 
        dp = self.config['dp']
        tp = self.config.get('tp', 1)
        # TP shards the weights, ZeRO-1 shards the optim states over DP
        return 3 * num_layers_in_stage * 4 * h**2 * self.FP32_BYTES / (dp * tp)

    def _attention_activation_per_gpu_stage(self, num_layers_in_stage: float) -> float:
        h = self.config['d_model']
        s = self.config['seqlen']
        b = self.config['mbs']
        num_heads = self.config['attention_heads']
        tp = self.config.get('tp', 1)
        
        input_act = self.FP16_BYTES * b * s * h
        # QKV outputs and attention context are divided by TP
        qkv = self.FP16_BYTES * 3 * b * s * h / tp
        attn_out = self.FP16_BYTES * b * s * h
        out_proj = self.FP16_BYTES * b * s * h / tp 
        layernorm = self.FP16_BYTES * b * s * h 
        residual_out = 1 #self.FP16_BYTES * b * s * h 
        
        attn_activation_per_layer = input_act + qkv + attn_out + out_proj + layernorm + residual_out 
        
        return attn_activation_per_layer * num_layers_in_stage

    def _model_param_moe_per_gpu_stage(self, num_layers_in_stage: float) -> float:
        h = self.config['d_model']
        num_experts = self.config['num_experts']
        expert_dim = self.config['expert_dim']
        ep = self.config['ep']
        tp = self.config.get('tp', 1)
        
        # Router is NOT sharded by TP
        router = self.FP16_BYTES * h * num_experts * num_layers_in_stage
        # Experts ARE sharded by TP
        experts = self.FP16_BYTES * (num_experts / ep) * h * expert_dim * 2 * num_layers_in_stage / tp 
        return router + experts

    def _model_grad_moe_per_gpu_stage(self, num_layers_in_stage: float) -> float:
        return self._model_param_moe_per_gpu_stage(num_layers_in_stage)

    def _model_optim_moe_per_gpu_stage(self, num_layers_in_stage: float) -> float:
        h = self.config['d_model']
        num_experts = self.config['num_experts']
        expert_dim = self.config['expert_dim']
        ep = self.config['ep']
        tp = self.config.get('tp', 1)
        dp = self.config['dp']
        
        # Router is NOT sharded by TP (retains original logic)
        router = 3 * self.FP32_BYTES * h * num_experts * num_layers_in_stage / dp 
        # Expert states scale down cleanly by TP for the slice this GPU owns
        # / (dp / ep) is divided by dp_moe w/ zero-1 
        experts = 3 * self.FP32_BYTES * (num_experts / ep) * h * expert_dim * 2 * num_layers_in_stage / (dp / ep / tp)
        return router + experts
        

    def _moe_activation_per_gpu_stage(self, num_layers_in_stage: float) -> float:
        h = self.config['d_model']
        s = self.config['seqlen']
        b = self.config['mbs']
        dp = self.config['dp']
        ep = self.config['ep']
        num_experts = self.config['num_experts']
        expert_dim = self.config['expert_dim']
        topk = self.config['topk']
        use_groupgemm = self.config.get('use_groupgemm', False)
        tp = self.config.get('tp', 1)
        c = self.config.get ("moe-train-capacity-factor", 1.25)
        
        num_experts_per_gpu = num_experts / ep 
        
            
        # for which ever EP there are in this distribution, there will be dp_non_moe streams inserting. 
        # E.g. 1024GPU, pp8-ep8-dp16 --> 128 GPUs --> model is splitted into 2x pp8-ep8, dp_moe=2
        # But for each individual pp stage, there are only 8 streams of input (dp / dp_moe)= ep 
        dp_non_moe = ep  
        per_gpu_max_assigned_tokens = b * s * dp_non_moe * topk / num_experts * num_experts_per_gpu #* c 
        
        # print (f'{per_gpu_max_assigned_tokens=}')
        
        # drop_factor = 0.65
        # 0.50 + 512/8/2 * 0.003
        drop_factor = 0.5 + num_experts_per_gpu * 0.003
        per_gpu_max_assigned_tokens = per_gpu_max_assigned_tokens * drop_factor
        
        
        # print (f'{b=}')
        # print (f'{s=}')
        # print (f'{dp_non_moe=}')
        # print (f'{topk=}')
        # print (f'{num_experts=}')
        # print (f'{num_experts_per_gpu=}')
        # print (f'{c=}')
        # print (f'After dropping tokens {per_gpu_max_assigned_tokens=}')
        # print (f'{per_gpu_assigned_tokens_after_drop=}')
        
        # Router 
        # input = 1 #self.FP16_BYTES * b * s * h 
        gate_output = self.FP16_BYTES * b * s * num_experts  
        softmax_input = self.FP32_BYTES * b * s * num_experts 
        input_fp32 = self.FP32_BYTES * b * s * h 
        
        # Input being transformed for rank-primary input before the all-to-all
        # Since measuring memory, and router is dynamic, we will account for the peak mem per expert 
        a2a_disp_input_buf = self.FP16_BYTES * per_gpu_max_assigned_tokens * h 
        
        up_proj_out = self.FP16_BYTES *per_gpu_max_assigned_tokens * expert_dim 
        
        gelu_out = self.FP16_BYTES * per_gpu_max_assigned_tokens * expert_dim 
        
        # output of gelu goes through down_proj and is transformed again to combine_a2a_buf 
        combine_buf = self.FP16_BYTES * per_gpu_max_assigned_tokens * h 
        
        # in memory viz, output is not materialized here. 
        combine_output = 1 # self.FP16_BYTES * b * s * h 
        
        layernorm = self.FP16_BYTES * b * s * h
        
        moe_activation_per_layer = input_fp32 + gate_output + softmax_input + \
                                    a2a_disp_input_buf + up_proj_out + gelu_out + \
                                    combine_buf + combine_output + layernorm 
        
        moe_activation_per_stage = moe_activation_per_layer * num_layers_in_stage 
        
        return moe_activation_per_stage
        
    def _embedding_param(self) -> float:
        tp = self.config.get('tp', 1)
        # In Megatron, the vocab matrix is sharded across the TP group
        return self.FP16_BYTES * (self.config['vocab_size'] / tp) * self.config['d_model']

    def _embedding_grad(self) -> float:
        return self._embedding_param()

    def _embedding_optim(self) -> float:
        return self.FP32_BYTES * 3 * self.config['vocab_size'] * self.config['d_model']
        
    def _embedding_act(self) -> float:
        return self.FP16_BYTES * self.config['mbs'] * self.config['seqlen'] * self.config['d_model']

    def _output_layer_param(self) -> float:
        return 0 if self.config.get('tied_embedding') else self._embedding_param()
        
    def _output_layer_grad(self) -> float:
        return 0 if self.config.get('tied_embedding') else self._embedding_grad()
        
    def _output_layer_optim(self) -> float:
        return 0 if self.config.get('tied_embedding') else self._embedding_optim()

    def _output_layer_act(self) -> float:
        b = self.config['mbs']
        s = self.config['seqlen']
        h = self.config['d_model']
        V = self.config['vocab_size']
        tp = self.config.get('tp', 1)
        
        # 1. The hidden states coming into the final layer
        input_act = self.FP16_BYTES * b * s * h
        
        # 2. Megatron VocabParallelCrossEntropy shards the V dimension by TP
        local_v = V / tp
        
        # 3. Logits (Forward):[B, S, V/TP] in FP32
        logits = self.FP32_BYTES * b * s * local_v
        
        # 4. Gradients of Logits (Backward):[B, S, V/TP] in FP32
        # Megatron avoids instantiating the Softmax 'probs' entirely to save memory.
        grad_logits = self.FP32_BYTES * b * s * local_v
        
        return input_act + logits + grad_logits

    
    

    # =========================================================================
    #  AGGREGATION
    # =========================================================================

    def calculate_stage_memory(self, stage_id: int) -> Dict[str, float]:
        """Calculates memory and maps to exact labels expected by plot_memory_usage."""
        pp_stages = self.config['pp_stages']
        if 'pp_partitioning' in self.config:
            layers_count = self.config['pp_partitioning'][stage_id]
        else:
            layers_count = self.config['num_layers'] / pp_stages

        param_attn = self._model_param_attention_per_gpu_stage(layers_count)
        param_moe = self._model_param_moe_per_gpu_stage(layers_count)
        grad_attn = self._model_grad_attention_per_gpu_stage(layers_count)
        grad_moe = self._model_grad_moe_per_gpu_stage(layers_count)
        optim_attn = self._model_optim_attention_per_gpu_stage(layers_count)
        optim_moe = self._model_optim_moe_per_gpu_stage(layers_count)

        param_special, grad_special, optim_special, special_act = 0.0, 0.0, 0.0, 0.0
        
        if stage_id == 0:
            param_special += self._embedding_param()
            grad_special += self._embedding_grad()
            optim_special += self._embedding_optim()
            special_act += self._embedding_act()
            
        if stage_id == pp_stages - 1:
            param_special += self._output_layer_param()
            grad_special += self._output_layer_grad()
            optim_special += self._output_layer_optim()
            special_act += self._output_layer_act()

        full_attn_per_layer = self._attention_activation_per_gpu_stage(1.0)
        full_moe_per_layer = self._moe_activation_per_gpu_stage(1.0)
        input_buffer = self.FP16_BYTES * self.config['mbs'] * self.config['seqlen'] * self.config['d_model']

        if 'dynamic_checkpointing' in self.config:
            num_ckpt = min(self.config['dynamic_checkpointing'][stage_id], layers_count)
        elif self.config.get('activation_checkpointing') != 'none':
            num_ckpt = layers_count
        else:
            num_ckpt = 0
            
        num_std = layers_count - num_ckpt

        # Standard layers pay full cost, checkpointed layers only pay for the tiny input buffer
        eff_attn = (num_std * full_attn_per_layer) + (num_ckpt * input_buffer)
        eff_moe = (num_std * full_moe_per_layer) + (num_ckpt * input_buffer)
        
        # Pipeline depth multiplier
        pipeline_multiplier = pp_stages - stage_id
        eff_attn_total = eff_attn * pipeline_multiplier
        eff_moe_total = eff_moe * pipeline_multiplier
        
        # Transient memory for the backward pass. It happens ONCE, so we do not multiply by pipeline_multiplier.
        active_backward_mem = (full_attn_per_layer + full_moe_per_layer) * 2.0 if num_ckpt > 0 else 0

        # Calculations for reporting metrics
        full_attn_act_total = full_attn_per_layer * layers_count * pipeline_multiplier
        full_moe_act_total = full_moe_per_layer * layers_count * pipeline_multiplier
        saved_attn = max(0, full_attn_act_total - eff_attn_total)
        saved_moe = max(0, full_moe_act_total - eff_moe_total)
        dynamic_frag = (active_backward_mem / self.GIGA) * 0.30

        return {
            "Model Parameters (Attention)": param_attn / self.GIGA,
            "Model Parameters (MoE)": param_moe / self.GIGA,
            "Model Parameters (Emb/Out)": param_special / self.GIGA,
            
            "Gradients (Attention)": grad_attn / self.GIGA,
            "Gradients (MoE)": grad_moe / self.GIGA,
            "Gradients (Emb/Out)": grad_special / self.GIGA,
            
            "Optimizer States (Attention)": optim_attn / self.GIGA,
            "Optimizer States (MoE)": optim_moe / self.GIGA,
            "Optimizer States (Emb/Out)": optim_special / self.GIGA,
            
            "Attention Activations": eff_attn_total / self.GIGA,
            "MoE Activations": eff_moe_total / self.GIGA,
            "Active Backward Memory": active_backward_mem / self.GIGA, 
            "Special Activations (Emb/Out)": special_act / self.GIGA,
            
            "Communication Buffers": 0.0,
            "System Overhead": 7.0 + dynamic_frag, 
            
            "Saved by Checkpointing (Attention)": saved_attn / self.GIGA,
            "Saved by Checkpointing (MoE)": saved_moe / self.GIGA,
            "_metadata": {
                "layers_per_stage": layers_count,
                "ckpt_layers_count": num_ckpt
            }
        }

    # =========================================================================
    #  DOWNSTREAM UTILITIES & VISUALIZATIONS
    # =========================================================================

    def dummy_breakdown_moe_act(self, total_moe_act_gb: float, num_layers_in_stage: float) -> Dict[str, float]:
        """
        Takes an observed/total MoE activation memory in GB and breaks it down 
        into its individual components based on their theoretical percentages.
        """
        h = self.config['d_model']
        s = self.config['seqlen']
        b = self.config['mbs']
        dp = self.config['dp']
        ep = self.config['ep']
        num_experts = self.config['num_experts']
        expert_dim = self.config['expert_dim']
        topk = self.config['topk']
        tp = self.config.get('tp', 1)
        
        num_experts_per_gpu = num_experts / ep 
        if int(tp) > 1: 
            s = s / tp 
            expert_dim /= tp 
            
        dp_non_moe = dp 
        per_gpu_max_assigned_tokens = b * s * dp_non_moe * topk / num_experts * num_experts_per_gpu 
        
        drop_factor = 0.5 + num_experts_per_gpu * 0.004
        per_gpu_max_assigned_tokens = per_gpu_max_assigned_tokens * drop_factor
        
        components = {
            "input_fp32 (Router)": self.FP32_BYTES * b * s * h,
            "gate_output": self.FP16_BYTES * b * s * num_experts,
            "softmax_input": self.FP32_BYTES * b * s * num_experts,
            "a2a_disp_input_buf": self.FP16_BYTES * per_gpu_max_assigned_tokens * h,
            "up_proj_out": self.FP16_BYTES * per_gpu_max_assigned_tokens * expert_dim,
            "gelu_out": self.FP16_BYTES * per_gpu_max_assigned_tokens * expert_dim,
            "layernorm": self.FP16_BYTES * b * s * h
        }
        
        for key in components:
            components[key] *= num_layers_in_stage

        theoretical_total = sum(components.values())
        if theoretical_total == 0:
            return {k: 0.0 for k in components.keys()}

        breakdown_gb = {}
        print(f"\n--- MoE Activation Breakdown (Allocating {total_moe_act_gb:.2f} GB) ---")
        for name, raw_bytes in components.items():
            percentage = raw_bytes / theoretical_total
            allocated_gb = percentage * total_moe_act_gb
            breakdown_gb[name] = allocated_gb
            print(f"{name:<25}: {allocated_gb:>6.2f} GB  ({percentage*100:>5.1f}%)")
            
        return breakdown_gb

    def debug_vs_profiled(self, profiled_data: List[float], model_name: str):
        """Prints a diff table and plots predicted vs profiled."""
        predicted_totals = []
        memory_breakdowns =[]
        
        for stage_id in range(self.config['pp_stages']):
            breakdown = self.calculate_stage_memory(stage_id)
            # Filter out metadata and "Saved by" stats to get actual physical memory
            clean_breakdown = {k: v for k, v in breakdown.items() if k != "_metadata" and "Saved by" not in k}
            memory_breakdowns.append(clean_breakdown)
            predicted_totals.append(sum(clean_breakdown.values()))
            
        print(f"\n{'='*65}")
        print(f"DEBUG REPORT: {model_name}")
        print(f"{'='*65}")
        print(f"{'Stage':<6} | {'Predicted (GB)':<15} | {'Profiled (GB)':<15} | {'Diff (Pred-Prof)':<15}")
        print("-" * 65)
        
        for i, (pred, prof) in enumerate(zip(predicted_totals, profiled_data)):
            diff = pred - prof
            flag = " <--- CHECK" if abs(diff) > 2.0 else ""
            print(f"{i:<6} | {pred:<15.2f} | {prof:<15.2f} | {diff:<+15.2f}{flag}")
            
        print("-" * 65)
        avg_diff = sum(abs(p - pr) for p, pr in zip(predicted_totals, profiled_data)) / len(profiled_data)
        print(f"Average Absolute Error: {avg_diff:.2f} GB\n")

        x = np.arange(self.config['pp_stages'])
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bottoms = np.zeros(self.config['pp_stages'])
        
        components = list(memory_breakdowns[0].keys())
        cmap = plt.cm.get_cmap('tab20', len(components))
        
        for idx, comp in enumerate(components):
            values = [b[comp] for b in memory_breakdowns]
            if sum(values) > 0.01:
                ax.bar(x - width/2, values, width, label=f"Pred: {comp}", bottom=bottoms, color=cmap(idx), edgecolor='black')
                bottoms += np.array(values)
            
        ax.bar(x + width/2, profiled_data, width, label='Profiled Total', color='gold', hatch='//', edgecolor='black', alpha=0.8)
        
        ax.set_title(f"Predicted vs Profiled Memory - {model_name}", fontweight='bold')
        ax.set_xlabel("Pipeline Stage")
        ax.set_ylabel("Memory (GB)")
        
        ax.set_xticks(x)
        ax.set_xticklabels([f"S{i}" for i in range(self.config['pp_stages'])])
        
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    def generate_report(self):
        print("="*80)
        print("GPU Memory Prediction Report")
        
        if 'dynamic_checkpointing' in self.config:
            print(f"Activation Checkpointing: Dynamic {self.config['dynamic_checkpointing']}")
        else:
            ckpt_str = self.config.get('activation_checkpointing', 'full')
            print(f"Activation Checkpointing: '{ckpt_str}'")
        
        if 'pp_partitioning' in self.config:
            print(f"Partitioning Strategy:    {self.config['pp_partitioning']}")
        else:
            print("Partitioning Strategy:    Even Split")
            
        print("="*80)

        for stage_id in range(self.config['pp_stages']):
            memory_breakdown = self.calculate_stage_memory(stage_id)
            metadata = memory_breakdown.pop("_metadata", {})
            total_memory = sum(v for k, v in memory_breakdown.items() if "Saved by" not in k)
            layers = metadata.get("layers_per_stage", "N/A")
            ckpt_count = metadata.get("ckpt_layers_count", "N/A")

            print(f"\n--- PP Stage {stage_id} (Layers: {layers} | Ckpt: {ckpt_count}) ---")
            
            param_total = memory_breakdown['Model Parameters (Attention)'] + memory_breakdown['Model Parameters (MoE)'] + memory_breakdown['Model Parameters (Emb/Out)']
            grad_total = memory_breakdown['Gradients (Attention)'] + memory_breakdown['Gradients (MoE)'] + memory_breakdown['Gradients (Emb/Out)']
            optim_total = memory_breakdown['Optimizer States (Attention)'] + memory_breakdown['Optimizer States (MoE)'] + memory_breakdown['Optimizer States (Emb/Out)']
            
            print(f"  {'Model Parameters':<30}: {param_total:7.2f} GB")
            print(f"  {'Gradients':<30}: {grad_total:7.2f} GB")
            print(f"  {'Optimizer States':<30}: {optim_total:7.2f} GB")
            print(f"  Activations (Effective):")

            print(f"    {'Attention Blocks':<25}: {memory_breakdown['Attention Activations']:7.2f} GB")
            if memory_breakdown.get('Saved by Checkpointing (Attention)', 0) > 0:
                print(f"      (Saved: {memory_breakdown['Saved by Checkpointing (Attention)']:7.2f} GB)")
                
            print(f"    {'MoE Blocks':<25}: {memory_breakdown['MoE Activations']:7.2f} GB")
            if memory_breakdown.get('Saved by Checkpointing (MoE)', 0) > 0:
                print(f"      (Saved: {memory_breakdown['Saved by Checkpointing (MoE)']:7.2f} GB)")
            
            if memory_breakdown.get('Active Backward Memory', 0) > 0:
                print(f"    {'Active Backward Mem':<25}: {memory_breakdown['Active Backward Memory']:7.2f} GB")

            print(f"    {'Embedding/Output':<25}: {memory_breakdown['Special Activations (Emb/Out)']:7.2f} GB")
            
            total_acts = (memory_breakdown['Attention Activations'] + memory_breakdown['MoE Activations'] + 
                          memory_breakdown['Special Activations (Emb/Out)'] + memory_breakdown.get('Active Backward Memory', 0))
            print(f"    {'-- Total Activation --':<25}: {total_acts:7.2f} GB")
            
            print(f"  {'Communication Buffers':<30}: {memory_breakdown['Communication Buffers']:7.2f} GB")
            print(f"  {'System Overhead':<30}: {memory_breakdown['System Overhead']:7.2f} GB")
            print("-" * 75)
            print(f"  {'Total Estimated Memory':<30}: {total_memory:7.2f} GB")
        print("\n" + "="*80)

    def calculate_model_size_in_billions(self) -> float:
        c = self.config
        attn_params = 4 * c['d_model']**2
        moe_params = (c['num_experts'] * 2 * c['d_model'] * c['expert_dim']) + (c['d_model'] * c['num_experts'])
        core_params = c['num_layers'] * (attn_params + moe_params)
        embed_params = c['vocab_size'] * c['d_model']
        total_params = core_params + embed_params + (0 if c.get('tied_embedding') else embed_params)
        return total_params / 1024**3
    
    def _generate_plot_title(self, base_title: str) -> str:
        size_str = f"{self.calculate_model_size_in_billions():.1f}B"
        schedule_str = "1F1B"

        if 'dynamic_checkpointing' in self.config:
            ckpt_str = f"Dynamic {self.config['dynamic_checkpointing']}"
        else:
            ckpt_str = self.config.get('activation_checkpointing', 'full')
            if ckpt_str != 'full': ckpt_str = f"'{ckpt_str}'"
        
        schedule_str += f" w/ {ckpt_str} CKPT"
        part_str = str(self.config['pp_partitioning']) if 'pp_partitioning' in self.config else "Even"
            
        return f"{size_str} Model - {base_title}\nDP={self.config['dp']}, EP={self.config['ep']}, TP={self.config.get('tp', 1)} PP={self.config['pp_stages']} ({part_str}) | Schedule: {schedule_str}"

    def plot_memory_usage(self, memory_limit_gb=64):
        stage_ids = list(range(self.config['pp_stages']))
        first_stage_data = self.calculate_stage_memory(0)
        first_stage_data.pop("_metadata", None)
        all_components = list(first_stage_data.keys())
        memory_data = {key:[] for key in all_components}

        for stage_id in stage_ids:
            memory_breakdown = self.calculate_stage_memory(stage_id)
            for component, memory in memory_breakdown.items():
                if component in memory_data:
                    memory_data[component].append(memory)

        active_components =[]
        for comp in all_components:
            if "Saved by" not in comp and sum(memory_data[comp]) > 0.01: 
                active_components.append(comp)

        desired_stacking_order =[
            'Model Parameters (Attention)', 'Model Parameters (MoE)', 'Model Parameters (Emb/Out)',
            'Gradients (Attention)', 'Gradients (MoE)', 'Gradients (Emb/Out)',
            'Optimizer States (Attention)', 'Optimizer States (MoE)', 'Optimizer States (Emb/Out)',
            'Attention Activations', 'MoE Activations', 'Special Activations (Emb/Out)',
            'Active Backward Memory', 'Communication Buffers', 'System Overhead'
        ]
        
        active_components.sort(key=lambda x: desired_stacking_order.index(x) if x in desired_stacking_order else 999)

        fig, ax = plt.subplots(figsize=(14, 9))
        
        custom_colors = {
            'Model Parameters (Attention)': '#1f77b4', 'Model Parameters (MoE)': '#aec7e8', 'Model Parameters (Emb/Out)': '#6baed6',
            'Gradients (Attention)': '#2ca02c', 'Gradients (MoE)': '#98df8a', 'Gradients (Emb/Out)': '#74c476',
            'Optimizer States (Attention)': '#d62728', 'Optimizer States (MoE)': '#ff9896', 'Optimizer States (Emb/Out)': '#fd8d3c',
            'Attention Activations': '#9467bd', 'MoE Activations': '#c5b0d5', 'Special Activations (Emb/Out)': '#8c564b',
            'Active Backward Memory': '#FF6B6B', 'Communication Buffers': '#e377c2', 'System Overhead': '#7f7f7f'
        }
        
        fallback_cmap = plt.cm.get_cmap('tab20', len(active_components))
        component_colors = {comp: custom_colors.get(comp, fallback_cmap(i)) for i, comp in enumerate(active_components)}

        total_stack_height = np.zeros(len(stage_ids))
        
        for component in active_components:
            memories = memory_data[component]
            
            p = ax.bar(stage_ids, memories, label=component, bottom=total_stack_height, width=0.6,
                       color=component_colors[component])
            total_stack_height += np.array(memories)
            
            labels =[f'{mem:.1f}' if mem > 1.0 else '' for mem in memories]
            ax.bar_label(p, labels=labels, label_type='center', color='white', fontweight='bold', fontsize=9)

        ax.set_xlabel("Pipeline Stage", fontweight='bold')
        ax.set_ylabel("Memory (GB)", fontweight='bold')

        base_title = "Predicted GPU Memory Usage per Pipeline Stage"
        
        if 'dynamic_checkpointing' in self.config and self.config['dynamic_checkpointing']:
            from itertools import groupby
            checkpoints = self.config['dynamic_checkpointing']
            grouped_checkpoints = "+".join([f"[{k}]*{len(list(g))}" for k, g in groupby(checkpoints)])
            title = f"{base_title}\nCheckpointing Schedule: {grouped_checkpoints}"
        else:
            title = self._generate_plot_title(base_title)
            
        ax.set_title(title, fontweight='bold', fontsize=14)
        ax.set_xticks(stage_ids)
        
        xtick_labels =[]
        for i in stage_ids:
            L = self.config['pp_partitioning'][i] if 'pp_partitioning' in self.config else int(self.config['num_layers'] / self.config['pp_stages'])
            if 'dynamic_checkpointing' in self.config:
                AC = self.config['dynamic_checkpointing'][i]
            else:
                ckpt_setting = self.config.get('activation_checkpointing', 'full')
                AC = L if ckpt_setting in['attention', 'layer', 'moe', 'full'] and ckpt_setting != 'none' else 0
            xtick_labels.append(f"Stage {i}\n({L}L - {AC}AC)")
            
        ax.set_xticklabels(xtick_labels)
        
        if memory_limit_gb > 0:
            ax.axhline(y=memory_limit_gb, color='r', linestyle='--', linewidth=2, label=f'Hardware Limit ({memory_limit_gb}GB)')

        ax.legend(title="Memory Component", loc='upper left', bbox_to_anchor=(1.05, 1.0))
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        effective_totals = np.zeros(len(stage_ids))
        for comp in active_components:
            effective_totals += np.array(memory_data[comp])

        for i, total in enumerate(effective_totals):
            ax.text(i, total + 0.5, f"Total: {total:.1f}", ha='center', fontweight='bold')

        ax.set_ylim(bottom=0, top=max(max(effective_totals)*1.2, memory_limit_gb * 1.1))
        plt.tight_layout()
        plt.show()

    def plot_memory_vs_mbs(self, mbs_values, memory_limit_gb=64):
        original_mbs = self.config['mbs']
        original_gbs = self.config['gbs']
        acc_steps = original_gbs // (self.config['dp'] * original_mbs)

        sample_calc = self.calculate_stage_memory(0)
        sample_calc.pop("_metadata", None)
        components = list(sample_calc.keys())
        memory_results = {}

        try:
            for mbs in mbs_values:
                self.config['mbs'] = mbs
                self.config['gbs'] = self.config['dp'] * mbs * acc_steps
                self._validate_config()
                
                memory_results[mbs] = {key:[] for key in components}
                for stage_id in range(self.config['pp_stages']):
                    mem_breakdown = self.calculate_stage_memory(stage_id)
                    for comp in components:
                        if "Saved by" not in comp:
                            memory_results[mbs][comp].append(mem_breakdown[comp])
        finally:
            self.config['mbs'] = original_mbs
            self.config['gbs'] = original_gbs

        num_mbs = len(mbs_values)
        num_stages = self.config['pp_stages']
        x = np.arange(num_mbs)
        width = 0.8 / num_stages

        fig, ax = plt.subplots(figsize=(18, 10))
        
        plot_components =[c for c in components if "Saved by" not in c]
        colors = plt.cm.get_cmap('viridis', len(plot_components))
        component_colors = {comp: colors(i) for i, comp in enumerate(plot_components)}
        
        total_bar_heights = np.zeros((num_mbs, num_stages))

        for stage_idx in range(num_stages):
            offset = width * (stage_idx - (num_stages - 1) / 2)
            bottoms_for_current_stage = np.zeros(num_mbs)

            for component in plot_components:
                mem_values = np.array([memory_results[mbs][component][stage_idx] for mbs in mbs_values])
                
                rects = ax.bar(x + offset, mem_values, width,
                               label=component if stage_idx == 0 else "",
                               bottom=bottoms_for_current_stage,
                               color=component_colors[component],
                               edgecolor='black',
                               linewidth=0.7)
                
                labels =[f'{val:.1f}' if val > 1.5 else '' for val in mem_values]
                ax.bar_label(rects, labels=labels, label_type='center',
                            color='white', fontsize=8, fontweight='bold')

                bottoms_for_current_stage += mem_values
            
            total_bar_heights[:, stage_idx] = bottoms_for_current_stage

        title = self._generate_plot_title("GPU Memory vs. Micro-Batch Size")
        ax.set_title(title, fontweight='bold', fontsize=16, pad=20)
        ax.set_xlabel("Micro-Batch Size (mbs)", fontweight='bold', fontsize=12)
        ax.set_ylabel("Total Memory (GB)", fontweight='bold', fontsize=12)
        ax.set_xticks(x, mbs_values)

        for i in range(num_mbs):
            for stage_idx in range(num_stages):
                offset = width * (stage_idx - (num_stages - 1) / 2)
                ax.text(x[i] + offset, -0.05, f"S{stage_idx}",
                        transform=ax.get_xaxis_transform(),
                        ha='center', va='top', fontsize=10, color='dimgray')
        
        if memory_limit_gb > 0:
            ax.axhline(y=memory_limit_gb, color='r', linestyle='--', linewidth=2, 
                       label=f'Hardware Limit ({memory_limit_gb}GB)')

        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.set_ylim(bottom=0, top=max(np.max(total_bar_heights) * 1.1, memory_limit_gb * 1.1))
        
        ax.legend(title="Memory Component", loc='upper left', bbox_to_anchor=(1.02, 1))
        fig.tight_layout()
        plt.show()

    def find_best_parallelism_config(self, ep_size, max_gpus=128, mbs=1, memory_limit_gb=64):
        valid_configs =[]
        for pp in range(1, (max_gpus // ep_size) + 1):
            total_gpus = ep_size * pp
            if self.config['num_layers'] % pp == 0:
                valid_configs.append({'ep': ep_size, 'pp': pp, 'total_gpus': total_gpus})

        if not valid_configs:
            print(f"No valid EP/PP configurations found for EP={ep_size} and max_gpus={max_gpus} "
                f"with {self.config['num_layers']} layers.")
            return

        original_config = self.config.copy()
        all_results = {}

        sample_calc = self.calculate_stage_memory(0)
        sample_calc.pop("_metadata", None)
        components =[c for c in sample_calc.keys() if "Saved by" not in c]

        try:
            for config in valid_configs:
                ep, pp = config['ep'], config['pp']

                self.config['ep'] = ep
                self.config['pp_stages'] = pp
                self.config['dp'] = ep
                self.config['mbs'] = mbs
                if 'pp_partitioning' in self.config:
                    del self.config['pp_partitioning']

                acc_steps = original_config['gbs'] // (self.config['dp'] * self.config['mbs'])
                self.config['gbs'] = self.config['dp'] * self.config['mbs'] * acc_steps
                self._validate_config()

                config_key = f"EP={ep}, PP={pp}"
                all_results[config_key] = {comp:[] for comp in components}

                for stage_id in range(pp):
                    mem_breakdown = self.calculate_stage_memory(stage_id)
                    for comp in components:
                        all_results[config_key][comp].append(mem_breakdown[comp])

        finally:
            self.config = original_config

        num_configs = len(valid_configs)
        ncols = min(num_configs, 3)
        nrows = math.ceil(num_configs / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 6 * nrows), squeeze=False)
        axes = axes.flatten()

        colors = plt.cm.get_cmap('viridis', len(components))
        component_colors = {comp: colors(i) for i, comp in enumerate(components)}

        for i, (config_key, memory_data) in enumerate(all_results.items()):
            ax = axes[i]
            num_stages = len(memory_data[components[0]])
            stage_ids = list(range(num_stages))

            bottom = np.zeros(num_stages)
            for component, memories in memory_data.items():
                p = ax.bar(stage_ids, memories, label=component, bottom=bottom, width=0.7, color=component_colors[component])
                bottom += np.array(memories)

            peak_memory = np.max(bottom)

            ax.set_title(f"{config_key} | Peak Memory: {peak_memory:.1f} GB", fontweight='bold')
            ax.set_xlabel("Pipeline Stage")
            ax.set_ylabel("Memory (GB)")
            ax.set_xticks(stage_ids)
            ax.grid(axis='y', linestyle='--', alpha=0.7)

            if memory_limit_gb > 0:
                ax.axhline(y=memory_limit_gb, color='r', linestyle='--', linewidth=2)

            ax.set_ylim(bottom=0, top=max(peak_memory * 1.15, memory_limit_gb * 1.15))

        for i in range(num_configs, len(axes)):
            axes[i].set_visible(False)

        handles =[mpatches.Patch(color=color, label=comp) for comp, color in component_colors.items()]
        fig.legend(handles=handles, title="Memory Component", loc='upper right', bbox_to_anchor=(0.98, 0.95))

        suptitle = self._generate_plot_title(f"Parallelism Options for EP={ep_size} (MBS={mbs})")
        fig.suptitle(suptitle, fontsize=20, fontweight='bold', y=1.0)

        plt.tight_layout(rect=[0, 0, 0.9, 0.96])
        plt.show()
        
    def query_stage_memory(self, stage_id: int, num_layers: int, num_ckpt_layers: int, micro_batch_size: int) -> float:
        """
        Calculates the estimated memory usage (GB) for a specific stage configuration.
        Designed to be called iteratively by an external Planner.
        """
        use_groupgemm = self.config.get('use_groupgemm')
        original_mbs = self.config.get('mbs')
        original_gbs = self.config.get('gbs')
        original_parts = self.config.get('pp_partitioning')
        original_dyn_ckpt = self.config.get('dynamic_checkpointing')

        try:
            if num_layers > self.config['num_layers']:
                return float('inf')

            self.config['mbs'] = micro_batch_size
            self.config['gbs'] = micro_batch_size * self.config['dp'] * 256 

            temp_parts = [0] * self.config['pp_stages'] 
            temp_parts[stage_id] = num_layers
            
            remainder = self.config['num_layers'] - num_layers
            if remainder > 0:
                neighbor_idx = (stage_id + 1) % self.config['pp_stages']
                temp_parts[neighbor_idx] += remainder

            self.config['pp_partitioning'] = temp_parts

            temp_ckpt = [0] * self.config['pp_stages']
            temp_ckpt[stage_id] = num_ckpt_layers
            self.config['dynamic_checkpointing'] = temp_ckpt

            result = self.calculate_stage_memory(stage_id)

            total_gb = sum(v for k, v in result.items() if "Saved by" not in k and k != "_metadata")
            return total_gb

        except Exception as e:
            return float('inf')

        finally:
            self.config['mbs'] = original_mbs
            self.config['gbs'] = original_gbs
            if original_parts is None: self.config.pop('pp_partitioning', None)
            else: self.config['pp_partitioning'] = original_parts
            
            if original_dyn_ckpt is None: self.config.pop('dynamic_checkpointing', None)
            else: self.config['dynamic_checkpointing'] = original_dyn_ckpt