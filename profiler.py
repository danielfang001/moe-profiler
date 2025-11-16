"""
Simple MoE Profiler for Router Middleware Research
Easy to use, notebook-friendly, minimal complexity
"""

import torch
import pandas as pd
import numpy as np
import time
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path

# Centralized log path (can be overridden by environment)
LOG_PATH = Path(os.environ.get('MOEPROFILER_LOG', '/root/moeprofiler_debug.log'))

# ============================================
# 1. SIMPLE METRIC TRACKER
# ============================================

@dataclass
class Metrics:
    """Enhanced metrics container with statistical tracking"""
    # Raw measurements
    flops_per_token: List[float] = field(default_factory=list)
    router_flops_per_token: List[float] = field(default_factory=list)
    expert_flops_per_token: List[float] = field(default_factory=list)
    active_experts: List[int] = field(default_factory=list)
    k_per_token: List[float] = field(default_factory=list)  # Average k (active experts) per token in batch
    k_distribution: List[List[int]] = field(default_factory=list)  # Full k distribution per forward pass
    latency_ms: List[float] = field(default_factory=list)
    expert_loads: Dict[int, int] = field(default_factory=lambda: {i: 0 for i in range(8)})
    router_confidence: List[float] = field(default_factory=list)
    semantic_confidence: List[float] = field(default_factory=list)
    memory_mb: List[float] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)

    # Metadata
    device: str = 'cuda:0'
    step_count: int = 0

    def to_df(self):
        """Convert to pandas DataFrame for easy analysis"""
        n = len(self.flops_per_token)

        # Pad semantic_confidence if not populated
        semantic_conf = self.semantic_confidence if len(self.semantic_confidence) == n else [0.0] * n

        return pd.DataFrame({
            'step': range(n),
            'flops_total': self.flops_per_token,
            'flops_router': self.router_flops_per_token,
            'flops_expert': self.expert_flops_per_token,
            'active_experts': self.active_experts,
            'k_avg': self.k_per_token,
            'latency_ms': self.latency_ms,
            'router_conf': self.router_confidence,
            'semantic_conf': semantic_conf,
            'memory_mb': self.memory_mb,
            'timestamp': self.timestamps,
            'device': self.device
        })

# ============================================
# 2. SIMPLE ROUTER WRAPPER
# ============================================

class SimpleRouterWrapper(torch.nn.Module):
    """Wraps your MoE router to track metrics
    """

    def __init__(self, original_router, num_experts=8, expert_dim=None, hidden_dim=None,
                 warmup_steps=5, sampling_rate=1, name: Optional[str]=None):
        """
        Args:
            original_router: The original router to wrap
            num_experts: Number of experts
            expert_dim: Expert FFN dimension
            hidden_dim: Model hidden dimension
            warmup_steps: Number of initial steps to skip metric collection or profiling, allowing the system to "warm up" for more stable measurements.
            sampling_rate: Frequency for collecting metrics (e.g., every Nth step), helping reduce overhead by not recording data on every forward pass.
        """
        super().__init__()
        self.router = original_router
        # Optional name of the wrapped module (set by MoEProfiler)
        self.name = name
        self.num_experts = num_experts
        self.expert_dim = expert_dim  # Expert FFN dimension
        self.hidden_dim = hidden_dim  # Model hidden dimension
        self.warmup_steps = warmup_steps
        self.sampling_rate = sampling_rate
        self.current_step = 0

        # Auto-detect dimensions if not provided
        if self.hidden_dim is None:
            self._auto_detect_dimensions()

        self.metrics = Metrics()
        self.enabled = True  # Toggle profiling on/off

        # CUDA events for precise timing
        self.use_cuda_events = torch.cuda.is_available()
        if self.use_cuda_events:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)

    def _auto_detect_dimensions(self):
        """Auto-detect model dimensions from router parameters"""
        try:
            for param in self.router.parameters():
                if len(param.shape) == 2:
                    self.hidden_dim = param.shape[0]
                    break
            if self.hidden_dim is None:
                self.hidden_dim = 4096  # Default fallback
        except:
            self.hidden_dim = 4096

        # Assume expert FFN dim is 4x hidden for MoE models
        if self.expert_dim is None:
            self.expert_dim = self.hidden_dim * 4
        
    def forward(self, x):
        if not self.enabled:
            return self.router(x)

        self.current_step += 1

        # Skip warmup steps
        in_warmup = self.current_step <= self.warmup_steps
        should_profile = (self.current_step % self.sampling_rate == 0) and not in_warmup

        if not should_profile:
            return self.router(x)

        # Start timing with CUDA events or fallback to CPU timing
        if self.use_cuda_events:
            self.start_event.record()

        start_time = time.perf_counter()

        # Get routing decision (call the wrapped router)
        router_output = self.router(x)
        
        # Case 1: Router returns (weights, indices) tuple
        if isinstance(router_output, tuple):
            routing_weights, expert_indices = router_output
        # Case 2: Router returns only logits (e.g., Mixtral gate, OlMoE)
        else:
            router_logits = router_output
            # Compute routing weights and select top-k experts
            routing_probs = torch.nn.functional.softmax(router_logits, dim=-1)
            # Assume top_k=2 for Mixtral, adjust if needed
            top_k = getattr(self.router, 'top_k', 2)
            routing_weights, expert_indices = torch.topk(routing_probs, 8, dim=-1)

        # Calculate confidence (entropy-based)
        psum = torch.sum(routing_probs**2)
        pcutoff = (64/63)*psum-(1/63)
        print("pcutoff",pcutoff,flush=True)
        # entropy = -torch.sum(routing_weights * torch.log(routing_weights + 1e-10), dim=-1).mean()
        # max_entropy = torch.log(torch.tensor(self.num_experts, dtype=torch.float32))
        # confidence = float(1.0 - (entropy / max_entropy).item())
        self.metrics.router_confidence.append(pcutoff)

        # Synchronize and record latency
        if self.use_cuda_events:
            self.end_event.record()
            torch.cuda.synchronize()
            latency = self.start_event.elapsed_time(self.end_event)  # Already in ms
        else:
            latency = (time.perf_counter() - start_time) * 1000

        self.metrics.latency_ms.append(latency)
        self.metrics.timestamps.append(time.time())

        # Get batch and sequence dimensions
        batch_size = x.shape[0]
        seq_len = x.shape[1] if len(x.shape) > 2 else 1
        hidden_dim = x.shape[-1]
        num_tokens = batch_size * seq_len

        # Calculate k per token (dynamic expert selection)
        # expert_indices shape: [batch_size, seq_len, k] or [num_tokens, k]
        expert_indices_flat = expert_indices.reshape(-1, expert_indices.shape[-1])

        # Count number of active experts per token (some might be masked/inactive)
        # Assuming -1 or negative values indicate inactive experts
        k_per_token_list = []
        for token_experts in expert_indices_flat:
            active_k = (token_experts >= 0).sum().item()
            k_per_token_list.append(active_k)

        # Store k distribution and average k
        self.metrics.k_distribution.append(k_per_token_list)
        avg_k = sum(k_per_token_list) / len(k_per_token_list) if len(k_per_token_list) > 0 else 0
        self.metrics.k_per_token.append(avg_k)

        sorted_probs, indices = torch.sort(routing_probs, descending=True, dim=1)  # Sort along dim=1
        cumsum = torch.cumsum(sorted_probs, dim=1)  # Cumsum along dim=1
        mask = cumsum <= pcutoff
        k = torch.sum(mask.float(), dim=1).item() + 1  # Sum along dim=1, then convert to scalar
        print("activated experts",k)
        print("sorted_probs",sorted_probs)
        print("cumsum",cumsum)

        self.metrics.active_experts.append(k)

        # Accurate FLOPs calculation with dynamic k
        # Router FLOPs: Linear projection (input @ weight) + softmax
        # Linear: 2 * num_tokens * hidden_dim * num_experts (multiply-add operations)
        # Softmax: ~5 * num_tokens * num_experts (exp, sum, divide)
        router_flops = (2 * num_tokens * hidden_dim * self.num_experts +
                        5 * num_tokens * self.num_experts)

        # Expert FLOPs: Accounts for dynamic k per token
        # Each expert is a 2-layer FFN: hidden -> expert_dim -> hidden
        # up_proj: 2 * hidden_dim * expert_dim
        # down_proj: 2 * expert_dim * hidden_dim
        # Sum across all tokens with their respective k values
        expert_flops = sum(k * (2 * hidden_dim * self.expert_dim +
                               2 * self.expert_dim * hidden_dim)
                          for k in k_per_token_list)

        total_flops = router_flops + expert_flops
        flops_per_token = total_flops / num_tokens
        router_flops_per_token = router_flops / num_tokens
        expert_flops_per_token = expert_flops / num_tokens

        self.metrics.flops_per_token.append(flops_per_token)
        self.metrics.router_flops_per_token.append(router_flops_per_token)
        self.metrics.expert_flops_per_token.append(expert_flops_per_token)

        # Track expert loads
        for expert_id in expert_indices.flatten().tolist():
            if 0 <= expert_id < self.num_experts:
                self.metrics.expert_loads[expert_id] += 1

        # Memory tracking
        if torch.cuda.is_available():
            self.metrics.memory_mb.append(torch.cuda.memory_allocated() / 1e6)

        self.metrics.step_count = self.current_step

        # Return in the same format as the original router
        if isinstance(router_output, tuple):
            return routing_weights, expert_indices
        else:
            # Return only logits to match Mixtral gate behavior
            return router_logits
    
    def reset_metrics(self):
        """Clear all recorded metrics"""
        self.metrics = Metrics()
        self.current_step = 0

    def _calculate_gini_coefficient(self, values):
        """
        Calculate Gini coefficient for load imbalance
        0 = perfect equality, 1 = perfect inequality
        """
        values = np.array(sorted(values))
        n = len(values)
        if n == 0 or values.sum() == 0:
            return 0.0
        cumsum = np.cumsum(values)
        return (2 * np.sum((np.arange(1, n + 1)) * values)) / (n * cumsum[-1]) - (n + 1) / n

    def _calculate_entropy(self, values):
        """
        Calculate normalized entropy of distribution
        0 = concentrated, 1 = uniform
        """
        values = np.array(values)
        if values.sum() == 0:
            return 0.0
        probs = values / values.sum()
        probs = probs[probs > 0]  # Remove zeros
        entropy = -np.sum(probs * np.log(probs))
        max_entropy = np.log(len(values))  # Maximum entory occurs when uniform distribution
        return entropy / max_entropy if max_entropy > 0 else 0.0 # Normalized entropy

    def _calculate_cv(self, values):
        """
        Calculate coefficient of variation (CV = std / mean)
        Measures relative variability
        """
        values = np.array(values)
        if len(values) == 0 or values.mean() == 0:
            return 0.0
        return values.std() / values.mean()

    def get_summary(self):
        """Get comprehensive statistical summary of metrics"""
        df = self.metrics.to_df()
        if len(df) == 0:
            return "No metrics collected yet"

        # Expert load statistics
        expert_loads = np.array(list(self.metrics.expert_loads.values()))

        # k distribution statistics
        k_values = np.array(self.metrics.k_per_token)

        return {
            # FLOPs statistics
            'flops_total_mean': df['flops_total'].mean(),
            'flops_total_std': df['flops_total'].std(),
            'flops_total_p50': df['flops_total'].median(),
            'flops_total_p95': df['flops_total'].quantile(0.95),
            'flops_total_p99': df['flops_total'].quantile(0.99),
            'flops_router_mean': df['flops_router'].mean(),
            'flops_expert_mean': df['flops_expert'].mean(),

            # Latency statistics
            'latency_mean_ms': df['latency_ms'].mean(),
            'latency_std_ms': df['latency_ms'].std(),
            'latency_min_ms': df['latency_ms'].min(),
            'latency_max_ms': df['latency_ms'].max(),
            'latency_p50_ms': df['latency_ms'].median(),
            'latency_p95_ms': df['latency_ms'].quantile(0.95),
            'latency_p99_ms': df['latency_ms'].quantile(0.99),

            # Expert utilization
            'active_experts_mean': df['active_experts'].mean(),
            'active_experts_std': df['active_experts'].std(),

            # Dynamic k statistics
            'k_mean': k_values.mean() if len(k_values) > 0 else 0,
            'k_std': k_values.std() if len(k_values) > 0 else 0,
            'k_min': k_values.min() if len(k_values) > 0 else 0,
            'k_max': k_values.max() if len(k_values) > 0 else 0,

            # Load balancing metrics
            'expert_load_mean': expert_loads.mean(),
            'expert_load_std': expert_loads.std(),
            'expert_load_cv': self._calculate_cv(expert_loads),
            'expert_load_gini': self._calculate_gini_coefficient(expert_loads),
            'expert_load_entropy': self._calculate_entropy(expert_loads),

            # Confidence metrics
            'router_confidence_mean': df['router_conf'].mean(),
            'router_confidence_std': df['router_conf'].std(),

            # Memory
            'memory_mean_mb': df['memory_mb'].mean() if 'memory_mb' in df and len(df['memory_mb']) > 0 else 0,
            'memory_max_mb': df['memory_mb'].max() if 'memory_mb' in df and len(df['memory_mb']) > 0 else 0,

            # Overall statistics
            'total_steps': len(df),
            'total_tokens_profiled': len(df) * (df.index[0] if len(df) > 0 else 0),
        }

# ============================================
# 3. SIMPLE PROFILER
# ============================================

class MoEProfiler:
    """Simple profiler for experiments"""
    
    def __init__(self, model):
        self.model = model
        self.wrappers = []
        # Ensure log file exists (best-effort) so we can diagnose missing logs
        try:
            LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(str(LOG_PATH), 'a') as _f:
                _f.write(f"{time.time():.3f}\tPID={os.getpid()}\tLOG_INIT\n")
        except Exception:
            # Non-fatal: if /tmp isn't writable in this environment we'll fall back to stderr-only logging
            try:
                os.write(2, (f"LOG_INIT_FAILED PID={os.getpid()}\n").encode())
            except Exception:
                pass

        self._wrap_routers()
        
    def _wrap_routers(self):
        """Find and wrap all MoE routers in the model"""
        for name, module in self.model.named_modules():
            # Skip if already wrapped
            if isinstance(module, SimpleRouterWrapper):
                continue

            # Check module type names for MoE-specific classes
            module_type = type(module).__name__

            # Match only gate/router modules (not the entire MoE block)
            # - Mixtral: 'gate' in 'block_sparse_moe.gate'
            # - OLMoE: 'gate' in 'mlp.gate' (Linear module inside OlmoeSparseMoeBlock)
            # - General: 'router' in name
            is_gate_or_router = (
                ('router' in name.lower() and 'gate_proj' not in name.lower()) or
                (name.endswith('.gate') and 'gate_proj' not in name.lower())
            )

            if is_gate_or_router:
                print(f"Wrapping router: {name} (type: {module_type})")
                # Replace with wrapper and record the original module name
                wrapper = SimpleRouterWrapper(module, name=name)
                self.wrappers.append(wrapper)

                # Determine parent module and child attribute name and set the wrapper
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                if parent_name:
                    parent = self.model.get_submodule(parent_name)
                else:
                    parent = self.model

                setattr(parent, child_name, wrapper)

                # Move wrapper to model device (if model has parameters/buffers on a device)
                model_device = None
                try:
                    model_device = next(self.model.parameters()).device
                except StopIteration:
                    try:
                        model_device = next(self.model.buffers()).device
                    except StopIteration:
                        model_device = None

                if model_device is not None:
                    try:
                        wrapper.to(model_device)
                        print(f"Moved wrapper '{name}' to device {model_device}")
                    except Exception:
                        # Non-fatal: don't break wrapping if move fails
                        print(f"Warning: failed to move wrapper '{name}' to device {model_device}")
                    
    def start(self):
        """Start profiling"""
        for wrapper in self.wrappers:
            wrapper.enabled = True
            wrapper.reset_metrics()
    
    def stop(self):
        """Stop profiling"""
        for wrapper in self.wrappers:
            wrapper.enabled = False
    
    def get_metrics(self):
        """Get all collected metrics"""
        all_metrics = {}
        for i, wrapper in enumerate(self.wrappers):
            all_metrics[f'layer_{i}'] = wrapper.metrics
        return all_metrics
    
    def save_csv(self, filepath='metrics.csv'):
        """Save metrics to CSV"""
        dfs = []
        for i, wrapper in enumerate(self.wrappers):
            df = wrapper.metrics.to_df()
            df['layer'] = i
            dfs.append(df)
        
        if dfs:
            combined = pd.concat(dfs)
            combined.to_csv(filepath, index=False)
            print(f"Saved {len(combined)} rows to {filepath}")

    def dump_wrapper_states(self):
        """Return and log a compact state summary for all wrappers.

        This writes a best-effort line to /tmp/moeprofiler_debug.log and to
        stderr (unbuffered), then returns the list of dicts so calling code can
        inspect it programmatically.
        """
        info = []
        for w in self.wrappers:
            info.append({
                'name': getattr(w, 'name', '<unknown>'),
                'enabled': bool(getattr(w, 'enabled', False)),
                'current_step': int(getattr(w, 'current_step', 0))
            })
        try:
            with open(str(LOG_PATH), 'a') as f:
                f.write(f"{time.time():.3f}\tPID={os.getpid()}\tDUMP_WRAPPERS\t{info}\n")
        except Exception:
            pass
        try:
            os.write(2, (f"DUMP_WRAPPERS: {info}\n").encode())
        except Exception:
            pass
        return info
    
    def print_summary(self):
        """Print comprehensive statistical summary"""
        print("\n" + "="*60)
        print("=== MoE Profiling Summary ===")
        print("="*60)

        for i, wrapper in enumerate(self.wrappers):
            print(f"\n{'─'*60}")
            print(f"Layer {i}:")
            print('─'*60)
            summary = wrapper.get_summary()
            if isinstance(summary, dict):
                # Organize output by category
                print("\n  FLOPs Metrics:")
                for k, v in summary.items():
                    if k.startswith('flops_'):
                        print(f"    {k:30s}: {v:>12.2f}" if isinstance(v, float) else f"    {k:30s}: {v:>12}")

                print("\n  Latency Metrics:")
                for k, v in summary.items():
                    if k.startswith('latency_'):
                        print(f"    {k:30s}: {v:>12.2f} ms" if isinstance(v, float) else f"    {k:30s}: {v:>12}")

                print("\n  Expert Utilization:")
                for k, v in summary.items():
                    if k.startswith('k_') or k.startswith('active_experts'):
                        print(f"    {k:30s}: {v:>12.2f}" if isinstance(v, float) else f"    {k:30s}: {v:>12}")

                print("\n  Load Balancing:")
                for k, v in summary.items():
                    if k.startswith('expert_load'):
                        print(f"    {k:30s}: {v:>12.2f}" if isinstance(v, float) else f"    {k:30s}: {v:>12}")

                print("\n  Memory & Confidence:")
                for k, v in summary.items():
                    if k.startswith('memory_') or k.startswith('router_confidence'):
                        print(f"    {k:30s}: {v:>12.2f}" if isinstance(v, float) else f"    {k:30s}: {v:>12}")

                print("\n  Overall Statistics:")
                for k, v in summary.items():
                    if k.startswith('total_'):
                        print(f"    {k:30s}: {v:>12.2f}" if isinstance(v, float) else f"    {k:30s}: {v:>12}")
                
            else:
                print(f"  {summary}")

        print("\n" + "="*60)

# ============================================
# 4. BENCHMARK RUNNER
# ============================================

class SimpleBenchmark:
    """Run and compare different configurations"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.results = {}
        
    def run_config(self, name, input_texts, config_fn=None):
        """Run a single configuration"""
        print(f"\nRunning config: {name}")
        
        # Apply configuration
        if config_fn:
            config_fn(self.model)
        
        # Setup profiler
        profiler = MoEProfiler(self.model)
        profiler.start()
        
        # Run inference
        for text in input_texts:
            inputs = self.tokenizer(text, return_tensors='pt')
            with torch.no_grad():
                outputs = self.model(**inputs)
        
        # Get results
        profiler.stop()
        metrics = profiler.get_metrics()
        
        # Save
        profiler.save_csv(f"{name}_metrics.csv")
        self.results[name] = metrics
        
        profiler.print_summary()
        
    def compare(self):
        """Compare all configurations"""
        comparison = []
        
        for config_name, metrics in self.results.items():
            total_flops = 0
            total_latency = 0
            total_tokens = 0
            
            for layer_metrics in metrics.values():
                df = layer_metrics.to_df()
                if len(df) > 0:
                    total_flops += df['flops'].sum()
                    total_latency += df['latency_ms'].sum()
                    total_tokens += len(df)
            
            comparison.append({
                'config': config_name,
                'total_flops': total_flops,
                'avg_latency_ms': total_latency / total_tokens if total_tokens > 0 else 0,
                'total_tokens': total_tokens
            })
        
        df = pd.DataFrame(comparison)
        df.to_csv('comparison.csv', index=False)
        print("\n=== Configuration Comparison ===")
        print(df.to_string())
        return df

# ============================================
# 5. YOUR CUSTOM MIDDLEWARE (Example)
# ============================================

class YourRouterMiddleware(torch.nn.Module):
    """Your custom router with confidence-based selection"""
    
    def __init__(self, base_router, confidence_threshold=0.7):
        super().__init__()
        self.router = base_router
        self.confidence_threshold = confidence_threshold
        
    def forward(self, x):
        # Get base routing
        weights, indices = self.router(x)
        
        # Calculate confidence (max weight)
        confidence = weights.max(dim=-1)[0]
        
        # Skip routing if confidence too low
        mask = confidence < self.confidence_threshold
        indices[mask] = -1  # Skip these tokens
        
        # Track semantic confidence (simplified)
        # You can add your actual semantic confidence logic here
        semantic_conf = torch.cosine_similarity(
            x.mean(dim=1), 
            x.mean(dim=1).roll(1, dims=0)
        ).mean()
        
        # Store in wrapper if it exists
        if hasattr(self.router, 'metrics'):
            self.router.metrics.semantic_confidence.append(semantic_conf.item())
        
        return weights, indices

# ============================================
# 6. USAGE EXAMPLE
# ============================================

def example_usage():
    """
    Complete example of how to use this in your notebook
    """
    
    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
    
    # Test texts
    test_texts = [
        "What is machine learning?",
        "Explain quantum computing",
        "How does photosynthesis work?"
    ]
    
    # Create benchmark runner
    benchmark = SimpleBenchmark(model, tokenizer)
    
    # Run baseline
    benchmark.run_config("baseline", test_texts)
    
    # Run with your middleware
    def apply_middleware(model):
        for name, module in model.named_modules():
            if 'router' in name.lower():
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent = model.get_submodule(parent_name)
                custom = YourRouterMiddleware(module, confidence_threshold=0.8)
                setattr(parent, child_name, custom)
    
    benchmark.run_config("with_middleware", test_texts, apply_middleware)
    
    # Compare results
    comparison = benchmark.compare()
    
    print("\nDone! Check these files:")
    print("- baseline_metrics.csv")
    print("- with_middleware_metrics.csv")
    print("- comparison.csv")

# ============================================
# 7. MULTI-GPU SUPPORT (Simple)
# ============================================

def setup_multi_gpu(model):
    """Simple multi-GPU setup"""
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
    elif torch.cuda.is_available():
        model = model.cuda()
    return model

# ============================================
# 8. NOTEBOOK HELPERS
# ============================================

def quick_profile(model, tokenizer, text="Hello world", num_iterations=10):
    """One-line profiling for notebooks"""
    profiler = MoEProfiler(model)
    profiler.start()
    
    inputs = tokenizer(text, return_tensors='pt')
    for _ in range(num_iterations):
        with torch.no_grad():
            model(**inputs)
    
    profiler.stop()
    profiler.print_summary()
    profiler.save_csv('quick_profile.csv')
    return profiler

# For Colab
def colab_setup():
    """Setup for Google Colab"""
    # Install dependencies
    print("!pip install transformers torch pandas")
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("No GPU available, using CPU")

if __name__ == "__main__":
    print("MoE Profiler loaded! Use example_usage() to see how it works.")