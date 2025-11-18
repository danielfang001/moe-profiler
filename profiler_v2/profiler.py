"""
MoE Profiler - Architecture-Agnostic Profiling for Mixture of Experts Models

Main profiler class that auto-detects MoE architecture and applies
appropriate wrappers for comprehensive metrics collection.
"""

import torch
import torch.nn as nn
import pandas as pd
from typing import Optional, Callable, List
from .architectures import detect_architecture
from .wrappers import RouterWrapper
from .metrics import get_metrics_summary


class MOEProfiler:
    """
    Main profiler for MoE models with architecture auto-detection.

    Automatically detects the MoE architecture (Mixtral, OLMoE, etc.)
    and applies appropriate wrappers to collect comprehensive metrics.
    """

    def __init__(
        self,
        model: nn.Module,
        selection_fn: Optional[Callable] = None,
        warmup_steps: int = 5,
        sampling_rate: int = 1,
        use_cuda_events: bool = True,
    ):
        """
        Initialize profiler for an MoE model.

        Args:
            model: The MoE model to profile
            selection_fn: Optional custom selection function for dynamic k
            warmup_steps: Number of initial steps to skip
            sampling_rate: Collect metrics every Nth step
            use_cuda_events: Use CUDA events for timing (more accurate)
        """
        self.model = model
        self.selection_fn = selection_fn
        self.warmup_steps = warmup_steps
        self.sampling_rate = sampling_rate
        self.use_cuda_events = use_cuda_events

        self.wrappers: List[RouterWrapper] = []
        self.architecture_info = {}

        # Wrap MoE modules
        self._wrap_moe_modules()

    def _wrap_moe_modules(self):
        """
        Scan model and wrap MoE modules with appropriate wrappers.

        Auto-detects architecture and applies correct wrapper type.
        """
        print("Scanning model for MoE modules...")

        for name, module in list(self.model.named_modules()):
            if name == '':
                continue

            # Skip if already wrapped
            if isinstance(module, RouterWrapper):
                continue

            # Try to detect architecture
            try:
                # Pass model config for automatic parameter extraction
                model_config = getattr(self.model, 'config', None)
                handler = detect_architecture(name, module, model_config)

                # Found a supported architecture!
                print(f"Found MoE module: {name}")
                print(f"  Type: {type(module).__name__}")
                print(f"  Architecture: {handler.get_specs()['architecture']}")
                print(f"  Wrapper type: {handler.get_wrapper_type()}")
                print(f"  Num experts: {handler.get_num_experts()}")
                print(f"  Hidden dim: {handler.get_hidden_dim(module)}")
                print(f"  Expert dim: {handler.get_expert_dim(module)}")
                print(f"  Default top-k: {handler.get_default_top_k()}")
                if model_config is not None:
                    print(f"  Config source: model.config (auto-detected)")

                # Create wrapper
                wrapper = RouterWrapper(
                    module=module,
                    handler=handler,
                    selection_fn=self.selection_fn,
                    warmup_steps=self.warmup_steps,
                    sampling_rate=self.sampling_rate,
                    name=name,
                )

                # Replace module with wrapper
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]

                if parent_name:
                    parent = self.model.get_submodule(parent_name)
                else:
                    parent = self.model

                setattr(parent, child_name, wrapper)

                # Move to same device as model
                try:
                    model_device = next(self.model.parameters()).device
                    wrapper.to(model_device)
                except Exception:
                    pass

                self.wrappers.append(wrapper)

                # Store architecture info
                self.architecture_info[name] = handler.get_specs()

            except ValueError:
                # Not a supported MoE module, skip
                continue

        if len(self.wrappers) == 0:
            print("Warning: No MoE modules found in model!")
        else:
            print(f"\nSuccessfully wrapped {len(self.wrappers)} MoE modules")

    def get_metrics_df(self) -> pd.DataFrame:
        """
        Get combined metrics from all wrappers as a DataFrame.

        Returns:
            DataFrame with metrics from all layers
        """
        all_dfs = []

        for i, wrapper in enumerate(self.wrappers):
            df = wrapper.get_metrics_df()
            if len(df) > 0:
                df['layer'] = i
                df['layer_name'] = wrapper.name
                all_dfs.append(df)

        if len(all_dfs) == 0:
            return pd.DataFrame()

        combined_df = pd.concat(all_dfs, ignore_index=True)
        return combined_df

    def get_summary(self) -> dict:
        """
        Get comprehensive summary statistics across all layers.

        Returns:
            Dictionary with aggregated statistics
        """
        df = self.get_metrics_df()

        if len(df) == 0:
            return {"error": "No metrics collected yet"}

        # Aggregate across all layers
        summary = {}

        # FLOPs statistics
        summary['flops_total_mean'] = df['flops_total'].mean()
        summary['flops_total_std'] = df['flops_total'].std()
        summary['flops_total_p50'] = df['flops_total'].median()
        summary['flops_total_p95'] = df['flops_total'].quantile(0.95)
        summary['flops_total_p99'] = df['flops_total'].quantile(0.99)

        # Latency statistics
        summary['latency_mean_ms'] = df['latency_ms'].mean()
        summary['latency_std_ms'] = df['latency_ms'].std()
        summary['latency_p50_ms'] = df['latency_ms'].median()
        summary['latency_p95_ms'] = df['latency_ms'].quantile(0.95)
        summary['latency_p99_ms'] = df['latency_ms'].quantile(0.99)

        # Dynamic k statistics
        summary['k_mean'] = df['k_avg'].mean()
        summary['k_std'] = df['k_avg'].std()
        summary['k_min'] = df['k_avg'].min()
        summary['k_max'] = df['k_avg'].max()

        # Active experts
        summary['active_experts_mean'] = df['active_experts'].mean()
        summary['active_experts_std'] = df['active_experts'].std()

        # Confidence
        summary['router_confidence_mean'] = df['router_conf'].mean()
        summary['router_confidence_std'] = df['router_conf'].std()

        # Overall stats
        summary['total_layers'] = df['layer'].nunique()
        summary['total_steps'] = len(df)

        return summary

    def generate_report(self) -> str:
        """
        Generate a human-readable profiling report.

        Returns:
            Formatted string report
        """
        summary = self.get_summary()

        if 'error' in summary:
            return summary['error']

        report = []
        report.append("=" * 80)
        report.append("MoE Profiling Report")
        report.append("=" * 80)
        report.append("")

        # Architecture info
        report.append("Architecture Information:")
        for layer_name, info in self.architecture_info.items():
            report.append(f"  {layer_name}:")
            report.append(f"    Type: {info['architecture']}")
            report.append(f"    Experts: {info['num_experts']}")
            report.append(f"    Default top-k: {info['default_top_k']}")
            report.append(f"    Wrapper: {info['wrapper_type']}")
        report.append("")

        # Metrics summary
        report.append("Performance Metrics:")
        report.append(f"  Total layers profiled: {summary['total_layers']}")
        report.append(f"  Total steps: {summary['total_steps']}")
        report.append("")

        report.append("FLOPs Statistics:")
        report.append(f"  Mean: {summary['flops_total_mean']:.2e}")
        report.append(f"  Std:  {summary['flops_total_std']:.2e}")
        report.append(f"  P50:  {summary['flops_total_p50']:.2e}")
        report.append(f"  P95:  {summary['flops_total_p95']:.2e}")
        report.append(f"  P99:  {summary['flops_total_p99']:.2e}")
        report.append("")

        report.append("Latency Statistics (ms):")
        report.append(f"  Mean: {summary['latency_mean_ms']:.2f}")
        report.append(f"  Std:  {summary['latency_std_ms']:.2f}")
        report.append(f"  P50:  {summary['latency_p50_ms']:.2f}")
        report.append(f"  P95:  {summary['latency_p95_ms']:.2f}")
        report.append(f"  P99:  {summary['latency_p99_ms']:.2f}")
        report.append("")

        report.append("Dynamic k Statistics:")
        report.append(f"  Mean k: {summary['k_mean']:.2f}")
        report.append(f"  Std k:  {summary['k_std']:.2f}")
        report.append(f"  Min k:  {summary['k_min']:.2f}")
        report.append(f"  Max k:  {summary['k_max']:.2f}")
        report.append("")

        report.append("Expert Utilization:")
        report.append(f"  Mean active experts per layer: {summary['active_experts_mean']:.2f}")
        report.append(f"  Std active experts:  {summary['active_experts_std']:.2f}")
        report.append("")

        report.append("Routing Confidence:")
        report.append(f"  Mean: {summary['router_confidence_mean']:.3f}")
        report.append(f"  Std:  {summary['router_confidence_std']:.3f}")
        report.append("")

        report.append("=" * 80)

        return "\n".join(report)

    def reset_metrics(self):
        """Reset metrics for all wrappers."""
        for wrapper in self.wrappers:
            wrapper.reset_metrics()

    def enable(self):
        """Enable profiling for all wrappers."""
        for wrapper in self.wrappers:
            wrapper.enable()

    def disable(self):
        """Disable profiling for all wrappers."""
        for wrapper in self.wrappers:
            wrapper.disable()

    def get_per_layer_summary(self) -> pd.DataFrame:
        """
        Get summary statistics per layer.

        Returns:
            DataFrame with per-layer statistics
        """
        df = self.get_metrics_df()

        if len(df) == 0:
            return pd.DataFrame()

        # Group by layer and compute statistics
        per_layer = df.groupby('layer_name').agg({
            'flops_total': ['mean', 'std', 'median'],
            'latency_ms': ['mean', 'std', 'median'],
            'k_avg': ['mean', 'std', 'min', 'max'],
            'active_experts': ['mean', 'std'],
            'router_conf': ['mean', 'std'],
        }).reset_index()

        per_layer.columns = ['_'.join(col).strip('_') for col in per_layer.columns.values]

        return per_layer

    def get_k_statistics(self) -> dict:
        """
        Get comprehensive k-value statistics from all wrappers.

        Returns:
            Dictionary with k statistics:
            - k_values: All k values across all layers
            - k_mean, k_std, k_min, k_max: Aggregate statistics
            - per_layer: Dictionary of k statistics per layer
            - distribution: Full k distribution list
        """
        all_k_values = []
        all_k_distributions = []
        per_layer_stats = {}

        for i, wrapper in enumerate(self.wrappers):
            k_vals = wrapper.metrics.k_per_token
            k_dist = wrapper.metrics.k_distribution

            if len(k_vals) > 0:
                all_k_values.extend(k_vals)
                all_k_distributions.extend(k_dist)

                import numpy as np
                k_array = np.array(k_vals)
                per_layer_stats[wrapper.name] = {
                    'mean': float(k_array.mean()),
                    'std': float(k_array.std()),
                    'min': float(k_array.min()),
                    'max': float(k_array.max()),
                    'count': len(k_vals)
                }

        if len(all_k_values) == 0:
            return {
                'error': 'No k statistics collected yet',
                'k_values': [],
                'k_mean': 0,
                'k_std': 0,
                'k_min': 0,
                'k_max': 0,
                'per_layer': {},
                'distribution': []
            }

        import numpy as np
        k_array = np.array(all_k_values)

        return {
            'k_values': all_k_values,
            'k_mean': float(k_array.mean()),
            'k_std': float(k_array.std()),
            'k_min': float(k_array.min()),
            'k_max': float(k_array.max()),
            'per_layer': per_layer_stats,
            'distribution': all_k_distributions,
            'total_samples': len(all_k_values)
        }

    def quick_test(self, prompts: list, tokenizer, max_new_tokens: int = 50, show_progress: bool = True):
        """
        Run quick inference test on multiple prompts to compare routing behavior.

        Args:
            prompts: List of prompt strings to test
            tokenizer: Tokenizer for encoding prompts
            max_new_tokens: Maximum tokens to generate per prompt
            show_progress: Print progress messages

        Returns:
            List of dictionaries with per-prompt statistics
        """
        results = []

        for i, prompt in enumerate(prompts):
            if show_progress:
                print(f"\nTesting prompt {i+1}/{len(prompts)}: \"{prompt[:50]}...\"")

            # Reset metrics for this prompt
            self.reset_metrics()

            # Run inference
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(next(self.model.parameters()).device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )

            # Collect k statistics for this prompt
            k_stats = self.get_k_statistics()

            if 'error' not in k_stats:
                result = {
                    'prompt': prompt,
                    'k_mean': k_stats['k_mean'],
                    'k_std': k_stats['k_std'],
                    'k_min': k_stats['k_min'],
                    'k_max': k_stats['k_max'],
                    'k_values': k_stats['k_values'],
                    'num_samples': k_stats['total_samples'],
                    'per_layer': k_stats['per_layer']
                }
            else:
                result = {
                    'prompt': prompt,
                    'k_mean': 0,
                    'k_std': 0,
                    'k_min': 0,
                    'k_max': 0,
                    'k_values': [],
                    'num_samples': 0,
                    'per_layer': {}
                }

            results.append(result)

            if show_progress:
                if result['num_samples'] > 0:
                    print(f"  ✓ Collected {result['num_samples']} samples")
                    print(f"  K stats: mean={result['k_mean']:.2f}, std={result['k_std']:.2f}, "
                          f"range=[{result['k_min']:.1f}, {result['k_max']:.1f}]")
                else:
                    print(f"  ⚠ No statistics collected!")

        return results

    def compare_routing_behavior(self, results: list, verbose: bool = True):
        """
        Compare routing behavior across multiple prompts to verify dynamic routing.

        Args:
            results: List of results from quick_test()
            verbose: Print detailed comparison

        Returns:
            Dictionary with comparison results
        """
        import numpy as np

        if len(results) < 2:
            return {'error': 'Need at least 2 prompts to compare'}

        comparison = {
            'num_prompts': len(results),
            'all_identical': True,
            'differences_found': []
        }

        if verbose:
            print("\n" + "=" * 80)
            print("ROUTING BEHAVIOR COMPARISON")
            print("=" * 80)

        # Compare each prompt against the first
        ref_k = np.array(results[0]['k_values'])

        for i in range(1, len(results)):
            curr_k = np.array(results[i]['k_values'])

            # Check if k-values are identical
            if len(ref_k) == len(curr_k) and np.allclose(ref_k, curr_k, atol=1e-6):
                if verbose:
                    print(f"\n⚠ Prompt {i+1} has IDENTICAL k-values to Prompt 1")
                    print(f"  First 10 k-values: {curr_k[:10].tolist()}")
            else:
                comparison['all_identical'] = False
                if verbose:
                    print(f"\n✓ Prompt {i+1} has DIFFERENT k-values from Prompt 1")

                    # Show some statistics
                    if len(curr_k) > 0 and len(ref_k) > 0:
                        mean_diff = abs(curr_k.mean() - ref_k.mean())
                        print(f"  Mean k difference: {mean_diff:.3f}")
                        if len(curr_k) == len(ref_k):
                            max_diff = abs(curr_k - ref_k).max()
                            print(f"  Max pointwise difference: {max_diff:.3f}")

                comparison['differences_found'].append(i)

        # Print conclusion
        if verbose:
            print("\n" + "=" * 80)
            if comparison['all_identical']:
                print("❌ PROBLEM: All prompts produce IDENTICAL routing behavior!")
                print("\nThis suggests:")
                print("  - The routing may not be dynamic")
                print("  - There might be caching or determinism issues")
                print("  - The model might not be using the wrapped MoE modules")
            else:
                print("✓ SUCCESS: Different prompts produce DIFFERENT routing behavior")
                print(f"\n{len(comparison['differences_found'])}/{len(results)-1} prompts "
                      "showed unique routing patterns.")
                print("This confirms dynamic k-selection is working correctly!")
            print("=" * 80)

        return comparison

    def print_routing_stats(self):
        """
        Print comprehensive routing statistics in a readable format.

        Similar to the groupmate's statistics display.
        """
        k_stats = self.get_k_statistics()

        if 'error' in k_stats:
            print(k_stats['error'])
            return

        print("\n" + "=" * 80)
        print("ROUTING STATISTICS")
        print("=" * 80)

        print(f"\nTotal samples: {k_stats['total_samples']}")
        print(f"\nAggregate K Statistics:")
        print(f"  Mean k: {k_stats['k_mean']:.2f}")
        print(f"  Std k:  {k_stats['k_std']:.2f}")
        print(f"  Min k:  {k_stats['k_min']:.1f}")
        print(f"  Max k:  {k_stats['k_max']:.1f}")

        if k_stats['per_layer']:
            print(f"\nPer-Layer Statistics:")
            for layer_name, stats in k_stats['per_layer'].items():
                print(f"\n  {layer_name}:")
                print(f"    Mean k: {stats['mean']:.2f}")
                print(f"    Std k:  {stats['std']:.2f}")
                print(f"    Range:  [{stats['min']:.1f}, {stats['max']:.1f}]")
                print(f"    Samples: {stats['count']}")

        # Show first few k-values as example
        if len(k_stats['k_values']) > 0:
            print(f"\nFirst 20 k-values: {k_stats['k_values'][:20]}")

        print("\n" + "=" * 80)
