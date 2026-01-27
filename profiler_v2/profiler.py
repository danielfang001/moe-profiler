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
        enable_multi_gpu: bool = True,
    ):
        """
        Initialize profiler for an MoE model.

        Args:
            model: The MoE model to profile
            selection_fn: Optional custom selection function for dynamic k
            warmup_steps: Number of initial steps to skip
            sampling_rate: Collect metrics every Nth step
            use_cuda_events: Use CUDA events for timing (more accurate)
            enable_multi_gpu: Enable multi-GPU profiling support (auto-detects device_map)
        """
        self.model = model
        self.selection_fn = selection_fn
        self.warmup_steps = warmup_steps
        self.sampling_rate = sampling_rate
        self.use_cuda_events = use_cuda_events
        self.enable_multi_gpu = enable_multi_gpu

        self.wrappers: List[RouterWrapper] = []
        self.architecture_info = {}
        self.device_map = {}  # Maps module names to devices

        # Detect devices being used
        self._detect_device_map()

        # Wrap MoE modules
        self._wrap_moe_modules()

    def _detect_device_map(self):
        """
        Detect which devices modules are on (for multi-GPU support).
        
        Builds a device_map dictionary that tracks which device each module is on.
        This is automatically done for models loaded with device_map='auto'.
        """
        if not self.enable_multi_gpu:
            return

        unique_devices = set()
        for name, module in self.model.named_modules():
            try:
                device = next(module.parameters()).device
                self.device_map[name] = device
                unique_devices.add(str(device))
            except StopIteration:
                # Module has no parameters, skip
                continue

        if len(unique_devices) > 1:
            print(f"Multi-GPU setup detected: {len(unique_devices)} devices")
            print(f"Devices: {sorted(unique_devices)}")
            for device_str in sorted(unique_devices):
                modules_on_device = [name for name, dev in self.device_map.items() 
                                    if str(dev) == device_str]
                print(f"  {device_str}: {len(modules_on_device)} modules")

    def _wrap_moe_modules(self):
        """
        Scan model and wrap MoE modules with appropriate wrappers.

        Auto-detects architecture and applies correct wrapper type.
        Supports models distributed across multiple GPUs via device_map.
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

                # Move wrapper to same device as the wrapped module (for multi-GPU)
                try:
                    if self.enable_multi_gpu and name in self.device_map:
                        module_device = self.device_map[name]
                    else:
                        module_device = next(self.model.parameters()).device
                    wrapper.to(module_device)
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

        # Multi-GPU info
        device_summary = self.get_per_device_summary()
        if device_summary and len(device_summary) > 1:
            report.append("Multi-GPU Information:")
            report.append(f"  Number of devices: {len(device_summary)}")
            for device, stats in sorted(device_summary.items()):
                report.append(f"  {device}:")
                report.append(f"    Layers: {stats['num_layers']}")
                report.append(f"    Steps: {stats['num_steps']}")
                report.append(f"    Avg Latency: {stats['latency_mean_ms']:.2f}ms")
                report.append(f"    Avg Memory: {stats['memory_mean_mb']:.1f}MB")
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

    def enable_logit_capture(self):
        """
        Enable raw router logit capture for all wrappers.

        Use this before inference to save raw router logits for analysis.
        Call save_logits_to_file() after inference to export to pickle.

        Example:
            >>> profiler.enable_logit_capture()
            >>> outputs = model.generate(**inputs, max_new_tokens=50)
            >>> profiler.save_logits_to_file('router_logits.pkl')
        """
        for wrapper in self.wrappers:
            wrapper.save_logits = True
        print(f"Enabled logit capture for {len(self.wrappers)} wrappers")

    def disable_logit_capture(self):
        """Disable raw router logit capture for all wrappers."""
        for wrapper in self.wrappers:
            wrapper.save_logits = False
        print(f"Disabled logit capture for {len(self.wrappers)} wrappers")

    def get_raw_logits(self) -> dict:
        """
        Get all captured raw router logits from all wrappers.

        Returns:
            Dictionary mapping wrapper name to list of logit dictionaries.
            Each logit dict contains:
                - 'logits': Tensor of shape [num_tokens, num_experts]
                - 'shape': Original input shape (batch, seq_len, hidden)
                - 'step': Step number when captured

        Example:
            >>> logits = profiler.get_raw_logits()
            >>> for layer_name, logit_list in logits.items():
            ...     print(f"{layer_name}: {len(logit_list)} captures")
            ...     for capture in logit_list:
            ...         print(f"  Shape: {capture['logits'].shape}")
        """
        logits_dict = {}
        for wrapper in self.wrappers:
            if len(wrapper.raw_logits) > 0:
                logits_dict[wrapper.name] = wrapper.raw_logits
        return logits_dict

    def clear_logits(self):
        """Clear all captured logits from all wrappers."""
        for wrapper in self.wrappers:
            wrapper.raw_logits = []
        print(f"Cleared logits from {len(self.wrappers)} wrappers")

    def get_expert_loads_by_layer(self) -> dict:
        """
        Get expert loads (selection counts) for each layer.

        Returns:
            Dictionary mapping layer name to expert_loads dict.
            {
                'model.layers.0.mlp': {0: 1245, 1: 893, 2: 1102, ...},
                'model.layers.1.mlp': {0: 1100, 1: 950, ...},
                ...
            }
        """
        loads_by_layer = {}
        for wrapper in self.wrappers:
            loads_by_layer[wrapper.name] = dict(wrapper.metrics.expert_loads)
        return loads_by_layer

    def get_k_distribution_by_layer(self) -> dict:
        """
        Get k-value distributions (experts selected per token) for each layer.

        Returns:
            Dictionary mapping layer name to list of k values per forward pass.
            {
                'model.layers.0.mlp': [[5, 8, 3, ...], [6, 7, ...], ...],
                ...
            }
        """
        k_dist_by_layer = {}
        for wrapper in self.wrappers:
            k_dist_by_layer[wrapper.name] = wrapper.metrics.k_distribution
        return k_dist_by_layer

    def save_load_balancing_data(self, filename: str, config_name: str = "unnamed"):
        """
        Save comprehensive load balancing data to a pickle file.

        Args:
            filename: Output file path (e.g., 'load_balancing_top8.pkl')
            config_name: Name for this configuration (e.g., 'top-8', 'elbow-8')

        Saves:
            {
                'config_name': str,
                'architecture_info': dict,
                'expert_loads_by_layer': {layer_name: {expert_id: count}},
                'k_distribution_by_layer': {layer_name: [[k values per token]]},
                'k_stats_by_layer': {layer_name: {'mean': x, 'std': y, ...}},
                'summary': dict with aggregate stats,
                'metrics_df': DataFrame with all per-step metrics
            }
        """
        import pickle
        import os
        import numpy as np

        expert_loads = self.get_expert_loads_by_layer()
        k_dist = self.get_k_distribution_by_layer()

        # Compute k stats per layer
        k_stats_by_layer = {}
        for layer_name, k_lists in k_dist.items():
            all_ks = []
            for k_list in k_lists:
                all_ks.extend(k_list)
            if len(all_ks) > 0:
                k_arr = np.array(all_ks)
                k_stats_by_layer[layer_name] = {
                    'mean': float(k_arr.mean()),
                    'std': float(k_arr.std()),
                    'min': float(k_arr.min()),
                    'max': float(k_arr.max()),
                    'total_tokens': len(all_ks),
                }
            else:
                k_stats_by_layer[layer_name] = {
                    'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'total_tokens': 0
                }

        # Get metrics dataframe
        df = self.get_metrics_df()

        save_data = {
            'config_name': config_name,
            'architecture_info': self.architecture_info,
            'expert_loads_by_layer': expert_loads,
            'k_distribution_by_layer': k_dist,
            'k_stats_by_layer': k_stats_by_layer,
            'summary': self.get_summary(),
            'metrics_df': df.to_dict() if len(df) > 0 else {},
        }

        with open(filename, 'wb') as f:
            pickle.dump(save_data, f)

        file_size_mb = os.path.getsize(filename) / 1024 / 1024
        total_loads = sum(sum(loads.values()) for loads in expert_loads.values())

        print(f"\n Saved load balancing data to: {filename}")
        print(f"  Config: {config_name}")
        print(f"  File size: {file_size_mb:.2f} MB")
        print(f"  Layers: {len(expert_loads)}")
        print(f"  Total expert selections: {total_loads}")
        print(f"\nTo load:")
        print(f"  import pickle")
        print(f"  with open('{filename}', 'rb') as f:")
        print(f"      data = pickle.load(f)")

    def save_logits_to_file(self, filename: str):
        """
        Save all captured router logits to a pickle file.

        Args:
            filename: Output file path (e.g., 'router_logits.pkl')

        The saved file contains:
            {
                'model_config': dict,  # Model configuration
                'num_wrappers': int,   # Number of MoE layers
                'router_logits': {     # Per-layer logits
                    'layer_name': [
                        {
                            'logits': Tensor[num_tokens, num_experts],
                            'shape': (batch, seq_len, hidden),
                            'step': int
                        },
                        ...
                    ],
                    ...
                }
            }

        Example:
            >>> profiler.enable_logit_capture()
            >>> outputs = model.generate(**inputs, max_new_tokens=50)
            >>> profiler.save_logits_to_file('olmoe_router_logits.pkl')
            >>>
            >>> # Later, load and analyze:
            >>> import pickle
            >>> with open('olmoe_router_logits.pkl', 'rb') as f:
            ...     data = pickle.load(f)
            >>> for layer, logits in data['router_logits'].items():
            ...     print(f"{layer}: {len(logits)} forward passes")
        """
        import pickle
        import os

        logits_dict = self.get_raw_logits()

        if not logits_dict:
            print("⚠️  No logits captured! Enable with enable_logit_capture() first.")
            return

        # Prepare save data
        save_data = {
            'model_config': {
                'architecture': self.architecture_info,
                'num_wrappers': len(self.wrappers),
            },
            'router_logits': logits_dict,
        }

        # Save to file
        with open(filename, 'wb') as f:
            pickle.dump(save_data, f)

        file_size_mb = os.path.getsize(filename) / 1024 / 1024
        total_captures = sum(len(logits) for logits in logits_dict.values())

        print(f"\n✓ Saved router logits to: {filename}")
        print(f"  File size: {file_size_mb:.2f} MB")
        print(f"  Layers: {len(logits_dict)}")
        print(f"  Total captures: {total_captures}")
        print(f"\nTo load:")
        print(f"  import pickle")
        print(f"  with open('{filename}', 'rb') as f:")
        print(f"      data = pickle.load(f)")

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

    def get_per_device_summary(self) -> dict:
        """
        Get summary statistics per device (for multi-GPU profiling).

        Returns:
            Dictionary mapping device strings to per-device statistics
        """
        df = self.get_metrics_df()

        if len(df) == 0:
            return {}

        device_summaries = {}

        # Group by device if multi-GPU
        if 'device' in df.columns:
            for device in df['device'].unique():
                device_df = df[df['device'] == device]

                if len(device_df) > 0:
                    device_summaries[device] = {
                        'num_layers': device_df['layer_name'].nunique(),
                        'num_steps': len(device_df),
                        'flops_total_mean': device_df['flops_total'].mean(),
                        'flops_total_std': device_df['flops_total'].std(),
                        'latency_mean_ms': device_df['latency_ms'].mean(),
                        'latency_std_ms': device_df['latency_ms'].std(),
                        'latency_p95_ms': device_df['latency_ms'].quantile(0.95),
                        'latency_p99_ms': device_df['latency_ms'].quantile(0.99),
                        'k_mean': device_df['k_avg'].mean(),
                        'k_std': device_df['k_avg'].std(),
                        'memory_mean_mb': device_df['memory_mb'].mean() if 'memory_mb' in device_df else 0,
                        'memory_max_mb': device_df['memory_mb'].max() if 'memory_mb' in device_df else 0,
                    }

        return device_summaries

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

    def print_device_stats(self):
        """
        Print per-device profiling statistics (for multi-GPU setups).

        Useful for identifying load imbalance across devices.
        """
        device_summary = self.get_per_device_summary()

        if not device_summary:
            print("No multi-GPU data available")
            return

        print("\n" + "=" * 80)
        print("PER-DEVICE STATISTICS")
        print("=" * 80)

        print(f"\nTotal devices: {len(device_summary)}\n")

        for device, stats in sorted(device_summary.items()):
            print(f"Device: {device}")
            print(f"  Layers: {stats['num_layers']}")
            print(f"  Steps: {stats['num_steps']}")
            print(f"  FLOPs (mean): {stats['flops_total_mean']:.2e}")
            print(f"  Latency (mean): {stats['latency_mean_ms']:.2f}ms")
            print(f"  Latency (p95):  {stats['latency_p95_ms']:.2f}ms")
            print(f"  Latency (p99):  {stats['latency_p99_ms']:.2f}ms")
            print(f"  Memory (mean):  {stats['memory_mean_mb']:.1f}MB")
            print(f"  Memory (max):   {stats['memory_max_mb']:.1f}MB")
            print(f"  K (mean):       {stats['k_mean']:.2f}")
            print()

        print("=" * 80)

    def set_selection_fn(self, selection_fn: Callable):
        """
        Apply a selection function to all existing wrappers.

        Use this to switch selection strategies after the profiler has been created.

        Args:
            selection_fn: Selection function to apply (e.g., kneedle_selector)

        Example:
            >>> from profiler_v2.selectors import kneedle_selector
            >>> profiler.set_selection_fn(kneedle_selector)
        """
        for wrapper in self.wrappers:
            wrapper.selection_fn = selection_fn
        self.selection_fn = selection_fn
        print(f"Applied selection function to {len(self.wrappers)} wrappers")

    def remove_selection_fn(self):
        """
        Remove selection function from all wrappers (revert to default routing).

        Use this to test baseline routing behavior.

        Example:
            >>> profiler.remove_selection_fn()
        """
        for wrapper in self.wrappers:
            wrapper.selection_fn = None
        self.selection_fn = None
        print(f"Removed selection function from {len(self.wrappers)} wrappers")
