"""
Benchmark Runner for MoE Profiler v2

Provides easy comparison between different routing configurations:
- Baseline (default routing)
- Custom selectors (kneedle, confidence-based, etc.)
"""

import torch
import pandas as pd
from typing import Callable, List, Optional, Dict, Any
from .profiler import MOEProfiler


class MoEBenchmark:
    """
    Run and compare different MoE routing configurations.

    Automatically tests:
    - Baseline (default routing)
    - Custom selectors (kneedle, etc.)

    And compares:
    - FLOPs (total, router, expert)
    - Latency
    - K-values (mean, std, distribution)
    - Expert utilization
    - Routing confidence
    """

    def __init__(self, model, tokenizer, profiler: Optional[MOEProfiler] = None):
        """
        Initialize benchmark runner.

        Args:
            model: The MoE model to benchmark
            tokenizer: Tokenizer for the model
            profiler: Optional existing profiler (if already wrapped)
        """
        self.model = model
        self.tokenizer = tokenizer

        # Use existing profiler or create new one
        if profiler is not None:
            self.profiler = profiler
            print("Using existing profiler (modules already wrapped)")
        else:
            print("Creating new profiler...")
            self.profiler = MOEProfiler(model)

        self.results = {}
        self.configs_run = []

    def run_config(
        self,
        name: str,
        test_prompts: List[str],
        selection_fn: Optional[Callable] = None,
        max_new_tokens: int = 50,
        **generate_kwargs
    ):
        """
        Run a single configuration and collect metrics.

        Args:
            name: Name for this configuration (e.g., "baseline", "kneedle")
            test_prompts: List of prompts to test
            selection_fn: Selection function to apply (None for baseline)
            max_new_tokens: Max tokens to generate per prompt
            **generate_kwargs: Additional arguments for model.generate()

        Returns:
            Dictionary with results for this configuration
        """
        print(f"\n{'='*80}")
        print(f"Running config: {name}")
        print(f"{'='*80}")

        # Apply or remove selection function
        if selection_fn is not None:
            print(f"Applying custom selector...")
            self.profiler.set_selection_fn(selection_fn)
        else:
            print(f"Using baseline routing (no selector)...")
            self.profiler.remove_selection_fn()

        # Reset metrics
        self.profiler.reset_metrics()

        # Run inference on all prompts
        all_outputs = []
        for i, prompt in enumerate(test_prompts):
            print(f"\nPrompt {i+1}/{len(test_prompts)}: \"{prompt[:50]}...\"")

            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(next(self.model.parameters()).device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    **generate_kwargs
                )

            all_outputs.append(outputs)
            print(f"  ✓ Generated {outputs.shape[1]} tokens")

        # Collect metrics
        print(f"\nCollecting metrics for config: {name}")

        df = self.profiler.get_metrics_df()
        summary = self.profiler.get_summary()
        k_stats = self.profiler.get_k_statistics()

        # Store results
        result = {
            'name': name,
            'metrics_df': df,
            'summary': summary,
            'k_stats': k_stats,
            'num_prompts': len(test_prompts),
            'selection_fn': selection_fn.__name__ if selection_fn else 'baseline',
            'outputs': all_outputs
        }

        self.results[name] = result
        self.configs_run.append(name)

        # Print quick summary
        if 'error' not in summary:
            print(f"\n--- Quick Summary for {name} ---")
            print(f"  Total steps:     {summary['total_steps']}")
            print(f"  Mean FLOPs:      {summary['flops_total_mean']:.2e}")
            print(f"  Mean latency:    {summary['latency_mean_ms']:.2f} ms")
            print(f"  Mean k:          {summary['k_mean']:.2f} ± {summary['k_std']:.2f}")
            print(f"  K range:         [{summary['k_min']:.1f}, {summary['k_max']:.1f}]")

        return result

    def run_baseline(self, test_prompts: List[str], max_new_tokens: int = 50, **kwargs):
        """
        Run baseline configuration (default routing).

        Args:
            test_prompts: List of prompts to test
            max_new_tokens: Max tokens to generate
            **kwargs: Additional generate arguments
        """
        return self.run_config(
            name="baseline",
            test_prompts=test_prompts,
            selection_fn=None,
            max_new_tokens=max_new_tokens,
            **kwargs
        )

    def run_with_selector(
        self,
        name: str,
        selector: Callable,
        test_prompts: List[str],
        max_new_tokens: int = 50,
        **kwargs
    ):
        """
        Run configuration with a custom selector.

        Args:
            name: Name for this config (e.g., "kneedle")
            selector: Selection function
            test_prompts: List of prompts to test
            max_new_tokens: Max tokens to generate
            **kwargs: Additional generate arguments
        """
        return self.run_config(
            name=name,
            test_prompts=test_prompts,
            selection_fn=selector,
            max_new_tokens=max_new_tokens,
            **kwargs
        )

    def compare(self, save_csv: bool = True) -> pd.DataFrame:
        """
        Compare all configurations that have been run.

        Args:
            save_csv: Whether to save comparison to CSV

        Returns:
            DataFrame with comparison metrics
        """
        if len(self.results) < 2:
            print("Need at least 2 configurations to compare!")
            if len(self.results) == 1:
                print(f"Only have: {list(self.results.keys())}")
            return pd.DataFrame()

        print(f"\n{'='*80}")
        print(f"COMPARING {len(self.results)} CONFIGURATIONS")
        print(f"{'='*80}")

        comparison_rows = []

        for config_name, result in self.results.items():
            summary = result['summary']
            k_stats = result['k_stats']

            if 'error' in summary or 'error' in k_stats:
                continue

            row = {
                'config': config_name,
                'selector': result['selection_fn'],

                # FLOPs
                'flops_total_mean': summary['flops_total_mean'],
                'flops_total_std': summary['flops_total_std'],

                # Latency
                'latency_mean_ms': summary['latency_mean_ms'],
                'latency_std_ms': summary['latency_std_ms'],

                # K statistics
                'k_mean': summary['k_mean'],
                'k_std': summary['k_std'],
                'k_min': summary['k_min'],
                'k_max': summary['k_max'],

                # Expert utilization
                'active_experts_mean': summary['active_experts_mean'],

                # Confidence
                'router_confidence_mean': summary['router_confidence_mean'],

                # Metadata
                'total_steps': summary['total_steps'],
                'num_prompts': result['num_prompts']
            }

            comparison_rows.append(row)

        comparison_df = pd.DataFrame(comparison_rows)

        # Calculate relative improvements vs baseline
        if 'baseline' in self.results:
            baseline_idx = comparison_df[comparison_df['config'] == 'baseline'].index[0]
            baseline_flops = comparison_df.loc[baseline_idx, 'flops_total_mean']
            baseline_latency = comparison_df.loc[baseline_idx, 'latency_mean_ms']
            baseline_k = comparison_df.loc[baseline_idx, 'k_mean']

            comparison_df['flops_reduction_%'] = 100 * (1 - comparison_df['flops_total_mean'] / baseline_flops)
            comparison_df['latency_reduction_%'] = 100 * (1 - comparison_df['latency_mean_ms'] / baseline_latency)
            comparison_df['k_reduction'] = baseline_k - comparison_df['k_mean']

        # Print comparison
        print("\n" + "="*80)
        print("COMPARISON RESULTS")
        print("="*80)
        print(comparison_df.to_string(index=False))
        print("="*80)

        # Save to CSV
        if save_csv:
            comparison_df.to_csv('moe_benchmark_comparison.csv', index=False)
            print(f"\n✓ Saved comparison to: moe_benchmark_comparison.csv")

        # Print highlights
        if 'baseline' in self.results and len(comparison_df) > 1:
            print("\n" + "="*80)
            print("KEY INSIGHTS")
            print("="*80)

            for idx, row in comparison_df.iterrows():
                if row['config'] == 'baseline':
                    continue

                print(f"\n{row['config']} vs baseline:")
                print(f"  FLOPs:    {row['flops_reduction_%']:+.1f}% "
                      f"({'saved' if row['flops_reduction_%'] > 0 else 'increased'})")
                print(f"  Latency:  {row['latency_reduction_%']:+.1f}% "
                      f"({'faster' if row['latency_reduction_%'] > 0 else 'slower'})")
                print(f"  Mean k:   {row['k_reduction']:+.2f} experts/token "
                      f"({'reduced' if row['k_reduction'] > 0 else 'increased'})")

        return comparison_df

    def save_all_metrics(self, prefix: str = "benchmark"):
        """
        Save detailed metrics for all configurations.

        Args:
            prefix: Prefix for output files
        """
        for config_name, result in self.results.items():
            df = result['metrics_df']
            if len(df) > 0:
                filename = f"{prefix}_{config_name}_metrics.csv"
                df.to_csv(filename, index=False)
                print(f"Saved {config_name} metrics to: {filename}")

    def print_detailed_comparison(self):
        """Print detailed per-layer comparison."""
        if len(self.results) < 2:
            print("Need at least 2 configurations to compare!")
            return

        print(f"\n{'='*80}")
        print("DETAILED PER-LAYER COMPARISON")
        print(f"{'='*80}")

        for config_name, result in self.results.items():
            print(f"\n--- {config_name} ---")

            df = result['metrics_df']
            if len(df) > 0:
                per_layer = df.groupby('layer_name').agg({
                    'flops_total': 'mean',
                    'latency_ms': 'mean',
                    'k_avg': ['mean', 'std'],
                    'active_experts': 'mean'
                })
                print(per_layer.to_string())


def quick_benchmark(
    model,
    tokenizer,
    test_prompts: List[str],
    selectors: Dict[str, Optional[Callable]] = None,
    profiler: Optional[MOEProfiler] = None,
    max_new_tokens: int = 50
) -> pd.DataFrame:
    """
    Quick one-liner to run benchmark with multiple selectors.

    Args:
        model: MoE model
        tokenizer: Tokenizer
        test_prompts: List of prompts to test
        selectors: Dict of {name: selector_fn}, e.g., {"kneedle": kneedle_selector}
        profiler: Optional existing profiler
        max_new_tokens: Max tokens to generate

    Returns:
        Comparison DataFrame

    Example:
        >>> from profiler_v2 import quick_benchmark
        >>> from profiler_v2.selectors import kneedle_selector
        >>>
        >>> comparison = quick_benchmark(
        ...     model, tokenizer,
        ...     test_prompts=["What is AI?", "Explain quantum physics"],
        ...     selectors={"kneedle": kneedle_selector}
        ... )
    """
    benchmark = MoEBenchmark(model, tokenizer, profiler)

    # Run baseline
    benchmark.run_baseline(test_prompts, max_new_tokens=max_new_tokens)

    # Run custom selectors
    if selectors:
        for name, selector in selectors.items():
            benchmark.run_with_selector(
                name=name,
                selector=selector,
                test_prompts=test_prompts,
                max_new_tokens=max_new_tokens
            )

    # Compare
    return benchmark.compare()
