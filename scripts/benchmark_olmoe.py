import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from profiler_v2 import MOEProfiler

from transformers import OlmoeForCausalLM, AutoTokenizer
import torch

from profiler_v2 import AccuracyBenchmark
from functools import partial
from profiler_v2.selectors import kneedle_selector

model = OlmoeForCausalLM.from_pretrained(
    "allenai/OLMoE-1B-7B-0924-Instruct",
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0924-Instruct")

profiler = MOEProfiler(model)

# Create benchmark
acc_bench = AccuracyBenchmark(model, tokenizer, profiler)

def run_benchmark(benchmark_name, selection_fn_name, **selection_params):
    """Run a benchmark with specified selection function."""
    dataset = None
    
    if benchmark_name == "arc_easy":
        dataset = acc_bench.load_benchmark("arc_easy", split="test")
    elif benchmark_name == "arc_challenge":
        dataset = acc_bench.load_benchmark("arc_challenge", split="test")
    elif benchmark_name == "mmlu":
        dataset = acc_bench.load_benchmark("mmlu", mmlu_mode="full")
    elif benchmark_name == "hellaswag":
        dataset = acc_bench.load_benchmark("hellaswag", split="test")
    elif benchmark_name == "piqa":
        dataset = acc_bench.load_benchmark("piqa", split="test")
    elif benchmark_name == "winogrande":
        dataset = acc_bench.load_benchmark("winogrande", split="validation")
    else:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")
    
    # Determine selection function
    if selection_fn_name == "baseline":
        selection_fn = None
        run_name = "baseline"
    elif selection_fn_name == "kneedle":
        k_max = selection_params.get("k_max", 8)
        selection_fn = partial(kneedle_selector, k_max=k_max)
        run_name = f"kneedle_k{k_max}"
    else:
        raise ValueError(f"Unknown selection function: {selection_fn_name}")
    
    acc_bench.run_evaluation(run_name, dataset, benchmark_name, selection_fn=selection_fn)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OLMoE benchmark evaluations")
    parser.add_argument("--benchmark", default="mmlu", 
                        choices=["arc_easy", "arc_challenge", "mmlu"],
                        help="Benchmark to run (default: mmlu)")
    parser.add_argument("--selector", default="kneedle",
                        choices=["baseline", "kneedle"],
                        help="Selection function to use (default: kneedle)")
    parser.add_argument("--k-max", type=int, default=8,
                        help="Maximum k value for kneedle selector (default: 8)")
    
    args = parser.parse_args()
    
    # Run the benchmark
    run_benchmark(args.benchmark, args.selector, k_max=args.k_max)

    # Display device mapping for MoE gates
    print("=" * 80)
    print("GATE DEVICE MAPPING")
    print("=" * 80)
    for layer_name, device_str in sorted(profiler.device_map.items()):
        if 'gate' in layer_name.lower() and 'gate_proj' not in layer_name.lower():
            print(f"  {layer_name:50s} → {device_str}")
    print("=" * 80)
    print()

    # Print comprehensive report
    print(profiler.generate_report())

    # Print per-device statistics
    print()
    profiler.print_device_stats()

    # Get per-device metrics programmatically
    device_summary = profiler.get_per_device_summary()
    print("\n" + "=" * 80)
    print("PER-DEVICE SUMMARY (Programmatic Access)")
    print("=" * 80)
    for device, stats in sorted(device_summary.items()):
        print(f"\n{device}:")
        print(f"  Avg Latency: {stats['latency_mean_ms']:.2f}ms (P95: {stats['latency_p95_ms']:.2f}ms)")
        print(f"  Avg Memory:  {stats['memory_mean_mb']:.1f}MB (Max: {stats['memory_max_mb']:.1f}MB)")
        print(f"  Avg FLOPs:   {stats['flops_total_mean']:.2e}")
        print(f"  Avg K:       {stats['k_mean']:.2f}")

    # Export full metrics to DataFrame for further analysis
    df = profiler.get_metrics_df()
    print(f"\n\nTotal metrics collected: {len(df)} samples across {df['device'].nunique()} devices")
    print(f"Columns: {', '.join(df.columns)}")

    # Save to CSV for offline analysis
    csv_file = "moe_profiling_results.csv"
    df.to_csv(csv_file, index=False)
    print(f"Saved metrics to: {csv_file}")