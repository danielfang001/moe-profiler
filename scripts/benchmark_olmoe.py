import sys
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

def arc_easy():
    dataset = acc_bench.load_benchmark("arc_easy", split="test")

    kneedle_k8 = partial(kneedle_selector, k_max=8)
    acc_bench.run_evaluation("kneedle_k8", dataset, "arc_easy", selection_fn=kneedle_k8)
    # acc_bench.run_evaluation_with_analysis("kneedle_k8", dataset, "arc_easy", selection_fn=kneedle_k8)
    # acc_bench.save_analysis_to_file("kneedle_k8", "arc_easy_k8_analysis.pkl")

def arc_challenge():
    dataset = acc_bench.load_benchmark("arc_challenge", split="test")

    kneedle_k8 = partial(kneedle_selector, k_max=8)
    acc_bench.run_evaluation("kneedle_k8", dataset, "arc_challenge", selection_fn=kneedle_k8)
    # acc_bench.run_evaluation_with_analysis("kneedle_k8", dataset, "arc_challenge", selection_fn=kneedle_k8)
    # acc_bench.save_analysis_to_file("kneedle_k8", "arc_challenge_k8_analysis.pkl")

def mmlu_8():
    dataset = acc_bench.load_benchmark("mmlu", mmlu_mode="full")

    kneedle_k8 = partial(kneedle_selector, k_max=8)
    acc_bench.run_evaluation("kneedle_k8", dataset, "mmlu", selection_fn=kneedle_k8)
    # acc_bench.run_evaluation_with_analysis("kneedle_k8", dataset, "mmlu", selection_fn=kneedle_k8)
    # acc_bench.save_analysis_to_file("kneedle_k8", "mmlu_k8_analysis.pkl")

def mmlu_16():
    dataset = acc_bench.load_benchmark("mmlu", mmlu_mode="full")

    kneedle_k16 = partial(kneedle_selector, k_max=16)
    acc_bench.run_evaluation("kneedle_k16", dataset, "mmlu", selection_fn=kneedle_k16)
    # acc_bench.run_evaluation_with_analysis("kneedle_k16", dataset, "mmlu", selection_fn=kneedle_k16)
    # acc_bench.save_analysis_to_file("kneedle_k16", "mmlu_k16_analysis.pkl")

def mmlu_baseline():
    dataset = acc_bench.load_benchmark("mmlu", mmlu_mode="full")

    acc_bench.run_evaluation("baseline", dataset, "mmlu", selection_fn=None)
    # acc_bench.run_evaluation_with_analysis("baseline", dataset, "mmlu", selection_fn=None)
    # acc_bench.save_analysis_to_file("baseline", "mmlu_baseline_analysis.pkl")

mmlu_8()

######################################################################

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