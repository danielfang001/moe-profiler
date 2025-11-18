"""
Example: Using Routing Comparison Utilities in profiler_v2

Demonstrates how to:
1. Run quick tests on multiple prompts
2. Compare routing behavior across prompts
3. Print comprehensive routing statistics
4. Verify dynamic k-selection is working
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from profiler_v2 import MOEProfiler, selectors

print("="*80)
print("Routing Comparison Example with profiler_v2")
print("="*80)
print()

# Load model
print("Loading OLMoE model...")
model_name = "allenai/OLMoE-1B-7B-0924-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Model loaded!")
print()

# Create profiler with kneedle selector
print("Initializing profiler with kneedle selector...")
profiler = MOEProfiler(
    model,
    selection_fn=selectors.kneedle_selector,
    warmup_steps=0,
    sampling_rate=1
)
print()

# Method 1: Quick test with multiple prompts
print("="*80)
print("Method 1: Quick Test with Multiple Prompts")
print("="*80)

test_prompts = [
    "Hello, how are you?",
    "Explain quantum computing in simple terms.",
    "Write a Python function to calculate fibonacci numbers.",
]

results = profiler.quick_test(test_prompts, tokenizer, max_new_tokens=30)

# Compare routing behavior
comparison = profiler.compare_routing_behavior(results)

# Method 2: Print detailed routing statistics
print("\n" + "="*80)
print("Method 2: Detailed Routing Statistics")
print("="*80)

# Reset and run a longer inference
profiler.reset_metrics()

prompt = "Explain the concept of mixture of experts in neural networks."
inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
inputs = {k: v.to(model.device) for k, v in inputs.items()}

print(f"\nRunning inference on: \"{prompt}\"")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id
    )

# Print comprehensive routing stats
profiler.print_routing_stats()

# Method 3: Get k statistics programmatically
print("\n" + "="*80)
print("Method 3: Programmatic Access to K Statistics")
print("="*80)

k_stats = profiler.get_k_statistics()

print(f"\nTotal samples collected: {k_stats['total_samples']}")
print(f"Overall k statistics:")
print(f"  Mean: {k_stats['k_mean']:.3f}")
print(f"  Std:  {k_stats['k_std']:.3f}")
print(f"  Range: [{k_stats['k_min']:.1f}, {k_stats['k_max']:.1f}]")

print(f"\nNumber of layers: {len(k_stats['per_layer'])}")

# Show per-layer breakdown
print("\nPer-layer k-values:")
for layer_name, stats in list(k_stats['per_layer'].items())[:3]:  # Show first 3 layers
    print(f"  {layer_name}: mean={stats['mean']:.2f}, std={stats['std']:.2f}")

print("\n" + "="*80)
print("Example Complete!")
print("="*80)

# Usage summary
print("\n" + "="*80)
print("USAGE SUMMARY")
print("="*80)
print("""
The profiler_v2 provides three main utilities for routing analysis:

1. profiler.quick_test(prompts, tokenizer)
   - Test multiple prompts and collect k statistics for each
   - Automatically resets metrics between prompts
   - Returns list of per-prompt results

2. profiler.compare_routing_behavior(results)
   - Compare routing across prompts
   - Detect if routing is truly dynamic or identical
   - Useful for debugging

3. profiler.print_routing_stats()
   - Pretty-print comprehensive routing statistics
   - Shows aggregate and per-layer statistics
   - Easy to use for quick inspection

4. profiler.get_k_statistics()
   - Programmatic access to k statistics
   - Returns dictionary with all statistics
   - Useful for custom analysis

These utilities help verify that:
- Dynamic k-selection is working correctly
- Different prompts produce different routing patterns
- The profiler is capturing routing decisions properly
""")
