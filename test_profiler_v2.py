"""
Test script for profiler_v2 with OLMoE

Tests the new architecture-aware profiler with OLMoE model.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Import profiler_v2
from profiler_v2 import MOEProfiler, selectors

print("="*80)
print("Testing profiler_v2 with OLMoE")
print("="*80)
print()

# Load OLMoE model with quantization
print("Loading OLMoE model with 8-bit quantization...")
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

print(f"Model device map: {model.hf_device_map}")
print()

# Test 1: Basic profiling without custom selector
print("="*80)
print("Test 1: Basic Profiling (Default Routing)")
print("="*80)
print()

profiler = MOEProfiler(model, use_cuda_events=True, warmup_steps=0, sampling_rate=1)
print()

# Run inference
prompts = ["Explain the concept of mixture of experts in neural networks."]
inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
inputs = {k: v.to(model.device) for k, v in inputs.items()}

print("Running inference...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id
    )

print("Inference complete!")
print()

# Get results
print("Generating report...")
report = profiler.generate_report()
print(report)
print()

df = profiler.get_metrics_df()
print(f"Generated {len(df)} rows of metrics")
print()
print("First few rows:")
print(df.head())
print()

# Per-layer summary
print("Per-layer summary:")
per_layer = profiler.get_per_layer_summary()
print(per_layer)
print()

# Test 2: With custom selector (kneedle)
print("="*80)
print("Test 2: With Kneedle Selector (Dynamic k)")
print("="*80)
print()

# Reload model (fresh start)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True
)

profiler_kneedle = MOEProfiler(
    model,
    selection_fn=selectors.kneedle_selector,
    use_cuda_events=True,
    warmup_steps=0,
    sampling_rate=1
)
print()

print("Running inference with kneedle selector...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id
    )

print("Inference complete!")
print()

# Get results
print("Generating report...")
report = profiler_kneedle.generate_report()
print(report)
print()

df_kneedle = profiler_kneedle.get_metrics_df()
print(f"Generated {len(df_kneedle)} rows of metrics")
print()

# Compare k distributions
print("="*80)
print("Comparison: Default vs Kneedle")
print("="*80)
print()

print("Default routing:")
print(f"  Mean k: {df['k_avg'].mean():.2f}")
print(f"  Std k:  {df['k_avg'].std():.2f}")
print(f"  Min k:  {df['k_avg'].min():.2f}")
print(f"  Max k:  {df['k_avg'].max():.2f}")
print()

print("Kneedle selector:")
print(f"  Mean k: {df_kneedle['k_avg'].mean():.2f}")
print(f"  Std k:  {df_kneedle['k_avg'].std():.2f}")
print(f"  Min k:  {df_kneedle['k_avg'].min():.2f}")
print(f"  Max k:  {df_kneedle['k_avg'].max():.2f}")
print()

print("="*80)
print("All tests completed successfully!")
print("="*80)
