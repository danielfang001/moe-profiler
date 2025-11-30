"""
Example: Multi-GPU Profiling with MoE Profiler

This script demonstrates how to profile MoE models distributed across multiple GPUs.
Works with models loaded via device_map='auto' (supported by transformers).

Requirements:
    - transformers
    - torch
    - accelerate (for device_map='auto')

Example usage:
    python example_multi_gpu.py
"""

import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent directory to path to import profiler_v2
sys.path.insert(0, str(Path(__file__).parent.parent))
from profiler_v2 import MOEProfiler, selectors


def main():
    # Model selection
    model_name = "allenai/OLMoE-1B-7B-0924-Instruct"
    
    print(f"Loading model: {model_name}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print()

    # Load model with automatic device distribution
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",  # Automatically distributes across GPUs
        torch_dtype=torch.float16,  # Optional: use half precision for memory efficiency
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create profiler (auto-detects multi-GPU setup)
    print("Creating profiler...")
    profiler = MOEProfiler(model, selection_fn=selectors.kneedle_selector, enable_multi_gpu=True)
    print()

    # Display device mapping for MoE gates
    print("=" * 80)
    print("GATE DEVICE MAPPING")
    print("=" * 80)
    for layer_name, device_str in sorted(profiler.device_map.items()):
        if 'gate' in layer_name.lower() and 'gate_proj' not in layer_name.lower():
            print(f"  {layer_name:50s} → {device_str}")
    print("=" * 80)
    print()

    # Test prompts to profile
    test_prompts = [
        "Hello, how are you?",
        "Explain quantum computing in simple terms.",
        "Write a Python function to sort a list.",
    ]

    # Run profiling on multiple prompts
    print("Running inference and collecting metrics...")
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n[{i}/{len(test_prompts)}] Processing: {prompt[:50]}...")
        
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        )
        
        # Move inputs to first GPU (model handles distribution)
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate output
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        print(f"  Generated {len(outputs[0])} tokens")

    print("\n" + "=" * 80)
    print("PROFILING RESULTS")
    print("=" * 80)

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


if __name__ == "__main__":
    main()
