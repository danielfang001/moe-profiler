# Using MoE Profiler in Google Colab

A step-by-step guide for profiling MoE models in Google Colab notebooks.

---

## Quick Start (5 minutes)

### 1. Setup Environment

```python
# Install dependencies
!pip install torch transformers pandas numpy accelerate -q

# Upload your profiler file or clone from repo
# Option A: Upload main.py to Colab
# Option B: Clone from git
!git clone https://github.com/your-repo/moe-profiler.git
import sys
sys.path.append('/content/moe-profiler')

# Import the profiler
from profiler import MoEProfiler, SimpleRouterWrapper, quick_profile
```

### 2. Load Your MoE Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Example: Mixtral-8x7B (requires GPU with sufficient memory)
model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"

# Load with device_map for automatic GPU placement
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",           # Automatic GPU placement
    torch_dtype=torch.float16,   # Use FP16 to save memory
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
```

### 3. Profile Your Model

```python
# Create profiler (automatically wraps all routers)
profiler = MoEProfiler(model)

# Start profiling
profiler.start()

# Run some inference
test_prompts = [
    "What is machine learning?",
    "Explain quantum computing in simple terms.",
    "How does photosynthesis work?"
]

for prompt in test_prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50)

# Stop profiling
profiler.stop()

# View results
profiler.print_summary()
```

---

## Advanced Usage

### Custom Router Middleware

If you have custom router logic with dynamic expert selection:

```python
from profiler import SimpleRouterWrapper

# Manually configure wrapper for specific layer
layer_idx = 0
original_router = model.model.layers[layer_idx].block_sparse_moe.gate

# Create wrapper with custom settings
custom_wrapper = SimpleRouterWrapper(
    original_router=original_router,
    num_experts=8,
    expert_dim=14336,      # Mixtral FFN dimension
    hidden_dim=4096,       # Mixtral hidden dimension
    warmup_steps=5,        # Skip first 5 steps
    sampling_rate=1        # Profile every step
)

# Replace the router
model.model.layers[layer_idx].block_sparse_moe.gate = custom_wrapper

# Run inference
profiler.start()
# ... your inference code ...
profiler.stop()

# Get detailed stats
summary = custom_wrapper.get_summary()
print(f"Average k per token: {summary['k_mean']:.2f}")
print(f"Load balance Gini: {summary['expert_load_gini']:.3f}")
print(f"P99 latency: {summary['latency_p99_ms']:.2f} ms")
```

### Export to CSV and Download

```python
# Save metrics to CSV
profiler.save_csv('mixtral_baseline.csv')

# Download the file (in Colab)
from google.colab import files
files.download('mixtral_baseline.csv')
```

### Visualize Results in Notebook

```python
import matplotlib.pyplot as plt
import pandas as pd

# Get metrics as DataFrame
metrics = profiler.get_metrics()
df = metrics['layer_0'].to_df()

# Plot FLOPs over time
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(df['flops_total'], label='Total')
plt.plot(df['flops_router'], label='Router')
plt.plot(df['flops_expert'], label='Expert')
plt.xlabel('Step')
plt.ylabel('FLOPs per Token')
plt.legend()
plt.title('FLOPs Distribution')

# Plot latency
plt.subplot(1, 3, 2)
plt.plot(df['latency_ms'])
plt.xlabel('Step')
plt.ylabel('Latency (ms)')
plt.title('Router Latency')

# Plot k distribution
plt.subplot(1, 3, 3)
plt.plot(df['k_avg'])
plt.xlabel('Step')
plt.ylabel('Average k')
plt.title('Active Experts per Token')

plt.tight_layout()
plt.show()
```

### Expert Load Heatmap

```python
import seaborn as sns

# Get expert loads
wrapper = profiler.wrappers[0]
expert_loads = list(wrapper.metrics.expert_loads.values())

# Plot heatmap
plt.figure(figsize=(10, 2))
sns.heatmap([expert_loads],
            annot=True,
            fmt='d',
            cmap='YlOrRd',
            xticklabels=[f'E{i}' for i in range(len(expert_loads))],
            yticklabels=['Load'],
            cbar_kws={'label': 'Token Count'})
plt.title('Expert Load Distribution')
plt.show()

# Calculate balance metrics
print(f"Gini Coefficient: {wrapper._calculate_gini_coefficient(expert_loads):.3f}")
print(f"Entropy (normalized): {wrapper._calculate_entropy(expert_loads):.3f}")
print(f"CV: {wrapper._calculate_cv(expert_loads):.3f}")
```

---

## Benchmarking Different Configurations

### Compare Baseline vs Custom Middleware

```python
from profiler import SimpleBenchmark

# Test texts
test_texts = [
    "Explain the theory of relativity",
    "What are the benefits of renewable energy?",
    "How do neural networks learn?"
]

# Create benchmark runner
benchmark = SimpleBenchmark(model, tokenizer)

# Run baseline
print("Running baseline...")
benchmark.run_config("baseline", test_texts)

# Define middleware modification
def apply_custom_middleware(model):
    """
    Example: Custom router that uses confidence threshold
    """
    for layer in model.model.layers:
        if hasattr(layer, 'block_sparse_moe'):
            original_gate = layer.block_sparse_moe.gate

            # Wrap with custom logic (pseudo-code)
            # You would implement your YourRouterMiddleware here
            layer.block_sparse_moe.gate = SimpleRouterWrapper(
                original_gate,
                num_experts=8,
                warmup_steps=3
            )

# Run with middleware
print("Running with custom middleware...")
benchmark.run_config("custom_middleware", test_texts, apply_custom_middleware)

# Compare results
comparison = benchmark.compare()
print("\nComparison:")
print(comparison)

# Download comparison
comparison.to_csv('comparison.csv')
files.download('comparison.csv')
```

---

## Memory-Efficient Profiling (Large Models)

For models that barely fit in GPU memory:

```python
# Reduce profiling overhead with sampling
profiler = MoEProfiler(model)

# Manually configure wrappers for lower overhead
for wrapper in profiler.wrappers:
    wrapper.warmup_steps = 10      # Skip more warmup
    wrapper.sampling_rate = 10     # Only profile every 10th step

profiler.start()

# Run inference with gradient checkpointing
model.gradient_checkpointing_enable()

# Your inference code here
# ...

profiler.stop()
profiler.print_summary()
```

---

## One-Liner Quick Profile

For rapid testing:

```python
# Quick profile function (defined in main.py)
profiler = quick_profile(
    model,
    tokenizer,
    text="What is the meaning of life?",
    num_iterations=10
)

# Results automatically printed and saved to quick_profile.csv
files.download('quick_profile.csv')
```

---

## Multi-GPU Setup (T4/V100/A100)

If using multiple GPUs in Colab:

```python
# Check available GPUs
!nvidia-smi

import torch
print(f"Available GPUs: {torch.cuda.device_count()}")

# Load model with device_map for multi-GPU
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="balanced",        # Balance across GPUs
    torch_dtype=torch.float16
)

# Profile will automatically track which GPU each layer is on
profiler = MoEProfiler(model)
profiler.start()

# Your inference code
# ...

profiler.stop()

# View per-layer device placement
for i, wrapper in enumerate(profiler.wrappers):
    print(f"Layer {i}: {wrapper.metrics.device}")
```

---

## Tracking k Distribution Over Time

For dynamic expert selection analysis:

```python
profiler = MoEProfiler(model)
profiler.start()

# Run inference
# ...

profiler.stop()

# Analyze k distribution
wrapper = profiler.wrappers[0]  # First layer
k_dist = wrapper.metrics.k_distribution

# Flatten all k values
all_k_values = [k for step_k_list in k_dist for k in step_k_list]

# Plot histogram
plt.figure(figsize=(10, 5))
plt.hist(all_k_values, bins=range(0, 10), alpha=0.7, edgecolor='black')
plt.xlabel('Number of Active Experts (k)')
plt.ylabel('Token Count')
plt.title('Distribution of k Across All Tokens')
plt.grid(axis='y', alpha=0.3)
plt.show()

# Statistics
print(f"Min k: {min(all_k_values)}")
print(f"Max k: {max(all_k_values)}")
print(f"Mean k: {sum(all_k_values)/len(all_k_values):.2f}")
print(f"Tokens with k=0: {all_k_values.count(0)}")
print(f"Tokens with k=1: {all_k_values.count(1)}")
print(f"Tokens with k=2: {all_k_values.count(2)}")
```

---

## Exporting for Analysis

### Export All Metrics

```python
profiler.save_csv('all_layers.csv')

# Also save summary statistics
import json

all_summaries = {}
for i, wrapper in enumerate(profiler.wrappers):
    all_summaries[f'layer_{i}'] = wrapper.get_summary()

with open('summary_stats.json', 'w') as f:
    json.dump(all_summaries, f, indent=2)

files.download('all_layers.csv')
files.download('summary_stats.json')
```

### Per-Layer Analysis

```python
# Get metrics for specific layer
layer_0_metrics = profiler.get_metrics()['layer_0']
df = layer_0_metrics.to_df()

# Save with timestamp
from datetime import datetime
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
df.to_csv(f'layer_0_metrics_{timestamp}.csv', index=False)
```

---

## Troubleshooting

### Issue: Out of Memory

```python
# Solution 1: Use smaller model or reduce batch size
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    max_memory={0: "15GB"}  # Limit memory per GPU
)

# Solution 2: Increase sampling rate
for wrapper in profiler.wrappers:
    wrapper.sampling_rate = 20  # Profile less frequently
```

### Issue: Router Not Found

```python
# Debug: Print all module names
for name, module in model.named_modules():
    print(name)

# Manually wrap if router has different name
for name, module in model.named_modules():
    if 'your_router_name' in name:
        wrapper = SimpleRouterWrapper(module)
        # Set it back in the model
        parent_name = '.'.join(name.split('.')[:-1])
        child_name = name.split('.')[-1]
        parent = model.get_submodule(parent_name)
        setattr(parent, child_name, wrapper)
```

### Issue: Incorrect FLOPs

```python
# Manually specify dimensions
wrapper = SimpleRouterWrapper(
    router,
    num_experts=8,
    expert_dim=14336,  # Check model config
    hidden_dim=4096    # Check model config
)

# Verify dimensions
print(f"Hidden dim: {wrapper.hidden_dim}")
print(f"Expert dim: {wrapper.expert_dim}")
```

---

## Complete Example Notebook

```python
# ===== CELL 1: Setup =====
!pip install torch transformers pandas numpy matplotlib seaborn -q

import sys
sys.path.append('/content')  # If uploaded main.py here

from profiler import MoEProfiler
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ===== CELL 2: Load Model =====
model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# ===== CELL 3: Profile =====
profiler = MoEProfiler(model)
profiler.start()

prompts = ["Explain AI", "What is quantum computing?"]
for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=30)

profiler.stop()

# ===== CELL 4: Results =====
profiler.print_summary()
profiler.save_csv('results.csv')

# ===== CELL 5: Visualize =====
import matplotlib.pyplot as plt

df = profiler.wrappers[0].metrics.to_df()

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].plot(df['flops_total'])
axes[0, 0].set_title('Total FLOPs per Token')

axes[0, 1].plot(df['latency_ms'])
axes[0, 1].set_title('Latency (ms)')

axes[1, 0].plot(df['k_avg'])
axes[1, 0].set_title('Average k per Token')

expert_loads = list(profiler.wrappers[0].metrics.expert_loads.values())
axes[1, 1].bar(range(len(expert_loads)), expert_loads)
axes[1, 1].set_title('Expert Load Distribution')

plt.tight_layout()
plt.show()

# ===== CELL 6: Download =====
from google.colab import files
files.download('results.csv')
```

---

## Tips for Research

1. **Always run warmup**: Set `warmup_steps=5` minimum to exclude compilation
2. **Use sampling for long runs**: `sampling_rate=10` reduces overhead by 90%
3. **Save raw data**: Always export CSV for post-processing
4. **Track k distribution**: Critical for dynamic routing analysis
5. **Compare multiple configs**: Use `SimpleBenchmark` for A/B testing
6. **Monitor load balance**: Use Gini + CV + Entropy for complete picture
7. **Check device placement**: Important for multi-GPU analysis

---

## Next: Multi-GPU Setup

For distributed profiling across multiple GPUs, see `MULTI_GPU_SETUP.md` (coming in Phase 2).
