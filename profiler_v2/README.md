# MoE Profiler v2

Architecture-agnostic profiling for Mixture of Experts models with automatic detection and clean separation of concerns.

## Features

- **Architecture Auto-Detection**: Automatically detects Mixtral, OLMoE, and other MoE architectures
- **Clean Separation**: Each architecture has its own handler with specific wrapping logic
- **Flexible Wrapping**:
  - Gate-only wrapping for Mixtral-like architectures
  - Full block wrapping for OLMoE-like architectures
- **Dynamic k Support**: Track and visualize dynamic expert selection
- **Custom Selectors**: Built-in selection functions (topk, kneedle, cumsum, entropy, gap_ratio)
- **Comprehensive Metrics**: FLOPs, latency, expert loads, confidence, k distribution
- **CUDA-Accurate Timing**: Uses CUDA events for precise GPU timing

## Quick Start

### Basic Usage

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from profiler_v2 import MOEProfiler

# Load your MoE model
model = AutoModelForCausalLM.from_pretrained(
    "allenai/OLMoE-1B-7B-0924-Instruct",
    quantization_config=BitsAndBytesConfig(load_in_8bit=True),
    device_map="auto",
    trust_remote_code=True
)

# Create profiler (auto-detects architecture)
profiler = MOEProfiler(model)

# Run inference
outputs = model.generate(**inputs, max_new_tokens=50)

# Get metrics
df = profiler.get_metrics_df()
print(profiler.generate_report())
```

### With Custom Selection

```python
from profiler_v2 import MOEProfiler, selectors

# Use kneedle selector for dynamic k
profiler = MOEProfiler(
    model,
    selection_fn=selectors.kneedle_selector
)
```

### Available Selectors

- `selectors.topk_selector`: Standard top-k selection
- `selectors.kneedle_selector`: Elbow detection for dynamic k
- `selectors.cumsum_selector`: Cumulative mass threshold
- `selectors.entropy_selector`: Entropy-based k selection
- `selectors.gap_ratio_selector`: Gap ratio detection

## Architecture

```
profiler_v2/
├── __init__.py              # Main exports
├── profiler.py              # MOEProfiler class
├── metrics.py               # Metrics tracking
├── wrappers.py              # RouterWrapper (unified)
├── selectors.py             # Selection functions
└── architectures/
    ├── __init__.py          # Handler registry
    ├── base.py              # BaseArchitectureHandler
    ├── mixtral.py           # Mixtral handler
    └── olmoe.py             # OLMoE handler
```

## Supported Architectures

| Architecture | Detection | Wrapper Type | Specs |
|-------------|-----------|--------------|-------|
| **Mixtral** | `block_sparse_moe.gate` | Gate-only | 8 experts, top-2, gate returns logits |
| **OLMoE** | `.mlp` (SparseMoeBlock) | Full block | 64 experts, top-8, gate is nn.Linear |

## Adding New Architectures

To add support for a new MoE architecture:

1. Create a new handler in `architectures/` (e.g., `deepseek.py`)
2. Implement `BaseArchitectureHandler` interface
3. Register in `architectures/__init__.py`

Example:

```python
from .base import BaseArchitectureHandler

class DeepSeekHandler(BaseArchitectureHandler):
    def can_handle(self, module_name, module):
        # Detection logic
        return 'deepseek_moe' in module_name.lower()

    def get_num_experts(self):
        return 16  # DeepSeek-specific

    # ... implement other methods
```

## Metrics

The profiler tracks:

- **FLOPs**: Router FLOPs, expert FLOPs, total FLOPs (per token)
- **Latency**: CUDA-accurate timing (ms)
- **Dynamic k**: Average k per token, k distribution
- **Expert Utilization**: Expert loads, active experts
- **Load Balancing**: Gini coefficient, CV, entropy
- **Confidence**: Routing confidence metrics
- **Memory**: GPU memory usage

## Routing Analysis Utilities

New in v2: Utilities for analyzing and comparing routing behavior across prompts.

### Quick Test on Multiple Prompts

```python
# Test routing behavior across different prompts
test_prompts = [
    "Hello, how are you?",
    "Explain quantum computing.",
    "Write Python code for sorting.",
]

results = profiler.quick_test(test_prompts, tokenizer, max_new_tokens=30)
```

### Compare Routing Behavior

```python
# Verify that routing is actually dynamic (not identical across prompts)
comparison = profiler.compare_routing_behavior(results)

# This will print:
# ✓ SUCCESS: Different prompts produce DIFFERENT routing behavior
# or
# ❌ PROBLEM: All prompts produce IDENTICAL routing behavior!
```

### Print Routing Statistics

```python
# Pretty-print comprehensive routing statistics
profiler.print_routing_stats()

# Output:
# ================================================================================
# ROUTING STATISTICS
# ================================================================================
# Total samples: 1234
#
# Aggregate K Statistics:
#   Mean k: 3.45
#   Std k:  1.23
#   Min k:  1.0
#   Max k:  8.0
#
# Per-Layer Statistics:
#   model.layers.0.mlp:
#     Mean k: 3.42
#     Std k:  1.20
#     ...
```

### Programmatic Access

```python
# Get k statistics as a dictionary
k_stats = profiler.get_k_statistics()

print(f"Mean k: {k_stats['k_mean']}")
print(f"Per-layer stats: {k_stats['per_layer']}")
```

### Complete Example

See `example_routing_comparison.py` for a complete working example.

## Comparison with v1

| Feature | v1 (profiler.py) | v2 (profiler_v2/) |
|---------|------------------|-------------------|
| Architecture Detection | Manual/heuristic | Automatic with handlers |
| Code Organization | Monolithic | Modular (handlers, wrappers, metrics) |
| Adding Architectures | Modify core code | Add new handler file |
| Hardcoded Values | Many (8 experts, 4096 dims) | None (from handlers) |
| Bugs | Yes (line 195, 358, etc.) | Fixed |
| Wrapper Logic | if/else branches | Delegated to handlers |

## Testing

Run the test script:

```bash
python test_profiler_v2.py
```

This will test:
1. Basic profiling with default routing
2. Custom selector (kneedle) profiling
3. Metrics collection and reporting
4. Per-layer analysis

## Migration from v1

Old code:
```python
from profiler import MOEProfiler
profiler = MOEProfiler(model, num_experts=8, expert_dim=14336)
```

New code:
```python
from profiler_v2 import MOEProfiler
profiler = MOEProfiler(model)  # Auto-detects everything!
```
