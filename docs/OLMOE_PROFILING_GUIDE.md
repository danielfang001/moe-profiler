# Using moe-profiler with OLMoE-1B-7B

This guide shows how to use the moe-profiler with the OLMoE model to profile the MoE router and expert selection.

## Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from profiler import MoEProfiler

# Load the OLMoE model
model = AutoModelForCausalLM.from_pretrained("allenai/OLMoE-1B-7B-0924-Instruct")
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0924-Instruct")

# Create profiler (auto-detects all routers)
profiler = MoEProfiler(model)
profiler.start()

# Run inference
text = "Explain quantum computing"
inputs = tokenizer(text, return_tensors='pt')
with torch.no_grad():
    outputs = model(**inputs)

# Print results
profiler.stop()
profiler.print_summary()
profiler.save_csv('olmoe_metrics.csv')
```

## Understanding the Layer Structure

OLMoE uses this naming convention:

```
model.layers[i].mlp.gate          <- OlmoeTopKRouter (MoE Router)
model.layers[i].mlp.experts       <- OlmoeExperts (Expert Pool)
```

The profiler automatically finds and wraps all modules with 'gate' in their name.

## Router Output Format

The OlmoeTopKRouter returns:
```python
(router_scores, router_indices)

# Where:
# router_scores: [num_tokens, num_experts] softmax-normalized weights
# router_indices: [num_tokens, k] expert indices selected per token (k=8 for OLMoE)
```

## Key Metrics Explained

### FLOPs Metrics
- `flops_total`: Total FLOPs including routing and expert computation
- `flops_router`: FLOPs for the routing gate computation
- `flops_expert`: FLOPs for processing tokens through selected experts

### Expert Utilization
- `k_mean`: Average number of experts selected per token (should be ~8)
- `active_experts`: Number of unique experts used across all tokens
- `expert_load_*`: Load balancing statistics (std, Gini coefficient, entropy)

### Routing Confidence
- `router_confidence`: Entropy-based confidence in routing decisions
  - Higher = more confident (lower entropy)
  - Useful for detecting routing behavior

## Configuration Parameters

For OLMoE-1B-7B:
- 64 total experts per layer
- 8 experts selected per token (k=8)
- 16 transformer layers
- 2048 hidden dimension
- Router loss coefficient: 0.01

## Profiling Different Layers

To profile only specific layers:

```python
# Get metrics for each layer
metrics_dict = profiler.get_metrics()

for layer_name, metrics in metrics_dict.items():
    print(f"\n{layer_name}:")
    summary = metrics.get_summary()
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
```

## Comparing Configurations

```python
from profiler import SimpleBenchmark

benchmark = SimpleBenchmark(model, tokenizer)

# Baseline
benchmark.run_config("baseline", test_texts)

# With custom configuration
def custom_config(model):
    # Modify model behavior here
    pass

benchmark.run_config("custom", test_texts, custom_config)

# Compare
comparison = benchmark.compare()
```

## Advanced: Custom Router Wrapper

To add your own routing middleware:

```python
class YourOLMoERouterMiddleware(torch.nn.Module):
    def __init__(self, base_router):
        super().__init__()
        self.router = base_router
    
    def forward(self, x):
        # Get base routing
        router_scores, expert_indices = self.router(x)
        
        # Modify routing (e.g., confidence-based selection)
        confidence = router_scores.max(dim=-1)[0]
        mask = confidence < 0.5
        expert_indices[mask] = -1  # Skip low-confidence tokens
        
        return router_scores, expert_indices
```

## Example: Profile OLMoE with Different Sequence Lengths

```python
test_texts = {
    "short": "Hello",
    "medium": "Explain how machine learning works in detail.",
    "long": "Provide a comprehensive explanation of transformer architectures, "
            "including attention mechanisms, positional encodings, and how they "
            "enable modern language models to process sequential data effectively."
}

for name, text in test_texts.items():
    profiler = MoEProfiler(model)
    profiler.start()
    
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        model(**inputs)
    
    profiler.stop()
    profiler.save_csv(f'olmoe_{name}_seq_metrics.csv')
    print(f"\n{name.upper()}:")
    profiler.print_summary()
```

## Troubleshooting

### Issue: "No routers found"
- Check that model.layers[i].mlp.gate exists
- The profiler looks for modules with 'gate' or 'router' in their name
- Verify model is OLMoE: `print(model.config.model_type)`

### Issue: "Router output format unexpected"
- OlmoeTopKRouter returns (router_scores, expert_indices)
- If getting single tensor, the router might be a different implementation
- Add output_router_logits=True to model config if needed

### Issue: "Metrics all zeros"
- Ensure model is on GPU for accurate CUDA timing
- Check warmup_steps parameter (first N steps are skipped)
- Verify sampling_rate is not filtering all data

## References

- OLMoE Paper: https://arxiv.org/abs/2409.02060
- HuggingFace Model: https://huggingface.co/allenai/OLMoE-1B-7B-0924-Instruct
- Transformers Source: https://github.com/huggingface/transformers/tree/main/src/transformers/models/olmoe

## Module Naming Pattern

The profiler automatically detects these modules:

```
model.layers.0.mlp.gate          <- Layer 0 router
model.layers.1.mlp.gate          <- Layer 1 router
...
model.layers.15.mlp.gate         <- Layer 15 router
```

Each .gate is an OlmoeTopKRouter instance that handles expert selection.
