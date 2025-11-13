# OLMoE-1B-7B Quick Reference Card

## The 30-Second Answer

**Question: What is the exact MoE router layer name in OLMoE?**

```python
model.layers[i].mlp.gate
```

That's it. The router is called `.gate` and it's an instance of `OlmoeTopKRouter`.

## The Module Hierarchy (Copy-Paste Ready)

```
model                               (OlmoeForCausalLM)
├── layers[0-15]                   (OlmoeDecoderLayer × 16)
│   ├── self_attn                  (OlmoeAttention)
│   ├── mlp                         (OlmoeSparseMoeBlock) ← MoE CONTAINER
│   │   ├── gate                   (OlmoeTopKRouter) ← THE ROUTER
│   │   └── experts                (OlmoeExperts) ← Expert pool
│   ├── input_layernorm            (OlmoeRMSNorm)
│   └── post_attention_layernorm   (OlmoeRMSNorm)
├── embed_tokens                   (Embedding)
├── norm                           (OlmoeRMSNorm)
└── rotary_emb                     (OlmoeRotaryEmbedding)
```

## Code Snippets (Use These)

### Access the router
```python
router = model.layers[0].mlp.gate
print(type(router).__name__)  # Output: OlmoeTopKRouter
```

### Call the router directly
```python
# Input: hidden states [num_tokens, 2048]
# Output: (router_scores, router_indices)
scores, indices = router(hidden_states)
print(scores.shape)   # [num_tokens, 64]
print(indices.shape)  # [num_tokens, 8]
```

### Get all routers
```python
routers = []
for i, layer in enumerate(model.layers):
    routers.append(layer.mlp.gate)
print(f"Found {len(routers)} routers")  # Should be 16
```

### Profile with moe-profiler
```python
from profiler import MoEProfiler

profiler = MoEProfiler(model)
# Auto-detects model.layers[i].mlp.gate
profiler.start()
# ... run model ...
profiler.print_summary()
```

## Key Numbers (OLMoE-1B-7B)

| Parameter | Value |
|-----------|-------|
| Total Layers | 16 |
| Total Experts per Layer | 64 |
| Experts Selected per Token | 8 |
| Hidden Dimension | 2048 |
| Expert FFN Dimension | 2048 |
| Router Weight Matrix Shape | [64, 2048] |
| Vocab Size | 50,304 |
| Max Sequence Length | 4,096 |

## Router Algorithm (Simplified)

```
Input: hidden_states [num_tokens, 2048]
       ↓
Linear: @ weight [64, 2048]
       ↓
Logits: [num_tokens, 64]
       ↓
Softmax: normalize across experts
       ↓
Top-8: select top 8 highest values
       ↓
Output: (router_scores, router_indices)
        - scores: [num_tokens, 64] (sparse, only top-8 filled)
        - indices: [num_tokens, 8] (which experts selected)
```

## Output Format

### router_scores (routing weights)
- Shape: `[num_tokens, num_experts=64]`
- Type: torch.FloatTensor
- Value: softmax-normalized weights
- Note: Sparse (mostly zeros, only top-8 positions filled)

### router_indices (expert selection)
- Shape: `[num_tokens, k=8]`
- Type: torch.LongTensor
- Value: expert IDs (0-63)
- Note: Indices into the 64 experts

## Where Is It Used?

The router output is used by the OlmoeExperts class:

```
Hidden States → OlmoeTopKRouter → (scores, indices) → OlmoeExperts → Output
```

1. Router decides which experts to use
2. Experts processes tokens through selected experts only
3. Weighted sum of expert outputs

## Tensor Dimensions Through the Layer

```
Input to router:
  [batch_size, seq_len, 2048] → reshaped to [batch_size*seq_len, 2048]

Router outputs:
  router_scores: [batch_size*seq_len, 64]
  router_indices: [batch_size*seq_len, 8]

Experts processes:
  hidden_states: [batch_size*seq_len, 2048]
  top_k_index: [batch_size*seq_len, 8]
  top_k_weights: [batch_size*seq_len, 8]

Expert output:
  [batch_size*seq_len, 2048] → reshaped back to [batch_size, seq_len, 2048]
```

## Configuration Access

```python
# From model config
model.config.num_experts              # 64
model.config.num_experts_per_tok      # 8
model.config.hidden_size              # 2048
model.config.intermediate_size        # 2048
model.config.num_hidden_layers        # 16
model.config.router_aux_loss_coef     # 0.01

# From router directly
router = model.layers[0].mlp.gate
router.num_experts                    # 64
router.top_k                          # 8
router.hidden_dim                     # 2048
router.weight.shape                   # torch.Size([64, 2048])
```

## Common Operations

### Check if it's OLMoE
```python
assert model.config.model_type == "olmoe"
```

### Iterate through all routers
```python
for layer_idx, layer in enumerate(model.layers):
    router = layer.mlp.gate
    experts = layer.mlp.experts
    print(f"Layer {layer_idx}: {router.num_experts} experts, "
          f"select {router.top_k} per token")
```

### Get router parameters
```python
router = model.layers[0].mlp.gate
for name, param in router.named_parameters():
    print(f"{name}: {param.shape}")
    # Output: weight: torch.Size([64, 2048])
```

### Manual routing for debugging
```python
hidden_states = torch.randn(100, 2048)  # 100 tokens
router = model.layers[0].mlp.gate

scores, indices = router(hidden_states)
print(f"Scores shape: {scores.shape}")        # [100, 64]
print(f"Indices shape: {indices.shape}")      # [100, 8]
print(f"Max score: {scores.max():.4f}")       # Should be ~1.0
print(f"Min nonzero score: {scores[scores > 0].min():.4f}")
print(f"Unique experts used: {indices.unique().tolist()}")
```

## Important Notes

1. **Output is a tuple**: Router returns `(scores, indices)`, not a single tensor
2. **Named 'gate'**: In OLMoE, it's `.gate`, not `.router`
3. **Sparse routing**: Only top-8 experts used per token (rest are zero)
4. **Token-based**: Routing per token, not per layer
5. **Dropless**: All selected tokens use exactly k=8 experts

## For Your moe-profiler

The profiler already handles OLMoE correctly because it:
1. Searches for 'gate' in module names ✓
2. Detects tuple output (scores, indices) ✓
3. Computes FLOPs with dynamic k=8 ✓
4. Tracks expert loads ✓

```python
profiler = MoEProfiler(model)
# Automatically finds and wraps:
# - model.layers[0].mlp.gate
# - model.layers[1].mlp.gate
# - ... etc
profiler.start()
# Run model...
profiler.print_summary()
```

## Model Loading Example

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "allenai/OLMoE-1B-7B-0924-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Access router
router = model.layers[0].mlp.gate
print(f"Router class: {type(router).__name__}")
# Output: OlmoeTopKRouter
```

## Debugging Checklist

- [ ] Model type is "olmoe": `print(model.config.model_type)`
- [ ] Router exists: `print(model.layers[0].mlp.gate)`
- [ ] Router is OlmoeTopKRouter: `print(type(model.layers[0].mlp.gate).__name__)`
- [ ] Weight shape is [64, 2048]: `print(model.layers[0].mlp.gate.weight.shape)`
- [ ] Forward returns tuple: `isinstance(router(x), tuple)`
- [ ] Router returns 2 tensors: `len(router(x)) == 2`

---

**TL;DR**: Use `model.layers[i].mlp.gate` to access OLMoE routers. It's an `OlmoeTopKRouter` that returns `(routing_weights, expert_indices)`.
