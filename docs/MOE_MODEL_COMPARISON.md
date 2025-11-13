# MoE Model Architecture Comparison

Comparison of router naming and structure across different MoE models in the transformers library.

## OLMoE vs Other MoE Models

### OLMoE-1B-7B (AllenAI)

```
Router Class: OlmoeTopKRouter
Module Path: model.layers[i].mlp.gate
Return Type: (router_scores, router_indices)
Naming Convention: 'gate'

Architecture:
- 64 total experts per layer
- 8 experts per token (top-k=8)
- 16 layers
- 2048 hidden dimension
- Uses softmax + top-k selection
- Token-based dropless routing
```

### Mixtral-8x7B (Mistral)

```
Router Class: MixtralSparseMoeBlock (contains router)
Module Path: model.layers[i].block_sparse_moe
Return Type: Router returns logits (model handles top-k internally)
Naming Convention: 'block_sparse_moe'

Architecture:
- 8 experts per layer
- 2 experts per token (top-k=2)
- Variable layers (32 for 8x7B)
- 4096 hidden dimension
- Uses softmax + top-k selection
- Expert pruning capable
```

### Key Differences

| Feature | OLMoE | Mixtral |
|---------|-------|---------|
| Router Module Name | `OlmoeTopKRouter` | Router inside `MixtralSparseMoeBlock` |
| Access Pattern | `model.layers[i].mlp.gate` | `model.layers[i].block_sparse_moe` |
| Output Format | (scores, indices) tuple | Single logits (post-processing needed) |
| Experts Per Token | 8 | 2 |
| Total Experts | 64 | 8 |
| Routing Algorithm | Dropless token-based | Standard softmax top-k |
| Expert Structure | 3D tensors per layer | Standard nn.Module experts |

## How to Detect Router Type

```python
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("allenai/OLMoE-1B-7B-0924-Instruct")

# Method 1: Check config
print(model.config.model_type)  # 'olmoe'

# Method 2: Inspect module structure
for name, module in model.named_modules():
    if 'gate' in name.lower() or 'router' in name.lower():
        print(f"{name}: {module.__class__.__name__}")

# Method 3: Direct access
router = model.layers[0].mlp.gate
print(router.__class__.__name__)  # OlmoeTopKRouter
```

## Profiling Compatibility

The moe-profiler is designed to work with multiple MoE models:

1. **Auto-detection**: Searches for 'gate' or 'router' in module names
2. **Flexible output handling**:
   - Accepts (weights, indices) tuples
   - Accepts single logits (applies softmax + top-k)
3. **Configuration**: Requires num_experts and (optionally) hidden_dim

## Router Implementation Details

### OLMoE Router Algorithm

```
1. Input: hidden_states [num_tokens, hidden_dim]
2. Linear projection: router_logits = W @ hidden_states
   - W shape: [num_experts=64, hidden_dim]
3. Softmax: router_probs = softmax(router_logits)
4. Top-k selection: 
   - router_scores, router_indices = topk(router_probs, k=8)
5. Optional renormalization:
   - if norm_topk_prob: router_scores /= sum(router_scores)
6. Scatter to dense: Create full sparse matrix with top-k values
7. Output: (router_scores, router_indices)
```

### Mixtral Router Algorithm

```
1. Input: hidden_states [num_tokens, hidden_dim]
2. Linear projection: router_logits = W @ hidden_states
   - W shape: [num_experts=8, hidden_dim]
3. Softmax + Top-2:
   - weights, selected_experts = topk(softmax(router_logits), k=2)
4. Output: Router logits (post-processing in model forward)
```

## Integration with moe-profiler

The SimpleRouterWrapper handles different formats:

```python
router_output = self.router(x)

# Case 1: OLMoE format (returns tuple)
if isinstance(router_output, tuple) and len(router_output) == 2:
    routing_weights, expert_indices = router_output

# Case 2: Mixtral format (returns logits)
else:
    router_logits = router_output
    routing_weights = torch.nn.functional.softmax(router_logits, dim=-1)
    top_k = getattr(self.router, 'top_k', 2)
    routing_weights, expert_indices = torch.topk(routing_weights, top_k, dim=-1)
```

## Parameter Inspection

```python
# OLMoE router parameters
router = model.layers[0].mlp.gate
for name, param in router.named_parameters():
    print(f"{name}: {param.shape}")
# Output:
# weight: torch.Size([64, 2048])  <- Router linear layer weights

# Expert parameters
experts = model.layers[0].mlp.experts
for name, param in experts.named_parameters():
    print(f"{name}: {param.shape}")
# Output:
# gate_up_proj: torch.Size([64, 4096, 2048])  <- 64 experts, up-proj
# down_proj: torch.Size([64, 2048, 4096])     <- 64 experts, down-proj
```

## Configuration Extraction

```python
# From config
print(f"Num experts: {model.config.num_experts}")
print(f"Experts per token: {model.config.num_experts_per_tok}")
print(f"Hidden size: {model.config.hidden_size}")
print(f"Intermediate size: {model.config.intermediate_size}")

# From module
router = model.layers[0].mlp.gate
print(f"Router hidden dim: {router.hidden_dim}")
print(f"Router top_k: {router.top_k}")
print(f"Router num experts: {router.num_experts}")
```

## Summary Table

| Aspect | OLMoE | Mixtral | Notes |
|--------|-------|---------|-------|
| Router Class | OlmoeTopKRouter | MixtralSparseMoeBlock | Different classes |
| Named Path | .mlp.gate | .block_sparse_moe | Different naming |
| Output Type | Tuple (scores, idx) | Logits only | Different formats |
| Experts/Token | 8 | 2 | OLMoE uses more experts |
| Total Experts | 64 | 8 | OLMoE has more experts |
| Routing Loss | 0.01 coef | Similar | Both have auxiliary loss |
| Architecture | Fine-grained (64 small) | Coarse (8 larger) | Different scaling |

