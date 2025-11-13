# OLMoE-1B-7B Model Architecture Research

## Exact Module Names and Hierarchy

### Complete Layer Path Structure

For a loaded OLMoE model, the exact paths for MoE router/gate are:

```
model.layers[i]              → OlmoeDecoderLayer
  .self_attn                 → OlmoeAttention
  .mlp                       → OlmoeSparseMoeBlock (contains MoE logic)
    .gate                    → OlmoeTopKRouter (ROUTER/GATE for expert selection)
    .experts                 → OlmoeExperts (expert FFN layers)
  .input_layernorm           → OlmoeRMSNorm
  .post_attention_layernorm  → OlmoeRMSNorm
```

### The MoE Router Component (The Answer to Question 1)

**Exact Module Name for MoE Router/Gate:** `OlmoeTopKRouter`
- **Path in model:** `model.layers[i].mlp.gate`
- **Class location:** `transformers.models.olmoe.modeling_olmoe.OlmoeTopKRouter`

### Module Structure for MoE Layers (Answer to Question 2)

**MoE Layer Class:** `OlmoeSparseMoeBlock`
- **Path in model:** `model.layers[i].mlp`
- **Contains:**
  - `.gate` - OlmoeTopKRouter instance
  - `.experts` - OlmoeExperts instance

### Expert Routing Return Values (Answer to Question 4)

The OlmoeTopKRouter returns:
```python
(router_scores, router_indices)  # Tuple of 2 tensors
```

Where:
- `router_scores`: Softmax-normalized routing weights [batch_size * seq_len, num_experts]
- `router_indices`: Top-k expert indices selected per token [batch_size * seq_len, k]

### Naming Convention (Answer to Question 5)

**Uses: 'gate' naming convention**

The router is explicitly named `.gate` (not `.router`) in the OlmoeSparseMoeBlock class:
```python
self.gate = OlmoeTopKRouter(config)
```

## Complete Class Definitions

### 1. OlmoeTopKRouter (Router/Gate Implementation)

```python
class OlmoeTopKRouter(nn.Module):
    """Routes tokens to top-k experts based on routing logits"""
    
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok  # Usually 8
        self.num_experts = config.num_experts     # Usually 64
        self.norm_topk_prob = config.norm_topk_prob
        self.hidden_dim = config.hidden_size
        # Single parameter for routing: weights matrix
        self.weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_dim))

    def forward(self, hidden_states: torch.Tensor):
        """
        Args:
            hidden_states: [batch_size * seq_len, hidden_dim]
        
        Returns:
            router_scores: [batch_size * seq_len, num_experts] (softmax normalized)
            router_indices: [batch_size * seq_len, top_k] (expert indices)
        """
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        
        # Compute routing logits via linear projection
        router_logits = F.linear(hidden_states, self.weight)
        
        # Apply softmax to get probabilities
        router_logits = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
        
        # Select top-k experts
        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)
        
        # Optionally renormalize top-k probabilities
        if self.norm_topk_prob:
            router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
        
        router_top_value = router_top_value.to(router_logits.dtype)
        
        # Create scores matrix by scattering top-k values
        router_scores = torch.zeros_like(router_logits).scatter_(1, router_indices, router_top_value)
        
        return router_scores, router_indices
```

### 2. OlmoeExperts (Expert Collection)

```python
class OlmoeExperts(nn.Module):
    """Collection of expert weights stored as 3D tensors"""

    def __init__(self, config: OlmoeConfig):
        super().__init__()
        self.num_experts = config.num_local_experts  # Per-device experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.intermediate_size
        
        # Expert parameters: [num_experts, dimension_out, dimension_in]
        self.gate_up_proj = nn.Parameter(
            torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim)
        )
        self.down_proj = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim)
        )
        self.act_fn = ACT2FN[config.hidden_act]  # Usually SiLU

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,      # Expert indices from router
        top_k_weights: torch.Tensor,    # Expert weights from router
    ) -> torch.Tensor:
        """
        Process tokens through selected experts only
        
        Args:
            hidden_states: [num_tokens, hidden_dim]
            top_k_index: [num_tokens, k] expert indices per token
            top_k_weights: [num_tokens, k] routing weights per token
        
        Returns:
            final_hidden_states: [num_tokens, hidden_dim]
        """
        # Initialize output tensor
        final_hidden_states = torch.zeros_like(hidden_states)
        num_experts = top_k_weights.shape[1]
        
        # Create mask for which experts are active
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=num_experts + 1)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        # Process each active expert
        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == num_experts:
                continue
            
            # Find tokens routed to this expert
            _, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            
            # Expert MLP forward: [hidden_dim] -> [2*intermediate_dim] -> [hidden_dim]
            gate, up = nn.functional.linear(
                current_state, 
                self.gate_up_proj[expert_idx]
            ).chunk(2, dim=-1)
            
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = nn.functional.linear(
                current_hidden_states, 
                self.down_proj[expert_idx]
            )
            
            # Weight by router scores
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, expert_idx, None]
            
            # Accumulate to output
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states
```

### 3. OlmoeSparseMoeBlock (Complete MoE Layer)

```python
class OlmoeSparseMoeBlock(nn.Module):
    """Sparse Mixture of Experts block combining router and experts"""
    
    def __init__(self, config):
        super().__init__()
        self.gate = OlmoeTopKRouter(config)
        self.experts = OlmoeExperts(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, sequence_length, hidden_dim]
        
        Returns:
            final_hidden_states: [batch_size, sequence_length, hidden_dim]
        """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        
        # Flatten for routing
        hidden_states = hidden_states.view(-1, hidden_dim)
        
        # Route through gate to select experts
        top_k_weights, top_k_index = self.gate(hidden_states)
        
        # Process through selected experts
        final_hidden_states = self.experts(hidden_states, top_k_index, top_k_weights).reshape(
            batch_size, sequence_length, hidden_dim
        )
        
        return final_hidden_states
```

### 4. OlmoeDecoderLayer (Full Transformer Layer)

```python
class OlmoeDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: OlmoeConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        # Attention
        self.self_attn = OlmoeAttention(config)

        # MoE (replaces traditional FFN)
        self.mlp = OlmoeSparseMoeBlock(config)

        # Layer normalization
        self.input_layernorm = OlmoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = OlmoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        
        # Pre-attention normalization
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self-attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        # Pre-MoE normalization and MoE forward
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        
        if output_attentions:
            outputs += (self_attn_weights,)
        if output_router_logits:
            outputs += (router_logits,)

        outputs += (present_key_value,)

        return outputs
```

## Key Configuration Parameters

For OlmoeConfig:

```python
class OlmoeConfig(PreTrainedConfig):
    model_type = "olmoe"
    
    # Expert configuration
    num_experts: int = 64                    # Total routed experts per layer
    num_experts_per_tok: int = 8            # Number of experts selected per token (k value)
    norm_topk_prob: bool = True             # Renormalize top-k probabilities
    router_aux_loss_coef: float = 0.01      # Auxiliary loss coefficient
    
    # Model architecture
    hidden_size: int = 2048                 # Hidden dimension
    intermediate_size: int = 2048           # Expert FFN intermediate dimension
    num_hidden_layers: int = 16             # Number of decoder layers
    num_attention_heads: int = 16           # Attention heads
    num_key_value_heads: Optional[int] = None
    
    # Other parameters
    vocab_size: int = 50304
    max_position_embeddings: int = 4096
    hidden_act: str = "silu"
    attention_dropout: float = 0.0
    output_router_logits: bool = False      # Whether to return router logits
```

## Summary: Exact Answers to Your Questions

1. **Exact module name for MoE router/gate that does expert selection:**
   - Class: `OlmoeTopKRouter`
   - Module path: `model.layers[i].mlp.gate`

2. **Module structure for the MoE layers:**
   - Container: `OlmoeSparseMoeBlock` at `model.layers[i].mlp`
   - Contains: `.gate` (OlmoeTopKRouter) and `.experts` (OlmoeExperts)

3. **HuggingFace/Transformers Implementation:**
   - File: `transformers/models/olmoe/modeling_olmoe.py`
   - All classes defined there: OlmoeTopKRouter, OlmoeExperts, OlmoeSparseMoeBlock

4. **Module that returns (routing_weights, expert_indices):**
   - The `OlmoeTopKRouter.forward()` method
   - Returns: `(router_scores, router_indices)` tuple
   - `router_scores`: [batch_size*seq_len, num_experts] softmax-normalized weights
   - `router_indices`: [batch_size*seq_len, k] expert indices

5. **Naming convention used:**
   - Uses **'gate'** naming (not 'router')
   - Found at: `model.layers[i].mlp.gate`
   - This is the standard convention in the OLMoE implementation

## Model Architecture at a Glance

- **64 total experts** per layer
- **8 experts per token** (k=8, num_experts_per_tok=8)
- **16 layers total** (num_hidden_layers=16)
- **2048 hidden dimension** (hidden_size=2048)
- **Sparse routing:** Only selected experts process each token
- **Token-based routing algorithm:** Dropless, fine-grained routing

For profiling with the moe-profiler, you would wrap:
```python
model.layers[i].mlp.gate
```

This is the key router module that determines which experts are activated for each token.
