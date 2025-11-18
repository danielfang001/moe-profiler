"""
OLMoE Architecture Handler

Handles OLMoE (allenai/OLMoE-1B-7B-0924-Instruct) where:
- Gate is mlp.gate (simple nn.Linear, returns logits only)
- Must wrap entire mlp block (OlmoeSparseMoeBlock)
- 64 experts, top-8 routing
- hidden_dim=2048, expert_dim=4096
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Any
from .base import BaseArchitectureHandler


class OLMoEHandler(BaseArchitectureHandler):
    """Handler for OLMoE architecture."""

    def __init__(self):
        super().__init__()
        # Defaults (will be overridden by config if available)
        self.num_experts_default = 64
        self.default_top_k_default = 8
        self.hidden_dim_default = 2048
        self.expert_dim_default = 1024  # FIXED: correct value from config

    def can_handle(self, module_name: str, module: nn.Module) -> bool:
        """
        Detect OLMoE architecture.

        For OLMoE, we need to detect the entire MLP block:
        - Module name ends with '.mlp'
        - Module type contains 'SparseMoe' and 'Block'
        - Has 'gate' and 'experts' attributes
        """
        module_type = type(module).__name__

        # Check if it's an OLMoE sparse MoE block
        if 'SparseMoe' in module_type and 'Block' in module_type:
            # Verify it has the expected structure
            if hasattr(module, 'gate') and hasattr(module, 'experts'):
                return True

        # Alternative check: module name pattern
        if module_name.endswith('.mlp'):
            if hasattr(module, 'gate') and hasattr(module, 'experts'):
                # Check if gate is nn.Linear (OLMoE signature)
                if isinstance(module.gate, nn.Linear):
                    return True

        return False

    def get_num_experts(self) -> int:
        """Get number of experts from config or default."""
        if self.config is not None and hasattr(self.config, 'num_experts'):
            return self.config.num_experts
        return self.num_experts_default

    def get_hidden_dim(self, module: nn.Module) -> int:
        """
        Extract hidden dimension from config, module, or default.

        Priority:
        1. config.hidden_size
        2. module.gate.in_features (gate is nn.Linear)
        3. module.gate.weight.shape[1]
        4. Default
        """
        # Try config first
        if self.config is not None and hasattr(self.config, 'hidden_size'):
            return self.config.hidden_size

        # Try module
        if hasattr(module, 'gate'):
            gate = module.gate
            if hasattr(gate, 'in_features'):
                return gate.in_features
            if hasattr(gate, 'weight'):
                # weight shape: [num_experts, hidden_dim]
                return gate.weight.shape[1]

        return self.hidden_dim_default

    def get_expert_dim(self, module: nn.Module) -> int:
        """
        Expert intermediate dimension for OLMoE.

        Priority:
        1. config.intermediate_size (correct!)
        2. Infer from experts module
        3. Default
        """
        # Try config first (BEST)
        if self.config is not None and hasattr(self.config, 'intermediate_size'):
            return self.config.intermediate_size

        # Try to infer from module
        if hasattr(module, 'experts'):
            # Try to get from first expert's structure
            experts = module.experts
            if hasattr(experts, 'mlp'):
                # OlmoeExperts might have mlp attribute
                mlp = experts.mlp
                if hasattr(mlp, 'w1') and hasattr(mlp.w1, 'weight'):
                    # w1 is up_proj: [hidden, expert_dim]
                    return mlp.w1.weight.shape[0]

        return self.expert_dim_default

    def get_default_top_k(self) -> int:
        """Get default top-k from config or default."""
        if self.config is not None and hasattr(self.config, 'num_experts_per_tok'):
            return self.config.num_experts_per_tok
        return self.default_top_k_default

    def parse_router_output(self, output: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parse OLMoE gate output.

        OLMoE gate (nn.Linear) returns logits only.
        The routing logic happens in OlmoeSparseMoeBlock.forward().

        Since we wrap the entire block, we need to process logits ourselves:
        1. Apply softmax
        2. Select top-k
        3. Return (weights, indices)
        """
        if isinstance(output, tuple) and len(output) == 2:
            # Already processed
            return output[0], output[1]

        # Output is raw logits from gate
        router_logits = output

        # Apply softmax
        routing_probs = torch.nn.functional.softmax(router_logits, dim=-1, dtype=torch.float)

        # Select top-k
        routing_weights, expert_indices = torch.topk(
            routing_probs, self.default_top_k, dim=-1
        )

        # Renormalize (OLMoE does this)
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

        return routing_weights, expert_indices

    def compute_confidence(self, output: Any, routing_weights: torch.Tensor) -> float:
        """
        Compute routing confidence for OLMoE.

        Use entropy over the selected top-k experts' weights.
        """
        if isinstance(output, tuple):
            # If tuple, use the weights
            weights = output[0] if len(output) > 0 else routing_weights
        else:
            weights = routing_weights

        # Compute entropy over the routing weights
        # Add small epsilon to avoid log(0)
        entropy = -(weights * torch.log(weights + 1e-10)).sum(dim=-1).mean()

        # Normalize by max entropy for top-k
        max_entropy = torch.log(torch.tensor(float(self.default_top_k)))
        confidence = float(1.0 - (entropy / max_entropy).item())

        return confidence

    def get_wrapper_type(self) -> str:
        """
        OLMoE requires full block wrapping.

        The gate returns only logits, so we need to wrap the entire
        OlmoeSparseMoeBlock to intercept and control expert routing.
        """
        return 'block'

    def get_specs(self) -> Dict[str, Any]:
        """Return all OLMoE specifications."""
        return {
            'architecture': 'olmoe',
            'num_experts': self.get_num_experts(),  # Use getter to get config value
            'default_top_k': self.get_default_top_k(),  # Use getter
            'wrapper_type': 'block',
            'gate_output_format': 'logits',
            'hidden_dim_default': self.hidden_dim_default,
            'expert_dim_default': self.expert_dim_default,
        }
