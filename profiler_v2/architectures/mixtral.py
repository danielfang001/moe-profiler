"""
Mixtral Architecture Handler

Handles Mixtral-8x7B and similar architectures where:
- Gate is in block_sparse_moe.gate
- Gate returns logits only (not tuple)
- 8 experts, top-2 routing
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Any
from .base import BaseArchitectureHandler


class MixtralHandler(BaseArchitectureHandler):
    """Handler for Mixtral-style MoE architectures."""

    def __init__(self):
        super().__init__()
        # Defaults (will be overridden by config if available)
        self.num_experts_default = 8
        self.default_top_k_default = 2
        self.hidden_dim_default = 4096
        self.expert_dim_default = 14336  # Typical for Mixtral
        self.expert_dim_multiplier = 3.5  # Fallback: 14336 / 4096

    def can_handle(self, module_name: str, module: nn.Module) -> bool:
        """
        Detect Mixtral architecture.

        Looks for:
        - 'block_sparse_moe' in module name (the entire MoE block)
        - Module type contains 'Mixtral'
        - Module has 'gate' and 'experts' attributes
        """
        module_type = type(module).__name__

        # Check if it's a Mixtral sparse MoE block
        if 'mixtral' in module_type.lower():
            # Verify it has the expected structure
            if hasattr(module, 'gate') and hasattr(module, 'experts'):
                return True

        # Alternative check: module name pattern
        if 'block_sparse_moe' in module_name.lower():
            # Make sure it's the block itself, not a submodule
            if module_name.endswith('block_sparse_moe'):
                # Verify structure
                if hasattr(module, 'gate') and hasattr(module, 'experts'):
                    return True

        return False

    def get_num_experts(self) -> int:
        """Get number of experts from config or default."""
        if self.config is not None and hasattr(self.config, 'num_local_experts'):
            return self.config.num_local_experts
        return self.num_experts_default

    def get_hidden_dim(self, module: nn.Module) -> int:
        """
        Extract hidden dimension from config, module, or default.

        Priority:
        1. config.hidden_size
        2. module.in_features
        3. module.weight.shape[1]
        4. Default
        """
        # Try config first
        if self.config is not None and hasattr(self.config, 'hidden_size'):
            return self.config.hidden_size

        # Try module
        if hasattr(module, 'in_features'):
            return module.in_features
        if hasattr(module, 'weight'):
            # weight shape: [num_experts, hidden_dim]
            return module.weight.shape[1]
        return self.hidden_dim_default

    def get_expert_dim(self, module: nn.Module) -> int:
        """
        Expert intermediate dimension.

        Priority:
        1. config.intermediate_size
        2. Fallback: hidden_dim * multiplier
        3. Default
        """
        # Try config first
        if self.config is not None and hasattr(self.config, 'intermediate_size'):
            return self.config.intermediate_size

        # Fallback: compute from hidden_dim
        hidden_dim = self.get_hidden_dim(module)
        if hidden_dim != self.hidden_dim_default:
            return int(hidden_dim * self.expert_dim_multiplier)

        return self.expert_dim_default

    def get_default_top_k(self) -> int:
        """Get default top-k from config or default."""
        if self.config is not None and hasattr(self.config, 'num_experts_per_tok'):
            return self.config.num_experts_per_tok
        return self.default_top_k_default

    def parse_router_output(self, output: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parse Mixtral gate output.

        Mixtral gate returns logits only (tensor), not tuple.
        We need to:
        1. Apply softmax to get probabilities
        2. Select top-k experts
        3. Return (routing_weights, expert_indices)
        """
        if isinstance(output, tuple):
            # If it's already a tuple, just return it
            return output[0], output[1]

        # Output is logits tensor
        router_logits = output
        routing_probs = torch.nn.functional.softmax(router_logits, dim=-1)

        # Select top-k experts
        top_k = self.get_default_top_k()  # Use getter to get config value
        routing_weights, expert_indices = torch.topk(
            routing_probs, top_k, dim=-1
        )

        return routing_weights, expert_indices

    def compute_confidence(self, output: Any, routing_weights: torch.Tensor) -> float:
        """
        Compute routing confidence based on entropy.

        For Mixtral, we have access to the full logits,
        so we can compute entropy over all experts.
        """
        if isinstance(output, tuple):
            # If tuple, output[0] might be weights
            router_logits = output[0]
        else:
            router_logits = output

        # Compute probability distribution
        routing_probs = torch.nn.functional.softmax(router_logits, dim=-1)

        # Compute entropy
        entropy = -(routing_probs * torch.log(routing_probs + 1e-10)).sum(dim=-1).mean()

        # Normalize entropy to [0, 1] range
        # Max entropy = log(num_experts)
        max_entropy = torch.log(torch.tensor(float(self.get_num_experts())))
        confidence = float(1.0 - (entropy / max_entropy).item())

        return confidence

    def get_wrapper_type(self) -> str:
        """Mixtral uses block wrapping for full control over dynamic k."""
        return 'block'

    def get_specs(self) -> Dict[str, Any]:
        """Return all Mixtral specifications."""
        return {
            'architecture': 'mixtral',
            'num_experts': self.get_num_experts(),  # Use getter to get config value
            'default_top_k': self.get_default_top_k(),  # Use getter
            'wrapper_type': 'block',
            'gate_output_format': 'logits',
            'hidden_dim_default': self.hidden_dim_default,
            'expert_dim_default': self.expert_dim_default,
            'expert_dim_multiplier': self.expert_dim_multiplier,
        }
