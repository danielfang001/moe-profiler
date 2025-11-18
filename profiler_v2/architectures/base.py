"""
Base Architecture Handler for MoE Models

Defines the interface that all architecture-specific handlers must implement.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional
import torch.nn as nn


class BaseArchitectureHandler(ABC):
    """Abstract base class for MoE architecture handlers."""

    def __init__(self):
        self.config = None

    def set_config(self, config):
        """
        Set the model config for extracting architecture parameters.

        Args:
            config: Model config (e.g., model.config from HuggingFace)
        """
        self.config = config

    @abstractmethod
    def can_handle(self, module_name: str, module: nn.Module) -> bool:
        """
        Check if this handler can handle the given module.

        Args:
            module_name: Full name of the module (e.g., 'model.layers.0.mlp')
            module: The PyTorch module instance

        Returns:
            True if this handler can handle this module
        """
        pass

    @abstractmethod
    def get_num_experts(self) -> int:
        """Return the number of experts in this architecture."""
        pass

    @abstractmethod
    def get_hidden_dim(self, module: nn.Module) -> int:
        """
        Get the hidden dimension for this architecture.

        Args:
            module: The module to inspect

        Returns:
            Hidden dimension size
        """
        pass

    @abstractmethod
    def get_expert_dim(self, module: nn.Module) -> int:
        """
        Get the expert FFN intermediate dimension.

        Args:
            module: The module to inspect

        Returns:
            Expert intermediate dimension size
        """
        pass

    @abstractmethod
    def get_default_top_k(self) -> int:
        """Return the default number of experts per token (k)."""
        pass

    @abstractmethod
    def parse_router_output(self, output: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parse router output into (routing_weights, expert_indices).

        Different architectures return different formats:
        - Some return logits only
        - Some return (weights, indices) tuple

        Args:
            output: Raw output from the router/gate

        Returns:
            Tuple of (routing_weights, expert_indices)
        """
        pass

    @abstractmethod
    def compute_confidence(self, output: Any, routing_weights: torch.Tensor) -> float:
        """
        Compute routing confidence metric.

        Args:
            output: Raw router output
            routing_weights: Parsed routing weights

        Returns:
            Confidence score (0-1)
        """
        pass

    @abstractmethod
    def get_wrapper_type(self) -> str:
        """
        Return the type of wrapper needed for this architecture.

        Returns:
            'gate' for gate-only wrapping (e.g., Mixtral)
            'block' for full block wrapping (e.g., OLMoE)
        """
        pass

    @abstractmethod
    def get_specs(self) -> Dict[str, Any]:
        """
        Return all architecture specifications as a dictionary.

        Returns:
            Dict with keys: num_experts, default_top_k, wrapper_type, etc.
        """
        pass
