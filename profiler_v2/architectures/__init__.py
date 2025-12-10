"""
Architecture Handlers for MoE Models

Auto-registers all available handlers and provides detection utilities.
"""

from .base import BaseArchitectureHandler
from .olmoe import OLMoEHandler

# Registry of all available handlers
HANDLERS = [
    OLMoEHandler(),
]


def detect_architecture(module_name: str, module, model_config=None) -> BaseArchitectureHandler:
    """
    Detect which architecture handler can handle the given module.

    Args:
        module_name: Full module name (e.g., 'model.layers.0.mlp')
        module: PyTorch module instance
        model_config: Optional model config for extracting parameters

    Returns:
        The appropriate handler with config set

    Raises:
        ValueError: If no handler can handle this module
    """
    for handler in HANDLERS:
        if handler.can_handle(module_name, module):
            # Set config if available
            if model_config is not None:
                handler.set_config(model_config)
            return handler

    raise ValueError(
        f"No architecture handler found for module: {module_name} "
        f"(type: {type(module).__name__})"
    )


__all__ = [
    'BaseArchitectureHandler',
    'OLMoEHandler',
    'HANDLERS',
    'detect_architecture',
]
