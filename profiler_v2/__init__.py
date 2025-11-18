"""
MoE Profiler v2 - Architecture-Agnostic Profiling for Mixture of Experts

A clean, modular profiler that automatically detects MoE architectures
and applies appropriate instrumentation.

Basic Usage:
    ```python
    from profiler_v2 import MOEProfiler
    from transformers import AutoModelForCausalLM

    # Load your MoE model
    model = AutoModelForCausalLM.from_pretrained("allenai/OLMoE-1B-7B-0924-Instruct")

    # Create profiler (auto-detects architecture)
    profiler = MOEProfiler(model, use_cuda_events=True)

    # Run inference
    outputs = model.generate(**inputs, max_new_tokens=50)

    # Get metrics
    df = profiler.get_metrics_df()
    print(profiler.generate_report())
    ```

With Custom Selection:
    ```python
    from profiler_v2 import MOEProfiler, selectors

    # Use kneedle selector for dynamic k
    profiler = MOEProfiler(
        model,
        selection_fn=selectors.kneedle_selector
    )
    ```
"""

from .profiler import MOEProfiler
from .metrics import Metrics, get_metrics_summary
from .wrappers import RouterWrapper
from . import selectors
from . import architectures

__version__ = "2.0.0"

__all__ = [
    "MOEProfiler",
    "Metrics",
    "RouterWrapper",
    "get_metrics_summary",
    "selectors",
    "architectures",
]
