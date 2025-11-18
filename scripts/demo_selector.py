"""Demo script: show SelectableRouterWrapper in action

Usage:
  python3 scripts/demo_selector.py [--use-hf]

If `--use-hf` is provided and `transformers` + a compatible model are available,
this will attempt to load an OLMoE or Mixtral model and attach the selector.
Otherwise the script runs a small toy MoE model to demonstrate the wrapper.

This script requires PyTorch to be installed to run the real-model path. The
toy-model path uses PyTorch if available; otherwise it falls back to a
numpy-only simulation.
"""

import argparse
import sys
import torch
import os

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from profiler import (
    MoEProfiler,
    example_topk_selector,
    attach_selector_to_model,
    SelectableRouterWrapper,
)
from profiler import kneedle_selector, cumsum_selector, entropy_selector, gap_ratio_selector


def run_toy_model_demo(use_torch=True):
    """Create a tiny toy MoE-like model and run one forward pass to exercise wrapper."""
    if use_torch and torch is None:
        print("PyTorch not available; falling back to numpy demo.")
        use_torch = False

    if use_torch:
        import torch.nn as nn

        class DummyGate(nn.Module):
            def __init__(self, num_experts=8):
                super().__init__()
                self.num_experts = num_experts

            def forward(self, x):
                # x: [batch, seq, hidden]
                batch = x.shape[0]
                seq = x.shape[1] if x.dim() > 2 else 1
                num_tokens = batch * seq
                logits = torch.randn(num_tokens, self.num_experts)
                # Return logits (this simulates a gate returning logits)
                return logits

        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                # gate will be replaced by profiler
                self.gate = DummyGate(num_experts=16)

        class Layer(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = MLP()

        class ToyMoEModel(nn.Module):
            def __init__(self, n_layers=1):
                super().__init__()
                self.layers = nn.ModuleList([Layer() for _ in range(n_layers)])

            def forward(self, x):
                # Call each layer's gate to trigger wrappers
                for l in self.layers:
                    _ = l.mlp.gate(x)
                # Return a fake LM output
                return {"logits": torch.zeros(x.shape[0], 1, 1)}

        model = ToyMoEModel(n_layers=1)

    else:
        # numpy-only simulation fallback: use profiler's no-torch demo
        print("Running numpy fallback demo (no PyTorch).")
        from profiler import run_selector_demo_no_torch
        run_selector_demo_no_torch()
        return

    # Attach selector wrappers to all gates and create profiler with chosen selector
    wrappers = attach_selector_to_model(model, selected_selector, name_match='gate')

    # Use profiler with the same selection_fn so it will install SelectableRouterWrapper too
    profiler = MoEProfiler(model, selection_fn=selected_selector)
    profiler.start()

    # Create dummy input and run one forward
    x = torch.randn(2, 3, 32)
    with torch.no_grad():
        _ = model(x)

    profiler.stop()

    print("\nProfiler wrappers created:")
    for i, w in enumerate(profiler.wrappers):
        print(f"Wrapper {i}: name={getattr(w, 'name', '<unknown>')} type={type(w).__name__}")
        last_probs = getattr(w, 'last_probs', None)
        last_indices = getattr(w, 'last_indices', None)
        print("  last_probs:", None if last_probs is None else getattr(last_probs, 'shape', str(type(last_probs))))
        print("  last_indices:", None if last_indices is None else getattr(last_indices, 'shape', str(type(last_indices))))


def run_hf_model_demo(model_name: str = "allenai/OLMoE-1B-7B-0924-Instruct"):
    """Attempt to load a HuggingFace model and attach the selector.

    Note: This requires `transformers` and PyTorch, and a lot of memory for large models.
    Use at your own risk or point to a smaller MoE model if available.
    """
    if torch is None:
        print("PyTorch is required to run the HuggingFace model demo. Aborting.")
        return

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:
        print("transformers not installed or failed to import:", e)
        return

    print(f"Loading model {model_name} (this may take a while and require significant RAM/GPU)...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Attach selector and profiler
    sel = globals().get('selected_selector', example_topk_selector)
    profiler = MoEProfiler(model, selection_fn=sel)
    profiler.start()

    text = "Hello world"
    # Hello world!
    # This is my first post on my brand new blog. I'm so excited to share my thoughts and experiences with you all.
    # As a language learner, I've always been fascinated by the power of words. The way they can inspire, motivate, and connect us with others is truly
    inputs = tokenizer(text, return_tensors='pt')
    inputs = {k: v for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_length=64)
        print(tokenizer.decode(out[0]))

    profiler.stop()

    print("Profiler wrappers created:")
    for i, w in enumerate(profiler.wrappers):
        print(f"Wrapper {i}: name={getattr(w, 'name', '<unknown>')} type={type(w).__name__}")
        print("  last_probs:", getattr(w, 'last_probs', None))
        print("  last_indices:", getattr(w, 'last_indices', None))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--use-hf', action='store_true', help='Attempt to load a HuggingFace OLMoE model')
    p.add_argument('--model-name', type=str, default='allenai/OLMoE-1B-7B-0924-Instruct', help='HF model id to load when --use-hf is set')
    p.add_argument('--selector', type=str, default='topk', help='Selector to use: topk|kneedle|cumsum|entropy|gap')
    args = p.parse_args()

    # pick selector
    sel_map = {
        'topk': example_topk_selector,
        'kneedle': kneedle_selector,
        'cumsum': cumsum_selector,
        'entropy': entropy_selector,
        'gap': gap_ratio_selector,
    }


    # pass selected_selector into run_toy_model_demo by setting a global variable
    selected_selector = sel_map.get(args.selector.lower(), example_topk_selector)
    globals()['selected_selector'] = selected_selector

    if args.use_hf:
        run_hf_model_demo(args.model_name)
    else:
        run_toy_model_demo(use_torch=(torch is not None))


if __name__ == '__main__':
    main()

