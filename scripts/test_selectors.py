"""Simple tests for selector functions in profiler.py

This is a lightweight test runner (no pytest dependency). It validates:
- each selector runs on numpy arrays and returns expected shapes
- k is capped at k_max
- returned indices are in the expected range or -1
- optional torch path if torch is installed

Run:
  python3 scripts/test_selectors.py
"""
import sys
import numpy as np

try:
    import torch
except Exception:
    torch = None

# Ensure local repo root is first on sys.path so we import the local `profiler.py`
# instead of any installed `profiler` package in the virtualenv.
import os
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from profiler import (
    example_topk_selector,
    kneedle_selector,
    cumsum_selector,
    entropy_selector,
    gap_ratio_selector,
)


def check_numpy_selector(selector, name, k_max=8):
    print(f"Testing {name} (numpy path)")
    rng = np.random.RandomState(0)
    n_tokens = 10
    n_experts = 16
    logits = rng.randn(n_tokens, n_experts).astype(np.float32)
    exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp / np.sum(exp, axis=1, keepdims=True)

    out = selector(probs, None, None, None, k_max=k_max) if 'kneedle' in name or 'cumsum' in name or 'entropy' in name or 'gap' in name else selector(probs, None, None, None, k=k_max)

    if out is None:
        raise AssertionError(f"{name} returned None")

    vals, idx = out
    assert vals.shape == (n_tokens, k_max), f"vals shape {vals.shape} != {(n_tokens, k_max)}"
    assert idx.shape == (n_tokens, k_max), f"idx shape {idx.shape} != {(n_tokens, k_max)}"

    # indices must be -1 (for padded) or in [0, n_experts)
    flat_idx = idx.flatten()
    if isinstance(flat_idx, np.ndarray):
        bad = [int(i) for i in flat_idx if not (i == -1 or (0 <= int(i) < n_experts))]
        assert len(bad) == 0, f"Invalid indices in {name}: {bad}"

    # values non-negative and <=1
    assert np.all(vals >= -1e-6) and np.all(vals <= 1.0 + 1e-6), f"vals out of range for {name}"

    print(f"  {name} numpy test passed")


def check_torch_selector(selector, name, k_max=8):
    if torch is None:
        print(f"Skipping {name} torch test (torch not installed)")
        return

    print(f"Testing {name} (torch path)")
    torch.manual_seed(0)
    n_tokens = 10
    n_experts = 16
    logits = torch.randn(n_tokens, n_experts)
    probs = torch.softmax(logits, dim=-1)

    # call
    if 'topk' in name:
        vals, idx = selector(probs, None, None, None, k=k_max)
    else:
        vals, idx = selector(probs, None, None, None, k_max=k_max)

    assert tuple(vals.shape) == (n_tokens, k_max), f"vals shape {vals.shape}"
    assert tuple(idx.shape) == (n_tokens, k_max), f"idx shape {idx.shape}"

    # indices in range or -1
    flat_idx = idx.flatten().cpu().numpy()
    bad = [int(i) for i in flat_idx if not (i == -1 or (0 <= int(i) < n_experts))]
    assert len(bad) == 0, f"Invalid indices in {name} torch: {bad}"

    print(f"  {name} torch test passed")


def main():
    selectors = [
        (example_topk_selector, 'topk'),
        (kneedle_selector, 'kneedle'),
        (cumsum_selector, 'cumsum'),
        (entropy_selector, 'entropy'),
        (gap_ratio_selector, 'gap'),
    ]

    for sel, name in selectors:
        check_numpy_selector(sel, name, k_max=8)
        check_torch_selector(sel, name, k_max=8)

    print('\nAll selector tests passed')


if __name__ == '__main__':
    try:
        main()
    except AssertionError as e:
        print('TEST FAILURE:', e)
        sys.exit(2)
    except Exception as e:
        print('ERROR during tests:', e)
        sys.exit(3)
    sys.exit(0)
