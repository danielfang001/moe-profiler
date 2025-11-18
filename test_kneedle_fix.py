"""
Test to verify kneedle selector matches proven implementation
"""

import torch
import numpy as np

# Test with a concrete example
test_probs = torch.tensor([[0.5, 0.3, 0.15, 0.05]])  # 1 token, 4 experts

print("Testing kneedle selector fix")
print("="*80)
print(f"Test probabilities: {test_probs}")
print()

# Expected behavior (from proven implementation):
print("Expected behavior:")
x = np.arange(4)
y = np.array([0.5, 0.3, 0.15, 0.05])  # sorted descending
x_norm = (x - x[0]) / (x[-1] - x[0])  # [0, 0.33, 0.67, 1]
y_norm = (y[0] - y) / (y[0] - y[-1])  # [0, 0.44, 0.78, 1]
dist = y_norm - x_norm  # [0, 0.11, 0.11, 0]
max_idx = np.argmax(dist)  # Should be 1 or 2
k_expected = max_idx + 1

print(f"x_norm: {x_norm}")
print(f"y_norm: {y_norm}")
print(f"distance: {dist}")
print(f"max_idx: {max_idx}")
print(f"k: {k_expected}")
print()

# Test our implementation
print("Testing profiler_v2 implementation:")
try:
    from profiler_v2.selectors import kneedle_selector

    weights, indices = kneedle_selector(test_probs, None, None, None, k_max=8)

    # Count non-negative indices to get k
    k_actual = (indices[0] >= 0).sum().item()

    print(f"Returned weights: {weights[0]}")
    print(f"Returned indices: {indices[0]}")
    print(f"k: {k_actual}")
    print()

    if k_actual == k_expected:
        print("✅ SUCCESS: kneedle selector matches proven implementation!")
    else:
        print(f"⚠️ MISMATCH: Expected k={k_expected}, got k={k_actual}")

except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()

print()
print("="*80)

# Test with multiple tokens
print("\nTesting with multiple tokens:")
test_probs_multi = torch.tensor([
    [0.5, 0.3, 0.15, 0.05],
    [0.4, 0.35, 0.15, 0.1],
    [0.7, 0.2, 0.05, 0.05],
])

try:
    weights, indices = kneedle_selector(test_probs_multi, None, None, None, k_max=8)

    for i in range(len(test_probs_multi)):
        k = (indices[i] >= 0).sum().item()
        print(f"Token {i}: probs={test_probs_multi[i].tolist()}, k={k}")

    print("\n✅ Multi-token test completed")

except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
