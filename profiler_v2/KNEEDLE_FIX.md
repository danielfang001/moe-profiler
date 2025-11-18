# Kneedle Selector Fix

## Issue

The original kneedle selector in `profiler_v2/selectors.py` used a different normalization than the proven implementation, leading to different k selection results.

## Root Cause

**Original (WRONG) implementation:**
```python
# y normalized to [1, 0] (not inverted)
y_norm = probs_sorted / (probs_sorted[:, :1] + 1e-12)

# Diagonal from (0,1) to (1,0)
diag = 1.0 - x_norm

# Distance to diagonal
dist = (y_norm - diag) / math.sqrt(2.0)
```

**Proven (CORRECT) implementation:**
```python
# y normalized to [0, 1] (INVERTED from original values)
y_norm = (probs_sorted[:, :1] - probs_sorted) / (probs_sorted[:, :1] - probs_sorted[:, -1:] + 1e-12)

# Distance on y=x diagonal
dist = y_norm - x_norm
```

## Key Differences

| Aspect | Original | Fixed (Proven) |
|--------|----------|----------------|
| **y normalization** | `y / y_max` → 1 to ~0 | `(y_max - y) / (y_max - y_min)` → 0 to 1 |
| **Coordinate system** | y=1-x diagonal | y=x diagonal |
| **Distance formula** | `(y_norm - (1-x_norm)) / sqrt(2)` | `y_norm - x_norm` |

## Why the Fix is Important

The kneedle algorithm finds the "elbow" by looking for the point furthest from a diagonal line:

- **Original**: Measured distance to diagonal from (0,1) to (1,0) with non-inverted y
- **Proven**: Measures distance to diagonal from (0,0) to (1,1) with inverted y

These produce **different results**! The proven implementation correctly inverts y-values so that:
- Both axes go from 0 to 1
- The elbow is detected on the y=x diagonal
- This matches standard kneedle algorithm behavior

## Example

With probabilities `[0.5, 0.3, 0.15, 0.05]`:

**Proven implementation:**
- x_norm = [0, 0.33, 0.67, 1]
- y_norm = [0, 0.44, 0.78, 1]  (inverted!)
- distance = [0, 0.11, 0.11, 0]
- k = 2 (elbow at index 1)

**Original (wrong):**
- x_norm = [0, 0.33, 0.67, 1]
- y_norm = [1, 0.6, 0.3, 0.1]  (not inverted)
- distance = different values
- k = different result

## Changes Made

### File: `profiler_v2/selectors.py`

**Lines 86-93 (Torch path):**
```python
# normalize x and y (matching proven kneedle implementation)
x_norm = torch.linspace(0, 1, steps=n_experts, device=device).unsqueeze(0).expand(probs_sorted.size(0), -1)
# normalize y: INVERT so it goes from 0 to 1 (matching proven implementation)
# y_norm = (y_max - y) / (y_max - y_min)
y_norm = (probs_sorted[:, :1] - probs_sorted) / (probs_sorted[:, :1] - probs_sorted[:, -1:] + 1e-12)

# distance: find max of (y_norm - x_norm) to detect elbow on y=x diagonal
dist = y_norm - x_norm
```

**Lines 127-130 (NumPy path):**
```python
# INVERT y normalization to match proven implementation: (y_max - y) / (y_max - y_min)
ys = (svals[0] - svals) / (svals[0] - svals[-1] + 1e-12)
# Distance: find max of (y_norm - x_norm) on y=x diagonal
dist = ys - xs
```

**Lines 56-64 (Updated docstring):**
```python
"""
Kneedle elbow detection selector (proven implementation).

For each token:
  - Sort probabilities descending
  - Normalize index x in [0,1]
  - Normalize probs y in [0,1] by INVERTING: y_norm = (y_max - y) / (y_max - y_min)
  - Compute distance: y_norm - x_norm (finds elbow on y=x diagonal)
  - Choose the index with maximum distance as the elbow (k = idx+1)
  - Cap k by `k_max` and at least 1
"""
```

## Testing

Run the test script to verify the fix:

```bash
python test_kneedle_fix.py
```

This test compares the fixed implementation against the expected behavior from the proven implementation.

## Source

The proven implementation comes from the groupmate's working monkey-patch code that has been validated with OLMoE models in production research experiments.
