# Automatic Config Extraction

## Overview

profiler_v2 now **automatically extracts architecture parameters** from HuggingFace model configs. No manual configuration needed!

## How It Works

When you create a profiler:
```python
profiler = MOEProfiler(model)  # That's it!
```

The profiler automatically:
1. Gets `model.config` from the HuggingFace model
2. Passes it to architecture handlers
3. Handlers extract correct dimensions from config
4. Falls back to defaults if config unavailable

## What Gets Extracted

### For OLMoE

From `model.config`:
- `num_experts` → `config.num_experts` (64)
- `hidden_size` → `config.hidden_size` (2048)
- `intermediate_size` → `config.intermediate_size` (1024) **← Critical for correct FLOPs!**
- `num_experts_per_tok` → `config.num_experts_per_tok` (8)

### For Mixtral

From `model.config`:
- `num_local_experts` → `config.num_local_experts` (8)
- `hidden_size` → `config.hidden_size` (4096)
- `intermediate_size` → `config.intermediate_size` (14336)
- `num_experts_per_tok` → `config.num_experts_per_tok` (2)

## Example Output

When initializing profiler, you'll see:
```
Scanning model for MoE modules...
Found MoE module: model.layers.0.mlp
  Type: OlmoeSparseMoeBlock
  Architecture: olmoe
  Wrapper type: block
  Num experts: 64
  Hidden dim: 2048
  Expert dim: 1024
  Default top-k: 8
  Config source: model.config (auto-detected)
```

Notice **Expert dim: 1024** - automatically extracted from config!

## Bug Fixed

### Before (Hardcoded)
```python
self.expert_dim_default = 4096  # WRONG for OLMoE!
```
- FLOPs calculation was **4x too high**
- Didn't work for different model variants

### After (Config Extraction)
```python
# Automatically reads config.intermediate_size = 1024
expert_dim = handler.get_expert_dim(module)  # Returns 1024 ✓
```
- Correct FLOPs automatically
- Works for any model configuration

## Priority Order

For each parameter, handlers check in order:
1. **Model config** (best, most accurate)
2. **Module introspection** (infer from module structure)
3. **Default values** (fallback for safety)

Example for `expert_dim`:
```python
def get_expert_dim(self, module):
    # 1. Try config (BEST)
    if self.config and hasattr(self.config, 'intermediate_size'):
        return self.config.intermediate_size  # ✓ Returns 1024

    # 2. Try module inspection
    if hasattr(module, 'experts'):
        # ... inspect expert structure ...

    # 3. Fallback to default
    return self.expert_dim_default  # 1024
```

## Backwards Compatibility

If you don't have a HuggingFace config (e.g., custom model), the profiler still works with defaults:
```python
profiler = MOEProfiler(custom_model)  # Uses defaults if no config
```

## Testing

Run the test to verify config extraction:
```bash
python test_config_extraction.py
```

This test:
- ✓ Verifies correct extraction from config
- ✓ Shows FLOPs calculation impact
- ✓ Compares old (wrong) vs new (correct) values

## Impact

For OLMoE-1B-7B:
- Old profiler: **4x overestimate** in expert FLOPs
- New profiler: **Correct** FLOPs from config

Per-token expert FLOPs:
- Old (wrong): k × 4096 × 2 × 2048 = **67.1M FLOPs** (4x too high)
- New (correct): k × 1024 × 2 × 2048 = **16.8M FLOPs** ✓

## Config Fields Reference

### OLMoE config.json
```json
{
  "hidden_size": 2048,           // Model hidden dimension
  "intermediate_size": 1024,     // Expert FFN dimension
  "num_experts": 64,             // Total experts
  "num_experts_per_tok": 8       // Default k
}
```

### Mixtral config.json
```json
{
  "hidden_size": 4096,           // Model hidden dimension
  "intermediate_size": 14336,    // Expert FFN dimension (3.5x)
  "num_local_experts": 8,        // Total experts
  "num_experts_per_tok": 2       // Default k
}
```

## Summary

✅ **Automatic**: No manual config needed
✅ **Accurate**: Correct dimensions from model config
✅ **Flexible**: Falls back to defaults gracefully
✅ **Compatible**: Works with any HuggingFace MoE model

Just do: `profiler = MOEProfiler(model)` and it works! 🎯
