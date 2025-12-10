# Elbow-Based Routing

A profiler + wrapper for Mixture of Experts (MoE) models that implements elbow-based routing and measures performance metrics.

## Quick Start

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

## Key Scripts

### Benchmarking

**Run accuracy benchmarks** (OLMoE):
```bash
python scripts/benchmark_olmoe.py --benchmark mmlu --selector kneedle --k-max 8
```
Available benchmarks: `arc_easy`, `arc_challenge`, `mmlu`, `hellaswag`, `piqa`, `winogrande`

### Router Analysis

**Analyze router trends** across layers:
```bash
python scripts/olmoe_router_trends.py
```

### Advanced Usage

**Multi-GPU profiling**:
```bash
python scripts/example_multi_gpu.py
```

## Project Structure

- `profiler_v2/` - Main profiler library
  - `profiler.py` - Core profiler class
  - `selectors.py` - Expert selection strategies
  - `benchmark.py` - Performance benchmarking
  - `accuracy_benchmark.py` - Accuracy evaluation
  - `architectures/` - Architecture-specific support
- `scripts/` - Example scripts and analysis tools
- `notebooks/`
  - `elbowanalysis.ipynb` - Elbow and elbow angle analysis 
