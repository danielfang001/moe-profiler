"""
MoE Profiler for Router Middleware Research
"""

import torch
import pandas as pd
import numpy as np
import math
import time
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path

LOG_PATH = Path(os.environ.get('MOEPROFILER_LOG', '/root/moeprofiler_debug.log'))


def _to_numpy(x):
    """Safely convert a tensor-like or array-like object to a NumPy array.

    Handles PyTorch tensors, objects with `numpy()` or `__array__`, and plain
    NumPy arrays. Returns `None` if conversion fails.
    """
    if x is None:
        return None
    try:
        # PyTorch Tensor path
        if hasattr(x, 'detach'):
            y = x.detach()
            try:
                y = y.cpu()
            except Exception:
                pass
            if hasattr(y, 'numpy'):
                return y.numpy()
            try:
                return np.asarray(y)
            except Exception:
                return None

        # Objects exposing numpy()
        if hasattr(x, 'numpy'):
            try:
                return x.numpy()
            except Exception:
                pass

        # Fallback to numpy.asarray for array-like
        return np.asarray(x)
    except Exception:
        try:
            return np.asarray(x)
        except Exception:
            return None


# ============================================
# 1. SIMPLE METRIC TRACKER
# ============================================

@dataclass
class Metrics:
    """Metrics container with statistical tracking"""
    # Raw measurements
    flops_per_token: List[float] = field(default_factory=list)
    router_flops_per_token: List[float] = field(default_factory=list)
    expert_flops_per_token: List[float] = field(default_factory=list)
    active_experts: List[int] = field(default_factory=list)
    k_per_token: List[float] = field(default_factory=list)  # Average k (active experts) per token in batch
    k_distribution: List[List[int]] = field(default_factory=list)  # Full k distribution per forward pass
    latency_ms: List[float] = field(default_factory=list)
    expert_loads: Dict[int, int] = field(default_factory=lambda: {i: 0 for i in range(8)})
    router_confidence: List[float] = field(default_factory=list)
    semantic_confidence: List[float] = field(default_factory=list)
    memory_mb: List[float] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)

    # Metadata
    device: str = 'cuda:0'
    step_count: int = 0

    def to_df(self):
        """Convert to pandas DataFrame for easy analysis"""
        n = len(self.flops_per_token)

        # Pad semantic_confidence if not populated
        semantic_conf = self.semantic_confidence if len(self.semantic_confidence) == n else [0.0] * n

        return pd.DataFrame({
            'step': range(n),
            'flops_total': self.flops_per_token,
            'flops_router': self.router_flops_per_token,
            'flops_expert': self.expert_flops_per_token,
            'active_experts': self.active_experts,
            'k_avg': self.k_per_token,
            'latency_ms': self.latency_ms,
            'router_conf': self.router_confidence,
            'semantic_conf': semantic_conf,
            'memory_mb': self.memory_mb,
            'timestamp': self.timestamps,
            'device': self.device
        })

# ============================================
# 2. SIMPLE ROUTER WRAPPER
# ============================================

class SimpleRouterWrapper(torch.nn.Module):
    """Wraps your MoE router to track metrics
    """

    def __init__(self, original_router, num_experts=8, expert_dim=None, hidden_dim=None,
                 warmup_steps=5, sampling_rate=1, name: Optional[str]=None):
        """
        Args:
            original_router: The original router to wrap
            num_experts: Number of experts
            expert_dim: Expert FFN dimension
            hidden_dim: Model hidden dimension
            warmup_steps: Number of initial steps to skip metric collection or profiling, allowing the system to "warm up" for more stable measurements.
            sampling_rate: Frequency for collecting metrics (e.g., every Nth step), helping reduce overhead by not recording data on every forward pass.
        """
        super().__init__()
        self.router = original_router
        # Optional name of the wrapped module (set by MoEProfiler)
        self.name = name
        self.num_experts = num_experts
        self.expert_dim = expert_dim  # Expert FFN dimension
        self.hidden_dim = hidden_dim  # Model hidden dimension
        self.warmup_steps = warmup_steps
        self.sampling_rate = sampling_rate
        self.current_step = 0

        # Auto-detect dimensions if not provided
        if self.hidden_dim is None:
            self._auto_detect_dimensions()

        self.metrics = Metrics()
        self.enabled = True  # Toggle profiling on/off

        # CUDA events for precise timing
        self.use_cuda_events = torch.cuda.is_available()
        if self.use_cuda_events:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)

    def _auto_detect_dimensions(self):
        """Auto-detect model dimensions from router parameters"""
        try:
            for param in self.router.parameters():
                if len(param.shape) == 2:
                    self.hidden_dim = param.shape[0]
                    break
            if self.hidden_dim is None:
                self.hidden_dim = 4096  # Default fallback
        except:
            self.hidden_dim = 4096

        # Assume expert FFN dim is 4x hidden for MoE models
        if self.expert_dim is None:
            self.expert_dim = self.hidden_dim * 4

    def forward(self, x):
        if not self.enabled:
            return self.router(x)

        self.current_step += 1

        # Skip warmup steps
        in_warmup = self.current_step <= self.warmup_steps
        should_profile = (self.current_step % self.sampling_rate == 0) and not in_warmup

        if not should_profile:
            return self.router(x)

        # Start timing with CUDA events or fallback to CPU timing
        if self.use_cuda_events:
            self.start_event.record()

        start_time = time.perf_counter()

        # Get routing decision (call the wrapped router)
        router_output = self.router(x)

        # Case 1: Router returns (weights, indices) tuple
        if isinstance(router_output, tuple):
            routing_weights, expert_indices = router_output
        # Case 2: Router returns only logits (e.g., Mixtral gate, OlMoE)
        else:
            router_logits = router_output
            # Compute routing weights and select top-k experts
            routing_probs = torch.nn.functional.softmax(router_logits, dim=-1)
            # Assume top_k=2 for Mixtral, adjust if needed
            top_k = getattr(self.router, 'top_k', 2)
            routing_weights, expert_indices = torch.topk(routing_probs, 8, dim=-1)

        # Calculate confidence (entropy-based)
        psum = torch.sum(routing_probs**2)
        pcutoff = (64/63)*psum-(1/63)
        print("pcutoff",pcutoff,flush=True)
        # entropy = -torch.sum(routing_weights * torch.log(routing_weights + 1e-10), dim=-1).mean()
        # max_entropy = torch.log(torch.tensor(self.num_experts, dtype=torch.float32))
        # confidence = float(1.0 - (entropy / max_entropy).item())
        self.metrics.router_confidence.append(pcutoff)

        # Synchronize and record latency
        if self.use_cuda_events:
            self.end_event.record()
            torch.cuda.synchronize()
            latency = self.start_event.elapsed_time(self.end_event)  # Already in ms
        else:
            latency = (time.perf_counter() - start_time) * 1000

        self.metrics.latency_ms.append(latency)
        self.metrics.timestamps.append(time.time())

        # Get batch and sequence dimensions
        batch_size = x.shape[0]
        seq_len = x.shape[1] if len(x.shape) > 2 else 1
        hidden_dim = x.shape[-1]
        num_tokens = batch_size * seq_len

        # Calculate k per token (dynamic expert selection)
        # expert_indices shape: [batch_size, seq_len, k] or [num_tokens, k]
        expert_indices_flat = expert_indices.reshape(-1, expert_indices.shape[-1])

        # Count number of active experts per token (some might be masked/inactive)
        # Assuming -1 or negative values indicate inactive experts
        k_per_token_list = []
        for token_experts in expert_indices_flat:
            active_k = (token_experts >= 0).sum().item()
            k_per_token_list.append(active_k)

        # Store k distribution and average k
        self.metrics.k_distribution.append(k_per_token_list)
        avg_k = sum(k_per_token_list) / len(k_per_token_list) if len(k_per_token_list) > 0 else 0
        self.metrics.k_per_token.append(avg_k)

        sorted_probs, indices = torch.sort(routing_probs, descending=True, dim=1)  # Sort along dim=1
        cumsum = torch.cumsum(sorted_probs, dim=1)  # Cumsum along dim=1
        mask = cumsum <= pcutoff
        k = torch.sum(mask.float(), dim=1).item() + 1  # Sum along dim=1, then convert to scalar
        print("activated experts",k)
        print("sorted_probs",sorted_probs)
        print("cumsum",cumsum)

        self.metrics.active_experts.append(k)

        # Accurate FLOPs calculation with dynamic k
        # Router FLOPs: Linear projection (input @ weight) + softmax
        # Linear: 2 * num_tokens * hidden_dim * num_experts (multiply-add operations)
        # Softmax: ~5 * num_tokens * num_experts (exp, sum, divide)
        router_flops = (2 * num_tokens * hidden_dim * self.num_experts +
                        5 * num_tokens * self.num_experts)

        # Expert FLOPs: Accounts for dynamic k per token
        # TODO: check this. Each expert is a 2-layer FFN: hidden -> expert_dim -> hidden (Mixtral)
        # up_proj: 2 * hidden_dim * expert_dim
        # down_proj: 2 * expert_dim * hidden_dim
        # Sum across all tokens with their respective k values
        expert_flops = sum(k * (2 * hidden_dim * self.expert_dim +
                               2 * self.expert_dim * hidden_dim)
                          for k in k_per_token_list)

        total_flops = router_flops + expert_flops
        flops_per_token = total_flops / num_tokens
        router_flops_per_token = router_flops / num_tokens
        expert_flops_per_token = expert_flops / num_tokens

        self.metrics.flops_per_token.append(flops_per_token)
        self.metrics.router_flops_per_token.append(router_flops_per_token)
        self.metrics.expert_flops_per_token.append(expert_flops_per_token)

        # Track expert loads
        for expert_id in expert_indices.flatten().tolist():
            if 0 <= expert_id < self.num_experts:
                self.metrics.expert_loads[expert_id] += 1

        # Memory tracking
        if torch.cuda.is_available():
            self.metrics.memory_mb.append(torch.cuda.memory_allocated() / 1e6)

        self.metrics.step_count = self.current_step

        # Return in the same format as the original router
        if isinstance(router_output, tuple):
            return routing_weights, expert_indices
        else:
            # Return only logits to match Mixtral gate behavior
            return router_logits

    def reset_metrics(self):
        """Clear all recorded metrics"""
        self.metrics = Metrics()
        self.current_step = 0


class SelectableRouterWrapper(torch.nn.Module):
    """Wrap a router and allow injecting a custom selection function.

    The `selection_fn` should have signature:
        selection_fn(routing_probs, expert_indices, inputs, router_wrapper) -> (weights, indices) or indices

    - `routing_probs`: tensor [num_tokens, num_experts]
    - `expert_indices`: original indices (or None) returned by the base router
    - `inputs`: the original input `x` passed to forward
    - `router_wrapper`: this wrapper instance (can hold state)

    The function may return:
      - a tuple `(weights, indices)` (preferred), or
      - a single `indices` tensor, in which case weights will be taken from `routing_probs`.

    The wrapper preserves the original router return format when possible.
    """

    def __init__(self, base_router, selection_fn=None, num_experts: int = 8, name: Optional[str] = None):
        super().__init__()
        self.base = base_router
        self.selection_fn = selection_fn
        self.num_experts = num_experts
        self.name = name

        # State for inspection/testing
        self.last_probs = None
        self.last_indices = None
        self.last_weights = None
        # Minimal profiler-compatible state
        self.metrics = Metrics()
        self.enabled = True
        self.current_step = 0

    def forward(self, x):
        # Handle enabled flag for profiler compatibility
        if not getattr(self, 'enabled', True):
            return self.base(x)

        # Increment step counter for profiler compatibility
        self.current_step = getattr(self, 'current_step', 0) + 1

        # Call original router
        router_out = self.base(x)

        router_logits = None
        routing_probs = None
        orig_weights = None
        orig_indices = None

        if isinstance(router_out, tuple):
            orig_weights, orig_indices = router_out
            # Heuristic: if weights look like logits (outside 0..1), softmax
            try:
                if orig_weights.min() < 0 or orig_weights.max() > 1:
                    routing_probs = torch.nn.functional.softmax(orig_weights, dim=-1)
                else:
                    routing_probs = orig_weights
            except Exception:
                routing_probs = orig_weights
        else: # This is always the case for OLMoE and the toy demo
            router_logits = router_out
            routing_probs = torch.nn.functional.softmax(router_logits, dim=-1)

        # Store for inspection
        try:
            self.last_probs = routing_probs.detach().cpu()
        except Exception:
            self.last_probs = None

        # Let selection_fn decide final indices/weights
        final_weights = orig_weights
        final_indices = orig_indices

        if self.selection_fn is not None:
            sel_out = self.selection_fn(routing_probs, orig_indices, x, self)
            if isinstance(sel_out, tuple) and len(sel_out) >= 2:
                final_weights, final_indices = sel_out[0], sel_out[1]
            elif isinstance(sel_out, torch.Tensor):
                final_indices = sel_out
                final_weights = routing_probs
            elif sel_out is None:
                # keep original
                pass
            else:
                # Unknown return so keep original
                pass

        # Save last selection
        try:
            self.last_indices = final_indices.detach().cpu() if final_indices is not None else None
            self.last_weights = final_weights.detach().cpu() if final_weights is not None else None
        except Exception:
            self.last_indices = None
            self.last_weights = None

        # Return in same format as base router
        if isinstance(router_out, tuple):
            return final_weights, final_indices
        else:
            # base returned logits; pass them through unchanged so calling code expecting logits still works
            return router_logits

    def reset_metrics(self):
        """Reset profiler-compatible metrics and step counter."""
        self.metrics = Metrics()
        self.current_step = 0


def attach_selector_to_model(model, selection_fn, name_match: str = 'gate'):
    """Find modules with `name_match` in their module path and replace them with SelectableRouterWrapper.

    Returns a list of wrappers created.
    """
    wrappers = []
    for name, module in list(model.named_modules()):
        # skip top-level model entry (named_modules yields parent too); we want leaf modules
        if name == '':
            continue
        if name_match.lower() in name.lower():
            # Skip if already wrapped
            if isinstance(module, SelectableRouterWrapper):
                continue

            print(f"Attaching selector wrapper to: {name} (type: {type(module).__name__})")
            wrapper = SelectableRouterWrapper(module, selection_fn=selection_fn, name=name)
            # set on parent
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            if parent_name:
                parent = model.get_submodule(parent_name)
            else:
                parent = model
            setattr(parent, child_name, wrapper)
            # move to same device as model parameters if possible
            try:
                model_device = next(model.parameters()).device
                wrapper.to(model_device)
            except Exception:
                pass
            wrappers.append(wrapper)
    return wrappers


def example_topk_selector(routing_probs, orig_indices, x, router_wrapper, k: int = 8, threshold: float = 0.0):
    """Example selection function: pick top-k per token, optionally threshold low-confidence tokens by setting indices to -1.

    Returns (weights, indices) where weights are topk probs and indices are topk indices.
    """
    # routing_probs: [num_tokens, num_experts]
    if routing_probs is None:
        return None

    # Support both torch tensors and numpy arrays/shims
    if hasattr(routing_probs, 'detach'):
        vals, idx = torch.topk(routing_probs, k, dim=-1)
        if threshold is not None and threshold > 0.0:
            low_conf_mask = vals[:, 0] < threshold
            if low_conf_mask.any():
                idx[low_conf_mask, :] = -1
                vals[low_conf_mask, :] = 0.0
        return vals, idx
    else:
        # numpy path: routing_probs is array-like
        arr = routing_probs.numpy() if hasattr(routing_probs, 'numpy') else np.asarray(routing_probs)
        idx = np.argsort(-arr, axis=1)[:, :k]
        vals = np.take_along_axis(arr, idx, axis=1)
        if threshold is not None and threshold > 0.0:
            low_conf_mask = vals[:, 0] < threshold
            if low_conf_mask.any():
                idx[low_conf_mask, :] = -1
                vals[low_conf_mask, :] = 0.0
        return vals, idx


def kneedle_selector(routing_probs, orig_indices, x, router_wrapper, k_max: int = 8):
    """Select k per token using a kneedle elbow detection on sorted probabilities.

    For each token:
      - Sort probabilities descending
      - Normalize index x in [0,1] and probs y in [0,1]
      - Compute distance from point (x,y) to the line y = 1 - x (diagonal from (0,1) to (1,0))
      - Choose the index with maximum positive distance as the elbow (k = idx+1)
      - Cap k by `k_max` and at least 1

    Returns (vals, idx) where idx shape is [num_tokens, k] and vals are the corresponding probs.
    """
    if routing_probs is None:
        return None

    # routing_probs may be torch tensor or numpy-like shim
    is_torch = hasattr(routing_probs, 'detach') and not isinstance(routing_probs, (list, tuple))

    if is_torch:
        probs = routing_probs.detach()
        device = probs.device
        probs_sorted, indices_sorted = torch.sort(probs, descending=True, dim=-1)
        n_experts = probs_sorted.size(-1)

        # normalize x and y
        x = torch.linspace(0, 1, steps=n_experts, device=device).unsqueeze(0).expand(probs_sorted.size(0), -1)
        # normalize y to [0,1] by dividing by max per row (first element since sorted desc)
        y = probs_sorted / (probs_sorted[:, :1] + 1e-12)

        # diagonal line y = 1 - x
        diag = 1.0 - x
        # distance (signed) from point to diagonal (positive means above diagonal)
        dist = (y - diag) / math.sqrt(2.0)

        # For each token, find index of max distance
        max_vals, max_idx = torch.max(dist, dim=1)
        ks = (max_idx + 1).clamp(min=1)
        ks = torch.clamp(ks, max=k_max)

        # Build final indices & vals per token
        final_idxs = []
        final_vals = []
        for i in range(probs_sorted.size(0)):
            k = int(ks[i].item())
            idxs = indices_sorted[i, :k]
            vals = probs_sorted[i, :k]
            # pad to k_max
            if k < k_max:
                pad = k_max - k
                idxs = torch.cat([idxs, torch.full((pad,), -1, dtype=idxs.dtype, device=device)])
                vals = torch.cat([vals, torch.zeros((pad,), dtype=vals.dtype, device=device)])
            final_idxs.append(idxs)
            final_vals.append(vals)

        final_idxs = torch.stack(final_idxs, dim=0)
        final_vals = torch.stack(final_vals, dim=0)
        return final_vals, final_idxs

    else:
        # numpy shim
        arr = _to_numpy(routing_probs)
        n_tokens, n_experts = arr.shape
        idxs_out = []
        vals_out = []
        for i in range(n_tokens):
            row = arr[i]
            sidx = np.argsort(-row)
            svals = row[sidx]
            xs = np.linspace(0, 1, n_experts)
            ys = svals / (svals[0] + 1e-12)
            diag = 1.0 - xs
            dist = (ys - diag) / math.sqrt(2.0)
            max_idx = int(np.argmax(dist))
            k = max(1, min(k_max, max_idx + 1))
            sel_idx = sidx[:k].tolist()
            sel_vals = svals[:k].tolist()
            # pad
            if k < k_max:
                sel_idx += [-1] * (k_max - k)
                sel_vals += [0.0] * (k_max - k)
            idxs_out.append(sel_idx)
            vals_out.append(sel_vals)
        return np.array(vals_out), np.array(idxs_out)


def cumsum_selector(routing_probs, orig_indices, x, router_wrapper, mass_threshold: float = 0.9, k_max: int = 8):
    """Select minimum k per token so cumulative mass >= mass_threshold (cap at k_max)."""
    if routing_probs is None:
        return None
    if hasattr(routing_probs, 'detach'):
        probs = routing_probs.detach()
        probs_sorted, indices_sorted = torch.sort(probs, descending=True, dim=-1)
        cums = torch.cumsum(probs_sorted, dim=-1)
        ks = (cums >= mass_threshold).float().argmax(dim=-1) + 1
        ks = torch.clamp(ks, min=1, max=k_max)
        final_idxs = []
        final_vals = []
        for i in range(probs_sorted.size(0)):
            k = int(ks[i].item())
            idxs = indices_sorted[i, :k]
            vals = probs_sorted[i, :k]
            if k < k_max:
                pad = k_max - k
                idxs = torch.cat([idxs, torch.full((pad,), -1, dtype=idxs.dtype, device=idxs.device)])
                vals = torch.cat([vals, torch.zeros((pad,), dtype=vals.dtype, device=vals.device)])
            final_idxs.append(idxs)
            final_vals.append(vals)
        return torch.stack(final_vals, dim=0), torch.stack(final_idxs, dim=0)
    else:
        arr = _to_numpy(routing_probs)
        n_tokens, n_experts = arr.shape
        idxs_out = []
        vals_out = []
        for i in range(n_tokens):
            row = arr[i]
            sidx = np.argsort(-row)
            svals = row[sidx]
            cums = np.cumsum(svals)
            k = int(np.searchsorted(cums, mass_threshold) + 1)
            k = max(1, min(k_max, k))
            sel_idx = sidx[:k].tolist()
            sel_vals = svals[:k].tolist()
            if k < k_max:
                sel_idx += [-1] * (k_max - k)
                sel_vals += [0.0] * (k_max - k)
            idxs_out.append(sel_idx)
            vals_out.append(sel_vals)
        return np.array(vals_out), np.array(idxs_out)


def entropy_selector(routing_probs, orig_indices, x, router_wrapper, k_max: int = 8):
    """Select k based on entropy: effective_k = ceil(exp(entropy)), capped at k_max."""
    if routing_probs is None:
        return None
    if hasattr(routing_probs, 'detach'):
        probs = routing_probs.detach()
        eps = 1e-12
        ent = -torch.sum(probs * torch.log(probs + eps), dim=-1)
        eff_k = torch.ceil(torch.exp(ent)).long()
        eff_k = torch.clamp(eff_k, min=1, max=k_max)
        probs_sorted, indices_sorted = torch.sort(probs, descending=True, dim=-1)
        final_idxs = []
        final_vals = []
        for i in range(probs_sorted.size(0)):
            k = int(eff_k[i].item())
            idxs = indices_sorted[i, :k]
            vals = probs_sorted[i, :k]
            if k < k_max:
                pad = k_max - k
                idxs = torch.cat([idxs, torch.full((pad,), -1, dtype=idxs.dtype, device=idxs.device)])
                vals = torch.cat([vals, torch.zeros((pad,), dtype=vals.dtype, device=vals.device)])
            final_idxs.append(idxs)
            final_vals.append(vals)
        return torch.stack(final_vals, dim=0), torch.stack(final_idxs, dim=0)
    else:
        arr = _to_numpy(routing_probs)
        n_tokens, n_experts = arr.shape
        idxs_out = []
        vals_out = []
        for i in range(n_tokens):
            row = arr[i]
            eps = 1e-12
            ent = -np.sum(row * np.log(row + eps))
            k = int(np.ceil(np.exp(ent)))
            k = max(1, min(k_max, k))
            sidx = np.argsort(-row)
            svals = row[sidx]
            sel_idx = sidx[:k].tolist()
            sel_vals = svals[:k].tolist()
            if k < k_max:
                sel_idx += [-1] * (k_max - k)
                sel_vals += [0.0] * (k_max - k)
            idxs_out.append(sel_idx)
            vals_out.append(sel_vals)
        return np.array(vals_out), np.array(idxs_out)


def gap_ratio_selector(routing_probs, orig_indices, x, router_wrapper, ratio_threshold: float = 2.0, k_max: int = 8):
    """Select k where the ratio p_k / p_{k+1} exceeds ratio_threshold (cap at k_max).

    If no gap exceeds threshold, fall back to min(k_max, top-1..k) or top-1.
    """
    if routing_probs is None:
        return None
    if hasattr(routing_probs, 'detach'):
        probs = routing_probs.detach()
        probs_sorted, indices_sorted = torch.sort(probs, descending=True, dim=-1)
        n_experts = probs_sorted.size(-1)
        final_idxs = []
        final_vals = []
        for i in range(probs_sorted.size(0)):
            row = probs_sorted[i]
            k = 1
            for j in range(n_experts - 1):
                p_k = float(row[j].item())
                p_k1 = float(row[j+1].item())
                if p_k1 <= 0:
                    continue
                if (p_k / p_k1) >= ratio_threshold:
                    k = j + 1
                    break
            k = max(1, min(k, k_max))
            idxs = indices_sorted[i, :k]
            vals = probs_sorted[i, :k]
            if k < k_max:
                pad = k_max - k
                idxs = torch.cat([idxs, torch.full((pad,), -1, dtype=idxs.dtype, device=idxs.device)])
                vals = torch.cat([vals, torch.zeros((pad,), dtype=vals.dtype, device=vals.device)])
            final_idxs.append(idxs)
            final_vals.append(vals)
        return torch.stack(final_vals, dim=0), torch.stack(final_idxs, dim=0)
    else:
        arr = _to_numpy(routing_probs)
        n_tokens, n_experts = arr.shape
        idxs_out = []
        vals_out = []
        for i in range(n_tokens):
            row = np.sort(arr[i])[::-1]
            sidx = np.argsort(-arr[i])
            k = 1
            for j in range(n_experts - 1):
                p_k = row[j]
                p_k1 = row[j+1]
                if p_k1 <= 0:
                    continue
                if (p_k / p_k1) >= ratio_threshold:
                    k = j + 1
                    break
            k = max(1, min(k, k_max))
            sel_idx = sidx[:k].tolist()
            sel_vals = arr[i][sel_idx].tolist()
            if k < k_max:
                sel_idx += [-1] * (k_max - k)
                sel_vals += [0.0] * (k_max - k)
            idxs_out.append(sel_idx)
            vals_out.append(sel_vals)
        return np.array(vals_out), np.array(idxs_out)


def run_selector_demo():
    """Lightweight demo using a DummyRouter to verify selection wrapper behavior.

    Prints last_probs and last_indices for inspection.
    """
    class DummyRouter(torch.nn.Module):
        def __init__(self, num_experts=16):
            super().__init__()
            self.num_experts = num_experts

        def forward(self, x):
            # Create deterministic logits from input mean to aid testing
            batch = x.shape[0]
            seq = x.shape[1] if x.dim() > 2 else 1
            num_tokens = batch * seq
            # logits shaped [num_tokens, num_experts]
            logits = torch.randn(num_tokens, self.num_experts)
            return logits

    # Create dummy input
    x = torch.randn(2, 3, 16)  # batch=2, seq=3, hidden=16

    base = DummyRouter(num_experts=16)
    wrapper = SelectableRouterWrapper(base, selection_fn=lambda p, i, x, r: example_topk_selector(p, i, x, r, k=4, threshold=0.0), num_experts=16, name='demo.gate')
    out = wrapper(x)
    print("Wrapper returned (base-logs):", isinstance(out, torch.Tensor))
    print("Last probs shape:", None if wrapper.last_probs is None else wrapper.last_probs.shape)
    print("Last indices shape:", None if wrapper.last_indices is None else wrapper.last_indices.shape)
    print("Example last indices (first tokens):\n", wrapper.last_indices[:4])


def run_selector_demo_no_torch():
    """Fallback demo that does not require PyTorch — uses NumPy only.

    This simulates routing probabilities and runs the example_topk_selector
    to show how indices/weights would be selected.
    """
    import numpy as _np

    num_experts = 16
    batch = 2
    seq = 3
    num_tokens = batch * seq

    # Simulate routing_probs as softmaxed random logits
    rng = _np.random.RandomState(0)
    logits = rng.randn(num_tokens, num_experts).astype(_np.float32)
    exp = _np.exp(logits - _np.max(logits, axis=1, keepdims=True))
    probs = exp / _np.sum(exp, axis=1, keepdims=True)

    # Convert to a minimal stand-in that our selector expects (numpy -> torch-like tensor)
    # We'll adapt example_topk_selector which expects a tensor; create a tiny shim
    class Shim:
        def __init__(self, arr):
            self._arr = arr

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self._arr

        def __array__(self):
            return self._arr

    shim_probs = Shim(probs)

    # Adapted selector using numpy directly
    def numpy_topk_selector(shim_probs, orig_indices, x, router_wrapper, k: int = 8, threshold: float = 0.0):
        arr = shim_probs.numpy()
        idx = _np.argsort(-arr, axis=1)[:, :k]
        vals = _np.take_along_axis(arr, idx, axis=1)
        if threshold is not None and threshold > 0.0:
            low_conf = vals[:, 0] < threshold
            if low_conf.any():
                idx[low_conf, :] = -1
                vals[low_conf, :] = 0.0
        return vals, idx

    vals, idx = numpy_topk_selector(shim_probs, None, None, None, k=4, threshold=0.0)
    print("Simulated routing_probs shape:", probs.shape)
    print("Top-4 indices shape:", idx.shape)
    print("Example top-4 indices (first 4 tokens):\n", idx[:4])
    print("Example top-4 weights (first 4 tokens):\n", vals[:4])

    def _calculate_gini_coefficient(self, values):
        """
        Calculate Gini coefficient for load imbalance
        0 = perfect equality, 1 = perfect inequality
        """
        values = np.array(sorted(values))
        n = len(values)
        if n == 0 or values.sum() == 0:
            return 0.0
        cumsum = np.cumsum(values)
        return (2 * np.sum((np.arange(1, n + 1)) * values)) / (n * cumsum[-1]) - (n + 1) / n

    def _calculate_entropy(self, values):
        """
        Calculate normalized entropy of distribution
        0 = concentrated, 1 = uniform
        """
        values = np.array(values)
        if values.sum() == 0:
            return 0.0
        probs = values / values.sum()
        probs = probs[probs > 0]  # Remove zeros
        entropy = -np.sum(probs * np.log(probs))
        max_entropy = np.log(len(values))  # Maximum entory occurs when uniform distribution
        return entropy / max_entropy if max_entropy > 0 else 0.0 # Normalized entropy

    def _calculate_cv(self, values):
        """
        Calculate coefficient of variation (CV = std / mean)
        Measures relative variability
        """
        values = np.array(values)
        if len(values) == 0 or values.mean() == 0:
            return 0.0
        return values.std() / values.mean()

    def get_summary(self):
        """Get comprehensive statistical summary of metrics"""
        df = self.metrics.to_df()
        if len(df) == 0:
            return "No metrics collected yet"

        # Expert load statistics
        expert_loads = np.array(list(self.metrics.expert_loads.values()))

        # k distribution statistics
        k_values = np.array(self.metrics.k_per_token)

        return {
            # FLOPs statistics
            'flops_total_mean': df['flops_total'].mean(),
            'flops_total_std': df['flops_total'].std(),
            'flops_total_p50': df['flops_total'].median(),
            'flops_total_p95': df['flops_total'].quantile(0.95),
            'flops_total_p99': df['flops_total'].quantile(0.99),
            'flops_router_mean': df['flops_router'].mean(),
            'flops_expert_mean': df['flops_expert'].mean(),

            # Latency statistics
            'latency_mean_ms': df['latency_ms'].mean(),
            'latency_std_ms': df['latency_ms'].std(),
            'latency_min_ms': df['latency_ms'].min(),
            'latency_max_ms': df['latency_ms'].max(),
            'latency_p50_ms': df['latency_ms'].median(),
            'latency_p95_ms': df['latency_ms'].quantile(0.95),
            'latency_p99_ms': df['latency_ms'].quantile(0.99),

            # Expert utilization
            'active_experts_mean': df['active_experts'].mean(),
            'active_experts_std': df['active_experts'].std(),

            # Dynamic k statistics
            'k_mean': k_values.mean() if len(k_values) > 0 else 0,
            'k_std': k_values.std() if len(k_values) > 0 else 0,
            'k_min': k_values.min() if len(k_values) > 0 else 0,
            'k_max': k_values.max() if len(k_values) > 0 else 0,

            # Load balancing metrics
            'expert_load_mean': expert_loads.mean(),
            'expert_load_std': expert_loads.std(),
            'expert_load_cv': self._calculate_cv(expert_loads),
            'expert_load_gini': self._calculate_gini_coefficient(expert_loads),
            'expert_load_entropy': self._calculate_entropy(expert_loads),

            # Confidence metrics
            'router_confidence_mean': df['router_conf'].mean(),
            'router_confidence_std': df['router_conf'].std(),

            # Memory
            'memory_mean_mb': df['memory_mb'].mean() if 'memory_mb' in df and len(df['memory_mb']) > 0 else 0,
            'memory_max_mb': df['memory_mb'].max() if 'memory_mb' in df and len(df['memory_mb']) > 0 else 0,

            # Overall statistics
            'total_steps': len(df),
            'total_tokens_profiled': len(df) * (df.index[0] if len(df) > 0 else 0),
        }

# ============================================
# 3. SIMPLE PROFILER
# ============================================

class MoEProfiler:
    """Simple profiler for experiments"""

    def __init__(self, model, selection_fn: Optional[callable] = None, selector_name_match: str = 'gate'):
        self.model = model
        self.wrappers = []
        # Optional selector to inject custom expert-selection logic
        # If provided, profiler will replace matched router modules with SelectableRouterWrapper
        self.selection_fn = selection_fn
        self.selector_name_match = selector_name_match
        # Ensure log file exists (best-effort) so we can diagnose missing logs
        try:
            LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(str(LOG_PATH), 'a') as _f:
                _f.write(f"{time.time():.3f}\tPID={os.getpid()}\tLOG_INIT\n")
        except Exception:
            # Non-fatal: if /tmp isn't writable in this environment we'll fall back to stderr-only logging
            try:
                os.write(2, (f"LOG_INIT_FAILED PID={os.getpid()}\n").encode())
            except Exception:
                pass

        self._wrap_routers()

    def _wrap_routers(self):
        """Find and wrap all MoE routers in the model"""
        # helper to infer number of experts if available on the module
        def _infer(module):
            for attr in ('num_experts', 'n_experts', 'num_expert', 'n_expert', 'n'):
                try:
                    val = getattr(module, attr)
                    if isinstance(val, int) and val > 0:
                        return val
                except Exception:
                    continue
            return 8
        for name, module in self.model.named_modules():
            # Skip if already wrapped
            # This doesn't seem to ever happen
            if isinstance(module, SimpleRouterWrapper):
                continue

            # Check module type names for MoE-specific classes
            module_type = type(module).__name__

            # Match MoE routing modules:
            # - Mixtral: wrap 'block_sparse_moe.gate' (returns logits)
            # - OLMoE: wrap entire 'mlp' block (OlmoeSparseMoeBlock) to intercept expert selection
            # - General: 'router' in name

            is_moe_block = 'SparseMoe' in module_type and 'Block' in module_type
            is_gate = (name.endswith('.gate') and 'gate_proj' not in name.lower() and
                      'block_sparse_moe' in name.lower())
            is_router = 'router' in name.lower() and 'gate_proj' not in name.lower()

            is_gate_or_router = is_moe_block or is_gate or is_router

            if is_gate_or_router:
                print(f"Wrapping router: {name} (type: {module_type})")
                # Determine number of experts (best-effort)
                num_experts = _infer(module)

                # Replace with SelectableRouterWrapper if a selection function was provided,
                # otherwise fall back to the original SimpleRouterWrapper
                if self.selection_fn is not None:
                    wrapper = SelectableRouterWrapper(module, selection_fn=self.selection_fn, num_experts=num_experts, name=name)
                else:
                    wrapper = SimpleRouterWrapper(module, num_experts=num_experts, name=name)
                self.wrappers.append(wrapper)

                # Determine parent module and child attribute name and set the wrapper
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                if parent_name:
                    parent = self.model.get_submodule(parent_name)
                else:
                    parent = self.model

                setattr(parent, child_name, wrapper)

                # Move wrapper to model device (if model has parameters/buffers on a device)
                model_device = None
                try:
                    model_device = next(self.model.parameters()).device
                except StopIteration:
                    try:
                        model_device = next(self.model.buffers()).device
                    except StopIteration:
                        model_device = None

                if model_device is not None:
                    try:
                        wrapper.to(model_device)
                        print(f"Moved wrapper '{name}' to device {model_device}")
                    except Exception:
                        # Non-fatal: don't break wrapping if move fails
                        print(f"Warning: failed to move wrapper '{name}' to device {model_device}")

    def start(self):
        """Start profiling"""
        for wrapper in self.wrappers:
            wrapper.enabled = True
            wrapper.reset_metrics()

    def stop(self):
        """Stop profiling"""
        for wrapper in self.wrappers:
            wrapper.enabled = False

    def get_metrics(self):
        """Get all collected metrics"""
        all_metrics = {}
        for i, wrapper in enumerate(self.wrappers):
            all_metrics[f'layer_{i}'] = wrapper.metrics
        return all_metrics

    def save_csv(self, filepath='metrics.csv'):
        """Save metrics to CSV"""
        dfs = []
        for i, wrapper in enumerate(self.wrappers):
            df = wrapper.metrics.to_df()
            df['layer'] = i
            dfs.append(df)

        if dfs:
            combined = pd.concat(dfs)
            combined.to_csv(filepath, index=False)
            print(f"Saved {len(combined)} rows to {filepath}")

    def dump_wrapper_states(self):
        """Return and log a compact state summary for all wrappers.

        This writes a best-effort line to /tmp/moeprofiler_debug.log and to
        stderr (unbuffered), then returns the list of dicts so calling code can
        inspect it programmatically.
        """
        info = []
        for w in self.wrappers:
            info.append({
                'name': getattr(w, 'name', '<unknown>'),
                'enabled': bool(getattr(w, 'enabled', False)),
                'current_step': int(getattr(w, 'current_step', 0))
            })
        try:
            with open(str(LOG_PATH), 'a') as f:
                f.write(f"{time.time():.3f}\tPID={os.getpid()}\tDUMP_WRAPPERS\t{info}\n")
        except Exception:
            pass
        try:
            os.write(2, (f"DUMP_WRAPPERS: {info}\n").encode())
        except Exception:
            pass
        return info

    def print_summary(self):
        """Print comprehensive statistical summary"""
        print("\n" + "="*60)
        print("=== MoE Profiling Summary ===")
        print("="*60)

        for i, wrapper in enumerate(self.wrappers):
            print(f"\n{'─'*60}")
            print(f"Layer {i}:")
            print('─'*60)
            summary = wrapper.get_summary()
            if isinstance(summary, dict):
                # Organize output by category
                print("\n  FLOPs Metrics:")
                for k, v in summary.items():
                    if k.startswith('flops_'):
                        print(f"    {k:30s}: {v:>12.2f}" if isinstance(v, float) else f"    {k:30s}: {v:>12}")

                print("\n  Latency Metrics:")
                for k, v in summary.items():
                    if k.startswith('latency_'):
                        print(f"    {k:30s}: {v:>12.2f} ms" if isinstance(v, float) else f"    {k:30s}: {v:>12}")

                print("\n  Expert Utilization:")
                for k, v in summary.items():
                    if k.startswith('k_') or k.startswith('active_experts'):
                        print(f"    {k:30s}: {v:>12.2f}" if isinstance(v, float) else f"    {k:30s}: {v:>12}")

                print("\n  Load Balancing:")
                for k, v in summary.items():
                    if k.startswith('expert_load'):
                        print(f"    {k:30s}: {v:>12.2f}" if isinstance(v, float) else f"    {k:30s}: {v:>12}")

                print("\n  Memory & Confidence:")
                for k, v in summary.items():
                    if k.startswith('memory_') or k.startswith('router_confidence'):
                        print(f"    {k:30s}: {v:>12.2f}" if isinstance(v, float) else f"    {k:30s}: {v:>12}")

                print("\n  Overall Statistics:")
                for k, v in summary.items():
                    if k.startswith('total_'):
                        print(f"    {k:30s}: {v:>12.2f}" if isinstance(v, float) else f"    {k:30s}: {v:>12}")

            else:
                print(f"  {summary}")

        print("\n" + "="*60)

# ============================================
# 4. BENCHMARK RUNNER
# ============================================

class SimpleBenchmark:
    """Run and compare different configurations"""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.results = {}

    def run_config(self, name, input_texts, config_fn=None):
        """Run a single configuration"""
        print(f"\nRunning config: {name}")

        # Apply configuration
        if config_fn:
            config_fn(self.model)

        # Setup profiler
        profiler = MoEProfiler(self.model)
        profiler.start()

        # Run inference
        for text in input_texts:
            inputs = self.tokenizer(text, return_tensors='pt')
            with torch.no_grad():
                outputs = self.model(**inputs)

        # Get results
        profiler.stop()
        metrics = profiler.get_metrics()

        # Save
        profiler.save_csv(f"{name}_metrics.csv")
        self.results[name] = metrics

        profiler.print_summary()

    def compare(self):
        """Compare all configurations"""
        comparison = []

        for config_name, metrics in self.results.items():
            total_flops = 0
            total_latency = 0
            total_tokens = 0

            for layer_metrics in metrics.values():
                df = layer_metrics.to_df()
                if len(df) > 0:
                    total_flops += df['flops'].sum()
                    total_latency += df['latency_ms'].sum()
                    total_tokens += len(df)

            comparison.append({
                'config': config_name,
                'total_flops': total_flops,
                'avg_latency_ms': total_latency / total_tokens if total_tokens > 0 else 0,
                'total_tokens': total_tokens
            })

        df = pd.DataFrame(comparison)
        df.to_csv('comparison.csv', index=False)
        print("\n=== Configuration Comparison ===")
        print(df.to_string())
        return df

# ============================================
# 5. YOUR CUSTOM MIDDLEWARE (Example)
# ============================================

class YourRouterMiddleware(torch.nn.Module):
    """Your custom router with confidence-based selection"""

    def __init__(self, base_router, confidence_threshold=0.7):
        super().__init__()
        self.router = base_router
        self.confidence_threshold = confidence_threshold

    def forward(self, x):
        # Get base routing
        weights, indices = self.router(x)

        # Calculate confidence (max weight)
        confidence = weights.max(dim=-1)[0]

        # Skip routing if confidence too low
        mask = confidence < self.confidence_threshold
        indices[mask] = -1  # Skip these tokens

        # Track semantic confidence (simplified)
        # You can add your actual semantic confidence logic here
        semantic_conf = torch.cosine_similarity(
            x.mean(dim=1),
            x.mean(dim=1).roll(1, dims=0)
        ).mean()

        # Store in wrapper if it exists
        if hasattr(self.router, 'metrics'):
            self.router.metrics.semantic_confidence.append(semantic_conf.item())

        return weights, indices

# ============================================
# 6. USAGE EXAMPLE
# ============================================

def example_usage():
    """
    Complete example of how to use this in your notebook
    """

    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")

    # Test texts
    test_texts = [
        "What is machine learning?",
        "Explain quantum computing",
        "How does photosynthesis work?"
    ]

    # Create benchmark runner
    benchmark = SimpleBenchmark(model, tokenizer)

    # Run baseline
    benchmark.run_config("baseline", test_texts)

    # Run with your middleware
    def apply_middleware(model):
        for name, module in model.named_modules():
            if 'router' in name.lower():
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent = model.get_submodule(parent_name)
                custom = YourRouterMiddleware(module, confidence_threshold=0.8)
                setattr(parent, child_name, custom)

    benchmark.run_config("with_middleware", test_texts, apply_middleware)

    # Compare results
    comparison = benchmark.compare()

    print("\nDone! Check these files:")
    print("- baseline_metrics.csv")
    print("- with_middleware_metrics.csv")
    print("- comparison.csv")

# ============================================
# 7. MULTI-GPU SUPPORT (Simple)
# ============================================

def setup_multi_gpu(model):
    """Simple multi-GPU setup"""
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
    elif torch.cuda.is_available():
        model = model.cuda()
    return model

# ============================================
# 8. NOTEBOOK HELPERS
# ============================================

def quick_profile(model, tokenizer, text="Hello world", num_iterations=10):
    """One-line profiling for notebooks"""
    profiler = MoEProfiler(model)
    profiler.start()

    inputs = tokenizer(text, return_tensors='pt')
    for _ in range(num_iterations):
        with torch.no_grad():
            model(**inputs)

    profiler.stop()
    profiler.print_summary()
    profiler.save_csv('quick_profile.csv')
    return profiler

# For Colab
def colab_setup():
    """Setup for Google Colab"""
    # Install dependencies
    print("!pip install transformers torch pandas")

    # Check GPU
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("No GPU available, using CPU")

if __name__ == "__main__":
    print("MoE Profiler loaded! Use example_usage() to see how it works.")
