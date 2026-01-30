"""
Selection Functions for Dynamic Expert Routing

Collection of selector functions that implement different strategies
for choosing which experts to activate per token.
"""

import torch
import numpy as np
import math
from .metrics import _to_numpy


def topk_selector(routing_probs, orig_indices, x, router_wrapper, k: int = 8, threshold: float = 0.0):
    """
    Top-k selector: pick top-k experts per token.

    Args:
        routing_probs: [num_tokens, num_experts] probability distribution
        orig_indices: Original indices from router (unused here)
        x: Input hidden states (unused here)
        router_wrapper: Reference to the wrapper (unused here)
        k: Number of experts to select
        threshold: Optional confidence threshold (set indices to -1 for low confidence)

    Returns:
        Tuple of (weights, indices) where weights are top-k probs and indices are expert IDs
    """
    if routing_probs is None:
        return None

    # Support both torch tensors and numpy arrays
    if hasattr(routing_probs, 'detach'):
        vals, idx = torch.topk(routing_probs, k, dim=-1)
        if threshold is not None and threshold > 0.0:
            low_conf_mask = vals[:, 0] < threshold
            if low_conf_mask.any():
                idx[low_conf_mask, :] = -1
                vals[low_conf_mask, :] = 0.0
        return vals, idx
    else:
        # numpy path
        arr = routing_probs.numpy() if hasattr(routing_probs, 'numpy') else np.asarray(routing_probs)
        idx = np.argsort(-arr, axis=1)[:, :k]
        vals = np.take_along_axis(arr, idx, axis=1)
        if threshold is not None and threshold > 0.0:
            low_conf_mask = vals[:, 0] < threshold
            if low_conf_mask.any():
                idx[low_conf_mask, :] = -1
                vals[low_conf_mask, :] = 0.0
        return vals, idx

def geometric_kneedle_selector(routing_probs, orig_indices, x, router_wrapper, k_max: int = 8):
    """
    Kneedle elbow detection selector (proven implementation).

    For each token:
      - Sort probabilities descending
      - Normalize index x in [0,1]
      - Normalize probs y in [0,1] by INVERTING: y_norm = (y_max - y) / (y_max - y_min)
      - Compute distance: y_norm - x_norm (finds elbow on y=x diagonal)
      - Choose the index with maximum distance as the elbow (k = idx+1)
      - Cap k by `k_max` and at least 1

    Args:
        routing_probs: [num_tokens, num_experts] probability distribution
        orig_indices: Original indices (unused)
        x: Input hidden states (unused)
        router_wrapper: Reference to wrapper (unused)
        k_max: Maximum k value

    Returns:
        Tuple of (vals, idx) with dynamic k per token (padded to k_max)
    """
    if routing_probs is None:
        return None

    is_torch = hasattr(routing_probs, 'detach') and not isinstance(routing_probs, (list, tuple))

    if is_torch:
        probs = routing_probs.detach()
        device = probs.device
        probs_sorted, indices_sorted = torch.sort(probs, descending=True, dim=-1)
        n_experts = probs_sorted.size(-1)

        # normalize x and y (matching proven kneedle implementation)
        x_norm = torch.linspace(0, 1, steps=n_experts, device=device).unsqueeze(0).expand(probs_sorted.size(0), -1)
        # normalize y: INVERT so it goes from 0 to 1 (matching proven implementation)
        # y_norm = (y_max - y) / (y_max - y_min)
        y_norm = (probs_sorted[:, :1] - probs_sorted) / (probs_sorted[:, :1] - probs_sorted[:, -1:] + 1e-12)

        # distance: find max of (y_norm - x_norm) to detect elbow on y=x diagonal
        dist = y_norm - x_norm

        # For each token, find index of max distance
        max_vals, max_idx = torch.max(dist, dim=1)
        ks = (max_idx + 1).clamp(min=1, max=k_max)

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
                # to resolve expected sequence of equal length, need padding
                idxs = torch.cat([idxs, torch.full((pad,), -1, dtype=idxs.dtype, device=device)])
                vals = torch.cat([vals, torch.zeros((pad,), dtype=vals.dtype, device=device)])
            final_idxs.append(idxs)
            final_vals.append(vals)

        return torch.stack(final_vals, dim=0), torch.stack(final_idxs, dim=0)

    else:
        # numpy path
        arr = _to_numpy(routing_probs)
        n_tokens, n_experts = arr.shape
        idxs_out = []
        vals_out = []
        for i in range(n_tokens):
            row = arr[i]
            sidx = np.argsort(-row)
            svals = row[sidx]
            xs = np.linspace(0, 1, n_experts)
            # INVERT y normalization to match proven implementation: (y_max - y) / (y_max - y_min)
            ys = (svals[0] - svals) / (svals[0] - svals[-1] + 1e-12)
            # Distance: find max of (y_norm - x_norm) on y=x diagonal
            dist = ys - xs
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


def kneedle_selector(routing_probs, orig_indices, x, router_wrapper, k_max: int = 8):
    """
    Kneedle elbow detection selector.

    For each token:
      - Sort probabilities descending
      - Normalize index x in [0,1]
      - Normalize probs y in [0,1] by INVERTING: y_norm = (y_max - y) / (y_max - y_min)
      - Compute distance: y_norm - x_norm (finds elbow on y=x diagonal)
      - Choose the index with maximum distance as the elbow (k = idx+1)
      - Cap k by `k_max` and at least 1

    Args:
        routing_probs: [num_tokens, num_experts] probability distribution
        orig_indices: Original indices (unused)
        x: Input hidden states (unused)
        router_wrapper: Reference to wrapper (unused)
        k_max: Maximum k value

    Returns:
        Tuple of (vals, idx) with dynamic k per token (padded to k_max)
    """
    if routing_probs is None:
        return None

    is_torch = hasattr(routing_probs, 'detach') and not isinstance(routing_probs, (list, tuple))

    if is_torch:
        probs = routing_probs.detach()
        device = probs.device
        dtype = probs.dtype
        num_tokens = probs.size(0)
        n_experts = probs.size(-1)

        # Sort once
        probs_sorted, indices_sorted = torch.sort(probs, descending=True, dim=-1)

        # Elbow detection (vectorized)
        x_norm = torch.linspace(0, 1, steps=n_experts, device=device)
        y_max = probs_sorted[:, 0:1]
        y_min = probs_sorted[:, -1:]
        y_norm = (y_max - probs_sorted) / (y_max - y_min + 1e-12)
        dist = y_norm - x_norm
        ks = (dist.argmax(dim=1) + 1).clamp(min=1, max=k_max)

        # Just take top k_max and mask invalid positions
        # This avoids per-token loops entirely
        final_vals = probs_sorted[:, :k_max].clone()
        final_idxs = indices_sorted[:, :k_max].clone()

        # Create mask and apply in-place
        col_idx = torch.arange(k_max, device=device)
        mask = col_idx >= ks.unsqueeze(1)  # positions to zero out
        final_vals[mask] = 0.0
        final_idxs[mask] = -1

        return final_vals, final_idxs

    else:
        # numpy path
        arr = _to_numpy(routing_probs)
        n_tokens, n_experts = arr.shape
        idxs_out = []
        vals_out = []
        for i in range(n_tokens):
            row = arr[i]
            sidx = np.argsort(-row)
            svals = row[sidx]
            xs = np.linspace(0, 1, n_experts)
            # INVERT y normalization to match proven implementation: (y_max - y) / (y_max - y_min)
            ys = (svals[0] - svals) / (svals[0] - svals[-1] + 1e-12)
            # Distance: find max of (y_norm - x_norm) on y=x diagonal
            dist = ys - xs
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
    """
    Cumulative mass selector: select minimum k so cumulative probability >= mass_threshold.

    Args:
        routing_probs: [num_tokens, num_experts] probability distribution
        orig_indices: Original indices (unused)
        x: Input hidden states (unused)
        router_wrapper: Reference to wrapper (unused)
        mass_threshold: Cumulative probability threshold (e.g., 0.9 = 90%)
        k_max: Maximum k value

    Returns:
        Tuple of (vals, idx) with dynamic k per token (padded to k_max)
    """
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
    """
    Entropy-based selector: select k based on entropy.

    effective_k = ceil(exp(entropy)), capped at k_max

    Args:
        routing_probs: [num_tokens, num_experts] probability distribution
        orig_indices: Original indices (unused)
        x: Input hidden states (unused)
        router_wrapper: Reference to wrapper (unused)
        k_max: Maximum k value

    Returns:
        Tuple of (vals, idx) with dynamic k per token (padded to k_max)
    """
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
    """
    Gap ratio selector: select k where ratio p_k / p_{k+1} exceeds ratio_threshold.

    Args:
        routing_probs: [num_tokens, num_experts] probability distribution
        orig_indices: Original indices (unused)
        x: Input hidden states (unused)
        router_wrapper: Reference to wrapper (unused)
        ratio_threshold: Ratio threshold for gap detection
        k_max: Maximum k value

    Returns:
        Tuple of (vals, idx) with dynamic k per token (padded to k_max)
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
