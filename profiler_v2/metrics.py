"""
Metrics tracking for MoE Profiler

Contains the Metrics dataclass and helper functions for statistical analysis.
"""

import pandas as pd
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Dict, List


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


@dataclass
class Metrics:
    """Metrics container with statistical tracking"""
    # Raw measurements
    flops_per_token: List[float] = field(default_factory=list)
    router_flops_per_token: List[float] = field(default_factory=list)
    expert_flops_per_token: List[float] = field(default_factory=list)
    active_experts: List[int] = field(default_factory=list)
    k_per_token: List[float] = field(default_factory=list)  # Average k per token
    k_distribution: List[List[int]] = field(default_factory=list)  # Full k distribution
    latency_ms: List[float] = field(default_factory=list)
    expert_loads: Dict[int, int] = field(default_factory=dict)  # Will be initialized with num_experts
    router_confidence: List[float] = field(default_factory=list)
    semantic_confidence: List[float] = field(default_factory=list)
    memory_mb: List[float] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)

    # Metadata
    device: str = 'cuda:0'
    device_index: int = 0  # GPU device index for multi-GPU support
    step_count: int = 0

    def initialize_expert_loads(self, num_experts: int):
        """Initialize expert_loads dict with correct number of experts."""
        if not self.expert_loads:
            self.expert_loads = {i: 0 for i in range(num_experts)}

    def set_device(self, device):
        """Set the device for this metrics object (supports 'cuda:0', 'cuda:1', etc.)"""
        if isinstance(device, str):
            self.device = device
            # Extract device index
            if device.startswith('cuda:'):
                try:
                    self.device_index = int(device.split(':')[1])
                except (ValueError, IndexError):
                    self.device_index = 0
        elif isinstance(device, torch.device):
            self.device = str(device)
            self.device_index = device.index if device.index is not None else 0
        else:
            self.device = str(device)
            self.device_index = 0

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


def calculate_gini_coefficient(values):
    """
    Calculate Gini coefficient for load imbalance.
    0 = perfect equality, 1 = perfect inequality
    """
    values = np.array(sorted(values))
    n = len(values)
    if n == 0 or values.sum() == 0:
        return 0.0
    cumsum = np.cumsum(values)
    return (2 * np.sum((np.arange(1, n + 1)) * values)) / (n * cumsum[-1]) - (n + 1) / n


def calculate_entropy(values):
    """
    Calculate normalized entropy of distribution.
    0 = concentrated, 1 = uniform
    """
    values = np.array(values)
    if values.sum() == 0:
        return 0.0
    probs = values / values.sum()
    probs = probs[probs > 0]  # Remove zeros
    entropy = -np.sum(probs * np.log(probs))
    max_entropy = np.log(len(values))  # Maximum entropy occurs when uniform distribution
    return entropy / max_entropy if max_entropy > 0 else 0.0  # Normalized entropy


def calculate_cv(values):
    """
    Calculate coefficient of variation (CV = std / mean).
    Measures relative variability.
    """
    values = np.array(values)
    if len(values) == 0 or values.mean() == 0:
        return 0.0
    return values.std() / values.mean()


def get_metrics_summary(metrics: Metrics):
    """Get comprehensive statistical summary of metrics."""
    df = metrics.to_df()
    if len(df) == 0:
        return "No metrics collected yet"

    # Expert load statistics
    expert_loads = np.array(list(metrics.expert_loads.values()))

    # k distribution statistics
    k_values = np.array(metrics.k_per_token)

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
        'expert_load_cv': calculate_cv(expert_loads),
        'expert_load_gini': calculate_gini_coefficient(expert_loads),
        'expert_load_entropy': calculate_entropy(expert_loads),

        # Confidence metrics
        'router_confidence_mean': df['router_conf'].mean(),
        'router_confidence_std': df['router_conf'].std(),

        # Memory
        'memory_mean_mb': df['memory_mb'].mean() if 'memory_mb' in df and len(df['memory_mb']) > 0 else 0,
        'memory_max_mb': df['memory_mb'].max() if 'memory_mb' in df and len(df['memory_mb']) > 0 else 0,

        # Overall statistics
        'total_steps': len(df),
    }
