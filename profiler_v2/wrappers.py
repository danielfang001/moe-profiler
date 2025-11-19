"""
Router Wrappers for MoE Profiling

Provides flexible wrappers that can handle different MoE architectures:
- GateWrapper: For architectures where we wrap the gate/router module
- BlockWrapper: For architectures where we wrap the entire MoE block
"""

import torch
import torch.nn as nn
import time
from typing import Optional, Callable, Tuple, Any
from .metrics import Metrics
from .architectures.base import BaseArchitectureHandler


class RouterWrapper(nn.Module):
    """
    Unified router wrapper that adapts to different architectures.

    Supports:
    - Gate-only wrapping (e.g., Mixtral)
    - Full block wrapping (e.g., OLMoE)
    - Custom selection functions
    - CUDA-accurate timing
    - Comprehensive metrics collection
    """

    def __init__(
        self,
        module: nn.Module,
        handler: BaseArchitectureHandler,
        selection_fn: Optional[Callable] = None,
        warmup_steps: int = 5,
        sampling_rate: int = 1,
        name: Optional[str] = None,
    ):
        """
        Args:
            module: The module to wrap (gate or entire block)
            handler: Architecture-specific handler
            selection_fn: Optional custom selection function for dynamic k
            warmup_steps: Number of initial steps to skip
            sampling_rate: Collect metrics every Nth step
            name: Optional name for this wrapper
        """
        super().__init__()
        self.wrapped_module = module
        self.handler = handler
        self.selection_fn = selection_fn
        self.warmup_steps = warmup_steps
        self.sampling_rate = sampling_rate
        self.name = name

        # Get architecture specs
        self.wrapper_type = handler.get_wrapper_type()
        self.num_experts = handler.get_num_experts()
        self.hidden_dim = handler.get_hidden_dim(module)
        self.expert_dim = handler.get_expert_dim(module)
        self.default_top_k = handler.get_default_top_k()

        # Metrics tracking
        self.metrics = Metrics()
        self.metrics.initialize_expert_loads(self.num_experts)
        self.enabled = True
        self.current_step = 0

        # CUDA events for precise timing
        self.use_cuda_events = torch.cuda.is_available()
        if self.use_cuda_events:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)

        # For block wrappers, cache expert modules
        if self.wrapper_type == 'block':
            self._cache_expert_modules()

    def _cache_expert_modules(self):
        """Cache references to expert modules for block-level wrapping."""
        if hasattr(self.wrapped_module, 'experts'):
            self.experts = self.wrapped_module.experts
        else:
            self.experts = None

        if hasattr(self.wrapped_module, 'gate'):
            self.gate = self.wrapped_module.gate
        else:
            self.gate = None

    def forward(self, hidden_states, *args, **kwargs):
        """Forward pass with profiling."""
        if not self.enabled:
            return self.wrapped_module(hidden_states, *args, **kwargs)

        self.current_step += 1

        # Skip warmup and sampling
        in_warmup = self.current_step <= self.warmup_steps
        should_profile = (self.current_step % self.sampling_rate == 0) and not in_warmup

        if not should_profile:
            return self.wrapped_module(hidden_states, *args, **kwargs)

        # Start timing
        if self.use_cuda_events:
            self.start_event.record()
        start_time = time.perf_counter()

        # Route based on wrapper type
        if self.wrapper_type == 'gate':
            output = self._forward_gate(hidden_states, *args, **kwargs)
        else:  # block
            output = self._forward_block(hidden_states, *args, **kwargs)

        # End timing
        if self.use_cuda_events:
            self.end_event.record()
            torch.cuda.synchronize()
            latency = self.start_event.elapsed_time(self.end_event)
        else:
            latency = (time.perf_counter() - start_time) * 1000  # Convert to ms

        self.metrics.latency_ms.append(latency)
        self.metrics.timestamps.append(time.time())

        # Track memory
        if torch.cuda.is_available():
            memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
            self.metrics.memory_mb.append(memory_mb)

        return output

    def _forward_gate(self, hidden_states, *args, **kwargs):
        """
        Forward pass for gate-only wrapping (e.g., Mixtral).

        The gate returns routing logits or (weights, indices).
        We intercept, optionally apply custom selection, collect metrics.
        """
        # Call the wrapped gate
        gate_output = self.wrapped_module(hidden_states, *args, **kwargs)

        # Parse output using handler
        routing_weights, expert_indices = self.handler.parse_router_output(gate_output)

        # Apply custom selection if provided
        if self.selection_fn is not None:
            try:
                # Get routing probabilities for selector
                if isinstance(gate_output, tuple):
                    routing_probs = routing_weights
                else:
                    routing_probs = torch.nn.functional.softmax(gate_output, dim=-1)

                # Apply custom selector
                custom_weights, custom_indices = self.selection_fn(
                    routing_probs, expert_indices, hidden_states, self
                )

                if custom_weights is not None and custom_indices is not None:
                    routing_weights = custom_weights
                    expert_indices = custom_indices
            except Exception as e:
                print(f"Warning: Custom selector failed: {e}. Using default routing.")

        # Collect metrics
        self._collect_metrics(hidden_states, routing_weights, expert_indices, gate_output)

        # Return in original format
        return gate_output

    def _forward_block(self, hidden_states, *args, **kwargs):
        """
        Forward pass for full block wrapping (e.g., OLMoE).

        The block contains gate + experts. We need to:
        1. Call gate to get logits
        2. Apply custom selection
        3. Manually route to experts
        4. Return aggregated output
        """
        # Get routing logits from gate
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_flat = hidden_states.view(-1, hidden_dim)

        gate_logits = self.gate(hidden_flat)

        # Parse logits to get routing decision
        routing_weights, expert_indices = self.handler.parse_router_output(gate_logits)

        # If no custom selector, use original model routing but still collect metrics
        if self.selection_fn is None:
            # Collect metrics for baseline
            self._collect_metrics(hidden_flat, routing_weights, expert_indices, gate_logits)
            # Use original routing
            return self.wrapped_module(hidden_states, *args, **kwargs)

        # Apply custom selection (we know selection_fn is not None here)
        try:
            routing_probs = torch.nn.functional.softmax(gate_logits, dim=-1)
            custom_weights, custom_indices = self.selection_fn(
                routing_probs, expert_indices, hidden_flat, self
            )

            if custom_weights is not None and custom_indices is not None:
                routing_weights = custom_weights
                expert_indices = custom_indices
        except Exception as e:
            print(f"Warning: Custom selector failed: {e}. Using default routing.")

        # Collect metrics
        self._collect_metrics(hidden_flat, routing_weights, expert_indices, gate_logits)

        # Manually route to experts (similar to OlmoeSparseMoeBlock)
        num_tokens = hidden_flat.shape[0]
        final_hidden_states = torch.zeros(
            (num_tokens, hidden_dim),
            dtype=hidden_flat.dtype,
            device=hidden_flat.device
        )

        # Route each token to its selected experts
        for expert_idx in range(self.num_experts):
            # Find which tokens are routed to this expert
            expert_mask = (expert_indices == expert_idx)
            token_positions, k_indices = torch.where(expert_mask)

            if len(token_positions) == 0:
                continue  # No tokens for this expert

            # Get the tokens and their routing weights
            expert_input = hidden_flat[token_positions]
            expert_weights = routing_weights[token_positions, k_indices]

            # Call the expert
            expert_output = self.experts[expert_idx](expert_input)

            # Apply routing weights and accumulate
            weighted_output = expert_output * expert_weights[:, None]
            final_hidden_states.index_add_(0, token_positions, weighted_output.to(hidden_flat.dtype))

        # Reshape back
        output = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        router_logits = gate_logits.view(batch_size, sequence_length, -1)

        return output, router_logits

    def _collect_metrics(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        expert_indices: torch.Tensor,
        gate_output: Any
    ):
        """Collect comprehensive metrics."""
        # Calculate dynamic k per token
        expert_indices_flat = expert_indices.reshape(-1, expert_indices.shape[-1])
        k_per_token_list = []
        for token_experts in expert_indices_flat:
            # Count non-negative indices (padding uses -1)
            active_k = (token_experts >= 0).sum().item()
            k_per_token_list.append(active_k)

        # Store k distribution and average k
        self.metrics.k_distribution.append(k_per_token_list)
        avg_k = sum(k_per_token_list) / len(k_per_token_list) if len(k_per_token_list) > 0 else 0
        self.metrics.k_per_token.append(avg_k)

        # Calculate FLOPs
        num_tokens = hidden_states.shape[0]

        # Router FLOPs: 2 * tokens * hidden_dim * num_experts
        router_flops = 2 * num_tokens * self.hidden_dim * self.num_experts

        # Expert FLOPs: Accounts for dynamic k per token
        # Each expert does: 2 * hidden * expert_dim + 2 * expert_dim * hidden
        expert_flops = sum(
            k * (2 * self.hidden_dim * self.expert_dim + 2 * self.expert_dim * self.hidden_dim)
            for k in k_per_token_list
        )

        total_flops = router_flops + expert_flops

        self.metrics.flops_per_token.append(total_flops / num_tokens)
        self.metrics.router_flops_per_token.append(router_flops / num_tokens)
        self.metrics.expert_flops_per_token.append(expert_flops / num_tokens)

        # Track expert loads
        for token_experts in expert_indices_flat:
            for expert_id in token_experts:
                expert_id = int(expert_id.item())
                if expert_id >= 0:  # Skip padding
                    if expert_id not in self.metrics.expert_loads:
                        self.metrics.expert_loads[expert_id] = 0
                    self.metrics.expert_loads[expert_id] += 1

        # Count active experts
        active_experts = len(set(
            int(idx.item()) for idx in expert_indices_flat.flatten() if idx >= 0
        ))
        self.metrics.active_experts.append(active_experts)

        # Compute confidence
        confidence = self.handler.compute_confidence(gate_output, routing_weights)
        self.metrics.router_confidence.append(confidence)

        self.metrics.step_count += 1

    def get_metrics_df(self):
        """Get metrics as DataFrame."""
        return self.metrics.to_df()

    def reset_metrics(self):
        """Reset all metrics."""
        self.metrics = Metrics()
        self.metrics.initialize_expert_loads(self.num_experts)
        self.current_step = 0

    def enable(self):
        """Enable profiling."""
        self.enabled = True

    def disable(self):
        """Disable profiling."""
        self.enabled = False
