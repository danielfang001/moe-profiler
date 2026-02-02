import torch
import torch.nn.functional as F
import time
import numpy as np

dynamic_topk_stats = {
    "latency": {
        "dynamic_forward_times": [],
        "original_forward_times": [],
    }
}

def _cuda_sync_if_needed(x: torch.Tensor):
    if x.is_cuda:
        torch.cuda.synchronize()

def _top_k_dynamic_fast(routing_weights: torch.Tensor):
    device = routing_weights.device
    dtype = routing_weights.dtype
    N, E = routing_weights.shape

    sorted_vals, sorted_idx = torch.sort(routing_weights, dim=-1, descending=True)

    x_norm = torch.linspace(0, 1, E, device=device, dtype=dtype)

    y_first = sorted_vals[:, 0:1]
    y_last = sorted_vals[:, -1:]
    y_norm = (y_first - sorted_vals) / (y_first - y_last + 1e-12)

    elbow_scores = y_norm - x_norm.unsqueeze(0)
    elbow_indices = torch.argmax(elbow_scores, dim=1)

    ks = torch.clamp(elbow_indices + 1, min=1, max=8)

    MAX_K = 8
    mask = torch.arange(MAX_K, device=device).unsqueeze(0) < ks.unsqueeze(1)

    top_k_weights = sorted_vals[:, :MAX_K] * mask
    top_k_index = sorted_idx[:, :MAX_K]

    return top_k_weights, top_k_index


def forward_with_dynamic_topk_instrumented(self, hidden_states: torch.Tensor):
    # --- correct GPU timing ---
    _cuda_sync_if_needed(hidden_states)
    start = time.perf_counter()

    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)

    router_logits = self.gate(hidden_states)
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)

    routing_weights, selected_experts = _top_k_dynamic_fast(routing_weights)

    if getattr(self, "norm_topk_prob", False):
        denom = routing_weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        routing_weights = routing_weights / denom

    routing_weights = routing_weights.to(hidden_states.dtype)

    final_hidden_states = torch.zeros(
        (batch_size * sequence_length, hidden_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    # NOTE: this one_hot uses num_experts = self.num_experts if available, else len(self.experts)
    num_experts = getattr(self, "num_experts", len(self.experts))

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=num_experts).permute(2, 1, 0)

    for expert_idx in range(num_experts):
        expert_layer = self.experts[expert_idx]
        idx, top_x = torch.where(expert_mask[expert_idx])
        if top_x.numel() == 0:
            continue

        current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
        current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
        final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

    _cuda_sync_if_needed(final_hidden_states)
    dynamic_topk_stats["latency"]["dynamic_forward_times"].append(time.perf_counter() - start)

    return final_hidden_states, router_logits


def _get_moe_block():
    from transformers.models.olmoe.modeling_olmoe import OlmoeSparseMoeBlock
    return OlmoeSparseMoeBlock


def use_original_forward():
    OlmoeSparseMoeBlock = _get_moe_block()

    # Save true original exactly once
    if not hasattr(OlmoeSparseMoeBlock, "_forward_true_original"):
        OlmoeSparseMoeBlock._forward_true_original = OlmoeSparseMoeBlock.forward

    # Avoid double-wrapping
    if getattr(OlmoeSparseMoeBlock.forward, "__name__", "") == "forward_original_timed":
        print("✓ Already using ORIGINAL forward (timed)")
        return

    def forward_original_timed(self, hidden_states: torch.Tensor, *args, **kwargs):
        _cuda_sync_if_needed(hidden_states)
        start = time.perf_counter()

        out = OlmoeSparseMoeBlock._forward_true_original(self, hidden_states, *args, **kwargs)

        # out could be Tensor or (Tensor, router_logits); synchronize on the tensor
        if isinstance(out, tuple):
            _cuda_sync_if_needed(out[0])
        else:
            _cuda_sync_if_needed(out)

        dynamic_topk_stats["latency"]["original_forward_times"].append(time.perf_counter() - start)
        return out

    OlmoeSparseMoeBlock.forward = forward_original_timed
    print("✓ Switched to ORIGINAL forward (timed)")


def use_dynamic_forward():
    OlmoeSparseMoeBlock = _get_moe_block()

    if not hasattr(OlmoeSparseMoeBlock, "_forward_true_original"):
        OlmoeSparseMoeBlock._forward_true_original = OlmoeSparseMoeBlock.forward

    OlmoeSparseMoeBlock.forward = forward_with_dynamic_topk_instrumented
    print("✓ Switched to DYNAMIC top-k forward (timed)")


def reset_forward():
    OlmoeSparseMoeBlock = _get_moe_block()
    if hasattr(OlmoeSparseMoeBlock, "_forward_true_original"):
        OlmoeSparseMoeBlock.forward = OlmoeSparseMoeBlock._forward_true_original
        print("✓ Reset to true original forward")
    else:
        print("⚠ No saved original forward found (call use_* once first)")


def reset_stats():
    dynamic_topk_stats["latency"]["dynamic_forward_times"] = []
    dynamic_topk_stats["latency"]["original_forward_times"] = []
    print("✓ Statistics reset")


def compare_latencies():
    lat = dynamic_topk_stats["latency"]

    print("\n" + "=" * 70)
    print("LATENCY COMPARISON: Dynamic Top-K vs Original")
    print("=" * 70)

    if lat["dynamic_forward_times"]:
        d = np.array(lat["dynamic_forward_times"]) * 1000
        print("\nDynamic Top-K:")
        print(f"  Mean: {d.mean():.3f} ms ± {d.std():.3f} ms")
        print(f"  Total: {d.sum():.1f} ms")
        print(f"  Passes: {len(d)}")

    if lat["original_forward_times"]:
        o = np.array(lat["original_forward_times"]) * 1000
        print("\nOriginal:")
        print(f"  Mean: {o.mean():.3f} ms ± {o.std():.3f} ms")
        print(f"  Total: {o.sum():.1f} ms")
        print(f"  Passes: {len(o)}")

    if lat["dynamic_forward_times"] and lat["original_forward_times"]:
        d_mean = np.mean(lat["dynamic_forward_times"]) * 1000
        o_mean = np.mean(lat["original_forward_times"]) * 1000
        speed = o_mean / d_mean
        overhead = (d_mean - o_mean) / o_mean * 100.0

        print("\n" + "=" * 70)
        if speed > 1:
            print(f"  Dynamic is {speed:.2f}x FASTER")
        else:
            print(f"  Dynamic is {1/speed:.2f}x SLOWER")
        print(f"  Overhead: {overhead:+.2f}%")
        print("=" * 70)


print("\nUsage:")
print("  reset_stats()           - Clear statistics")
print("  compare_latencies()     - Show comparison")
print("  use_original_forward()  - Switch to original (timed)")
print("  use_dynamic_forward()   - Switch to dynamic (timed)")
print("  reset_forward()         - Restore the true original")



from transformers import OlmoeForCausalLM, AutoTokenizer
import torch

DEVICE = "cuda" 

# Load different ckpts 
model = OlmoeForCausalLM.from_pretrained("allenai/OLMoE-1B-7B-0924-Instruct").to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0924-Instruct")



from datasets import load_dataset
#Load MMLU dataset
dataset = load_dataset("cais/mmlu", "all")

# Get a subset for testing (e.g., 100 examples from validation set)
test_samples = dataset['test'].select(range(0,5))

def format_mmlu_prompt(example):
    """Format MMLU example as a prompt"""
    question = example["question"]
    choices = example["choices"]
    
    # Format as multiple choice
    prompt = f"Question: {question}\n"
    prompt += "Choices:\n"
    for i, choice in enumerate(choices):
        prompt += f"{chr(65+i)}. {choice}\n"  # A, B, C, D
    prompt += "Answer:"
    
    return prompt

# Test formatting
sample = test_samples[0]
print(format_mmlu_prompt(sample))

dynamic_topk_stats = {
    'latency': {
        'dynamic_forward_times': [],
        'original_forward_times': [],
    }
}

# Convert to list first
test_samples_list = [ex for ex in test_samples]

def run_mmlu_batch(samples, batch_size=20):
    """Process MMLU samples in batches"""
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i+batch_size]
        prompts = [format_mmlu_prompt(ex) for ex in batch]
        
        # Tokenize batch
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(model.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        if i % 40 == 0:
            print(f"Processed {i}/{len(samples)} examples")



# Test with dynamic
print("Testing DYNAMIC with batching...")
use_dynamic_forward()
run_mmlu_batch(test_samples_list)

# Test with original
use_original_forward()
print("\nTesting ORIGINAL with batching...")
run_mmlu_batch(test_samples_list)

compare_latencies()



import pickle

# save
with open("dynamiclatency.pkl", "wb") as f:
    pickle.dump(dynamic_topk_stats["latency"]["dynamic_forward_times"], f)

# save
with open("originallatency.pkl", "wb") as f:
    pickle.dump(dynamic_topk_stats["latency"]["original_forward_times"], f)