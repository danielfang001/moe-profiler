import re
import torch
import torch.nn.functional as F
import time
import numpy as np
import pickle
import argparse
from functools import partialmethod

from transformers import OlmoeForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
model = OlmoeForCausalLM.from_pretrained("allenai/OLMoE-1B-7B-0924-Instruct").to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0924-Instruct")

dynamic_topk_stats = {}

def _top_k_dynamic_fast_metrics(routing_weights: torch.Tensor):
    """
    Helper to calculate 'ks' (the dynamic cutoff) for metrics gathering.
    Same logic as the forward pass but isolated for reporting.
    """
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
    return ks


def forward_with_random_replacement(self, hidden_states: torch.Tensor, encroachment: int = 0):
    """
    Modified forward pass that:
    1. Calculates Dynamic Top-K cut-off (ks).
    2. Routes typically to Top-8 experts.
    3. BUT, for slots that would have been pruned by Dynamic Top-K (indices ks to 7),
       it acts as if it routed them, but replaces the target expert with a RANDOM expert
       from the 'tail' (ranks ks to 63).
    4. Crucially, it uses the WEIGHT of the original slot (e.g. the 7th best probability)
       applied to the random junk expert.
    
    Hypothesis: If experts ks..7 are truly interchangeable/irrelevant, replacing them
    with random experts (while keeping the weight magnitude) should not degrade accuracy
    significantly compared to Dynamic Top-K (which just zeroes them).
    """
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)

    # 1. Router Logits
    router_logits = self.gate(hidden_states)
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)

    # 2. Sort to get Ranks
    sorted_vals, sorted_idx = torch.sort(routing_weights, dim=-1, descending=True)

    # 3. Determine Dynamic K (cutoff)
    device = routing_weights.device
    num_experts = routing_weights.shape[1]
    
    x_norm = torch.linspace(0, 1, num_experts, device=device, dtype=routing_weights.dtype)
    y_first = sorted_vals[:, 0:1]
    y_last = sorted_vals[:, -1:]
    y_norm = (y_first - sorted_vals) / (y_first - y_last + 1e-12)

    elbow_scores = y_norm - x_norm.unsqueeze(0)
    elbow_indices = torch.argmax(elbow_scores, dim=1)
    ks = torch.clamp(elbow_indices + 1, min=1, max=8)

    # 4. Prepare Top-8 Selection
    # Start with the standard Top-8 experts and weights
    top_k_idx = sorted_idx[:, :8].clone()
    top_k_w = sorted_vals[:, :8].clone() # Keep weights of original positions!

    # 5. Apply Random Replacement for "Pruned" Slots
    # We iterate over the 8 slots.
    # Encroachment modification: we can encroach x slots above the dynamic threshold.
    # Replacement range: [max(0, ks - x), 8)
    
    for i in range(8):
        # Identify tokens where slot 'i' is below the cut (i.e., i >= ks - encroachment)
        # Note: ks is in [1, 8].
        cutoff_rank = torch.clamp(ks - encroachment, min=0)
        mask = (cutoff_rank <= i) # (N,) boolean
        
        if mask.any():
            # How many need replacement
            n_replace = mask.sum()
            
            # The pool of random experts we can pick from are those ranked [ks, 64)
            # For each token, the start of the "tail" is ks.
            # We want to pick a rank r in [ks, 64).
            
            # Get start ranks for the masked tokens
            start_ranks = ks[mask]
            
            # Length of the tail for each token
            lengths = 64 - start_ranks
            
            # Generate random offsets
            offsets = (torch.rand(n_replace, device=device) * lengths).long()
            
            # Calculate the chosen rank (index in sorted list)
            chosen_ranks = start_ranks + offsets
            
            # We need to fetch the expert IDs corresponding to these ranks from sorted_idx
            # sorted_idx is (N, 64). We need to gather (N_subset, 1) values.
            
            # Indices of the tokens we are modifying
            row_indices = torch.nonzero(mask, as_tuple=True)[0]
            
            # Select the new random experts
            new_experts = sorted_idx[row_indices, chosen_ranks]
            
            # Update the top_k_idx for this slot
            top_k_idx[row_indices, i] = new_experts

    # 6. Normalize weights (Standard implementation optional detail, likely True)
    if getattr(self, "norm_topk_prob", False):
        denom = top_k_w.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        top_k_w = top_k_w / denom

    top_k_w = top_k_w.to(hidden_states.dtype)

    # 7. Dispatch to Experts
    final_hidden_states = torch.zeros(
        (batch_size * sequence_length, hidden_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    # expert_mask shape: (num_experts, 8, N) after permute
    expert_mask = torch.nn.functional.one_hot(top_k_idx, num_classes=num_experts).permute(2, 1, 0)

    for expert_idx in range(num_experts):
        # idx: slot index (0..7)
        # top_x: batch index
        idx, top_x = torch.where(expert_mask[expert_idx])
        
        if top_x.numel() == 0:
            continue

        # Get inputs for this expert
        current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
        
        # Apply expert + weight
        # top_k_w[top_x, idx] gets the weight associated with the SLOT
        # (which is the original high weight of the pruned expert)
        current_hidden_states = self.experts[expert_idx](current_state) * top_k_w[top_x, idx, None]
        
        final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    
    return final_hidden_states, router_logits

def _get_moe_block():
    from transformers.models.olmoe.modeling_olmoe import OlmoeSparseMoeBlock
    return OlmoeSparseMoeBlock

def use_random_replacement_forward(encroachment=0):
    OlmoeSparseMoeBlock = _get_moe_block()
    if not hasattr(OlmoeSparseMoeBlock, "_forward_true_original"):
        OlmoeSparseMoeBlock._forward_true_original = OlmoeSparseMoeBlock.forward
    
    # Use partialmethod to bind the encroachment argument
    OlmoeSparseMoeBlock.forward = partialmethod(forward_with_random_replacement, encroachment=encroachment)
    print(f"✓ Switched to RANDOM REPLACEMENT forward (Experts < K=8 but >= K_dyn-{encroachment} are swapped with random tail experts)")

def reset_forward():
    OlmoeSparseMoeBlock = _get_moe_block()
    if hasattr(OlmoeSparseMoeBlock, "_forward_true_original"):
        OlmoeSparseMoeBlock.forward = OlmoeSparseMoeBlock._forward_true_original
        print("✓ Reset to true original forward")

# --- Dataset and Formatting Utilities ---

def extract_first_choice_letter(text: str) -> str | None:
    m = re.search(r"Answer:\s*[^A-Za-z]*([A-Da-d])\b", text)
    return m.group(1).upper() if m else None

def load_dataset_by_name(benchmark_name):
    if benchmark_name == "mmlu":
        return load_dataset("cais/mmlu", "all", split='test')
    elif benchmark_name == "arc_easy":
        return load_dataset("ai2_arc", "ARC-Easy", split='test')
    elif benchmark_name == "arc_challenge":
        return load_dataset("ai2_arc", "ARC-Challenge", split='test')
    elif benchmark_name == "hellaswag":
        return load_dataset("hellaswag", split='validation')
    elif benchmark_name == "piqa":
        return load_dataset("piqa", split='validation')
    elif benchmark_name == "winogrande":
        return load_dataset("winogrande", "winogrande_xl", split='validation')
    else:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")

def format_mmlu_prompt(example):
    question = example["question"]
    choices = example["choices"]
    prompt = f"Question: {question}\nChoices:\n"
    for i, choice in enumerate(choices):
        prompt += f"{chr(65+i)}. {choice}\n"
    prompt += "Answer:"
    return prompt

def format_arc_prompt(example):
    q = example.get("question")
    if isinstance(q, dict):
        question = q.get("stem", "")
        raw_choices = q.get("choices")
    else:
        question = q
        raw_choices = example.get("choices")
    pairs = []
    if isinstance(raw_choices, dict) and "label" in raw_choices and "text" in raw_choices:
        labels = raw_choices.get("label") or []
        texts = raw_choices.get("text") or []
        pairs = list(zip(labels, texts))
    elif isinstance(raw_choices, list):
        for item in raw_choices:
            if isinstance(item, dict):
                pairs.append((item.get("label"), item.get("text")))
            else:
                pairs.append((None, str(item)))
    prompt = f"Question: {question}\nChoices:\n"
    for i, (label, text) in enumerate(pairs):
        choice_label = label if label else chr(65 + i)
        prompt += f"{choice_label}. {text}\n"
    prompt += "Answer:"
    return prompt

def format_hellaswag_prompt(example):
    ctx = example.get("ctx") or example.get("context") or ""
    endings = example.get("endings") or example.get("choices") or []
    prompt = f"Context: {ctx}\nChoices:\n"
    for i, ending in enumerate(endings):
        prompt += f"{chr(65+i)}. {ending}\n"
    prompt += "Answer:"
    return prompt

def format_piqa_prompt(example):
    goal = example.get("goal") or ""
    sol1 = example.get("sol1") or ""
    sol2 = example.get("sol2") or ""
    prompt = f"Goal: {goal}\nChoices:\nA. {sol1}\nB. {sol2}\nAnswer:"
    return prompt

def format_winogrande_prompt(example):
    sentence = example.get("sentence") or ""
    option1 = example.get("option1") or ""
    option2 = example.get("option2") or ""
    prompt = f"Sentence: {sentence}\nChoices:\nA. {option1}\nB. {option2}\nAnswer:"
    return prompt

PROMPT_FORMATTERS = {
    "mmlu": format_mmlu_prompt,
    "arc_easy": format_arc_prompt,
    "arc_challenge": format_arc_prompt,
    "hellaswag": format_hellaswag_prompt,
    "piqa": format_piqa_prompt,
    "winogrande": format_winogrande_prompt,
}

def format_answer(sample, benchmark):
    if benchmark == "mmlu":
        return ['A', 'B', 'C', 'D'][sample['answer']]
    if benchmark == 'arc_easy' or benchmark == 'arc_challenge':
        return sample['answerKey']
    if benchmark == 'hellaswag':
        return ['A', 'B', 'C', 'D'][int(sample['label'])]
    if benchmark == 'piqa': 
        return ['A', 'B'][sample['label']]
    if benchmark == 'winogrande':
        return 'A' if sample['answer'] == '1' else 'B'

def run_accuracy(samples, format_prompt_fn, benchmark, batch_size=8):
    results = []
    print(f"Running inference on {len(samples)} samples (batch_size={batch_size})...")
    
    # Set padding side to left for generation and ensure pad token exists
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    start_time = time.time()
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i + batch_size]
        prompts = [format_prompt_fn(sample) for sample in batch]
        
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=2, pad_token_id=tokenizer.eos_token_id)
            
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        for j, text in enumerate(decoded_outputs):
            sample = batch[j]
            singleletter = extract_first_choice_letter(text)
            
            correct_answer = format_answer(sample, benchmark)
            is_correct = (singleletter == correct_answer)
            
            results.append({
                'correct': is_correct,
                'predicted': singleletter,
                'ground_truth': correct_answer
            })
        
        print(f"Processed {min(i + batch_size, len(samples))}/{len(samples)}", end='\r')

    print(f"\nCompleted in {time.time() - start_time:.2f}s")
    
    # Calculate accuracy
    num_correct = sum(1 for r in results if r['correct'])
    accuracy = num_correct / len(results) if results else 0
    print(f"Accuracy: {accuracy:.2%}")
    return accuracy, results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OLMoE Random Replacement Experiment")
    parser.add_argument("--benchmark", default="mmlu", 
                        choices=["arc_easy", "arc_challenge", "mmlu", "hellaswag", "piqa", "winogrande"],
                        help="Benchmark to run")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Number of samples to run")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for inference")
    parser.add_argument("--pruning-encroachment", type=int, default=0,
                        help="Offset to aggressively prune valid experts above the threshold (x=1 means prune K-1, etc)")
    args = parser.parse_args()

    print(f"Pruning Encroachment: {args.pruning_encroachment} (Replacing experts in range [K_{{dyn}}-{args.pruning_encroachment}, 8))")

    # Load benchmark
    print(f"Loading benchmark: {args.benchmark}")
    test_samples = load_dataset_by_name(args.benchmark)
    
    if args.num_samples is not None:
        n = min(args.num_samples, len(test_samples))
        test_samples = test_samples.select(range(n))
    
    test_samples_list = [ex for ex in test_samples]
    format_prompt_fn = PROMPT_FORMATTERS[args.benchmark]
    
    # Run Experiment
    use_random_replacement_forward(encroachment=args.pruning_encroachment)
    accuracy, _ = run_accuracy(test_samples_list, format_prompt_fn, args.benchmark, batch_size=args.batch_size)
    
    print("\nExperiment Complete.")
    print(f"Benchmark: {args.benchmark}")
    print(f"Samples: {len(test_samples_list)}")
    print(f"Accuracy with Random Replacement of Pruned Experts: {accuracy:.2%}")
