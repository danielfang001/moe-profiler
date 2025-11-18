"""Sample MMLU questions and save router logits from OLMoE model.

Usage:
  python3 scripts/sample_mmlu_logits.py [--num-samples 5] [--output logits.pt]

This script:
1. Loads allenai/OLMoE-1B-7B-0924-Instruct model
2. Samples a few MMLU questions
3. Hooks all router modules to capture their logits
4. Saves all router logits to a file for analysis
"""

import argparse
import sys
import torch
import os
import pickle
from collections import defaultdict

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)


def load_mmlu_samples(num_samples=5):
    """Load a few MMLU samples from HuggingFace datasets."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("datasets library not installed. Install with: pip install datasets")
        return []

    try:
        dataset = load_dataset('cais/mmlu', 'all', split='validation')
        # Randomly sample
        indices = torch.randperm(len(dataset))[:num_samples].tolist()
        samples = [dataset[i] for i in indices]
        return samples
    except Exception as e:
        print(f"Error loading MMLU: {e}")
        return []


def format_mmlu_question(sample):
    """Format a single MMLU sample as a question string."""
    question = sample['question']
    choices = sample['choices']

    # Format as multiple choice
    formatted = f"{question}\n"
    for i, choice in enumerate(choices):
        formatted += f"({chr(65 + i)}) {choice}\n"

    return formatted.strip()


def hook_router_logits(model, logits_dict, sample_idx):
    """Attach hooks to all router modules to capture their logits.

    Args:
        model: The transformer model
        logits_dict: Dictionary to store logits by (router_name, sample_idx)
        sample_idx: Current sample index (for tracking which sample produced the logits)

    Returns:
        List of hook handles that need to be removed later
    """
    handles = []

    def create_hook(router_name):
        def hook_fn(module, input, output):
            # output can be logits or (logits, indices) tuple
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output

            # Convert to CPU and store with sample index
            if logits is not None:
                key = (router_name, sample_idx)
                if key not in logits_dict:
                    logits_dict[key] = []
                logits_dict[key].append(logits.detach().cpu())

        return hook_fn

    # Find all router/gate modules (exclude expert.*.gate_proj which are projection layers)
    for name, module in model.named_modules():
        if ('gate' in name.lower() or 'router' in name.lower()) and 'gate_proj' not in name.lower():
            hook = module.register_forward_hook(create_hook(name))
            handles.append(hook)

    return handles


def run_mmlu_logits_capture(model_name="allenai/OLMoE-1B-7B-0924-Instruct",
                            num_samples=5,
                            output_file="mmlu_router_logits.pkl"):
    """Load model, sample MMLU, capture router logits."""

    print(f"Loading model {model_name}...")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("transformers library not installed")
        return

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()

    # Load MMLU samples
    print(f"\nLoading {num_samples} MMLU samples...")
    samples = load_mmlu_samples(num_samples=num_samples)

    if not samples:
        print("Failed to load MMLU samples")
        return

    print(f"Loaded {len(samples)} samples\n")

    # Setup logits collection: key = (router_name, sample_idx), value = list of logit tensors
    logits_dict = defaultdict(list)

    try:
        # Process each sample
        with torch.no_grad():
            for sample_idx, sample in enumerate(samples):
                print(f"Processing sample {sample_idx+1}/{len(samples)}...")

                question_text = format_mmlu_question(sample)
                print(f"  Question: {question_text[:100]}...")

                # Attach hooks for this sample
                handles = hook_router_logits(model, logits_dict, sample_idx)

                # Tokenize and run forward pass
                inputs = tokenizer(question_text, return_tensors='pt', truncation=True, max_length=512)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                # Forward pass (collect logits via hooks)
                _ = model(**inputs, output_hidden_states=False)

                # Remove hooks for this sample
                for handle in handles:
                    handle.remove()

                print(f"  Captured {len([k for k in logits_dict.keys() if k[1] == sample_idx])} routers")

    finally:
        print("\nHooks removed")

    # Reorganize: convert from (router_name, sample_idx) keys to per-sample dicts
    # Structure: {sample_idx: {router_name: logits_tensor}}
    organized_logits = defaultdict(dict)

    for (router_name, sample_idx), logits_list in logits_dict.items():
        # Concatenate all logits from this router on this sample
        try:
            stacked = torch.cat(logits_list, dim=0)
            organized_logits[sample_idx][router_name] = stacked
        except Exception as e:
            # If concatenation fails, keep as list
            organized_logits[sample_idx][router_name] = logits_list

    # Print summary
    print("\nLogits captured per sample:")
    for sample_idx in sorted(organized_logits.keys()):
        print(f"\nSample {sample_idx}:")
        for router_name, logits in organized_logits[sample_idx].items():
            if isinstance(logits, torch.Tensor):
                print(f"  {router_name}: shape {logits.shape}")
            else:
                print(f"  {router_name}: {len(logits)} sequences (variable shapes)")

    # Save to file
    print(f"\nSaving logits to {output_file}...")
    save_data = {
        'model_name': model_name,
        'num_samples': len(samples),
        'samples': samples,  # Keep original samples for reference
        'router_logits': dict(organized_logits),  # Convert defaultdict to dict
    }

    with open(output_file, 'wb') as f:
        pickle.dump(save_data, f)

    print(f"Saved! File size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")
    print(f"\nTo load and inspect:")
    print(f"  import pickle")
    print(f"  with open('{output_file}', 'rb') as f:")
    print(f"      data = pickle.load(f)")
    print(f"  for sample_idx, routers in data['router_logits'].items():")
    print(f"      print(f'Sample {{sample_idx}}: {{list(routers.keys())}}')")


def main():
    parser = argparse.ArgumentParser(description='Sample MMLU and capture router logits')
    parser.add_argument('--num-samples', type=int, default=5, help='Number of MMLU samples to process')
    parser.add_argument('--output', type=str, default='mmlu_router_logits.pkl', help='Output file for logits')
    parser.add_argument('--model-name', type=str, default='allenai/OLMoE-1B-7B-0924-Instruct',
                        help='Model name from HuggingFace Hub')
    args = parser.parse_args()

    run_mmlu_logits_capture(
        model_name=args.model_name,
        num_samples=args.num_samples,
        output_file=args.output
    )


if __name__ == '__main__':
    main()

