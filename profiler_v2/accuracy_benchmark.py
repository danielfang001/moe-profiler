"""
Accuracy Benchmark for MoE Profiler

Evaluates accuracy degradation when using different routing strategies.
Supports MMLU, ARC, HellaSwag, and other common benchmarks.
"""

import torch
import pandas as pd
from typing import Callable, List, Optional, Dict, Any
from tqdm import tqdm
from datasets import load_dataset


class AccuracyBenchmark:
    """
    Evaluate accuracy with different MoE routing strategies.

    Compares baseline vs custom selectors on standard benchmarks
    while tracking FLOPs and latency.
    """

    def __init__(self, model, tokenizer, profiler):
        """
        Initialize accuracy benchmark.

        Args:
            model: MoE model to evaluate
            tokenizer: Tokenizer
            profiler: MOEProfiler instance (already wrapped)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.profiler = profiler
        self.results = {}

        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def load_benchmark(
        self,
        name: str = "arc_easy",
        split: str = "test",
        num_samples: Optional[int] = None
    ):
        """
        Load a benchmark dataset.

        Args:
            name: Benchmark name
                - "arc_easy": ARC-Easy (2376 questions, easier)
                - "arc_challenge": ARC-Challenge (1172 questions, harder)
                - "mmlu": MMLU (specify subject)
                - "hellaswag": HellaSwag
                - "piqa": PIQA
            split: Dataset split ("test", "validation", etc.)
            num_samples: Limit to N samples (for quick testing)

        Returns:
            List of evaluation examples
        """
        print(f"Loading {name} ({split})...")

        if name == "arc_easy":
            dataset = load_dataset("allenai/ai2_arc", "ARC-Easy", split=split)
        elif name == "arc_challenge":
            dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split)
        elif name == "mmlu":
            # Default to a small subject for testing
            dataset = load_dataset("cais/mmlu", "abstract_algebra", split=split)
        elif name == "hellaswag":
            dataset = load_dataset("Rowan/hellaswag", split=split)
        elif name == "piqa":
            dataset = load_dataset("ybisk/piqa", split=split)
        else:
            raise ValueError(f"Unknown benchmark: {name}")

        # Limit samples if requested
        if num_samples is not None:
            dataset = dataset.select(range(min(num_samples, len(dataset))))

        print(f"Loaded {len(dataset)} examples")
        return dataset

    def format_arc_example(self, example):
        """Format ARC example as multiple choice prompt."""
        question = example['question']
        choices = example['choices']

        # Format: Q: ... A) ... B) ... C) ... D) ...
        prompt = f"Question: {question}\n"
        for i, (label, text) in enumerate(zip(choices['label'], choices['text'])):
            prompt += f"{label}) {text}\n"
        prompt += "Answer:"

        # Get correct answer
        answer_key = example['answerKey']

        return {
            'prompt': prompt,
            'answer': answer_key,
            'choices': choices['label']
        }

    def format_mmlu_example(self, example):
        """Format MMLU example as multiple choice prompt."""
        question = example['question']
        choices = example['choices']

        prompt = f"Question: {question}\nA) {choices[0]}\nB) {choices[1]}\nC) {choices[2]}\nD) {choices[3]}\nAnswer:"

        answer_idx = example['answer']
        answer_key = ['A', 'B', 'C', 'D'][answer_idx]

        return {
            'prompt': prompt,
            'answer': answer_key,
            'choices': ['A', 'B', 'C', 'D']
        }

    def evaluate_example(self, example_dict, max_new_tokens: int = 5):
        """
        Evaluate a single example.

        Args:
            example_dict: Dict with 'prompt', 'answer', 'choices'
            max_new_tokens: Max tokens to generate

        Returns:
            Dict with prediction and correctness
        """
        prompt = example_dict['prompt']
        correct_answer = example_dict['answer']
        choices = example_dict['choices']

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                temperature=None,
                top_p=None
            )

        # Decode
        generated_text = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        # Extract prediction (first letter that matches a choice)
        prediction = None
        for char in generated_text:
            if char.upper() in choices:
                prediction = char.upper()
                break

        # Check correctness
        correct = (prediction == correct_answer)

        return {
            'prompt': prompt,
            'prediction': prediction,
            'correct_answer': correct_answer,
            'generated': generated_text.strip(),
            'correct': correct
        }

    def run_evaluation(
        self,
        config_name: str,
        dataset,
        dataset_name: str,
        selection_fn: Optional[Callable] = None,
        max_new_tokens: int = 5,
        batch_size: int = 1  # Keep at 1 for simplicity
    ):
        """
        Run evaluation on a dataset with a specific routing configuration.

        Args:
            config_name: Name for this configuration (e.g., "baseline", "kneedle_k16")
            dataset: HuggingFace dataset
            dataset_name: Name of the benchmark (for formatting)
            selection_fn: Selection function (None for baseline)
            max_new_tokens: Max tokens to generate per example
            batch_size: Batch size (keep at 1 for now)

        Returns:
            Dict with results
        """
        print(f"\n{'='*80}")
        print(f"Evaluating: {config_name} on {dataset_name}")
        print(f"{'='*80}")

        # Apply selection function
        if selection_fn is not None:
            self.profiler.set_selection_fn(selection_fn)
        else:
            self.profiler.remove_selection_fn()

        # Reset metrics
        self.profiler.reset_metrics()

        # Evaluate all examples
        results_list = []
        correct_count = 0

        for i, example in enumerate(tqdm(dataset, desc=f"Evaluating {config_name}")):
            # Format example based on dataset
            if dataset_name.startswith("arc"):
                formatted = self.format_arc_example(example)
            elif dataset_name == "mmlu":
                formatted = self.format_mmlu_example(example)
            else:
                raise ValueError(f"Dataset formatting not implemented for: {dataset_name}")

            # Evaluate
            result = self.evaluate_example(formatted, max_new_tokens=max_new_tokens)
            results_list.append(result)

            if result['correct']:
                correct_count += 1

            # Print progress every 50 examples
            if (i + 1) % 50 == 0:
                acc = 100 * correct_count / (i + 1)
                print(f"  Progress: {i+1}/{len(dataset)} | Accuracy: {acc:.2f}%")

        # Calculate final accuracy
        accuracy = 100 * correct_count / len(dataset)

        # Get profiling metrics
        k_stats = self.profiler.get_k_statistics()
        summary = self.profiler.get_summary()

        # Store results
        result = {
            'config_name': config_name,
            'dataset_name': dataset_name,
            'num_examples': len(dataset),
            'correct': correct_count,
            'accuracy': accuracy,
            'results_list': results_list,
            'k_stats': k_stats,
            'summary': summary
        }

        self.results[config_name] = result

        # Print summary
        print(f"\n--- Results for {config_name} ---")
        print(f"  Accuracy:     {accuracy:.2f}% ({correct_count}/{len(dataset)})")
        print(f"  Mean k:       {k_stats.get('k_mean', 0):.2f}")
        print(f"  K range:      [{k_stats.get('k_min', 0):.1f}, {k_stats.get('k_max', 0):.1f}]")
        if 'error' not in summary:
            print(f"  Mean FLOPs:   {summary['flops_total_mean']:.2e}")
            print(f"  Mean latency: {summary['latency_mean_ms']:.2f} ms")

        return result

    def compare_results(self):
        """
        Compare accuracy and efficiency across all configurations.

        Returns:
            DataFrame with comparison
        """
        if len(self.results) < 2:
            print("Need at least 2 configurations to compare!")
            return pd.DataFrame()

        print(f"\n{'='*80}")
        print(f"ACCURACY COMPARISON")
        print(f"{'='*80}")

        comparison_rows = []

        for config_name, result in self.results.items():
            k_stats = result['k_stats']
            summary = result['summary']

            row = {
                'config': config_name,
                'accuracy_%': result['accuracy'],
                'correct': result['correct'],
                'total': result['num_examples'],
                'k_mean': k_stats.get('k_mean', 0),
                'k_std': k_stats.get('k_std', 0),
                'k_range': f"[{k_stats.get('k_min', 0):.0f}, {k_stats.get('k_max', 0):.0f}]",
            }

            if 'error' not in summary:
                row['flops_mean'] = summary['flops_total_mean']
                row['latency_ms'] = summary['latency_mean_ms']

            comparison_rows.append(row)

        comparison_df = pd.DataFrame(comparison_rows)

        # Calculate deltas vs baseline
        if 'baseline' in self.results:
            baseline_acc = self.results['baseline']['accuracy']
            baseline_flops = self.results['baseline']['summary'].get('flops_total_mean', 0)

            comparison_df['accuracy_delta_%'] = comparison_df['accuracy_%'] - baseline_acc
            if 'flops_mean' in comparison_df.columns:
                comparison_df['flops_reduction_%'] = 100 * (1 - comparison_df['flops_mean'] / baseline_flops)

        print(comparison_df.to_string(index=False))
        print(f"{'='*80}")

        # Save to CSV
        comparison_df.to_csv('accuracy_comparison.csv', index=False)
        print(f"\n✓ Saved comparison to: accuracy_comparison.csv")

        return comparison_df

    def save_detailed_results(self, prefix: str = "accuracy"):
        """Save detailed per-example results."""
        for config_name, result in self.results.items():
            # Save per-example results
            results_df = pd.DataFrame(result['results_list'])
            filename = f"{prefix}_{config_name}_results.csv"
            results_df.to_csv(filename, index=False)
            print(f"Saved {config_name} detailed results to: {filename}")


def quick_accuracy_test(
    model,
    tokenizer,
    profiler,
    benchmark: str = "arc_easy",
    num_samples: int = 100,
    selectors: Optional[Dict[str, Callable]] = None
):
    """
    Quick one-liner for accuracy testing.

    Args:
        model: MoE model
        tokenizer: Tokenizer
        profiler: MOEProfiler instance
        benchmark: Benchmark name ("arc_easy", "arc_challenge", "mmlu")
        num_samples: Number of examples to test (None = all)
        selectors: Dict of {name: selector_fn}

    Returns:
        Comparison DataFrame

    Example:
        >>> from profiler_v2 import quick_accuracy_test
        >>> from profiler_v2.selectors import kneedle_selector
        >>> from functools import partial
        >>>
        >>> comparison = quick_accuracy_test(
        ...     model, tokenizer, profiler,
        ...     benchmark="arc_easy",
        ...     num_samples=100,
        ...     selectors={"kneedle_k16": partial(kneedle_selector, k_max=16)}
        ... )
    """
    acc_bench = AccuracyBenchmark(model, tokenizer, profiler)

    # Load dataset
    dataset = acc_bench.load_benchmark(benchmark, split="test", num_samples=num_samples)

    # Run baseline
    acc_bench.run_evaluation("baseline", dataset, benchmark, selection_fn=None)

    # Run custom selectors
    if selectors:
        for name, selector in selectors.items():
            acc_bench.run_evaluation(name, dataset, benchmark, selection_fn=selector)

    # Compare
    return acc_bench.compare_results()
