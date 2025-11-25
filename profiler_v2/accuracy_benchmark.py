"""
Accuracy Benchmark for MoE Profiler

Evaluates accuracy degradation when using different routing strategies.
Supports MMLU, ARC, HellaSwag, and other common benchmarks.
"""

import torch
import pandas as pd
from typing import Callable, List, Optional, Dict, Any
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets


# MMLU subjects (all 57)
MMLU_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology",
    "high_school_statistics", "high_school_us_history",
    "high_school_world_history", "human_aging", "human_sexuality",
    "international_law", "jurisprudence", "logical_fallacies",
    "machine_learning", "management", "marketing", "medical_genetics",
    "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition",
    "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology",
    "us_foreign_policy", "virology", "world_religions"
]

# MMLU subject sets for different test modes
MMLU_QUICK = ["abstract_algebra", "anatomy", "astronomy"]  # 3 subjects, ~300-400 questions
MMLU_MEDIUM = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_mathematics", "computer_security", "philosophy"
]  # 10 subjects, ~1,500 questions


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
        num_samples: Optional[int] = None,
        mmlu_mode: str = "quick"
    ):
        """
        Load a benchmark dataset.

        Args:
            name: Benchmark name
                - "arc_easy": ARC-Easy (2376 questions, easier)
                - "arc_challenge": ARC-Challenge (1172 questions, harder)
                - "mmlu": MMLU (use mmlu_mode to specify scope)
                - "hellaswag": HellaSwag
                - "piqa": PIQA
            split: Dataset split ("test", "validation", etc.)
            num_samples: Limit to N samples (for quick testing, overrides mmlu_mode)
            mmlu_mode: For MMLU only, choose test scope:
                - "quick": 3 subjects (~300-400 questions)
                - "medium": 10 subjects (~1,500 questions)
                - "full": All 57 subjects (~14,000 questions)

        Returns:
            List of evaluation examples
        """
        print(f"Loading {name} ({split})...")

        if name == "arc_easy":
            dataset = load_dataset("allenai/ai2_arc", "ARC-Easy", split=split)
        elif name == "arc_challenge":
            dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split)
        elif name == "mmlu":
            # Select subjects based on mode
            if mmlu_mode == "quick":
                subjects = MMLU_QUICK
                print(f"  MMLU Quick mode: {len(subjects)} subjects")
            elif mmlu_mode == "medium":
                subjects = MMLU_MEDIUM
                print(f"  MMLU Medium mode: {len(subjects)} subjects")
            elif mmlu_mode == "full":
                subjects = MMLU_SUBJECTS
                print(f"  MMLU Full mode: {len(subjects)} subjects (this will take a while!)")
            else:
                raise ValueError(f"Invalid mmlu_mode: {mmlu_mode}. Use 'quick', 'medium', or 'full'")

            # Load and concatenate all selected subjects
            datasets = []
            for subject in tqdm(subjects, desc="Loading MMLU subjects"):
                subject_data = load_dataset("cais/mmlu", subject, split=split)
                datasets.append(subject_data)

            dataset = concatenate_datasets(datasets)
            print(f"  Loaded {len(dataset)} total MMLU questions from {len(subjects)} subjects")

        elif name == "hellaswag":
            dataset = load_dataset("Rowan/hellaswag", split=split)
        elif name == "piqa":
            dataset = load_dataset("ybisk/piqa", split=split)
        else:
            raise ValueError(f"Unknown benchmark: {name}")

        # Limit samples if requested (overrides mmlu_mode)
        if num_samples is not None:
            dataset = dataset.select(range(min(num_samples, len(dataset))))
            print(f"  Limited to {len(dataset)} examples")

        print(f"Final dataset size: {len(dataset)} examples")
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

    def run_evaluation_with_analysis(
        self,
        config_name: str,
        dataset,
        dataset_name: str,
        selection_fn: Optional[Callable] = None,
        max_new_tokens: int = 5,
        save_logits: bool = True,
    ):
        """
        Run comprehensive evaluation with detailed routing analysis.

        Captures:
        - Per-question accuracy, k-values, expert usage
        - Per-layer routing patterns
        - Router logits (optional)
        - Expert utilization across all questions
        - Per-subject analysis (for MMLU)

        Args:
            config_name: Name for this configuration
            dataset: HuggingFace dataset
            dataset_name: Dataset name (for formatting)
            selection_fn: Selection function (None for baseline)
            max_new_tokens: Max tokens to generate
            save_logits: Whether to capture raw router logits

        Returns:
            Dictionary with comprehensive analysis data
        """
        print(f"\n{'='*80}")
        print(f"Running Comprehensive Analysis: {config_name} on {dataset_name}")
        print(f"{'='*80}")

        # Apply selection function
        if selection_fn is not None:
            self.profiler.set_selection_fn(selection_fn)
        else:
            self.profiler.remove_selection_fn()

        # Enable logit capture if requested
        if save_logits:
            self.profiler.enable_logit_capture()

        # Storage for detailed results
        questions_data = []

        # Process each question
        for i, example in enumerate(tqdm(dataset, desc=f"Analyzing {config_name}")):
            # Reset metrics for this question
            self.profiler.reset_metrics()

            # Format example based on dataset
            if dataset_name.startswith("arc"):
                formatted = self.format_arc_example(example)
            elif dataset_name == "mmlu":
                formatted = self.format_mmlu_example(example)
            else:
                raise ValueError(f"Dataset formatting not implemented for: {dataset_name}")

            # Run evaluation
            result = self.evaluate_example(formatted, max_new_tokens=max_new_tokens)

            # Capture per-layer routing data
            layers_data = {}
            for wrapper in self.profiler.wrappers:
                # Get routing data for this question
                k_per_token = wrapper.metrics.k_per_token.copy()

                # Compute entropy per token if we have logits
                router_entropy = []
                if save_logits and len(wrapper.raw_logits) > 0:
                    for logit_data in wrapper.raw_logits:
                        logits = logit_data['logits']
                        probs = torch.nn.functional.softmax(logits, dim=-1)
                        entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)
                        router_entropy.extend(entropy.tolist())

                layers_data[wrapper.name] = {
                    'k_per_token': k_per_token,
                    'mean_k': float(sum(k_per_token) / len(k_per_token)) if k_per_token else 0,
                    'k_variance': float(torch.tensor(k_per_token).var()) if len(k_per_token) > 1 else 0,
                    'expert_load': dict(wrapper.metrics.expert_loads),  # How many times each expert used
                    'router_entropy': router_entropy,
                    'mean_entropy': float(sum(router_entropy) / len(router_entropy)) if router_entropy else 0,
                }

                # Save logits if enabled
                if save_logits:
                    layers_data[wrapper.name]['raw_logits'] = wrapper.raw_logits.copy()

            # Compute aggregates across layers
            all_k_values = []
            all_entropies = []
            for layer_data in layers_data.values():
                all_k_values.extend(layer_data['k_per_token'])
                all_entropies.extend(layer_data['router_entropy'])

            # Store comprehensive question data
            question_data = {
                'question_id': i,
                'question_text': formatted['prompt'][:200],  # Truncate for storage
                'subject': example.get('subject', dataset_name),
                'dataset': dataset_name,
                'correct_answer': result['correct_answer'],
                'prediction': result['prediction'],
                'generated': result['generated'],
                'correct': result['correct'],

                # Routing data
                'layers': layers_data,
                'avg_k_all_layers': float(sum(all_k_values) / len(all_k_values)) if all_k_values else 0,
                'max_k_all_layers': float(max(all_k_values)) if all_k_values else 0,
                'min_k_all_layers': float(min(all_k_values)) if all_k_values else 0,
                'avg_entropy_all_layers': float(sum(all_entropies) / len(all_entropies)) if all_entropies else 0,
            }

            questions_data.append(question_data)

            # Progress update every 50 questions
            if (i + 1) % 50 == 0:
                correct_so_far = sum(1 for q in questions_data if q['correct'])
                acc = 100 * correct_so_far / len(questions_data)
                avg_k = sum(q['avg_k_all_layers'] for q in questions_data) / len(questions_data)
                print(f"  Progress: {i+1}/{len(dataset)} | Acc: {acc:.1f}% | Avg k: {avg_k:.2f}")

        # Compute summary statistics
        accuracy = 100 * sum(1 for q in questions_data if q['correct']) / len(questions_data)

        # Get profiler metrics
        k_stats = self.profiler.get_k_statistics()
        summary = self.profiler.get_summary()

        # Aggregate expert usage across all questions
        expert_loads_total = {}
        for q in questions_data:
            for layer_name, layer_data in q['layers'].items():
                for expert_id, count in layer_data['expert_load'].items():
                    expert_loads_total[expert_id] = expert_loads_total.get(expert_id, 0) + count

        # Disable logit capture
        if save_logits:
            self.profiler.disable_logit_capture()

        # Store comprehensive results
        analysis_data = {
            'config_name': config_name,
            'dataset_name': dataset_name,
            'num_questions': len(questions_data),
            'accuracy': accuracy,
            'selection_fn': selection_fn.__name__ if selection_fn and hasattr(selection_fn, '__name__') else config_name,

            # All question data (main analysis data)
            'questions': questions_data,

            # Aggregate statistics
            'summary': summary,
            'k_stats': k_stats,
            'expert_loads_total': expert_loads_total,

            # Metadata
            'save_logits': save_logits,
            'model_config': self.profiler.architecture_info,
        }

        # Store in results
        self.results[config_name] = analysis_data

        # Print summary
        print(f"\n{'='*80}")
        print(f"Analysis Complete: {config_name}")
        print(f"{'='*80}")
        print(f"  Accuracy:     {accuracy:.2f}%")
        print(f"  Mean k:       {k_stats.get('k_mean', 0):.2f}")
        print(f"  K range:      [{k_stats.get('k_min', 0):.1f}, {k_stats.get('k_max', 0):.1f}]")
        print(f"  Questions:    {len(questions_data)}")
        print(f"  Logits saved: {save_logits}")
        print(f"{'='*80}")

        return analysis_data

    def save_analysis_to_file(self, config_name: str, filename: str):
        """
        Save comprehensive analysis data to pickle file.

        Args:
            config_name: Which configuration to save
            filename: Output pickle file

        Example:
            >>> acc_bench.run_evaluation_with_analysis("kneedle_k8", dataset, "mmlu", ...)
            >>> acc_bench.save_analysis_to_file("kneedle_k8", "mmlu_kneedle_analysis.pkl")
        """
        import pickle
        import os

        if config_name not in self.results:
            print(f"⚠️  Config '{config_name}' not found in results")
            return

        data = self.results[config_name]

        # Save to pickle
        with open(filename, 'wb') as f:
            pickle.dump(data, f)

        file_size_mb = os.path.getsize(filename) / 1024 / 1024

        print(f"\n✓ Saved analysis to: {filename}")
        print(f"  File size: {file_size_mb:.2f} MB")
        print(f"  Questions: {data['num_questions']}")
        print(f"  Layers: {len(data['model_config'])}")
        print(f"  Logits included: {data['save_logits']}")
        print(f"\nTo load:")
        print(f"  import pickle")
        print(f"  with open('{filename}', 'rb') as f:")
        print(f"      data = pickle.load(f)")
        print(f"  questions = data['questions']")
        print(f"  print(f'Accuracy: {{data[\"accuracy\"]:.2f}}%')")


def quick_accuracy_test(
    model,
    tokenizer,
    profiler,
    benchmark: str = "arc_easy",
    num_samples: Optional[int] = None,
    mmlu_mode: str = "quick",
    selectors: Optional[Dict[str, Callable]] = None
):
    """
    Quick one-liner for accuracy testing.

    Args:
        model: MoE model
        tokenizer: Tokenizer
        profiler: MOEProfiler instance
        benchmark: Benchmark name ("arc_easy", "arc_challenge", "mmlu")
        num_samples: Number of examples to test (None = use full dataset or mmlu_mode)
        mmlu_mode: For MMLU only, test scope:
            - "quick": 3 subjects (~300-400 questions)
            - "medium": 10 subjects (~1,500 questions)
            - "full": All 57 subjects (~14,000 questions)
        selectors: Dict of {name: selector_fn}

    Returns:
        Comparison DataFrame

    Example:
        >>> from profiler_v2 import quick_accuracy_test
        >>> from profiler_v2.selectors import kneedle_selector
        >>> from functools import partial
        >>>
        >>> # ARC-Easy (quick test)
        >>> comparison = quick_accuracy_test(
        ...     model, tokenizer, profiler,
        ...     benchmark="arc_easy",
        ...     num_samples=100,
        ...     selectors={"kneedle_k8": partial(kneedle_selector, k_max=8)}
        ... )
        >>>
        >>> # MMLU Quick mode (3 subjects, ~300 questions)
        >>> comparison = quick_accuracy_test(
        ...     model, tokenizer, profiler,
        ...     benchmark="mmlu",
        ...     mmlu_mode="quick",
        ...     selectors={"kneedle_k8": partial(kneedle_selector, k_max=8)}
        ... )
        >>>
        >>> # MMLU Full mode (57 subjects, ~14k questions)
        >>> comparison = quick_accuracy_test(
        ...     model, tokenizer, profiler,
        ...     benchmark="mmlu",
        ...     mmlu_mode="full",
        ...     selectors={"kneedle_k8": partial(kneedle_selector, k_max=8)}
        ... )
    """
    acc_bench = AccuracyBenchmark(model, tokenizer, profiler)

    # Load dataset
    dataset = acc_bench.load_benchmark(
        benchmark,
        split="test",
        num_samples=num_samples,
        mmlu_mode=mmlu_mode
    )

    # Run baseline
    acc_bench.run_evaluation("baseline", dataset, benchmark, selection_fn=None)

    # Run custom selectors
    if selectors:
        for name, selector in selectors.items():
            acc_bench.run_evaluation(name, dataset, benchmark, selection_fn=selector)

    # Compare
    return acc_bench.compare_results()
