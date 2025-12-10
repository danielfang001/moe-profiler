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
        mmlu_mode: Optional[str] = None,
        random_seed: int = 42
    ):
        """
        Load a benchmark dataset.

        Args:
            name: Benchmark name
                - "arc_easy": ARC-Easy (2376 questions, easier)
                - "arc_challenge": ARC-Challenge (1172 questions, harder)
                - "mmlu": MMLU (use mmlu_mode to specify scope)
                - "hellaswag": HellaSwag (10,042 questions)
                - "piqa": PIQA (1,838 questions)
                - "winogrande": WinoGrande (1,267 questions in validation split)
            split: Dataset split ("test", "validation", etc.)
            num_samples: Limit to N samples (random sampling with fixed seed)
            mmlu_mode: For MMLU only, choose test scope:
                - "quick": 3 subjects (~300-400 questions)
                - "medium": 10 subjects (~1,500 questions)
                - "full": All 57 subjects (~14,000 questions)
                - None: If num_samples specified, uses "full" and random samples
            random_seed: Random seed for reproducible sampling (default: 42)

        Returns:
            List of evaluation examples
        """
        print(f"Loading {name} ({split})...")

        if name == "arc_easy":
            dataset = load_dataset("allenai/ai2_arc", "ARC-Easy", split=split)
        elif name == "arc_challenge":
            dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split)
        elif name == "mmlu":
            # If num_samples specified without mode, default to "full" for random sampling
            if mmlu_mode is None and num_samples is not None:
                mmlu_mode = "full"
                print(f"  num_samples specified without mmlu_mode, defaulting to 'full' for random sampling")
            elif mmlu_mode is None:
                mmlu_mode = "quick"  # Default to quick if nothing specified

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
            dataset = load_dataset("ybisk/piqa", split=split, trust_remote_code=True)
        elif name == "winogrande":
            dataset = load_dataset("allenai/winogrande", "winogrande_xl", split=split)
        else:
            raise ValueError(f"Unknown benchmark: {name}")

        # Limit samples if requested with random sampling
        if num_samples is not None:
            num_samples = min(num_samples, len(dataset))
            # Random sampling with fixed seed for reproducibility
            import random
            random.seed(random_seed)
            indices = random.sample(range(len(dataset)), num_samples)
            indices.sort()  # Sort to maintain some order
            dataset = dataset.select(indices)
            print(f"  Randomly sampled {len(dataset)} examples (seed={random_seed})")

        print(f"Final dataset size: {len(dataset)} examples")
        return dataset

    def format_arc_example(self, example):
        """Format ARC example as multiple choice prompt (OLMES style)."""
        question = example['question']
        choices = example['choices']
        
        # OLMES format: Question: {question}\n A. {choice1}\n B. {choice2}...\nAnswer:
        prompt = f"Question: {question}\n"
        for i, (label, text) in enumerate(zip(choices['label'], choices['text'])):
            # Ensure label is A, B, C, D, E...
            # ARC sometimes has 1, 2, 3, 4 or A, B, C, D, E
            # We map to A, B, C, D for consistency if needed, but ARC usually has labels.
            # OLMES uses " A. " prefix.
            prompt += f" {label}. {text}\n"
        prompt += "Answer:"

        # Get correct answer
        answer_key = example['answerKey']

        return {
            'prompt': prompt,
            'answer': answer_key,
            'choices': choices['label']
        }

    def format_mmlu_example(self, example):
        """Format MMLU example as multiple choice prompt (OLMES style)."""
        question = example['question']
        choices = example['choices']
        
        # OLMES format: Question: {question}\n A. {choice1}\n B. {choice2}...\nAnswer:
        prompt = f"Question: {question}\n"
        labels = ['A', 'B', 'C', 'D']
        for i, choice in enumerate(choices):
            prompt += f" {labels[i]}. {choice}\n"
        prompt += "Answer:"

        answer_idx = example['answer']
        answer_key = labels[answer_idx]

        return {
            'prompt': prompt,
            'answer': answer_key,
            'choices': labels
        }

    def format_hellaswag_example(self, example):
        """Format HellaSwag example for completion likelihood evaluation."""
        context = example['ctx']
        endings = example['endings']
        label = int(example['label'])
        
        # OLMES/Standard: Calculate P(Ending | Context)
        # Normalize by character length
        
        candidates = []
        for ending in endings:
            # Ensure space prefix if needed (standard for HellaSwag)
            target = " " + ending if not ending.startswith(" ") else ending
            candidates.append((context, target))
            
        return {
            'type': 'completion',
            'candidates': candidates,
            'answer_idx': label,
            'normalize': 'char',
            'prompt': context # For logging
        }

    def format_piqa_example(self, example):
        """Format PIQA example as multiple choice prompt (OLMES style)."""
        goal = example['goal']
        sol1 = example['sol1']
        sol2 = example['sol2']
        
        # OLMES format: Goal: {goal}\n A. {sol1}\n B. {sol2}\nAnswer:
        prompt = f"Goal: {goal}\n A. {sol1}\n B. {sol2}\nAnswer:"

        answer_idx = example['label']
        answer_key = 'A' if answer_idx == 0 else 'B'

        return {
            'prompt': prompt,
            'answer': answer_key,
            'choices': ['A', 'B']
        }

    def format_winogrande_example(self, example):
        """Format WinoGrande example for partial evaluation."""
        sentence = example['sentence']
        option1 = example['option1']
        option2 = example['option2']
        answer = example['answer'] # "1" or "2"
        
        # Standard Winogrande evaluation:
        # Context = Prefix + Option
        # Target = Suffix
        # Compare P(Suffix | Prefix + Option1) vs P(Suffix | Prefix + Option2)
        
        if "_" in sentence:
            parts = sentence.split("_")
            prefix = parts[0]
            suffix = "".join(parts[1:])
            
            # Ensure spacing is handled reasonably
            # If prefix ends with space, good. If option starts with space, good.
            # Usually simple concatenation works for Winogrande.
            
            return {
                'type': 'completion',
                'candidates': [
                    (prefix + option1, suffix),
                    (prefix + option2, suffix)
                ],
                'answer_idx': 0 if answer == '1' else 1,
                'normalize': False,
                'prompt': sentence # For logging
            }
        else:
            # Fallback to MCQ if no underscore (shouldn't happen in standard dataset)
            prompt = f"Fill in the blank: {sentence}\n A. {option1}\n B. {option2}\nAnswer:"
            return {
                'prompt': prompt,
                'answer': 'A' if answer == '1' else 'B',
                'choices': ['A', 'B']
            }

    def evaluate_completion(self, example_dict):
        """
        Evaluate using completion likelihood (for HellaSwag, Winogrande).
        """
        candidates = example_dict['candidates']
        answer_idx = example_dict['answer_idx']
        normalize = example_dict.get('normalize', False)
        
        # Prepare batch
        contexts = [c[0] for c in candidates]
        targets = [c[1] for c in candidates]
        full_texts = [c + t for c, t in zip(contexts, targets)]
        
        # Tokenize
        inputs = self.tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Get context lengths (without padding)
        # Note: Do NOT use return_tensors="pt" here because contexts have different lengths
        context_inputs = self.tokenizer(contexts, padding=False)
        context_lengths = [len(ids) for ids in context_inputs['input_ids']]
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits # (B, L, V)
            
        # Calculate log likelihood of targets
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inputs['input_ids'][..., 1:].contiguous()
        
        log_likelihoods = []
        
        for i in range(len(candidates)):
            # Range of interest in shifted arrays
            # Target starts at context_lengths[i] in original sequence
            # So in shifted (prediction) array, it starts at context_lengths[i] - 1
            
            start_idx = max(0, context_lengths[i] - 1)
            # End at real length - 1
            end_idx = inputs['attention_mask'][i].sum().item() - 1
            
            if start_idx >= end_idx:
                log_likelihoods.append(-float('inf'))
                continue
                
            target_logits = shift_logits[i, start_idx:end_idx, :]
            target_ids = shift_labels[i, start_idx:end_idx]
            
            # Gather log probs
            log_probs = torch.nn.functional.log_softmax(target_logits, dim=-1)
            target_log_probs = log_probs.gather(1, target_ids.unsqueeze(-1)).squeeze(-1)
            sum_log_prob = target_log_probs.sum().item()
            
            if normalize == 'char':
                sum_log_prob /= len(targets[i])
            elif normalize == 'token':
                sum_log_prob /= (end_idx - start_idx)
            
            log_likelihoods.append(sum_log_prob)
            
        best_idx = log_likelihoods.index(max(log_likelihoods))
        correct = (best_idx == answer_idx)
        
        return {
            'prompt': example_dict.get('prompt', full_texts[0]),
            'prediction': best_idx,
            'correct_answer': answer_idx,
            'generated': f"Choice {best_idx}", 
            'correct': correct,
            'log_likelihoods': log_likelihoods
        }

    def evaluate_example(self, example_dict, max_new_tokens: int = 1):
        """
        Evaluate a single example.
        Dispatches to appropriate method based on example type.
        """
        if example_dict.get('type') == 'completion':
            return self.evaluate_completion(example_dict)

        prompt = example_dict['prompt']
        correct_answer = example_dict['answer']
        choices = example_dict['choices']

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Get token IDs for choices (e.g., " A", " B", " C", " D")
        # Note: OLMES uses " A" (space + letter).
        choice_tokens = []
        for choice in choices:
            # Try " A" first
            token_str = f" {choice}"
            token_ids = self.tokenizer.encode(token_str, add_special_tokens=False)
            
            # If " A" is multiple tokens, take the last one or fallback to just "A"
            # Ideally it should be a single token.
            if len(token_ids) == 1:
                choice_tokens.append(token_ids[0])
            else:
                # Fallback to just the letter if " A" is weird
                token_ids = self.tokenizer.encode(choice, add_special_tokens=False)
                choice_tokens.append(token_ids[0])

        # Run forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, -1, :]  # Last token logits

        # Get logits for choice tokens
        choice_logits = logits[choice_tokens]
        probs = torch.nn.functional.softmax(choice_logits, dim=0)
        
        # Get prediction
        best_idx = torch.argmax(choice_logits).item()
        prediction = choices[best_idx]

        # Check correctness
        correct = (prediction == correct_answer)

        return {
            'prompt': prompt,
            'prediction': prediction,
            'correct_answer': correct_answer,
            'generated': prediction, # For compatibility
            'correct': correct,
            'probs': {c: p.item() for c, p in zip(choices, probs)}
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
            elif dataset_name == "hellaswag":
                formatted = self.format_hellaswag_example(example)
            elif dataset_name == "piqa":
                formatted = self.format_piqa_example(example)
            elif dataset_name == "winogrande":
                formatted = self.format_winogrande_example(example)
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
            elif dataset_name == "hellaswag":
                formatted = self.format_hellaswag_example(example)
            elif dataset_name == "piqa":
                formatted = self.format_piqa_example(example)
            elif dataset_name == "winogrande":
                formatted = self.format_winogrande_example(example)
            else:
                raise ValueError(f"Dataset formatting not implemented for: {dataset_name}")

            # Run evaluation
            result = self.evaluate_example(formatted, max_new_tokens=max_new_tokens)

            # Capture per-layer routing data
            layers_data = {}
            for wrapper in self.profiler.wrappers:
                # Get routing data for this question
                k_per_token = wrapper.metrics.k_per_token.copy()
                flops_per_token = wrapper.metrics.flops_per_token.copy()
                latency_ms = wrapper.metrics.latency_ms.copy()

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
                    'flops_per_token': flops_per_token,
                    'mean_flops': float(sum(flops_per_token) / len(flops_per_token)) if flops_per_token else 0,
                    'latency_ms': latency_ms,
                    'mean_latency_ms': float(sum(latency_ms) / len(latency_ms)) if latency_ms else 0,
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

        # Compute k_stats from per-question data (not from profiler, since we reset per question)
        all_avg_k = [q['avg_k_all_layers'] for q in questions_data if q['avg_k_all_layers'] > 0]
        all_max_k = [q['max_k_all_layers'] for q in questions_data if q['max_k_all_layers'] > 0]
        all_min_k = [q['min_k_all_layers'] for q in questions_data if q['min_k_all_layers'] > 0]

        k_stats = {
            'k_mean': float(sum(all_avg_k) / len(all_avg_k)) if all_avg_k else 0,
            'k_std': float(torch.tensor(all_avg_k).std()) if len(all_avg_k) > 1 else 0,
            'k_min': float(min(all_min_k)) if all_min_k else 0,
            'k_max': float(max(all_max_k)) if all_max_k else 0,
        }

        # Compute summary stats from per-question data (not from profiler, since we reset per question)
        all_flops = []
        all_latency = []
        for q in questions_data:
            for layer_data in q['layers'].values():
                if layer_data['mean_flops'] > 0:
                    all_flops.append(layer_data['mean_flops'])
                if layer_data['mean_latency_ms'] > 0:
                    all_latency.append(layer_data['mean_latency_ms'])

        summary = {
            'flops_total_mean': float(sum(all_flops) / len(all_flops)) if all_flops else 0,
            'flops_total_std': float(torch.tensor(all_flops).std()) if len(all_flops) > 1 else 0,
            'latency_mean_ms': float(sum(all_latency) / len(all_latency)) if all_latency else 0,
            'latency_std_ms': float(torch.tensor(all_latency).std()) if len(all_latency) > 1 else 0,
        }

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
        correct_count = sum(1 for q in questions_data if q['correct'])
        print(f"\n{'='*80}")
        print(f"Analysis Complete: {config_name}")
        print(f"{'='*80}")
        print(f"  Accuracy:     {accuracy:.2f}% ({correct_count}/{len(questions_data)})")
        print(f"  Mean k:       {k_stats.get('k_mean', 0):.2f}")
        print(f"  K range:      [{k_stats.get('k_min', 0):.1f}, {k_stats.get('k_max', 0):.1f}]")
        if 'error' not in summary:
            print(f"  Mean FLOPs:   {summary['flops_total_mean']:.2e}")
            print(f"  Mean latency: {summary['latency_mean_ms']:.2f} ms")
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
