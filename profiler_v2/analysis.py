"""
Analysis utilities for MoE routing behavior.

Functions to analyze:
- Expert utilization
- Per-subject patterns (MMLU)
- Hard vs easy routing
- High-k cases
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List, Dict, Any


def analyze_expert_utilization(questions: List[Dict], plot=True, save_path='expert_utilization.png'):
    """
    Analyze how often each expert is used.

    Args:
        questions: List of question dictionaries from analysis data
        plot: Whether to create visualization
        save_path: Where to save plot

    Returns:
        Dictionary with expert utilization statistics
    """
    expert_loads = defaultdict(int)
    expert_subjects = defaultdict(lambda: defaultdict(int))
    expert_layers = defaultdict(lambda: defaultdict(int))

    # Aggregate expert usage
    for q in questions:
        subject = q['subject']
        for layer_name, layer_data in q['layers'].items():
            for expert_id, count in layer_data['expert_load'].items():
                expert_loads[expert_id] += count
                expert_subjects[expert_id][subject] += count
                expert_layers[expert_id][layer_name] += count

    # Identify dead/underutilized experts
    total_load = sum(expert_loads.values())
    dead_threshold = total_load * 0.001  # < 0.1% of total load
    dead_experts = [e for e, load in expert_loads.items() if load < dead_threshold]

    # Compute load balance metrics
    loads = list(expert_loads.values())
    load_mean = np.mean(loads)
    load_std = np.std(loads)
    load_cv = load_std / load_mean if load_mean > 0 else 0  # Coefficient of variation

    results = {
        'expert_loads': dict(expert_loads),
        'dead_experts': dead_experts,
        'expert_subjects': {e: dict(s) for e, s in expert_subjects.items()},
        'expert_layers': {e: dict(l) for e, l in expert_layers.items()},
        'statistics': {
            'total_load': total_load,
            'mean_load': load_mean,
            'std_load': load_std,
            'cv_load': load_cv,
            'num_dead': len(dead_experts),
        }
    }

    # Print summary
    print("\n" + "="*80)
    print("EXPERT UTILIZATION ANALYSIS")
    print("="*80)
    print(f"\nTotal expert activations: {total_load}")
    print(f"Mean load per expert: {load_mean:.1f}")
    print(f"Std load: {load_std:.1f}")
    print(f"Coefficient of variation: {load_cv:.3f}")
    print(f"Dead experts (<0.1% load): {len(dead_experts)}")
    if dead_experts:
        print(f"  Dead expert IDs: {dead_experts}")

    # Top 10 and bottom 10
    sorted_experts = sorted(expert_loads.items(), key=lambda x: x[1], reverse=True)
    print(f"\nTop 10 most used experts:")
    for expert_id, load in sorted_experts[:10]:
        pct = 100 * load / total_load
        print(f"  Expert {expert_id}: {load} activations ({pct:.2f}%)")

    print(f"\nBottom 10 least used experts:")
    for expert_id, load in sorted_experts[-10:]:
        pct = 100 * load / total_load
        print(f"  Expert {expert_id}: {load} activations ({pct:.2f}%)")

    # Plot if requested
    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Bar chart
        expert_ids = sorted(expert_loads.keys())
        loads = [expert_loads[e] for e in expert_ids]

        ax1.bar(expert_ids, loads, color='skyblue', edgecolor='black')
        ax1.axhline(load_mean, color='red', linestyle='--', label=f'Mean: {load_mean:.0f}')
        ax1.set_xlabel('Expert ID')
        ax1.set_ylabel('Load (# tokens processed)')
        ax1.set_title('Expert Utilization Distribution')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        # Histogram
        ax2.hist(loads, bins=20, color='lightgreen', edgecolor='black')
        ax2.axvline(load_mean, color='red', linestyle='--', label=f'Mean: {load_mean:.0f}')
        ax2.set_xlabel('Load')
        ax2.set_ylabel('Number of Experts')
        ax2.set_title('Load Distribution Histogram')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved plot to: {save_path}")
        plt.close()

    return results


def plot_expert_usage_by_router(questions: List[Dict], save_path='expert_usage_by_router.png'):
    """
    Plot expert usage split by router/layer as a heatmap.

    Args:
        questions: List of question dictionaries from analysis data
        save_path: Where to save plot

    Returns:
        Dictionary with per-router expert loads
    """
    # Collect expert usage per router
    router_expert_loads = defaultdict(lambda: defaultdict(int))

    for q in questions:
        for layer_name, layer_data in q['layers'].items():
            for expert_id, count in layer_data['expert_load'].items():
                router_expert_loads[layer_name][expert_id] += count

    # Convert to DataFrame for heatmap
    routers = sorted(router_expert_loads.keys())
    all_expert_ids = set()
    for router_loads in router_expert_loads.values():
        all_expert_ids.update(router_loads.keys())
    expert_ids = sorted(all_expert_ids)

    # Create matrix: rows = experts, columns = routers
    matrix = []
    for expert_id in expert_ids:
        row = [router_expert_loads[router].get(expert_id, 0) for router in routers]
        matrix.append(row)

    matrix = np.array(matrix)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(max(12, len(routers)), max(8, len(expert_ids) * 0.3)))

    im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd')

    # Set ticks
    ax.set_xticks(np.arange(len(routers)))
    ax.set_yticks(np.arange(len(expert_ids)))
    ax.set_xticklabels(routers, rotation=45, ha='right')
    ax.set_yticklabels(expert_ids)

    # Labels
    ax.set_xlabel('Router/Layer', fontsize=12)
    ax.set_ylabel('Expert ID', fontsize=12)
    ax.set_title('Expert Usage by Router/Layer', fontsize=14, fontweight='bold')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Load (# tokens)', rotation=270, labelpad=20)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved router-wise expert usage plot to: {save_path}")
    plt.close()

    return {router: dict(loads) for router, loads in router_expert_loads.items()}


def analyze_per_subject(questions: List[Dict], save_path='subject_analysis.csv'):
    """
    Analyze routing patterns per MMLU subject.

    Args:
        questions: List of question dictionaries
        save_path: Where to save CSV

    Returns:
        DataFrame with per-subject statistics
    """
    subject_stats = defaultdict(lambda: {
        'questions': [],
        'k_values': [],
        'correct': [],
        'entropy': [],
        'expert_loads': defaultdict(int)
    })

    # Aggregate by subject
    for q in questions:
        subject = q['subject']
        subject_stats[subject]['questions'].append(q['question_id'])
        subject_stats[subject]['k_values'].append(q['avg_k_all_layers'])
        subject_stats[subject]['correct'].append(q['correct'])
        subject_stats[subject]['entropy'].append(q['avg_entropy_all_layers'])

        # Expert usage per subject
        for layer_data in q['layers'].values():
            for expert_id, count in layer_data['expert_load'].items():
                subject_stats[subject]['expert_loads'][expert_id] += count

    # Compute summary statistics
    summary = {}
    for subject, data in subject_stats.items():
        summary[subject] = {
            'num_questions': len(data['questions']),
            'accuracy': 100 * np.mean(data['correct']),
            'mean_k': np.mean(data['k_values']),
            'k_std': np.std(data['k_values']),
            'k_min': np.min(data['k_values']),
            'k_max': np.max(data['k_values']),
            'mean_entropy': np.mean(data['entropy']),
            'entropy_std': np.std(data['entropy']),
        }

    # Create DataFrame and sort
    df = pd.DataFrame(summary).T
    df = df.sort_values('mean_k', ascending=False)

    # Print
    print("\n" + "="*80)
    print("PER-SUBJECT ANALYSIS")
    print("="*80)
    print(df.to_string())
    print("="*80)

    # Save
    df.to_csv(save_path)
    print(f"\n✓ Saved to: {save_path}")

    # Insights
    print("\n📊 Key Insights:")
    highest_k = df['mean_k'].idxmax()
    lowest_k = df['mean_k'].idxmin()
    print(f"  Highest k: {highest_k} (k={df.loc[highest_k, 'mean_k']:.2f})")
    print(f"  Lowest k:  {lowest_k} (k={df.loc[lowest_k, 'mean_k']:.2f})")

    # Correlation between k and accuracy
    corr_k_acc = df['mean_k'].corr(df['accuracy'])
    print(f"  Correlation (k vs accuracy): {corr_k_acc:.3f}")
    if abs(corr_k_acc) > 0.5:
        direction = "higher" if corr_k_acc > 0 else "lower"
        print(f"    → {direction.capitalize()} k → {'higher' if corr_k_acc > 0 else 'lower'} accuracy")

    return df


def analyze_high_k_cases(questions: List[Dict], k_threshold=None, top_n=10):
    """
    Analyze cases where k is high.

    Args:
        questions: List of question dictionaries
        k_threshold: K threshold (default: 75th percentile)
        top_n: Number of examples to show

    Returns:
        List of high-k case dictionaries
    """
    # Determine threshold
    all_k_values = [q['avg_k_all_layers'] for q in questions]
    if k_threshold is None:
        k_threshold = np.percentile(all_k_values, 75)

    # Find high-k cases
    high_k_cases = []
    for q in questions:
        if q['avg_k_all_layers'] > k_threshold:
            high_k_cases.append({
                'question_id': q['question_id'],
                'subject': q['subject'],
                'question': q['question_text'],
                'avg_k': q['avg_k_all_layers'],
                'max_k': q['max_k_all_layers'],
                'entropy': q['avg_entropy_all_layers'],
                'correct': q['correct'],
            })

    # Sort by k
    high_k_cases.sort(key=lambda x: x['avg_k'], reverse=True)

    # Print summary
    print("\n" + "="*80)
    print(f"HIGH-K CASE ANALYSIS (k > {k_threshold:.1f})")
    print("="*80)
    print(f"\nFound {len(high_k_cases)} high-k cases ({100*len(high_k_cases)/len(questions):.1f}% of questions)")

    # Statistics
    high_k_accuracy = np.mean([c['correct'] for c in high_k_cases])
    low_k_cases = [q for q in questions if q['avg_k_all_layers'] <= k_threshold]
    low_k_accuracy = np.mean([q['correct'] for q in low_k_cases])

    print(f"\nAccuracy comparison:")
    print(f"  High-k cases: {100*high_k_accuracy:.1f}%")
    print(f"  Low-k cases:  {100*low_k_accuracy:.1f}%")
    print(f"  Difference:   {100*(high_k_accuracy - low_k_accuracy):+.1f}%")

    # Subject distribution
    subject_counts = defaultdict(int)
    for case in high_k_cases:
        subject_counts[case['subject']] += 1

    print(f"\nSubject distribution of high-k cases:")
    for subject, count in sorted(subject_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        pct = 100 * count / len(high_k_cases)
        print(f"  {subject}: {count} ({pct:.1f}%)")

    # Show examples
    print(f"\nTop {top_n} highest-k questions:")
    for i, case in enumerate(high_k_cases[:top_n], 1):
        print(f"\n{i}. Subject: {case['subject']}")
        print(f"   K: {case['avg_k']:.1f} (max: {case['max_k']:.0f})")
        print(f"   Entropy: {case['entropy']:.2f}")
        print(f"   Correct: {case['correct']}")
        print(f"   Q: {case['question'][:100]}...")

    return high_k_cases


def compare_configs(data_baseline: Dict, data_kneedle: Dict, save_path='config_comparison.png'):
    """
    Compare two configurations (e.g., baseline vs kneedle).

    Args:
        data_baseline: Analysis data from baseline
        data_kneedle: Analysis data from kneedle
        save_path: Where to save comparison plot

    Returns:
        Comparison dictionary
    """
    questions_baseline = data_baseline['questions']
    questions_kneedle = data_kneedle['questions']

    assert len(questions_baseline) == len(questions_kneedle), "Must have same number of questions"

    # Per-question comparison
    k_diffs = []
    entropy_diffs = []
    correctness_changes = []

    for qb, qk in zip(questions_baseline, questions_kneedle):
        k_diff = qk['avg_k_all_layers'] - qb['avg_k_all_layers']
        entropy_diff = qk['avg_entropy_all_layers'] - qb['avg_entropy_all_layers']

        k_diffs.append(k_diff)
        entropy_diffs.append(entropy_diff)

        if qb['correct'] != qk['correct']:
            correctness_changes.append({
                'question_id': qb['question_id'],
                'subject': qb['subject'],
                'baseline_correct': qb['correct'],
                'kneedle_correct': qk['correct'],
                'k_diff': k_diff,
            })

    # Summary statistics
    comparison = {
        'accuracy': {
            'baseline': data_baseline['accuracy'],
            'kneedle': data_kneedle['accuracy'],
            'delta': data_kneedle['accuracy'] - data_baseline['accuracy'],
        },
        'mean_k': {
            'baseline': data_baseline['k_stats']['k_mean'],
            'kneedle': data_kneedle['k_stats']['k_mean'],
            'delta': data_kneedle['k_stats']['k_mean'] - data_baseline['k_stats']['k_mean'],
        },
        'k_reduction': {
            'mean': np.mean(k_diffs),
            'std': np.std(k_diffs),
            'questions_reduced': sum(1 for d in k_diffs if d < 0),
            'questions_increased': sum(1 for d in k_diffs if d > 0),
        },
        'correctness_changes': correctness_changes,
    }

    # Print
    print("\n" + "="*80)
    print("CONFIGURATION COMPARISON")
    print("="*80)
    print(f"\nAccuracy:")
    print(f"  Baseline: {comparison['accuracy']['baseline']:.2f}%")
    print(f"  Kneedle:  {comparison['accuracy']['kneedle']:.2f}%")
    print(f"  Delta:    {comparison['accuracy']['delta']:+.2f}%")

    print(f"\nMean k:")
    print(f"  Baseline: {comparison['mean_k']['baseline']:.2f}")
    print(f"  Kneedle:  {comparison['mean_k']['kneedle']:.2f}")
    print(f"  Delta:    {comparison['mean_k']['delta']:+.2f}")

    print(f"\nK changes per question:")
    print(f"  Questions with k reduced:   {comparison['k_reduction']['questions_reduced']} ({100*comparison['k_reduction']['questions_reduced']/len(k_diffs):.1f}%)")
    print(f"  Questions with k increased: {comparison['k_reduction']['questions_increased']} ({100*comparison['k_reduction']['questions_increased']/len(k_diffs):.1f}%)")

    print(f"\nCorrectness changes: {len(correctness_changes)} questions changed")
    if correctness_changes:
        improved = sum(1 for c in correctness_changes if c['kneedle_correct'] and not c['baseline_correct'])
        degraded = sum(1 for c in correctness_changes if c['baseline_correct'] and not c['kneedle_correct'])
        print(f"  Improved: {improved}")
        print(f"  Degraded: {degraded}")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # K difference histogram
    axes[0, 0].hist(k_diffs, bins=30, color='lightblue', edgecolor='black')
    axes[0, 0].axvline(0, color='red', linestyle='--', label='No change')
    axes[0, 0].axvline(np.mean(k_diffs), color='green', linestyle='--', label=f'Mean: {np.mean(k_diffs):.2f}')
    axes[0, 0].set_xlabel('K difference (kneedle - baseline)')
    axes[0, 0].set_ylabel('Number of questions')
    axes[0, 0].set_title('K Change Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)

    # K scatter
    k_baseline = [q['avg_k_all_layers'] for q in questions_baseline]
    k_kneedle = [q['avg_k_all_layers'] for q in questions_kneedle]
    axes[0, 1].scatter(k_baseline, k_kneedle, alpha=0.5)
    axes[0, 1].plot([0, max(k_baseline)], [0, max(k_baseline)], 'r--', label='y=x')
    axes[0, 1].set_xlabel('Baseline k')
    axes[0, 1].set_ylabel('Kneedle k')
    axes[0, 1].set_title('K Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # Entropy difference
    axes[1, 0].hist(entropy_diffs, bins=30, color='lightgreen', edgecolor='black')
    axes[1, 0].axvline(0, color='red', linestyle='--')
    axes[1, 0].set_xlabel('Entropy difference (kneedle - baseline)')
    axes[1, 0].set_ylabel('Number of questions')
    axes[1, 0].set_title('Entropy Change Distribution')
    axes[1, 0].grid(axis='y', alpha=0.3)

    # Accuracy by k
    k_bins = np.linspace(min(k_baseline + k_kneedle), max(k_baseline + k_kneedle), 10)
    baseline_acc_by_k = []
    kneedle_acc_by_k = []
    for i in range(len(k_bins) - 1):
        baseline_in_bin = [q['correct'] for q in questions_baseline
                          if k_bins[i] <= q['avg_k_all_layers'] < k_bins[i+1]]
        kneedle_in_bin = [q['correct'] for q in questions_kneedle
                         if k_bins[i] <= q['avg_k_all_layers'] < k_bins[i+1]]
        baseline_acc_by_k.append(np.mean(baseline_in_bin) if baseline_in_bin else 0)
        kneedle_acc_by_k.append(np.mean(kneedle_in_bin) if kneedle_in_bin else 0)

    bin_centers = (k_bins[:-1] + k_bins[1:]) / 2
    axes[1, 1].plot(bin_centers, baseline_acc_by_k, 'o-', label='Baseline')
    axes[1, 1].plot(bin_centers, kneedle_acc_by_k, 's-', label='Kneedle')
    axes[1, 1].set_xlabel('K value')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Accuracy vs K')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved comparison plot to: {save_path}")
    plt.close()

    return comparison
