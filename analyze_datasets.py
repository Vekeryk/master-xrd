"""
Compare balanced vs non-balanced datasets.

Analyzes distribution uniformity using Chi-squared tests.
"""

import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt


# Parameter names and ranges (from model_common.py)
PARAM_NAMES = ['Dmax1', 'D01', 'L1', 'Rp1', 'D02', 'L2', 'Rp2']
RANGES = [
    (0.001, 0.030),  # Dmax1
    (0.002, 0.030),  # D01
    (1000e-8, 7000e-8),  # L1 (cm)
    (0, 7000e-8),  # Rp1 (cm)
    (0.002, 0.030),  # D02
    (1000e-8, 7000e-8),  # L2 (cm)
    (-6000e-8, 0),  # Rp2 (cm)
]


def load_dataset(path):
    """Load dataset from pickle file."""
    print(f"\nLoading: {path}")
    with open(path, 'rb') as f:
        data = pickle.load(f)

    X = data['X']
    Y = data['Y']

    print(f"  X shape: {X.shape}")
    print(f"  Y shape: {Y.shape}")

    return X, Y, data


def compute_uniformity_metrics(X, param_idx, param_name):
    """
    Compute Chi-squared uniformity test for a parameter.

    Lower Chi² = better uniformity.
    Target: Chi² < 10,000 (from DATASET_BIAS_ANALYSIS.md)
    """
    values = X[:, param_idx]

    # Get unique values and counts
    unique_vals, counts = np.unique(values, return_counts=True)

    # Chi-squared test (uniform distribution hypothesis)
    expected_count = len(values) / len(unique_vals)
    chi_squared = np.sum((counts - expected_count)**2 / expected_count)

    # Bias ratio (max/min frequency)
    bias_ratio = counts.max() / counts.min() if counts.min() > 0 else np.inf

    return {
        'unique_values': len(unique_vals),
        'chi_squared': chi_squared,
        'bias_ratio': bias_ratio,
        'min_count': counts.min(),
        'max_count': counts.max(),
        'counts': counts,
        'unique_vals': unique_vals,
    }


def analyze_dataset(dataset_path, name):
    """Analyze distribution uniformity of a dataset."""
    X, Y, data = load_dataset(dataset_path)

    print("\n" + "=" * 70)
    print(f"DISTRIBUTION ANALYSIS: {name}")
    print("=" * 70)

    results = {}

    for i, param_name in enumerate(PARAM_NAMES):
        metrics = compute_uniformity_metrics(X, i, param_name)
        results[param_name] = metrics

        # Print summary
        chi2 = metrics['chi_squared']
        bias = metrics['bias_ratio']

        # Assessment
        if chi2 < 10000:
            status = "✅ PASS"
        elif chi2 < 50000:
            status = "⚠️  WARN"
        else:
            status = "🔴 FAIL"

        print(f"\n{param_name}:")
        print(f"  Unique values: {metrics['unique_values']}")
        print(f"  Chi²:          {chi2:,.0f} {status}")
        print(f"  Bias ratio:    {bias:.2f}x")
        print(f"  Count range:   [{metrics['min_count']}, {metrics['max_count']}]")

    return results, X, Y


def compare_datasets(not_balanced_path, balanced_path):
    """Compare balanced vs not balanced datasets."""
    print("=" * 70)
    print("DATASET COMPARISON: BALANCED vs NOT BALANCED")
    print("=" * 70)

    # Analyze both
    results_nb, X_nb, Y_nb = analyze_dataset(not_balanced_path, "NOT BALANCED")
    results_b, X_b, Y_b = analyze_dataset(balanced_path, "BALANCED")

    # Comparison table
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    print(f"{'Parameter':<10} {'Not Balanced':<25} {'Balanced':<25} {'Improvement':<15}")
    print(f"{'':10} {'Chi²':<12} {'Bias':<12} {'Chi²':<12} {'Bias':<12} {'Chi²':<15}")
    print("-" * 70)

    improvements = {}

    for param_name in PARAM_NAMES:
        nb = results_nb[param_name]
        b = results_b[param_name]

        chi2_nb = nb['chi_squared']
        chi2_b = b['chi_squared']
        bias_nb = nb['bias_ratio']
        bias_b = b['bias_ratio']

        chi2_improve = (chi2_nb - chi2_b) / chi2_nb * 100 if chi2_nb > 0 else 0

        improvements[param_name] = {
            'chi2_improve': chi2_improve,
            'bias_improve': bias_nb - bias_b
        }

        print(f"{param_name:<10} "
              f"{chi2_nb:>11,.0f} {bias_nb:>11.1f}x "
              f"{chi2_b:>11,.0f} {bias_b:>11.1f}x "
              f"{chi2_improve:>14.1f}%")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    avg_chi2_nb = np.mean([results_nb[p]['chi_squared'] for p in PARAM_NAMES])
    avg_chi2_b = np.mean([results_b[p]['chi_squared'] for p in PARAM_NAMES])

    print(f"\nAverage Chi-squared:")
    print(f"  Not Balanced: {avg_chi2_nb:,.0f}")
    print(f"  Balanced:     {avg_chi2_b:,.0f}")
    print(f"  Improvement:  {(avg_chi2_nb - avg_chi2_b) / avg_chi2_nb * 100:.1f}%")

    # Visualization
    visualize_comparison(results_nb, results_b, X_nb, X_b)

    return results_nb, results_b, improvements


def visualize_comparison(results_nb, results_b, X_nb, X_b):
    """Create visualization comparing distributions."""

    # 1. Chi-squared comparison
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Distribution Comparison: Balanced vs Not Balanced (1M samples)',
                 fontsize=14, fontweight='bold')

    for i, param_name in enumerate(PARAM_NAMES):
        ax = axes[i // 4, i % 4]

        # Histogram comparison
        ax.hist(X_nb[:, i], bins=50, alpha=0.5, label='Not Balanced',
                color='red', edgecolor='black')
        ax.hist(X_b[:, i], bins=50, alpha=0.5, label='Balanced',
                color='green', edgecolor='black')

        chi2_nb = results_nb[param_name]['chi_squared']
        chi2_b = results_b[param_name]['chi_squared']

        ax.set_title(f'{param_name}\nχ²: {chi2_nb:,.0f} → {chi2_b:,.0f}',
                     fontsize=10)
        ax.set_xlabel('Value')
        ax.set_ylabel('Count')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Remove empty subplot
    axes[1, 3].remove()

    plt.tight_layout()
    plt.savefig('dataset_comparison_histograms.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Histograms saved to: dataset_comparison_histograms.png")

    # 2. Chi-squared bar chart
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    params = PARAM_NAMES
    chi2_nb_values = [results_nb[p]['chi_squared'] for p in params]
    chi2_b_values = [results_b[p]['chi_squared'] for p in params]

    x = np.arange(len(params))
    width = 0.35

    bars1 = ax.bar(x - width/2, chi2_nb_values, width, label='Not Balanced',
                   color='red', alpha=0.7, edgecolor='black')
    bars2 = ax.bar(x + width/2, chi2_b_values, width, label='Balanced',
                   color='green', alpha=0.7, edgecolor='black')

    ax.axhline(y=10000, color='orange', linestyle='--', linewidth=2,
               label='Target (χ² < 10,000)')

    ax.set_xlabel('Parameter', fontsize=12, fontweight='bold')
    ax.set_ylabel('Chi-squared Uniformity', fontsize=12, fontweight='bold')
    ax.set_title('Distribution Uniformity: Balanced vs Not Balanced (1M samples)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(params)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')

    # Add improvement percentages
    for i, (nb, b) in enumerate(zip(chi2_nb_values, chi2_b_values)):
        improve = (nb - b) / nb * 100 if nb > 0 else 0
        ax.text(i, max(nb, b) * 1.2, f'{improve:.0f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig('dataset_comparison_chi2.png', dpi=150, bbox_inches='tight')
    print(f"📊 Chi-squared comparison saved to: dataset_comparison_chi2.png")

    # 3. L2 special analysis (most problematic parameter)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Convert L2 from cm to Angstroms for readability
    l2_nb = X_nb[:, 5] * 1e8
    l2_b = X_b[:, 5] * 1e8

    # Histogram
    axes[0].hist(l2_nb, bins=50, alpha=0.6, label='Not Balanced',
                 color='red', edgecolor='black')
    axes[0].hist(l2_b, bins=50, alpha=0.6, label='Balanced',
                 color='green', edgecolor='black')
    axes[0].set_xlabel('L2 (Angstroms)', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Count', fontsize=11, fontweight='bold')
    axes[0].set_title(f'L2 Distribution (χ²: {results_nb["L2"]["chi_squared"]:,.0f} → {results_b["L2"]["chi_squared"]:,.0f})',
                      fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Value counts comparison
    unique_nb, counts_nb = np.unique(l2_nb, return_counts=True)
    unique_b, counts_b = np.unique(l2_b, return_counts=True)

    axes[1].plot(unique_nb, counts_nb, 'o-', color='red', label='Not Balanced',
                 linewidth=2, markersize=8, alpha=0.7)
    axes[1].plot(unique_b, counts_b, 's-', color='green', label='Balanced',
                 linewidth=2, markersize=8, alpha=0.7)
    axes[1].axhline(y=len(l2_b)/len(unique_b), color='green', linestyle='--',
                    alpha=0.5, label='Expected (balanced)')
    axes[1].set_xlabel('L2 Value (Angstroms)', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Count', fontsize=11, fontweight='bold')
    axes[1].set_title('L2 Frequency per Unique Value', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('dataset_comparison_L2.png', dpi=150, bbox_inches='tight')
    print(f"📊 L2 detailed comparison saved to: dataset_comparison_L2.png")


if __name__ == "__main__":
    # Dataset paths
    NOT_BALANCED_PATH = "datasets/dataset_1000000_dl400_jit.pkl"
    BALANCED_PATH = "datasets/dataset_1000000_dl100_balanced.pkl"

    # Check if files exist
    if not Path(NOT_BALANCED_PATH).exists():
        print(f"❌ Not found: {NOT_BALANCED_PATH}")
        exit(1)

    if not Path(BALANCED_PATH).exists():
        print(f"❌ Not found: {BALANCED_PATH}")
        exit(1)

    # Compare
    results_nb, results_b, improvements = compare_datasets(
        NOT_BALANCED_PATH,
        BALANCED_PATH
    )

    print("\n" + "=" * 70)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - dataset_comparison_histograms.png")
    print("  - dataset_comparison_chi2.png")
    print("  - dataset_comparison_L2.png")
