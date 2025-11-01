#!/usr/bin/env python3
"""
Test Dataset Uniformity
========================

Analyzes dataset to verify uniform distribution after proportional sampling fix.

Expected improvements:
- Chi-square test: p > 0.05 for all parameters (PASS)
- Imbalance ratio: ~1.1× (down from 4.26×)
- Edge samples: ~40% for L2, Rp2 (down from 60%)
"""

import numpy as np
import pickle
from scipy import stats
from collections import Counter
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from model_common
from model_common import PARAM_NAMES, RANGES


def load_dataset(filepath):
    """Load dataset from pickle file."""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data['X'], data.get('generation_params', {})


def chi_square_uniformity_test(values, n_bins=10):
    """
    Perform chi-square test for uniform distribution.

    Returns:
        chi2_stat: Chi-square statistic
        p_value: P-value (> 0.05 means uniform)
        bin_counts: Observed counts per bin
    """
    # Create bins
    hist, bin_edges = np.histogram(values, bins=n_bins)

    # Expected count (uniform distribution)
    expected = len(values) / n_bins

    # Chi-square test
    chi2_stat = np.sum((hist - expected)**2 / expected)
    df = n_bins - 1
    p_value = 1 - stats.chi2.cdf(chi2_stat, df)

    return chi2_stat, p_value, hist


def calculate_imbalance_ratio(X, n_bins_per_param=5):
    """
    Calculate imbalance ratio (max/min bin counts) for 7D stratification.

    Returns:
        imbalance_ratio: max_count / min_count
        bin_counts: dict with bin statistics
    """
    # Import here to avoid circular imports
    import dataset_stratified_7d
    create_parameter_bins = dataset_stratified_7d.create_parameter_bins
    get_multidim_bin_key = dataset_stratified_7d.get_multidim_bin_key

    # Create bins
    param_bins = create_parameter_bins(n_bins_per_param)

    # Count samples per 7D bin
    bin_counts = Counter()
    for i in range(len(X)):
        # Convert to Angstroms for L/Rp parameters
        combo = (
            X[i, 0],  # Dmax1
            X[i, 1],  # D01
            X[i, 2] * 1e8,  # L1 (cm → Å)
            X[i, 3] * 1e8,  # Rp1 (cm → Å)
            X[i, 4],  # D02
            X[i, 5] * 1e8,  # L2 (cm → Å)
            X[i, 6] * 1e8,  # Rp2 (cm → Å)
        )
        bin_key = get_multidim_bin_key(combo, param_bins)
        bin_counts[bin_key] += 1

    # Calculate imbalance
    counts = list(bin_counts.values())
    min_count = min(counts)
    max_count = max(counts)
    imbalance_ratio = max_count / min_count

    return imbalance_ratio, {
        'min': min_count,
        'max': max_count,
        'mean': np.mean(counts),
        'median': np.median(counts),
        'std': np.std(counts),
        'num_bins': len(bin_counts)
    }


def calculate_edge_percentage(X, param_idx, threshold=0.2):
    """
    Calculate percentage of samples in edge regions (bottom/top 20%).

    Args:
        X: Dataset parameters [N, 7]
        param_idx: Index of parameter (5=L2, 6=Rp2)
        threshold: Edge region threshold (0.2 = 20%)

    Returns:
        edge_percentage: Percentage of samples in edge regions
    """
    param_name = PARAM_NAMES[param_idx]
    param_range = RANGES[param_name]

    values = X[:, param_idx]

    # Define edge thresholds
    threshold_low = param_range[0] + threshold * (param_range[1] - param_range[0])
    threshold_high = param_range[1] - threshold * (param_range[1] - param_range[0])

    # Count edge samples
    edge_mask = (values < threshold_low) | (values > threshold_high)
    edge_count = np.sum(edge_mask)
    edge_percentage = 100 * edge_count / len(X)

    return edge_percentage


def analyze_dataset(filepath, comparison_file=None):
    """
    Analyze dataset uniformity and compare with baseline if provided.

    Args:
        filepath: Path to dataset to analyze
        comparison_file: Optional path to old dataset for comparison
    """
    print(f"\n{'='*80}")
    print(f"DATASET UNIFORMITY ANALYSIS")
    print(f"{'='*80}")

    # Load dataset
    X, gen_params = load_dataset(filepath)
    print(f"\n📦 Dataset: {filepath}")
    print(f"   Samples: {len(X):,}")
    print(f"   Proportional sampling: {gen_params.get('proportional_sampling', 'Unknown')}")
    print(f"   Generated: {gen_params.get('timestamp', 'Unknown')}")

    # 1. Chi-square test for each parameter
    print(f"\n{'='*80}")
    print(f"1. CHI-SQUARE UNIFORMITY TEST (p > 0.05 = PASS)")
    print(f"{'='*80}")
    print(f"{'Parameter':<10} {'Chi2':>10} {'p-value':>10} {'Status':>10}")
    print(f"{'-'*80}")

    chi_square_results = {}
    all_pass = True

    for i, param in enumerate(PARAM_NAMES):
        chi2, p_value, bin_counts = chi_square_uniformity_test(X[:, i], n_bins=10)
        chi_square_results[param] = (chi2, p_value)

        status = "✅ PASS" if p_value > 0.05 else "❌ FAIL"
        if p_value <= 0.05:
            all_pass = False

        print(f"{param:<10} {chi2:>10.2f} {p_value:>10.4f} {status:>10}")

    print(f"\n{'Overall:':<10} {'':>10} {'':>10} {'✅ ALL PASS' if all_pass else '❌ SOME FAIL':>10}")

    # 2. Imbalance ratio
    print(f"\n{'='*80}")
    print(f"2. 7D BIN IMBALANCE RATIO (lower is better)")
    print(f"{'='*80}")

    imbalance_ratio, bin_stats = calculate_imbalance_ratio(X, n_bins_per_param=3)

    print(f"   Imbalance ratio: {imbalance_ratio:.2f}× (max/min)")
    print(f"   Min bin count:   {bin_stats['min']:,}")
    print(f"   Max bin count:   {bin_stats['max']:,}")
    print(f"   Mean bin count:  {bin_stats['mean']:.1f}")
    print(f"   Median bin count: {bin_stats['median']:.0f}")
    print(f"   Std bin count:   {bin_stats['std']:.1f}")
    print(f"   Non-empty bins:  {bin_stats['num_bins']:,}")

    # 3. Edge sample percentage
    print(f"\n{'='*80}")
    print(f"3. EDGE SAMPLE PERCENTAGE (bottom/top 20%)")
    print(f"{'='*80}")
    print(f"{'Parameter':<10} {'Edge %':>10} {'Expected':>10} {'Status':>10}")
    print(f"{'-'*80}")

    # Check L2 and Rp2 (focus params for augmented sampling)
    edge_results = {}
    for param_idx, param_name in [(5, 'L2'), (6, 'Rp2')]:
        edge_pct = calculate_edge_percentage(X, param_idx, threshold=0.2)
        edge_results[param_name] = edge_pct

        # Expected: ~40% for 20% threshold (if uniform)
        expected = 40.0
        diff = abs(edge_pct - expected)
        status = "✅ GOOD" if diff < 5 else "⚠️  HIGH" if edge_pct > expected else "⚠️  LOW"

        print(f"{param_name:<10} {edge_pct:>9.1f}% {expected:>9.1f}% {status:>10}")

    # 4. Comparison with old dataset (if provided)
    if comparison_file:
        print(f"\n{'='*80}")
        print(f"4. COMPARISON WITH OLD DATASET")
        print(f"{'='*80}")

        X_old, gen_params_old = load_dataset(comparison_file)
        print(f"\n📦 Old dataset: {comparison_file}")
        print(f"   Samples: {len(X_old):,}")
        print(f"   Proportional sampling: {gen_params_old.get('proportional_sampling', False)}")

        # Old imbalance
        imbalance_old, bin_stats_old = calculate_imbalance_ratio(X_old, n_bins_per_param=3)

        # Old edge percentages
        edge_old_L2 = calculate_edge_percentage(X_old, 5, threshold=0.2)
        edge_old_Rp2 = calculate_edge_percentage(X_old, 6, threshold=0.2)

        print(f"\n{'Metric':<30} {'Old':>12} {'New':>12} {'Change':>12}")
        print(f"{'-'*80}")
        print(f"{'Imbalance ratio':<30} {imbalance_old:>11.2f}× {imbalance_ratio:>11.2f}× {(imbalance_ratio/imbalance_old - 1)*100:>10.1f}%")
        print(f"{'Edge % (L2)':<30} {edge_old_L2:>11.1f}% {edge_results['L2']:>11.1f}% {edge_results['L2'] - edge_old_L2:>10.1f}%")
        print(f"{'Edge % (Rp2)':<30} {edge_old_Rp2:>11.1f}% {edge_results['Rp2']:>11.1f}% {edge_results['Rp2'] - edge_old_Rp2:>10.1f}%")

        # Chi-square comparison
        print(f"\n{'Parameter':<10} {'Old p-value':>12} {'New p-value':>12} {'Status':>15}")
        print(f"{'-'*80}")
        for i, param in enumerate(PARAM_NAMES):
            chi2_old, p_old = chi_square_uniformity_test(X_old[:, i], n_bins=10)[:2]
            chi2_new, p_new = chi_square_results[param]

            status_old = "PASS" if p_old > 0.05 else "FAIL"
            status_new = "PASS" if p_new > 0.05 else "FAIL"
            status = f"{status_old} → {status_new}"

            print(f"{param:<10} {p_old:>12.4f} {p_new:>12.4f} {status:>15}")

    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"\n✅ Uniformity test: {'PASS' if all_pass else 'FAIL'} ({sum(1 for _, p in chi_square_results.values() if p > 0.05)}/7 params)")
    print(f"✅ Imbalance ratio: {imbalance_ratio:.2f}× {'(excellent <1.5)' if imbalance_ratio < 1.5 else '(good <3.0)' if imbalance_ratio < 3.0 else '(needs work)'}")
    print(f"✅ Edge samples: L2={edge_results['L2']:.1f}%, Rp2={edge_results['Rp2']:.1f}% {'(expected ~40%)' if abs(edge_results['L2'] - 40) < 10 else '(off from 40%)'}")

    if comparison_file:
        improvement = (imbalance_old - imbalance_ratio) / imbalance_old * 100
        print(f"\n🎯 Improvement: {improvement:.1f}% reduction in imbalance")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test dataset uniformity")
    parser.add_argument("dataset", help="Path to dataset to analyze")
    parser.add_argument("--compare", help="Path to old dataset for comparison", default=None)

    args = parser.parse_args()

    analyze_dataset(args.dataset, args.compare)
