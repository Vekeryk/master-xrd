#!/usr/bin/env python3
"""
Analyze Dataset Coverage for Experimental Parameters
=====================================================

Check how many samples in the 100k dataset are similar to the experimental case.
"""

import numpy as np
import pickle
import sys
from model_common import PARAM_NAMES, RANGES

# Experimental parameters
EXPERIMENT_PARAMS = [0.008094, 0.000943, 5200e-8, 3500e-8, 0.00255, 3000e-8, -50e-8]

def load_dataset(filepath):
    """Load dataset from pickle file."""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data['X']


def normalize_params(params, ranges):
    """Normalize parameters to [0, 1] range."""
    normalized = np.zeros_like(params)
    for i, (min_val, max_val) in enumerate(ranges.values()):
        normalized[..., i] = (params[..., i] - min_val) / (max_val - min_val)
    return normalized


def find_similar_samples(X, target_params, threshold_pct=10):
    """
    Find samples similar to target parameters.

    Args:
        X: Dataset parameters [N, 7]
        target_params: Target parameters [7]
        threshold_pct: Percentage threshold for each parameter (10% = within ±10% of range)

    Returns:
        indices: Indices of similar samples
        distances: Euclidean distances of similar samples
    """
    target = np.array(target_params)

    # Normalize both dataset and target to [0, 1]
    X_norm = normalize_params(X, RANGES)
    target_norm = normalize_params(target.reshape(1, -1), RANGES)[0]

    # Calculate absolute differences per parameter
    diffs = np.abs(X_norm - target_norm)

    # Threshold in normalized space
    threshold = threshold_pct / 100.0

    # Find samples where ALL parameters are within threshold
    within_threshold = np.all(diffs <= threshold, axis=1)
    indices = np.where(within_threshold)[0]

    # Calculate Euclidean distance in normalized space
    distances = np.linalg.norm(X_norm[indices] - target_norm, axis=1)

    return indices, distances, diffs


def analyze_coverage(dataset_path, experiment_params, thresholds=[5, 10, 15, 20, 25]):
    """Analyze dataset coverage around experimental parameters."""
    print(f"\n{'='*80}")
    print(f"DATASET COVERAGE ANALYSIS FOR EXPERIMENTAL PARAMETERS")
    print(f"{'='*80}")

    # Load dataset
    X = load_dataset(dataset_path)
    print(f"\n📦 Dataset: {dataset_path}")
    print(f"   Total samples: {len(X):,}")

    # Show experimental parameters
    print(f"\n🎯 Experimental parameters:")
    for i, (param, value) in enumerate(zip(PARAM_NAMES, experiment_params)):
        param_range = list(RANGES.values())[i]
        # Normalize to [0, 1]
        normalized = (value - param_range[0]) / (param_range[1] - param_range[0])
        print(f"   {param:8s}: {value:12.6e}  (normalized: {normalized:.3f}, range: [{param_range[0]:.6e}, {param_range[1]:.6e}])")

    # Check if parameters are within ranges
    print(f"\n🔍 Parameter range check:")
    all_in_range = True
    for i, (param, value) in enumerate(zip(PARAM_NAMES, experiment_params)):
        param_range = list(RANGES.values())[i]
        in_range = param_range[0] <= value <= param_range[1]
        status = "✅ IN" if in_range else "❌ OUT"
        if not in_range:
            all_in_range = False
        print(f"   {param:8s}: {status} of range")

    if not all_in_range:
        print(f"\n⚠️  WARNING: Some parameters are OUTSIDE the dataset range!")
        print(f"   Model may not generalize well to these values.")

    # Analyze coverage at different thresholds
    print(f"\n{'='*80}")
    print(f"COVERAGE AT DIFFERENT THRESHOLDS")
    print(f"{'='*80}")
    print(f"{'Threshold':<12} {'Count':>8} {'%':>8} {'Avg Distance':>14}")
    print(f"{'-'*80}")

    for threshold in thresholds:
        indices, distances, _ = find_similar_samples(X, experiment_params, threshold_pct=threshold)
        count = len(indices)
        percentage = 100 * count / len(X)
        avg_dist = np.mean(distances) if count > 0 else 0

        print(f"±{threshold:2d}% of range {count:>8,} {percentage:>7.3f}% {avg_dist:>14.6f}")

    # Detailed analysis at 10% threshold
    print(f"\n{'='*80}")
    print(f"DETAILED ANALYSIS AT ±10% THRESHOLD")
    print(f"{'='*80}")

    indices, distances, diffs = find_similar_samples(X, experiment_params, threshold_pct=10)

    print(f"\nSamples within ±10%: {len(indices):,} ({100*len(indices)/len(X):.3f}%)")

    if len(indices) > 0:
        # Show closest samples
        sorted_idx = np.argsort(distances)
        top_n = min(10, len(indices))

        print(f"\n🎯 {top_n} closest samples:")
        print(f"{'Rank':<6} {'Index':>8} {'Distance':>12} {'Parameters':>50}")
        print(f"{'-'*80}")

        for rank, idx in enumerate(sorted_idx[:top_n]):
            sample_idx = indices[idx]
            dist = distances[idx]
            params_str = "[" + ", ".join([f"{X[sample_idx, i]:.4e}" for i in range(7)]) + "]"
            print(f"{rank+1:<6} {sample_idx:>8} {dist:>12.6f} {params_str}")

        # Per-parameter deviation analysis
        print(f"\n📊 Per-parameter deviation statistics (normalized [0,1] space):")
        print(f"{'Parameter':<10} {'Min':>10} {'Max':>10} {'Mean':>10} {'Median':>10}")
        print(f"{'-'*80}")

        X_norm = normalize_params(X[indices], RANGES)
        target_norm = normalize_params(np.array(experiment_params).reshape(1, -1), RANGES)[0]
        param_diffs = np.abs(X_norm - target_norm)

        for i, param in enumerate(PARAM_NAMES):
            print(f"{param:<10} {param_diffs[:, i].min():>10.6f} {param_diffs[:, i].max():>10.6f} "
                  f"{param_diffs[:, i].mean():>10.6f} {np.median(param_diffs[:, i]):>10.6f}")
    else:
        print(f"\n⚠️  NO samples found within ±10% threshold!")
        print(f"\nThis suggests the experimental parameters are in a SPARSE region.")
        print(f"Model predictions may be unreliable for this case.")

        # Show which parameters are problematic
        print(f"\n📊 Parameter deviations from closest samples:")
        # Find closest sample overall
        X_norm = normalize_params(X, RANGES)
        target_norm = normalize_params(np.array(experiment_params).reshape(1, -1), RANGES)[0]
        distances_all = np.linalg.norm(X_norm - target_norm, axis=1)
        closest_idx = np.argmin(distances_all)

        print(f"\nClosest sample (index {closest_idx}, distance {distances_all[closest_idx]:.6f}):")
        param_diffs = np.abs(X_norm[closest_idx] - target_norm)

        print(f"{'Parameter':<10} {'Target':>12} {'Closest':>12} {'Diff (norm)':>14} {'Status':>10}")
        print(f"{'-'*80}")
        for i, param in enumerate(PARAM_NAMES):
            target_val = experiment_params[i]
            closest_val = X[closest_idx, i]
            diff_norm = param_diffs[i]
            status = "✅ CLOSE" if diff_norm < 0.1 else "⚠️  FAR"
            print(f"{param:<10} {target_val:>12.6e} {closest_val:>12.6e} {diff_norm:>14.6f} {status:>10}")

    # Recommendations
    print(f"\n{'='*80}")
    print(f"RECOMMENDATIONS")
    print(f"{'='*80}")

    if not all_in_range:
        print(f"\n❌ CRITICAL: Experimental parameters are OUTSIDE dataset range!")
        print(f"   ACTION: Extend dataset generation ranges or use different experimental values.")
    elif len(indices) == 0:
        print(f"\n⚠️  SPARSE REGION: No samples within ±10% of experimental parameters.")
        print(f"   OPTIONS:")
        print(f"   1. Generate MORE samples (200k+) to improve coverage")
        print(f"   2. Use targeted sampling around experimental region")
        print(f"   3. Accept higher uncertainty in predictions")
    elif len(indices) < 100:
        print(f"\n⚠️  LOW COVERAGE: Only {len(indices)} samples within ±10%.")
        print(f"   OPTIONS:")
        print(f"   1. Generate more samples (200k+) for better coverage")
        print(f"   2. Use augmented sampling focused on this region")
        print(f"   CURRENT: Model may work but with higher uncertainty")
    elif len(indices) < 1000:
        print(f"\n✅ MODERATE COVERAGE: {len(indices)} samples within ±10%.")
        print(f"   STATUS: Should be sufficient for reasonable predictions")
        print(f"   OPTIONAL: Generate 200k+ dataset for even better coverage")
    else:
        print(f"\n✅ EXCELLENT COVERAGE: {len(indices)} samples within ±10%!")
        print(f"   STATUS: Dataset has good coverage for these parameters")
        print(f"   Model should predict reliably")

    print(f"\n{'='*80}\n")

    return indices, distances


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze dataset coverage for experimental parameters")
    parser.add_argument("--dataset", default="datasets/dataset_100000_dl100_7d.pkl",
                        help="Path to dataset file")
    parser.add_argument("--params", nargs=7, type=float, default=None,
                        help="Override experimental parameters (7 values)")

    args = parser.parse_args()

    # Use provided params or default experimental params
    experiment_params = args.params if args.params else EXPERIMENT_PARAMS

    # Analyze coverage
    indices, distances = analyze_coverage(args.dataset, experiment_params)
