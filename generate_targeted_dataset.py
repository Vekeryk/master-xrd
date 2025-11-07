#!/usr/bin/env python3
"""
Generate Targeted Dataset Around Experimental Parameters
=========================================================

Creates a dataset with Gaussian sampling centered on experimental parameters.
This improves model performance for specific experimental regions.

Usage:
    python generate_targeted_dataset.py --n-samples 10000 --std-pct 15
"""

import numpy as np
import torch
import xrd
from helpers import get_device
from importlib import reload
import pickle
import os
from datetime import datetime
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from model_common import RANGES, PARAM_NAMES

reload(xrd)


# =============================================================================
# EXPERIMENTAL PARAMETERS (CENTER OF DISTRIBUTION)
# =============================================================================

EXPERIMENT_PARAMS = {
    'Dmax1': 0.008094,
    'D01': 0.000943,
    'L1': 5200e-8,      # 5200 Å in cm
    'Rp1': 3500e-8,     # 3500 Å in cm
    'D02': 0.00255,
    'L2': 3000e-8,      # 3000 Å in cm
    'Rp2': -50e-8,      # -50 Å in cm
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _generate_single_sample(args):
    """Generate a single XRD curve sample."""
    _Dmax1, _D01, _L1, _Rp1, _D02, _L2, _Rp2, dl = args

    params_obj = xrd.DeformationProfile(
        Dmax1=_Dmax1,
        D01=_D01,
        L1=_L1,
        Rp1=_Rp1,
        D02=_D02,
        L2=_L2,
        Rp2=_Rp2,
        Dmin=0.0001,
        dl=dl
    )

    curve, profile = xrd.compute_curve_and_profile(params_obj=params_obj)

    return ([_Dmax1, _D01, _L1, _Rp1, _D02, _L2, _Rp2], curve.Y_R_vseZ)


def sample_gaussian_with_constraints(center, param_ranges, std_pct=15, n_samples=10000, max_attempts=1000000):
    """
    Sample parameters from Gaussian distribution around center with physical constraints.

    Args:
        center: dict with center parameters
        param_ranges: dict with parameter ranges (min, max)
        std_pct: Standard deviation as percentage of parameter range
        n_samples: Number of samples to generate
        max_attempts: Maximum sampling attempts before giving up

    Returns:
        samples: array [n_samples, 7] with valid parameter combinations
    """
    print(f"\n{'='*80}")
    print(f"GAUSSIAN SAMPLING AROUND EXPERIMENTAL PARAMETERS")
    print(f"{'='*80}")

    # Prepare center and std for each parameter
    center_arr = np.array([center[p] for p in PARAM_NAMES])

    # Calculate std based on percentage of range
    std_arr = np.zeros(7)
    for i, param in enumerate(PARAM_NAMES):
        param_range = param_ranges[param]
        range_width = param_range[1] - param_range[0]
        std_arr[i] = (std_pct / 100.0) * range_width

    print(f"\n🎯 Center parameters:")
    for i, param in enumerate(PARAM_NAMES):
        print(
            f"   {param:8s}: {center_arr[i]:12.6e} ± {std_arr[i]:12.6e} ({std_pct}% of range)")

    print(f"\n🔒 Physical constraints:")
    print(f"   1. D01 <= Dmax1")
    print(f"   2. D01 + D02 <= 0.03")
    print(f"   3. Rp1 <= L1")
    print(f"   4. L2 <= L1")

    # Sample with rejection sampling
    samples = []
    attempts = 0
    rejected_range = 0
    rejected_constraint = 0

    pbar = tqdm(total=n_samples, desc="Sampling with constraints")

    while len(samples) < n_samples and attempts < max_attempts:
        # Generate candidate sample
        candidate = np.random.normal(center_arr, std_arr)
        attempts += 1

        # Check if within parameter ranges
        in_range = True
        for i, param in enumerate(PARAM_NAMES):
            param_range = param_ranges[param]
            if not (param_range[0] <= candidate[i] <= param_range[1]):
                in_range = False
                rejected_range += 1
                break

        if not in_range:
            continue

        # Check physical constraints
        Dmax1, D01, L1, Rp1, D02, L2, Rp2 = candidate

        if D01 > Dmax1:  # Constraint 1
            rejected_constraint += 1
            continue
        if D01 + D02 > 0.03:  # Constraint 2
            rejected_constraint += 1
            continue
        if Rp1 > L1:  # Constraint 3
            rejected_constraint += 1
            continue
        if L2 > L1:  # Constraint 4
            rejected_constraint += 1
            continue

        # Valid sample!
        samples.append(candidate)
        pbar.update(1)

    pbar.close()

    if len(samples) < n_samples:
        print(
            f"\n⚠️  WARNING: Could only generate {len(samples)} valid samples out of {n_samples} requested")
        print(f"   Attempts: {attempts:,}")
        print(f"   Rejected (out of range): {rejected_range:,}")
        print(f"   Rejected (constraints): {rejected_constraint:,}")
        print(f"   Acceptance rate: {100*len(samples)/attempts:.2f}%")
        print(f"\n   Consider:")
        print(f"   1. Reducing std_pct (currently {std_pct}%)")
        print(f"   2. Accepting fewer samples")
        print(f"   3. Relaxing constraints (if physically valid)")
    else:
        print(f"\n✅ Generated {len(samples):,} valid samples")
        print(f"   Attempts: {attempts:,}")
        print(
            f"   Rejected (out of range): {rejected_range:,} ({100*rejected_range/attempts:.1f}%)")
        print(
            f"   Rejected (constraints): {rejected_constraint:,} ({100*rejected_constraint/attempts:.1f}%)")
        print(f"   Acceptance rate: {100*len(samples)/attempts:.2f}%")

    samples_arr = np.array(samples)

    # Show distribution statistics
    print(f"\n📊 Generated distribution statistics:")
    print(f"{'Parameter':<10} {'Center':>12} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
    print(f"{'-'*80}")

    for i, param in enumerate(PARAM_NAMES):
        print(f"{param:<10} {center_arr[i]:>12.6e} {samples_arr[:, i].mean():>12.6e} "
              f"{samples_arr[:, i].std():>12.6e} {samples_arr[:, i].min():>12.6e} "
              f"{samples_arr[:, i].max():>12.6e}")

    return samples_arr


def generate_targeted_dataset(experiment_params, n_samples=10000, dl=100e-8,
                              std_pct=15, output_dir="datasets"):
    """
    Generate targeted dataset around experimental parameters.

    Args:
        experiment_params: dict with experimental parameters
        n_samples: Number of samples to generate
        dl: Layer thickness in cm
        std_pct: Standard deviation as % of parameter range
        output_dir: Output directory

    Returns:
        data: dict with X, Y, and metadata
    """
    print(f"\n{'='*80}")
    print(f"TARGETED DATASET GENERATION")
    print(f"{'='*80}")
    print(f"Target samples: {n_samples:,}")
    print(f"dl: {dl*1e8:.0f} Å")
    print(f"Std deviation: {std_pct}% of parameter range")

    # Sample parameters around experimental values
    param_samples = sample_gaussian_with_constraints(
        experiment_params, RANGES, std_pct=std_pct, n_samples=n_samples
    )

    print(f"\n{'='*80}")
    print(f"GENERATING XRD CURVES")
    print(f"{'='*80}")

    # Prepare arguments for parallel processing
    args_list = [(*params, dl) for params in param_samples]

    # Generate curves in parallel
    n_cores = cpu_count()
    print(f"Using {n_cores} CPU cores...")

    with Pool(n_cores) as pool:
        results = list(tqdm(
            pool.imap(_generate_single_sample, args_list),
            total=len(args_list),
            desc="Generating curves"
        ))

    # Collect results
    X_list = []
    Y_list = []

    for params, curve_y in results:
        X_list.append(params)
        Y_list.append(curve_y)

    X = np.array(X_list, dtype=np.float32)
    Y = np.array(Y_list, dtype=np.float32)

    print(f"\n✅ Generated dataset:")
    print(f"   X shape: {X.shape}")
    print(f"   Y shape: {Y.shape}")

    # Create crop parameters
    crop_params = {
        'm1': Y.shape[1],
        'start_ML': 50,
        'cropped_length': Y.shape[1] - 50,
        'note': f'Y contains FULL curves ({Y.shape[1]} points). Crop to [start_ML:m1] during loading.'
    }

    # Save dataset
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"dataset_{len(X)}_dl{int(dl*1e8)}_targeted.pkl"
    filepath = os.path.join(output_dir, filename)

    data = {
        'X': X,
        'Y': Y,
        'crop_params': crop_params,
        'generation_params': {
            'n_samples': n_samples,
            'dl': dl,
            'std_pct': std_pct,
            'method': 'Gaussian sampling around experimental parameters',
            'experiment_params': experiment_params,
            'timestamp': timestamp
        }
    }

    with open(filepath, 'wb') as f:
        pickle.dump(data, f, protocol=4)

    print(f"\n✅ Saved dataset to: {filepath}")
    print(f"   File size: {os.path.getsize(filepath) / (1024**2):.1f} MB")

    # Verify coverage of experimental parameters
    print(f"\n{'='*80}")
    print(f"COVERAGE VERIFICATION")
    print(f"{'='*80}")

    center_arr = np.array([experiment_params[p] for p in PARAM_NAMES])

    # Calculate distances to center (in normalized space)
    X_norm = np.zeros_like(X)
    center_norm = np.zeros_like(center_arr)

    for i, param in enumerate(PARAM_NAMES):
        param_range = RANGES[param]
        X_norm[:, i] = (X[:, i] - param_range[0]) / \
            (param_range[1] - param_range[0])
        center_norm[i] = (center_arr[i] - param_range[0]) / \
            (param_range[1] - param_range[0])

    distances = np.linalg.norm(X_norm - center_norm, axis=1)

    print(f"\nDistance to experimental parameters (normalized space):")
    print(f"   Min distance:    {distances.min():.6f}")
    print(f"   Max distance:    {distances.max():.6f}")
    print(f"   Mean distance:   {distances.mean():.6f}")
    print(f"   Median distance: {np.median(distances):.6f}")

    # Count samples within different thresholds
    for threshold in [0.05, 0.10, 0.15, 0.20]:
        count = np.sum(distances <= threshold)
        pct = 100 * count / len(X)
        print(f"   Within {threshold:.2f}: {count:>6,} ({pct:>5.1f}%)")

    return data


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate targeted dataset around experimental parameters")
    parser.add_argument("--n-samples", type=int, default=10000,
                        help="Number of samples to generate (default: 10000)")
    parser.add_argument("--dl", type=float, default=100e-8,
                        help="Layer thickness in cm (default: 100e-8)")
    parser.add_argument("--std-pct", type=float, default=15,
                        help="Standard deviation as %% of parameter range (default: 15)")
    parser.add_argument("--output-dir", type=str, default="datasets",
                        help="Output directory (default: datasets)")

    args = parser.parse_args()

    # Generate dataset
    dataset = generate_targeted_dataset(
        experiment_params=EXPERIMENT_PARAMS,
        n_samples=args.n_samples,
        dl=args.dl,
        std_pct=args.std_pct,
        output_dir=args.output_dir
    )

    print(f"\n{'='*80}")
    print(f"✅ TARGETED DATASET GENERATION COMPLETE")
    print(f"{'='*80}")
    print(f"\nNext steps:")
    print(f"1. Fine-tune existing model on this dataset:")
    print(f"   python model_train.py --dataset datasets/dataset_*_targeted_*.pkl \\")
    print(f"                         --load-model models/dataset_100000_dl100_7d_v3.pt \\")
    print(f"                         --epochs 50 --lr 1e-4")
    print(f"\n2. Or train from scratch for this specific region:")
    print(f"   python model_train.py --dataset datasets/dataset_*_targeted_*.pkl \\")
    print(f"                         --epochs 200")
    print(f"\n3. Or combine with general dataset (recommended):")
    print(f"   # Merge 90k general + 10k targeted in j_combine_datasets.ipynb")
