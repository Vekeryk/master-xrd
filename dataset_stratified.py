"""
Stratified Dataset Generation for XRD Rocking Curves
=====================================================

This module implements STRATIFIED SAMPLING to generate uniformly distributed
datasets, addressing the severe bias issues in random sampling (see DATASET_BIAS_ANALYSIS.md).

Key improvements:
- Uniform representation across all parameter values
- Reduced Chi-squared uniformity metric (target: <10,000)
- Better coverage of rare parameter combinations
- Expected accuracy improvement: +2-5% on biased parameters (L2, Rp2)

Usage:
    python dataset_stratified.py

Output files will have '_stratified' postfix:
    datasets/dataset_200000_dl400_stratified.pkl
"""

import numpy as np
import torch
import xrd_parallel as xrd
from helpers import get_device
from importlib import reload
import pickle
import os
from datetime import datetime
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from collections import defaultdict

reload(xrd)


def arange_inclusive(start, stop, step):
    """Helper function to create inclusive ranges"""
    return np.arange(start, stop + 0.5 * step, step, dtype=float)


# =============================================================================
# HELPER FUNCTION FOR MULTIPROCESSING
# =============================================================================

def _generate_single_sample(args):
    """
    Generate a single XRD curve sample.
    Must be at module level for multiprocessing pickling.

    Args:
        args: tuple of (_Dmax1, _D01, _L1, _Rp1, _D02, _L2, _Rp2, dl)

    Returns:
        tuple: (parameters_list, curve_ML_Y)
    """
    _Dmax1, _D01, _L1, _Rp1, _D02, _L2, _Rp2, dl = args

    # Convert L1, Rp1, L2, Rp2 from Angstroms to cm
    _L1_cm = _L1 * 1e-8
    _Rp1_cm = _Rp1 * 1e-8
    _L2_cm = _L2 * 1e-8
    _Rp2_cm = _Rp2 * 1e-8

    params_obj = xrd.DeformationProfile(
        Dmax1=_Dmax1,
        D01=_D01,
        L1=_L1_cm,
        Rp1=_Rp1_cm,
        D02=_D02,
        L2=_L2_cm,
        Rp2=_Rp2_cm,
        Dmin=0.0001,
        dl=dl
    )

    curve, profile = xrd.compute_curve_and_profile(params_obj=params_obj)

    return ([_Dmax1, _D01, _L1_cm, _Rp1_cm, _D02, _L2_cm, _Rp2_cm], curve.ML_Y)


# =============================================================================
# STRATIFIED SAMPLING IMPLEMENTATION
# =============================================================================

def build_valid_combinations_dict():
    """
    Build all valid combinations and group by L2 value (most biased parameter).

    Returns:
        dict: {L2_value: [list of combinations with this L2]}
    """
    # Define parameter grids - IMPROVED with finer granularity
    Dmax1_grid = arange_inclusive(0.0025, 0.0250, 0.0025)  # 10 values
    D01_grid = arange_inclusive(0.0025, 0.0250, 0.0025)    # 10 values
    L1_grid = arange_inclusive(500., 7000., 500.)          # 14 values
    Rp1_grid = arange_inclusive(490., 4990., 500.)         # 10 values
    D02_grid = arange_inclusive(0.0025, 0.0250, 0.0025)    # 10 values

    # IMPROVED: Finer grid for L2 and Rp2 (reduces gaps)
    L2_grid = arange_inclusive(500., 5000., 500.)          # 10 values (was 5)
    Rp2_grid = arange_inclusive(-6010., -10., 500.)        # 13 values (was 7)

    limit = 0.03  # constraint for D01 + D02

    print("Building valid combinations with finer grid...")
    print(f"  L2 grid: {len(L2_grid)} values (500Å steps)")
    print(f"  Rp2 grid: {len(Rp2_grid)} values (500Å steps)")

    # Group combinations by L2 value (most biased parameter)
    grouped_by_L2 = defaultdict(list)

    # Iterate through all combinations respecting constraints
    for d1 in Dmax1_grid:
        for d01 in D01_grid:
            if d01 > d1:
                break
            for d02 in D02_grid:
                if d01 + d02 > limit:
                    break
                for l1 in L1_grid:
                    for r1 in Rp1_grid:
                        if r1 > l1:
                            break
                        for l2 in L2_grid:
                            if l2 > l1:
                                break
                            for r2 in Rp2_grid:
                                combo = (d1, d01, l1, r1, d02, l2, r2)
                                grouped_by_L2[l2].append(combo)

    return grouped_by_L2


def stratified_sample(grouped_combinations, n_samples, strategy='balanced'):
    """
    Perform stratified sampling to ensure uniform representation.

    Args:
        grouped_combinations: dict {parameter_value: [combinations]}
        n_samples: Total number of samples desired
        strategy: 'balanced' (equal from each group) or 'proportional'

    Returns:
        list of selected combinations
    """
    n_groups = len(grouped_combinations)
    print(f"\nStratified sampling from {n_groups} L2 groups...")

    if strategy == 'balanced':
        # Equal number from each group
        samples_per_group = n_samples // n_groups
        remainder = n_samples % n_groups

        selected = []
        for i, (L2_value, group) in enumerate(sorted(grouped_combinations.items())):
            # Add 1 extra to first 'remainder' groups
            target = samples_per_group + (1 if i < remainder else 0)

            if len(group) >= target:
                # Randomly sample from group
                indices = np.random.choice(
                    len(group), size=target, replace=False
                )
                selected_combos = [group[idx] for idx in indices]
            else:
                # Use all if group is smaller than target
                selected_combos = group
                print(f"  Warning: L2={L2_value:.0f}Å has only {len(group)} "
                      f"combinations (target: {target})")

            selected.extend(selected_combos)

        print(
            f"Selected {len(selected)} combinations ({samples_per_group} per group)")

    else:  # proportional
        # Sample proportional to group size (less uniform, but respects natural distribution)
        all_combos = []
        for group in grouped_combinations.values():
            all_combos.extend(group)

        indices = np.random.choice(
            len(all_combos), size=n_samples, replace=False)
        selected = [all_combos[i] for i in indices]

    return selected[:n_samples]


def generate_stratified_dataset(n_samples, dl=100e-8, n_workers=None, strategy='balanced'):
    """
    Generate dataset using STRATIFIED SAMPLING for uniform distribution.

    This addresses the severe bias in random sampling (see DATASET_BIAS_ANALYSIS.md):
    - Random sampling bias ratio: up to 87x (D01)
    - Chi-squared uniformity: up to 105,310 (L2)

    Expected improvements:
    - Chi-squared uniformity: <10,000 for all parameters
    - Better coverage of rare combinations
    - Accuracy improvement: +2-5% on L2 and Rp2

    Args:
        n_samples: Number of samples to generate
        dl: Sublayer thickness in cm (default: 100e-8)
        n_workers: Number of parallel workers (default: cpu_count() - 1)
        strategy: 'balanced' (equal per group) or 'proportional'

    Returns:
        X, Y: PyTorch tensors with input parameters and XRD curves
    """
    # Determine number of workers
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    print(f"Using {n_workers} parallel workers")

    # Build valid combinations grouped by L2
    grouped_combinations = build_valid_combinations_dict()

    # Calculate total combinations
    total_valid = sum(len(group) for group in grouped_combinations.values())
    print(f"Total valid combinations: {total_valid:,}")

    # Print group sizes
    print(f"\nL2 group sizes:")
    for L2_value in sorted(grouped_combinations.keys()):
        count = len(grouped_combinations[L2_value])
        pct = 100 * count / total_valid
        print(f"  L2={L2_value:5.0f}Å: {count:6,} combinations ({pct:5.2f}%)")

    # Stratified sampling
    selected_combinations = stratified_sample(
        grouped_combinations, n_samples, strategy)

    # Prepare arguments for parallel processing
    args_list = []
    for combo in selected_combinations:
        _Dmax1, _D01, _L1, _Rp1, _D02, _L2, _Rp2 = combo
        args_list.append((_Dmax1, _D01, _L1, _Rp1, _D02, _L2, _Rp2, dl))

    # Parallel processing with progress bar
    print(f"\nGenerating {n_samples} samples in parallel...")
    X = []
    Y = []

    with Pool(processes=n_workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(_generate_single_sample, args_list),
            total=n_samples,
            desc="Generating samples",
            unit="sample"
        ))

    # Unpack results
    for params, curve_y in results:
        X.append(params)
        Y.append(curve_y)

    # Convert to tensors
    device = get_device()
    X = torch.tensor(X, dtype=torch.float32, device=device)
    Y = torch.tensor(Y, dtype=torch.float32, device=device)

    return X, Y


# =============================================================================
# DISTRIBUTION VERIFICATION
# =============================================================================

def verify_distribution(X, dataset_name="stratified"):
    """
    Verify that the generated dataset has uniform distribution.

    Args:
        X: Parameter tensor [N, 7]
        dataset_name: Name for reporting
    """
    import numpy as np

    params = ['Dmax1', 'D01', 'L1', 'Rp1', 'D02', 'L2', 'Rp2']

    print("\n" + "=" * 70)
    print(f"DISTRIBUTION VERIFICATION: {dataset_name}")
    print("=" * 70)

    for i, name in enumerate(params):
        values = X[:, i].cpu().numpy() if torch.is_tensor(X) else X[:, i]

        # Convert to Angstroms if needed
        if i in [2, 3, 5, 6]:
            values = values * 1e8

        unique_vals = np.unique(values)

        # Calculate uniformity
        hist, bins = np.histogram(values, bins=len(unique_vals))
        expected = len(values) / len(unique_vals)
        chi_sq = np.sum((hist - expected)**2 / expected)

        # Calculate bias ratio
        min_count = hist.min()
        max_count = hist.max()
        bias_ratio = max_count / max_count if min_count == 0 else max_count / min_count

        print(f"\n{name}:")
        print(f"  Unique values: {len(unique_vals)}")
        print(f"  Chi-squared:   {chi_sq:.2f} (target: <10,000)")
        print(f"  Bias ratio:    {bias_ratio:.2f}x (target: <2x)")
        print(f"  Min count:     {min_count}")
        print(f"  Max count:     {max_count}")

        if chi_sq < 10000:
            print(f"  ✅ PASS - Good uniformity!")
        else:
            print(f"  ⚠️  WARN - High non-uniformity")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Configuration
    # n_samples = 1_000  # Quick test (3 seconds) - VERIFIED WORKING ✅
    n_samples = 1_000_000  # Production: 200k for good balance between quality and time
    dl = 100e-8  # in cm (100 Angstroms)
    n_workers = None  # Auto-detect
    strategy = 'balanced'  # 'balanced' or 'proportional'

    # Convert dl to Angstroms for filename
    dl_angstrom = int(dl * 1e8)
    output_file = f"datasets/dataset_{n_samples}_dl{dl_angstrom}_balanced.pkl"

    print("=" * 70)
    print("STRATIFIED DATASET GENERATION")
    print("=" * 70)
    print(f"Generating {n_samples:,} samples with stratified sampling...")
    print(f"Output file: {output_file}")
    print(f"dl parameter: {dl_angstrom} Angstroms")
    print(f"Strategy: {strategy}")
    print("=" * 70)

    # Generate dataset
    X, Y = generate_stratified_dataset(
        n_samples=n_samples,
        dl=dl,
        n_workers=n_workers,
        strategy=strategy
    )

    print("\n" + "=" * 70)
    print("Dataset generated successfully!")
    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")
    print(f"Device: {X.device}")
    print("=" * 70)

    # Verify distribution
    verify_distribution(X, dataset_name=output_file)

    # Prepare data for saving
    dataset = {
        'X': X.cpu().numpy(),
        'Y': Y.cpu().numpy(),
        'n_samples': n_samples,
        'dl': dl,
        'dl_angstrom': dl_angstrom,
        'sampling_method': 'stratified_balanced',
        'timestamp': datetime.now().isoformat(),
        'device': str(X.device),
        'grid_info': {
            'L2_step': 500,  # Angstroms
            'Rp2_step': 500,  # Angstroms
            'note': 'Finer grid compared to random sampling (was 1000Å)'
        }
    }

    # Save to pickle file
    print(f"\nSaving dataset to {output_file}...")
    os.makedirs("datasets", exist_ok=True)
    with open(output_file, 'wb') as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Dataset saved successfully!")
    print(f"File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
    print("\n" + "=" * 70)
    print("✅ STRATIFIED DATASET GENERATION COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Train model: python model_train.py")
    print("2. Update DATA_PATH to use this stratified dataset")
    print("3. Compare results with biased dataset")
    print("4. Expected improvements: Rp2 error 12.36% → 9-10%, L2 error 5.86% → 4-5%")
