"""
7D Grid-Based Stratified Dataset Generation for XRD Rocking Curves
===================================================================

CORRECT APPROACH:
1. Generate GRID using arange_inclusive (discrete values from GRID_STEPS)
2. Apply physical constraints to filter valid combinations
3. Group valid combinations by 7D bins
4. Sample uniformly from each bin

This maintains:
- Discrete grid values (from GRID_STEPS)
- Physical constraints (D01 <= Dmax1, etc.)
- Uniform distribution across all 7 parameters

Usage:
    python dataset_stratified_7d.py
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
from collections import defaultdict

# Import from model_common.py - single source of truth!
from model_common import RANGES as MODEL_RANGES, GRID_STEPS, PARAM_NAMES

reload(xrd)


def arange_inclusive(start, stop, step):
    """Helper function to create inclusive ranges."""
    n_steps = round((stop - start) / step)
    return np.array([start + i * step for i in range(n_steps + 1)], dtype=float)


# =============================================================================
# HELPER FUNCTION FOR MULTIPROCESSING
# =============================================================================

def _generate_single_sample(args):
    """Generate a single XRD curve sample."""
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

    curve, profile = xrd.compute_curve_and_profile(
        params_obj=params_obj)

    return ([_Dmax1, _D01, _L1_cm, _Rp1_cm, _D02, _L2_cm, _Rp2_cm], curve.Y_R_vseZ)


# =============================================================================
# BINNING
# =============================================================================

def create_parameter_bins(n_bins_per_param=5):
    """Create bin edges for each parameter."""
    bins = {}

    for param in PARAM_NAMES:
        min_val, max_val = MODEL_RANGES[param]

        if param in ['L1', 'Rp1', 'L2', 'Rp2']:
            min_val_A = min_val * 1e8
            max_val_A = max_val * 1e8
            bins[param] = np.linspace(
                min_val_A, max_val_A, n_bins_per_param + 1)
        else:
            bins[param] = np.linspace(min_val, max_val, n_bins_per_param + 1)

    return bins


def get_bin_index(value, bin_edges):
    """Get bin index for a value given bin edges."""
    idx = np.digitize(value, bin_edges) - 1
    idx = max(0, min(len(bin_edges) - 2, idx))
    return idx


def get_multidim_bin_key(combo, param_bins):
    """
    Get 7D bin key for a parameter combination.

    Args:
        combo: tuple (Dmax1, D01, L1, Rp1, D02, L2, Rp2) in Angstroms for L/Rp

    Returns:
        tuple: (bin_idx_Dmax1, bin_idx_D01, ..., bin_idx_Rp2)
    """
    Dmax1, D01, L1, Rp1, D02, L2, Rp2 = combo

    bin_key = (
        get_bin_index(Dmax1, param_bins['Dmax1']),
        get_bin_index(D01, param_bins['D01']),
        get_bin_index(L1, param_bins['L1']),
        get_bin_index(Rp1, param_bins['Rp1']),
        get_bin_index(D02, param_bins['D02']),
        get_bin_index(L2, param_bins['L2']),
        get_bin_index(Rp2, param_bins['Rp2']),
    )

    return bin_key


# =============================================================================
# BUILD VALID GRID COMBINATIONS WITH MULTI-DIMENSIONAL GROUPING
# =============================================================================

def build_valid_combinations_7d(n_bins_per_param=5):
    """
    Build all valid GRID combinations and group by 7D bins.

    Returns:
        tuple: (grouped_combinations dict, param_bins dict)
    """
    print(f"\n{'='*80}")
    print(f"7D GRID-BASED STRATIFIED SAMPLING")
    print(f"{'='*80}")

    # Create bins for grouping
    param_bins = create_parameter_bins(n_bins_per_param)

    print(f"\n📊 Bin configuration ({n_bins_per_param} bins per parameter):")
    for param in PARAM_NAMES:
        print(f"  {param:8s}: {len(param_bins[param])-1} bins")

    # Generate DISCRETE grids from MODEL_RANGES + GRID_STEPS
    print(f"\n📐 Generating discrete grids from GRID_STEPS:")
    Dmax1_grid = arange_inclusive(
        MODEL_RANGES['Dmax1'][0], MODEL_RANGES['Dmax1'][1], GRID_STEPS['Dmax1'])
    D01_grid = arange_inclusive(
        MODEL_RANGES['D01'][0], MODEL_RANGES['D01'][1], GRID_STEPS['D01'])

    # L1, Rp1, L2, Rp2 in model_common.py in cm → convert to Å for grid
    L1_grid = arange_inclusive(
        MODEL_RANGES['L1'][0] * 1e8, MODEL_RANGES['L1'][1] * 1e8, GRID_STEPS['L1'])
    Rp1_grid = arange_inclusive(
        MODEL_RANGES['Rp1'][0] * 1e8, MODEL_RANGES['Rp1'][1] * 1e8, GRID_STEPS['Rp1'])

    D02_grid = arange_inclusive(
        MODEL_RANGES['D02'][0], MODEL_RANGES['D02'][1], GRID_STEPS['D02'])
    L2_grid = arange_inclusive(
        MODEL_RANGES['L2'][0] * 1e8, MODEL_RANGES['L2'][1] * 1e8, GRID_STEPS['L2'])
    Rp2_grid = arange_inclusive(
        MODEL_RANGES['Rp2'][0] * 1e8, MODEL_RANGES['Rp2'][1] * 1e8, GRID_STEPS['Rp2'])

    print(
        f"  Dmax1: {len(Dmax1_grid)} grid points (step {GRID_STEPS['Dmax1']})")
    print(f"  D01:   {len(D01_grid)} grid points (step {GRID_STEPS['D01']})")
    print(f"  L1:    {len(L1_grid)} grid points (step {GRID_STEPS['L1']} Å)")
    print(f"  Rp1:   {len(Rp1_grid)} grid points (step {GRID_STEPS['Rp1']} Å)")
    print(f"  D02:   {len(D02_grid)} grid points (step {GRID_STEPS['D02']})")
    print(f"  L2:    {len(L2_grid)} grid points (step {GRID_STEPS['L2']} Å)")
    print(f"  Rp2:   {len(Rp2_grid)} grid points (step {GRID_STEPS['Rp2']} Å)")

    limit = 0.03  # constraint for D01 + D02

    print(f"\n🔒 Physical constraints:")
    print(f"  1. D01 <= Dmax1")
    print(f"  2. D01 + D02 <= {limit}")
    print(f"  3. Rp1 <= L1")
    print(f"  4. L2 <= L1")

    # Group combinations by 7D bin key
    grouped_by_7d_bin = defaultdict(list)

    print(f"\n⏳ Building valid grid combinations with constraints...")

    total_combos = 0
    # Iterate through all grid combinations respecting constraints
    for d1 in tqdm(Dmax1_grid, desc="Processing Dmax1"):
        for d01 in D01_grid:
            if d01 > d1:  # Constraint 1
                break
            for d02 in D02_grid:
                if d01 + d02 > limit:  # Constraint 2
                    break
                for l1 in L1_grid:
                    for r1 in Rp1_grid:
                        if r1 > l1:  # Constraint 3
                            break
                        for l2 in L2_grid:
                            if l2 > l1:  # Constraint 4
                                break
                            for r2 in Rp2_grid:
                                combo = (d1, d01, l1, r1, d02, l2, r2)

                                # Get 7D bin key
                                bin_key = get_multidim_bin_key(
                                    combo, param_bins)

                                # Add to group
                                grouped_by_7d_bin[bin_key].append(combo)
                                total_combos += 1

    print(f"\n✅ Generated {total_combos:,} valid grid combinations")
    print(f"✅ Distributed across {len(grouped_by_7d_bin):,} non-empty 7D bins")
    print(
        f"   Average: {total_combos / len(grouped_by_7d_bin):.1f} combinations per bin")

    # Show bin statistics
    bin_sizes = [len(combos) for combos in grouped_by_7d_bin.values()]
    print(f"\n📊 Bin size distribution:")
    print(f"   Min:    {min(bin_sizes):,} combinations")
    print(f"   Max:    {max(bin_sizes):,} combinations")
    print(f"   Mean:   {np.mean(bin_sizes):.1f} combinations")
    print(f"   Median: {np.median(bin_sizes):.0f} combinations")
    print(f"   Std:    {np.std(bin_sizes):.1f} combinations")

    return grouped_by_7d_bin, param_bins


# =============================================================================
# 7D STRATIFIED SAMPLING
# =============================================================================

def stratified_sample_7d(grouped_combinations, n_samples):
    """
    Perform 7D stratified sampling from GRID combinations.

    Args:
        grouped_combinations: dict {7D_bin_key: [grid_combinations]}
        n_samples: Total number of samples desired

    Returns:
        list of selected grid combinations
    """
    n_bins = len(grouped_combinations)

    print(f"\n{'='*80}")
    print(f"7D STRATIFIED SAMPLING FROM GRID")
    print(f"{'='*80}")
    print(f"Non-empty bins: {n_bins:,}")
    print(f"Target samples: {n_samples:,}")

    # Strategy: Equal number from each bin
    samples_per_bin = n_samples // n_bins
    remainder = n_samples % n_bins

    print(f"Samples per bin: {samples_per_bin}")
    if remainder > 0:
        print(f"Extra samples for first {remainder} bins")

    selected = []
    bins_with_insufficient_samples = 0

    for i, (bin_key, group) in enumerate(sorted(grouped_combinations.items())):
        # Add 1 extra to first 'remainder' bins
        target = samples_per_bin + (1 if i < remainder else 0)

        if len(group) >= target:
            # Randomly sample from bin's GRID combinations
            indices = np.random.choice(len(group), size=target, replace=False)
            selected_combos = [group[idx] for idx in indices]
        else:
            # Use all if bin is smaller than target
            selected_combos = group
            bins_with_insufficient_samples += 1

        selected.extend(selected_combos)

    print(f"\n✅ Selected {len(selected):,} grid combinations")
    if bins_with_insufficient_samples > 0:
        print(
            f"⚠️  {bins_with_insufficient_samples} bins had fewer samples than target")
    else:
        print(f"✅ All bins met sampling target")

    return selected


# =============================================================================
# MAIN DATASET GENERATION
# =============================================================================

def generate_dataset_7d(n_samples=10000, dl=100e-8, n_bins_per_param=5, output_dir="datasets"):
    """Generate stratified dataset with 7D grid-based stratification."""
    print(f"\n{'='*80}")
    print(f"7D GRID-BASED STRATIFIED DATASET GENERATION")
    print(f"{'='*80}")
    print(f"Target samples: {n_samples:,}")
    print(f"dl: {dl*1e8:.0f} Å")
    print(f"Bins per parameter: {n_bins_per_param}")

    # Build valid grid combinations grouped by 7D bins
    grouped_combinations, param_bins = build_valid_combinations_7d(
        n_bins_per_param)

    # Perform stratified sampling
    selected_combinations = stratified_sample_7d(
        grouped_combinations, n_samples)

    print(f"\n{'='*80}")
    print(f"GENERATING XRD CURVES")
    print(f"{'='*80}")

    # Prepare arguments for parallel processing
    args_list = [(d1, d01, l1, r1, d02, l2, r2, dl)
                 for (d1, d01, l1, r1, d02, l2, r2) in selected_combinations]

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

    # Verify discrete values
    print(f"\n🔍 Verifying discrete grid values:")
    for i, param in enumerate(PARAM_NAMES):
        unique_count = len(np.unique(X[:, i]))
        print(f"   {param:8s}: {unique_count} unique values")

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
    filename = f"dataset_{len(X)}_dl{int(dl*1e8)}_7d.pkl"
    filepath = os.path.join(output_dir, filename)

    data = {
        'X': X,
        'Y': Y,
        'crop_params': crop_params,
        'generation_params': {
            'n_samples': n_samples,
            'dl': dl,
            'n_bins_per_param': n_bins_per_param,
            'method': '7D grid-based stratified sampling',
            'grid_steps': GRID_STEPS,
            'timestamp': timestamp
        }
    }

    with open(filepath, 'wb') as f:
        pickle.dump(data, f, protocol=4)

    print(f"\n✅ Saved dataset to: {filepath}")
    print(f"   File size: {os.path.getsize(filepath) / (1024**2):.1f} MB")

    return data


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Configuration
    N_SAMPLES = 200_000
    DL = 100e-8  # 100 Angstroms
    N_BINS_PER_PARAM = 3  # Start with 3 bins (3^7 = 2187 bins)

    # Generate dataset
    dataset = generate_dataset_7d(
        n_samples=N_SAMPLES,
        dl=DL,
        n_bins_per_param=N_BINS_PER_PARAM,
        output_dir="datasets"
    )

    print(f"\n{'='*80}")
    print(f"✅ DATASET GENERATION COMPLETE")
    print(f"{'='*80}")
    print(f"\nNext steps:")
    print(f"1. Run: jupyter notebook j_analyze_dataset.ipynb")
    print(f"2. Update DATASET_PATH to new file")
    print(f"3. Check uniformity: Chi-square p-value should be > 0.05")
    print(f"4. Verify unique values match grid sizes")
