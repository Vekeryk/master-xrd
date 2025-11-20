#!/usr/bin/env python3
"""
Generate 3D Stratified Dataset with Fixed Geometry Parameters
==============================================================

Creates a 3D stratified dataset varying only deformation parameters
(Dmax1, D01, D02) while keeping geometry parameters fixed (L1, Rp1, L2, Rp2).

Fixed parameters:
- L1 = 5400 Å
- Rp1 = 3450 Å
- L2 = 3000 Å
- Rp2 = -50 Å

Variable parameters:
- Dmax1: 0.00609 - 0.02126 (середина 0.01405)
- D01: 0.00074 - 0.00225 (середина 0.00149)
- D02: 0.00155 - 0.00857 (середина 0.00528)

Usage:
    python generate_fixed_geometry_dataset.py --n-samples 10000 --n-bins 5
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

reload(xrd)


# =============================================================================
# FIXED GEOMETRY PARAMETERS (Angstroms)
# =============================================================================

FIXED_PARAMS = {
    'L1': 5400.0,    # Å
    'Rp1': 3450.0,   # Å
    'L2': 3000.0,    # Å
    'Rp2': -50.0,    # Å
}

# =============================================================================
# VARIABLE DEFORMATION PARAMETER RANGES
# =============================================================================

VARIABLE_RANGES = {
    'Dmax1': (0.00609, 0.02126),  # середина 0.01405
    'D01': (0.00074, 0.00225),    # середина 0.00149
    'D02': (0.00155, 0.00857),    # середина 0.00528
}

VARIABLE_PARAM_NAMES = ['Dmax1', 'D01', 'D02']


# =============================================================================
# GRID STEPS FOR 3D STRATIFICATION
# =============================================================================

GRID_STEPS_3D = {
    'Dmax1': 0.0004,  # Very fine grid: ~38 points
    'D01': 0.00004,   # Ultra fine grid: ~38 points
    'D02': 0.0002,    # Very fine grid: ~36 points
    # Total: ~38 × 38 × 36 = ~52k theoretical combinations
    # After constraints: ~30-35k valid combinations
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def arange_inclusive(start, stop, step):
    """Helper function to create inclusive ranges."""
    n_steps = round((stop - start) / step)
    return np.array([start + i * step for i in range(n_steps + 1)], dtype=float)


def _generate_single_sample(args):
    """Generate a single XRD curve sample."""
    _Dmax1, _D01, _D02, dl = args

    # Use fixed geometry parameters
    _L1 = FIXED_PARAMS['L1'] * 1e-8    # Convert to cm
    _Rp1 = FIXED_PARAMS['Rp1'] * 1e-8  # Convert to cm
    _L2 = FIXED_PARAMS['L2'] * 1e-8    # Convert to cm
    _Rp2 = FIXED_PARAMS['Rp2'] * 1e-8  # Convert to cm

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

    # Return all 7 parameters for compatibility with model
    return ([_Dmax1, _D01, _L1, _Rp1, _D02, _L2, _Rp2], curve.Y_R_vseZ)


# =============================================================================
# 3D BINNING
# =============================================================================

def create_parameter_bins_3d(n_bins_per_param=5):
    """Create bin edges for 3 variable parameters."""
    bins = {}

    for param in VARIABLE_PARAM_NAMES:
        min_val, max_val = VARIABLE_RANGES[param]
        bins[param] = np.linspace(min_val, max_val, n_bins_per_param + 1)

    return bins


def get_bin_index(value, bin_edges):
    """Get bin index for a value given bin edges."""
    idx = np.digitize(value, bin_edges) - 1
    idx = max(0, min(len(bin_edges) - 2, idx))
    return idx


def get_3d_bin_key(combo, param_bins):
    """
    Get 3D bin key for a parameter combination.

    Args:
        combo: tuple (Dmax1, D01, D02)

    Returns:
        tuple: (bin_idx_Dmax1, bin_idx_D01, bin_idx_D02)
    """
    Dmax1, D01, D02 = combo

    bin_key = (
        get_bin_index(Dmax1, param_bins['Dmax1']),
        get_bin_index(D01, param_bins['D01']),
        get_bin_index(D02, param_bins['D02']),
    )

    return bin_key


# =============================================================================
# BUILD VALID 3D GRID COMBINATIONS
# =============================================================================

def build_valid_combinations_3d(n_bins_per_param=5):
    """
    Build all valid 3D grid combinations and group by bins.

    Returns:
        tuple: (grouped_combinations dict, param_bins dict)
    """
    print(f"\n{'='*80}")
    print(f"3D GRID-BASED STRATIFIED SAMPLING (FIXED GEOMETRY)")
    print(f"{'='*80}")

    # Show fixed parameters
    print(f"\n🔒 Fixed geometry parameters:")
    for param, value in FIXED_PARAMS.items():
        print(f"  {param:8s}: {value:>10.1f} Å")

    # Create bins for grouping
    param_bins = create_parameter_bins_3d(n_bins_per_param)

    print(f"\n📊 Bin configuration ({n_bins_per_param} bins per parameter):")
    for param in VARIABLE_PARAM_NAMES:
        print(f"  {param:8s}: {len(param_bins[param])-1} bins")

    # Generate DISCRETE grids for variable parameters
    print(f"\n📐 Generating discrete grids for variable parameters:")
    Dmax1_grid = arange_inclusive(
        VARIABLE_RANGES['Dmax1'][0],
        VARIABLE_RANGES['Dmax1'][1],
        GRID_STEPS_3D['Dmax1']
    )
    D01_grid = arange_inclusive(
        VARIABLE_RANGES['D01'][0],
        VARIABLE_RANGES['D01'][1],
        GRID_STEPS_3D['D01']
    )
    D02_grid = arange_inclusive(
        VARIABLE_RANGES['D02'][0],
        VARIABLE_RANGES['D02'][1],
        GRID_STEPS_3D['D02']
    )

    print(f"  Dmax1: {len(Dmax1_grid)} grid points (step {GRID_STEPS_3D['Dmax1']})")
    print(f"  D01:   {len(D01_grid)} grid points (step {GRID_STEPS_3D['D01']})")
    print(f"  D02:   {len(D02_grid)} grid points (step {GRID_STEPS_3D['D02']})")

    limit = 0.03  # constraint for D01 + D02

    print(f"\n🔒 Physical constraints:")
    print(f"  1. D01 <= Dmax1")
    print(f"  2. D01 + D02 <= {limit}")

    # Group combinations by 3D bin key
    grouped_by_3d_bin = defaultdict(list)

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

                combo = (d1, d01, d02)

                # Get 3D bin key
                bin_key = get_3d_bin_key(combo, param_bins)

                # Add to group
                grouped_by_3d_bin[bin_key].append(combo)
                total_combos += 1

    print(f"\n✅ Generated {total_combos:,} valid grid combinations")
    print(f"✅ Distributed across {len(grouped_by_3d_bin):,} non-empty 3D bins")
    print(f"   Average: {total_combos / len(grouped_by_3d_bin):.1f} combinations per bin")

    # Show bin statistics
    bin_sizes = [len(combos) for combos in grouped_by_3d_bin.values()]
    print(f"\n📊 Bin size distribution:")
    print(f"   Min:    {min(bin_sizes):,} combinations")
    print(f"   Max:    {max(bin_sizes):,} combinations")
    print(f"   Mean:   {np.mean(bin_sizes):.1f} combinations")
    print(f"   Median: {np.median(bin_sizes):.0f} combinations")
    print(f"   Std:    {np.std(bin_sizes):.1f} combinations")

    return grouped_by_3d_bin, param_bins


# =============================================================================
# 3D STRATIFIED SAMPLING
# =============================================================================

def stratified_sample_3d(grouped_combinations, n_samples):
    """
    Perform 3D stratified sampling from GRID combinations.

    Args:
        grouped_combinations: dict {3D_bin_key: [grid_combinations]}
        n_samples: Total number of samples desired

    Returns:
        list of selected grid combinations (Dmax1, D01, D02)
    """
    n_bins = len(grouped_combinations)

    print(f"\n{'='*80}")
    print(f"3D STRATIFIED SAMPLING FROM GRID")
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
        print(f"⚠️  {bins_with_insufficient_samples} bins had fewer samples than target")
    else:
        print(f"✅ All bins met sampling target")

    return selected


# =============================================================================
# MAIN DATASET GENERATION
# =============================================================================

def generate_dataset_3d_fixed_geometry(n_samples=10000, dl=100e-8, n_bins_per_param=5, output_dir="datasets"):
    """Generate 3D stratified dataset with fixed geometry parameters."""
    print(f"\n{'='*80}")
    print(f"3D STRATIFIED DATASET GENERATION (FIXED GEOMETRY)")
    print(f"{'='*80}")
    print(f"Target samples: {n_samples:,}")
    print(f"dl: {dl*1e8:.0f} Å")
    print(f"Bins per parameter: {n_bins_per_param}")

    # Build valid grid combinations grouped by 3D bins
    grouped_combinations, param_bins = build_valid_combinations_3d(n_bins_per_param)

    # Perform stratified sampling
    selected_combinations = stratified_sample_3d(grouped_combinations, n_samples)

    print(f"\n{'='*80}")
    print(f"GENERATING XRD CURVES")
    print(f"{'='*80}")

    # Prepare arguments for parallel processing
    args_list = [(d1, d01, d02, dl) for (d1, d01, d02) in selected_combinations]

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

    # Verify discrete values for variable parameters
    print(f"\n🔍 Verifying discrete grid values (variable parameters):")
    param_names_all = ['Dmax1', 'D01', 'L1', 'Rp1', 'D02', 'L2', 'Rp2']
    for i, param in enumerate(param_names_all):
        unique_count = len(np.unique(X[:, i]))
        if param in VARIABLE_PARAM_NAMES:
            print(f"   {param:8s}: {unique_count} unique values (VARIABLE)")
        else:
            print(f"   {param:8s}: {unique_count} unique values (FIXED)")

    # Show range statistics for variable parameters
    print(f"\n📊 Variable parameter statistics:")
    print(f"{'Parameter':<10} {'Min':>12} {'Max':>12} {'Mean':>12} {'Median':>12}")
    print(f"{'-'*60}")
    for i, param in enumerate(['Dmax1', 'D01', 'D02']):
        idx = param_names_all.index(param)
        print(f"{param:<10} {X[:, idx].min():>12.6f} {X[:, idx].max():>12.6f} "
              f"{X[:, idx].mean():>12.6f} {np.median(X[:, idx]):>12.6f}")

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
    filename = f"dataset_{len(X)}_dl{int(dl*1e8)}_3d_fixed_geom.pkl"
    filepath = os.path.join(output_dir, filename)

    data = {
        'X': X,
        'Y': Y,
        'crop_params': crop_params,
        'generation_params': {
            'n_samples': n_samples,
            'dl': dl,
            'n_bins_per_param': n_bins_per_param,
            'method': '3D grid-based stratified sampling with fixed geometry',
            'fixed_params': FIXED_PARAMS,
            'variable_ranges': VARIABLE_RANGES,
            'grid_steps': GRID_STEPS_3D,
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
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate 3D stratified dataset with fixed geometry parameters")
    parser.add_argument("--n-samples", type=int, default=10000,
                        help="Number of samples to generate (default: 10000)")
    parser.add_argument("--dl", type=float, default=100e-8,
                        help="Layer thickness in cm (default: 100e-8)")
    parser.add_argument("--n-bins", type=int, default=5,
                        help="Number of bins per parameter (default: 5, total bins = 5^3 = 125)")
    parser.add_argument("--output-dir", type=str, default="datasets",
                        help="Output directory (default: datasets)")

    args = parser.parse_args()

    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate dataset
    dataset = generate_dataset_3d_fixed_geometry(
        n_samples=args.n_samples,
        dl=args.dl,
        n_bins_per_param=args.n_bins,
        output_dir=args.output_dir
    )

    print(f"\n{'='*80}")
    print(f"✅ 3D DATASET GENERATION COMPLETE")
    print(f"{'='*80}")
    print(f"\nDataset characteristics:")
    print(f"  - 3D stratified sampling (Dmax1, D01, D02)")
    print(f"  - Fixed geometry: L1={FIXED_PARAMS['L1']:.0f}Å, Rp1={FIXED_PARAMS['Rp1']:.0f}Å, "
          f"L2={FIXED_PARAMS['L2']:.0f}Å, Rp2={FIXED_PARAMS['Rp2']:.0f}Å")
    print(f"  - {len(dataset['X']):,} samples")
    print(f"  - {args.n_bins**3} total bins ({args.n_bins} per parameter)")
    print(f"\nNext steps:")
    print(f"1. Train model on this 3D dataset:")
    dataset_filename = f'dataset_{len(dataset["X"])}_dl{int(args.dl*1e8)}_3d_fixed_geom.pkl'
    dataset_path = os.path.join(args.output_dir, dataset_filename)
    print(f"   python model_train.py --dataset {dataset_path}")
    print(f"\n2. Analyze dataset distribution:")
    print(f"   jupyter notebook j_analyze_dataset.ipynb")
