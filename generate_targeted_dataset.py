#!/usr/bin/env python3
"""
Generate Dense Grid Dataset Around Experimental Parameters
===========================================================

Creates a DENSE GRID of samples around experimental parameters.
Uses small steps in a narrow range around experiment (ignores global RANGES).

Usage:
    python generate_targeted_dataset.py --n-samples 10000 --range-pct 10 --step-pct 1
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

from model_common import PARAM_NAMES

reload(xrd)


# =============================================================================
# EXPERIMENTAL PARAMETERS (CENTER OF GRID)
# =============================================================================

# EXPERIMENT_PARAMS = {
#     'Dmax1': 0.008094,
#     'D01': 0.000943,
#     'L1': 5200e-8,      # 5200 Å in cm
#     'Rp1': 3500e-8,     # 3500 Å in cm
#     'D02': 0.00255,
#     'L2': 3000e-8,      # 3000 Å in cm
#     'Rp2': -50e-8,      # -50 Å in cm
# }
# EXPERIMENT_PARAMS = {
#     'Dmax1': 0.01926,
#     'D01': 0.00205,
#     'L1': 5460e-8,      # 5460 Å in cm
#     'Rp1': 3463e-8,     # 3463 Å in cm
#     'D02': 0.00767,
#     'L2': 4126e-8,      # 4126 Å in cm
#     'Rp2': -20e-8,      # -20 Å in cm
# }

EXPERIMENT_PARAMS = {
    'Dmax1': 0.01405,
    'D01': 0.00149,
    'L1': 5300e-8,      # 5460 Å in cm
    'Rp1': 3670e-8,     # 3463 Å in cm
    'D02': 0.00428,
    'L2': 3626e-8,      # 4126 Å in cm
    'Rp2': -20e-8,      # -20 Å in cm
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


def create_local_ranges(center, range_pct=10, range_pct_dict=None):
    """
    Create local ranges around center (±range_pct% of center value).

    Args:
        center: dict with center parameters
        range_pct: default percentage for all parameters (e.g., 10 = ±10%)
        range_pct_dict: dict with custom percentages per parameter (overrides range_pct)
                       Example: {'Dmax1': 20, 'D01': 15, 'L1': 5}

    Returns:
        local_ranges: dict with (min, max) for each parameter
    """
    local_ranges = {}

    for param, value in center.items():
        # Get percentage for this parameter (custom or default)
        pct = range_pct_dict.get(
            param, range_pct) if range_pct_dict else range_pct

        # For positive params: ±pct%
        if value > 0:
            delta = abs(value * pct / 100.0)
            local_ranges[param] = (value - delta, value + delta)
        # For negative params (like Rp2): ±pct% of absolute value
        elif value < 0:
            delta = abs(value * pct / 100.0)
            local_ranges[param] = (value - delta, value + delta)
        # For zero: use small absolute range
        else:
            delta = 0.0001 * pct / 100.0
            local_ranges[param] = (-delta, delta)

    return local_ranges


def calculate_grid_steps(local_ranges, step_pct=1, step_pct_dict=None):
    """
    Calculate grid step sizes (step_pct% of local range width).

    Args:
        local_ranges: dict with (min, max) for each parameter
        step_pct: default percentage of range width to use as step
        step_pct_dict: dict with custom step percentages per parameter
                      Example: {'Dmax1': 0.5, 'D01': 2, 'L1': 1}

    Returns:
        steps: dict with step size for each parameter
    """
    steps = {}

    for param, (min_val, max_val) in local_ranges.items():
        # Get step percentage for this parameter (custom or default)
        pct = step_pct_dict.get(param, step_pct) if step_pct_dict else step_pct

        range_width = max_val - min_val
        steps[param] = range_width * pct / 100.0

    return steps


def generate_dense_grid(center, local_ranges, steps, n_samples_target=10000, max_samples=100000):
    """
    Generate dense grid of parameters around center.

    Strategy:
    1. Create 1D grids for each parameter
    2. Sample random combinations (not full meshgrid - too many points!)
    3. Check physical constraints

    Args:
        center: dict with center parameters
        local_ranges: dict with (min, max) for each parameter
        steps: dict with step size for each parameter
        n_samples_target: target number of samples
        max_samples: maximum samples before stopping

    Returns:
        samples: array [n_samples, 7] with valid parameter combinations
    """
    print(f"\n{'='*80}")
    print(f"DENSE GRID GENERATION AROUND EXPERIMENTAL PARAMETERS")
    print(f"{'='*80}")

    # Create 1D grids for each parameter
    grids = {}
    n_points_per_param = {}

    print(f"\n📊 Grid configuration:")
    print(f"{'Parameter':<10} {'Center':>12} {'Min':>12} {'Max':>12} {'Step':>12} {'N_points':>10}")
    print(f"{'-'*80}")

    for param in PARAM_NAMES:
        min_val, max_val = local_ranges[param]
        step = steps[param]
        center_val = center[param]

        # Create 1D grid
        grid = np.arange(min_val, max_val + step / 2, step)
        grids[param] = grid
        n_points_per_param[param] = len(grid)

        print(f"{param:<10} {center_val:>12.6e} {min_val:>12.6e} {max_val:>12.6e} "
              f"{step:>12.6e} {len(grid):>10}")

    # Calculate theoretical full meshgrid size
    full_meshgrid_size = np.prod(list(n_points_per_param.values()))
    print(
        f"\n⚠️  Full meshgrid would have {full_meshgrid_size:,} points (too many!)")
    print(f"   Sampling {n_samples_target:,} random combinations instead...")

    print(f"\n🔒 Physical constraints:")
    print(f"   1. D01 <= Dmax1")
    print(f"   2. D01 + D02 <= 0.03")
    print(f"   3. Rp1 <= L1")
    print(f"   4. L2 <= L1")

    # Sample random combinations from grids
    samples = []
    attempts = 0
    rejected_constraint = 0

    # FIRST SAMPLE IS ALWAYS THE EXPERIMENT ITSELF!
    center_arr = np.array([center[p] for p in PARAM_NAMES])
    samples.append(center_arr)
    print(f"\n✅ Sample 0: EXPERIMENTAL PARAMETERS (always first!)")

    pbar = tqdm(total=n_samples_target, desc="Generating grid samples")
    pbar.update(1)  # Already added experiment as first sample

    while len(samples) < n_samples_target and attempts < max_samples:
        # Sample one random value from each parameter's grid
        candidate = np.array([
            np.random.choice(grids[param])
            for param in PARAM_NAMES
        ])
        attempts += 1

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

    if len(samples) < n_samples_target:
        print(
            f"\n⚠️  WARNING: Could only generate {len(samples)} valid samples out of {n_samples_target} requested")
        print(f"   Attempts: {attempts:,}")
        print(f"   Rejected (constraints): {rejected_constraint:,}")
        print(f"   Acceptance rate: {100*len(samples)/attempts:.2f}%")
        print(f"\n   Consider:")
        print(f"   1. Increasing max_samples")
        print(f"   2. Reducing n_samples_target")
        print(f"   3. Relaxing constraints (if physically valid)")
    else:
        print(f"\n✅ Generated {len(samples):,} valid samples")
        print(f"   Attempts: {attempts:,}")
        print(
            f"   Rejected (constraints): {rejected_constraint:,} ({100*rejected_constraint/attempts:.1f}%)")
        print(f"   Acceptance rate: {100*len(samples)/attempts:.2f}%")

    samples_arr = np.array(samples)

    # Show distribution statistics
    print(f"\n📊 Generated distribution statistics (Sample 0 = experiment):")
    print(f"{'Parameter':<10} {'Center':>12} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
    print(f"{'-'*80}")

    center_arr = np.array([center[p] for p in PARAM_NAMES])

    for i, param in enumerate(PARAM_NAMES):
        print(f"{param:<10} {center_arr[i]:>12.6e} {samples_arr[:, i].mean():>12.6e} "
              f"{samples_arr[:, i].std():>12.6e} {samples_arr[:, i].min():>12.6e} "
              f"{samples_arr[:, i].max():>12.6e}")

    return samples_arr


def generate_targeted_dataset(experiment_params, n_samples=10000, dl=100e-8,
                              range_pct=10, step_pct=1,
                              range_pct_dict=None, step_pct_dict=None,
                              output_dir="datasets"):
    """
    Generate dense grid dataset around experimental parameters.

    Args:
        experiment_params: dict with experimental parameters
        n_samples: Number of samples to generate
        dl: Layer thickness in cm
        range_pct: Default range around center as % of center value (e.g., 10 = ±10%)
        step_pct: Default grid step as % of local range width (e.g., 1 = 1% steps)
        range_pct_dict: Custom percentages per parameter (overrides range_pct)
        step_pct_dict: Custom step percentages per parameter (overrides step_pct)
        output_dir: Output directory

    Returns:
        data: dict with X, Y, and metadata
    """
    print(f"\n{'='*80}")
    print(f"DENSE GRID TARGETED DATASET GENERATION")
    print(f"{'='*80}")
    print(f"Target samples: {n_samples:,}")
    print(f"dl: {dl*1e8:.0f} Å")

    if range_pct_dict:
        print(f"Local ranges (custom per parameter):")
        for param, pct in range_pct_dict.items():
            print(f"  {param}: ±{pct}%")
    else:
        print(f"Local range: ±{range_pct}% of center value (all parameters)")

    if step_pct_dict:
        print(f"Grid steps (custom per parameter):")
        for param, pct in step_pct_dict.items():
            print(f"  {param}: {pct}% of range")
    else:
        print(f"Grid step: {step_pct}% of local range width (all parameters)")

    # Create local ranges around experiment (ignores global RANGES!)
    local_ranges = create_local_ranges(experiment_params,
                                       range_pct=range_pct,
                                       range_pct_dict=range_pct_dict)

    # Calculate grid steps
    steps = calculate_grid_steps(local_ranges,
                                 step_pct=step_pct,
                                 step_pct_dict=step_pct_dict)

    # Generate dense grid samples
    param_samples = generate_dense_grid(
        experiment_params, local_ranges, steps,
        n_samples_target=n_samples, max_samples=n_samples * 10
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
            'range_pct': range_pct,
            'step_pct': step_pct,
            'range_pct_dict': range_pct_dict,
            'step_pct_dict': step_pct_dict,
            'method': 'Dense grid around experimental parameters',
            'first_sample': 'EXPERIMENTAL PARAMETERS (always X[0])',
            'experiment_params': experiment_params,
            'local_ranges': local_ranges,
            'steps': steps,
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

    # Calculate distances to center (in local normalized space)
    X_norm = np.zeros_like(X)
    center_norm = np.zeros_like(center_arr)

    for i, param in enumerate(PARAM_NAMES):
        min_val, max_val = local_ranges[param]
        range_width = max_val - min_val
        if range_width > 0:
            X_norm[:, i] = (X[:, i] - min_val) / range_width
            center_norm[i] = (center_arr[i] - min_val) / range_width
        else:
            X_norm[:, i] = 0.5
            center_norm[i] = 0.5

    distances = np.linalg.norm(X_norm - center_norm, axis=1)

    print(f"\nDistance to experimental parameters (local normalized space):")
    print(f"   Min distance:    {distances.min():.6f}")
    print(f"   Max distance:    {distances.max():.6f}")
    print(f"   Mean distance:   {distances.mean():.6f}")
    print(f"   Median distance: {np.median(distances):.6f}")

    # Count samples within different thresholds
    for threshold in [0.05, 0.10, 0.15, 0.20, 0.30]:
        count = np.sum(distances <= threshold)
        pct = 100 * count / len(X)
        print(f"   Within {threshold:.2f}: {count:>6,} ({pct:>5.1f}%)")

    # Show how many unique values per parameter
    print(f"\n📊 Unique values per parameter:")
    for i, param in enumerate(PARAM_NAMES):
        n_unique = len(np.unique(X[:, i]))
        print(f"   {param:<10}: {n_unique:>4} unique values")

    return data


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate dense grid dataset around experimental parameters")
    parser.add_argument("--n-samples", type=int, default=1000,
                        help="Number of samples to generate (default: 10000)")
    parser.add_argument("--dl", type=float, default=100e-8,
                        help="Layer thickness in cm (default: 100e-8)")
    parser.add_argument("--range-pct", type=float, default=10,
                        help="Range around center as %% of center value (default: 10 = ±10%%)")
    parser.add_argument("--step-pct", type=float, default=5,
                        help="Grid step as %% of local range width (default: 1 = 1%% steps)")
    parser.add_argument("--output-dir", type=str, default="datasets",
                        help="Output directory (default: datasets)")

    args = parser.parse_args()

    # =============================================================================
    # CUSTOM PERCENTAGES PER PARAMETER (Optional)
    # =============================================================================
    # Якщо потрібно вказати різні відсотки для різних параметрів, розкоментуйте:

    # Приклад: custom ranges для кожного параметра
    range_pct_dict = {
        'Dmax1': 50,  # ±50% від центру
        'D01': 50,    # ±30% від центру
        'L1': 5,     # ±10% від центру
        'Rp1': 1,    # ±20% від центру
        'D02': 60,    # ±40% від центру
        'L2': 20,     # ±15% від центру
        'Rp2': 90,   # ±100% від центру
    }

    # Приклад: custom grid steps для кожного параметра
    step_pct_dict = {
        'Dmax1': 5,   # 2% від range width
        'D01': 5,     # 1% від range width
        'L1': 10,      # 5% від range width
        'Rp1': 80,     # 5% від range width
        'D02': 5,     # 2% від range width
        'L2': 10,      # 5% від range width
        'Rp2': 80,    # 10% від range width
    }

    # Generate dataset
    dataset = generate_targeted_dataset(
        experiment_params=EXPERIMENT_PARAMS,
        n_samples=args.n_samples,
        dl=args.dl,
        range_pct=args.range_pct,
        step_pct=args.step_pct,
        range_pct_dict=range_pct_dict,  # Розкоментуйте для custom ranges
        step_pct_dict=step_pct_dict,    # Розкоментуйте для custom steps
        output_dir=args.output_dir
    )

    print(f"\n{'='*80}")
    print(f"✅ DENSE GRID DATASET GENERATION COMPLETE")
    print(f"{'='*80}")
    print(f"\nDataset characteristics:")
    print(f"  - Dense grid with {args.step_pct}% steps")
    print(f"  - Covers ±{args.range_pct}% around experimental parameters")
    print(f"  - {len(dataset['X']):,} samples")
    print(f"\nNext steps:")
    print(f"1. Train model on this dense dataset:")
    print(f"   python train_with_curve_validation.py")
    print(f"\n2. Or fine-tune existing model:")
    print(f"   python model_train.py --load-model checkpoints/model.pt")
