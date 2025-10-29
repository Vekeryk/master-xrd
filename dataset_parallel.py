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
reload(xrd)


def arange_inclusive(start, stop, step):
    """Helper function to create inclusive ranges (from perebir.py)"""
    return np.arange(start, stop + 0.5 * step, step, dtype=float)


# =============================================================================
# HELPER FUNCTION FOR MULTIPROCESSING (must be at module level for pickling)
# =============================================================================

def _generate_single_sample(args):
    """
    Generate a single XRD curve sample.
    This function must be at module level for multiprocessing pickling.

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
# PARALLEL DATASET GENERATION
# =============================================================================

def generate_train_dataset(n_samples, dl=100e-8, n_workers=None):
    """
    Generate dataset using smart grid-based sampling with PARALLEL PROCESSING.

    Physical constraints respected:
    - D01 <= Dmax1
    - D01 + D02 <= 0.03
    - Rp1 <= L1
    - L2 <= L1

    Args:
        n_samples: Number of samples to generate
        dl: Sublayer thickness in cm (default: 100e-8)
        n_workers: Number of parallel workers (default: cpu_count() - 1)

    Returns:
        X, Y: PyTorch tensors with input parameters and XRD curves
    """
    # Determine number of workers
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    print(f"Using {n_workers} parallel workers")

    # Define parameter grids (from perebir.py - optimal grid)
    Dmax1_grid = arange_inclusive(0.0025, 0.0250, 0.0025)
    D01_grid = arange_inclusive(0.0025, 0.0250, 0.0025)
    L1_grid = arange_inclusive(500., 7000., 500.)
    Rp1_grid = arange_inclusive(490., 4990., 500.)
    D02_grid = arange_inclusive(0.0025, 0.0250, 0.0025)
    L2_grid = arange_inclusive(500., 5000., 1000.)
    Rp2_grid = arange_inclusive(-6010., -10., 1000.)

    limit = 0.03  # constraint for D01 + D02

    # Create all valid combinations using smart iteration logic
    print("Building valid combinations grid...")
    valid_combinations = []
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
                                valid_combinations.append(
                                    (d1, d01, l1, r1, d02, l2, r2))

    total_valid = len(valid_combinations)
    print(f"Total valid combinations: {total_valid:,}")

    # Randomly sample n_samples from valid combinations
    if n_samples > total_valid:
        print(
            f"Warning: Requested {n_samples} samples, but only {total_valid} valid combinations exist.")
        print(f"Generating all {total_valid} samples.")
        indices = range(total_valid)
        n_samples = total_valid
    else:
        indices = np.random.choice(total_valid, size=n_samples, replace=False)

    # Prepare arguments for parallel processing
    # Each argument is (Dmax1, D01, L1, Rp1, D02, L2, Rp2, dl)
    args_list = []
    for idx in indices:
        _Dmax1, _D01, _L1, _Rp1, _D02, _L2, _Rp2 = valid_combinations[idx]
        args_list.append((_Dmax1, _D01, _L1, _Rp1, _D02, _L2, _Rp2, dl))

    # Parallel processing with progress bar
    print(f"Generating {n_samples} samples in parallel...")
    X = []
    Y = []

    with Pool(processes=n_workers) as pool:
        # imap_unordered for better performance, wrapping with tqdm for progress
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


if __name__ == "__main__":
    # Hardcoded parameters
    n_samples = 100_000
    dl = 100e-8  # in cm (400 Angstroms)
    n_workers = None  # Auto-detect (cpu_count - 1)

    jit = True  # Enable JIT compilation
    jit_part = ''
    if jit:
        jit_part = '_jit'

    # Convert dl to Angstroms for filename
    dl_angstrom = int(dl * 1e8)
    output_file = f"datasets/dataset_{n_samples}_dl{dl_angstrom}{jit_part}.pkl"

    print(f"Generating {n_samples} samples with parallel processing...")
    print(f"Output file: {output_file}")
    print(f"dl parameter: {dl_angstrom} Angstroms")
    print("-" * 60)

    # Generate dataset
    X, Y = generate_train_dataset(n_samples, dl=dl, n_workers=n_workers)

    print("-" * 60)
    print(f"Dataset generated successfully!")
    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")
    print(f"Device: {X.device}")

    # Prepare data for saving
    dataset = {
        'X': X.cpu().numpy(),  # Convert to numpy for pickle
        'Y': Y.cpu().numpy(),
        'n_samples': n_samples,
        'dl': dl,
        'dl_angstrom': dl_angstrom,
        'timestamp': datetime.now().isoformat(),
        'device': str(X.device)
    }

    # Save to pickle file
    print(f"\nSaving dataset to {output_file}...")
    with open(output_file, 'wb') as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Dataset saved successfully!")
    print(f"File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
