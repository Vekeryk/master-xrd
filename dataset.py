import numpy as np
import torch
import xrd
from helpers import get_device
from importlib import reload
import pickle
import os
from datetime import datetime
from tqdm import tqdm

# Import from model_common.py - single source of truth!
from model_common import RANGES as MODEL_RANGES, GRID_STEPS

reload(xrd)


def arange_inclusive(start, stop, step):
    """Helper function to create inclusive ranges (from perebir.py)"""
    return np.arange(start, stop + 0.5 * step, step, dtype=float)


# Функція для генерації датасету з розумним перебором (smart grid logic from perebir.py)
def generate_train_dataset(n_samples, dl=100e-8):
    """
    Generate dataset using smart grid-based sampling that respects physical constraints:
    - D01 <= Dmax1
    - D01 + D02 <= 0.03
    - Rp1 <= L1
    - L2 <= L1
    """
    # Generate grids from MODEL_RANGES + GRID_STEPS (model_common.py - single source of truth!)
    Dmax1_grid = arange_inclusive(MODEL_RANGES['Dmax1'][0], MODEL_RANGES['Dmax1'][1], GRID_STEPS['Dmax1'])
    D01_grid = arange_inclusive(MODEL_RANGES['D01'][0], MODEL_RANGES['D01'][1], GRID_STEPS['D01'])
    L1_grid = arange_inclusive(MODEL_RANGES['L1'][0] * 1e8, MODEL_RANGES['L1'][1] * 1e8, GRID_STEPS['L1'])
    Rp1_grid = arange_inclusive(MODEL_RANGES['Rp1'][0] * 1e8, MODEL_RANGES['Rp1'][1] * 1e8, GRID_STEPS['Rp1'])
    D02_grid = arange_inclusive(MODEL_RANGES['D02'][0], MODEL_RANGES['D02'][1], GRID_STEPS['D02'])
    L2_grid = arange_inclusive(MODEL_RANGES['L2'][0] * 1e8, MODEL_RANGES['L2'][1] * 1e8, GRID_STEPS['L2'])
    Rp2_grid = arange_inclusive(MODEL_RANGES['Rp2'][0] * 1e8, MODEL_RANGES['Rp2'][1] * 1e8, GRID_STEPS['Rp2'])

    limit = 0.03  # constraint for D01 + D02

    X = []
    Y = []

    # Create all valid combinations using smart iteration logic
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
    else:
        indices = np.random.choice(total_valid, size=n_samples, replace=False)

    for idx in tqdm(indices, desc="Generating samples", unit="sample"):
        _Dmax1, _D01, _L1, _Rp1, _D02, _L2, _Rp2 = valid_combinations[idx]

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

        X.append([_Dmax1, _D01, _L1_cm, _Rp1_cm, _D02, _L2_cm, _Rp2_cm])
        Y.append(curve.ML_Y)

    device = get_device()
    X = torch.tensor(X, dtype=torch.float32, device=device)
    Y = torch.tensor(Y, dtype=torch.float32, device=device)

    return X, Y


if __name__ == "__main__":
    # Hardcoded parameters
    n_samples = 100_000
    dl = 400e-8  # in cm (400 Angstroms)

    # Convert dl to Angstroms for filename
    dl_angstrom = int(dl * 1e8)
    output_file = f"datasets/dataset_{n_samples}_dl{dl_angstrom}.pkl"

    print(f"Generating {n_samples} samples...")
    print(f"Output file: {output_file}")
    print("-" * 60)

    # Generate dataset
    X, Y = generate_train_dataset(n_samples, dl=dl)

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
        'timestamp': datetime.now().isoformat(),
        'device': str(X.device)
    }

    # Save to pickle file
    print(f"\nSaving dataset to {output_file}...")
    with open(output_file, 'wb') as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Dataset saved successfully!")
    print(f"File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
