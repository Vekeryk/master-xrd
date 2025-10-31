"""
Verify peak positions in synthetic XRD curves.

Critical diagnostic to validate fixed start_ML=50 truncation.
If peak positions vary significantly, adaptive truncation is required.
"""

import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import xrd_deprecated


def verify_peak_positions(dataset_path, n_samples=1000):
    """
    Check distribution of peak positions in dataset.

    Args:
        dataset_path: Path to pickle dataset
        n_samples: Number of samples to check (default: 1000)
    """
    print("=" * 70)
    print("PEAK POSITION VERIFICATION")
    print("=" * 70)
    print(f"Dataset: {dataset_path}")
    print(f"Samples to check: {n_samples}")
    print()

    # Load dataset
    if not Path(dataset_path).exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)

    X = data['X']
    dl = data.get('dl', 100e-8)

    # Limit samples
    n_check = min(n_samples, len(X))
    print(f"Checking {n_check} curves...")
    print()

    # Compute peak positions
    peak_positions = []
    peak_intensities = []

    for i in range(n_check):
        # Generate full curve (before truncation)
        curve, _ = xrd_deprecated.compute_curve_and_profile(
            array=X[i],
            dl=dl,
            m1=700,
            m10=20
        )

        full_curve = curve.Y_R_vseZ
        peak_idx = np.argmax(full_curve)
        peak_int = full_curve[peak_idx]

        peak_positions.append(peak_idx)
        peak_intensities.append(peak_int)

        if i % 100 == 0:
            print(f"  Progress: {i}/{n_check} curves checked...")

    peak_positions = np.array(peak_positions)
    peak_intensities = np.array(peak_intensities)

    # Statistics
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    print("Peak Position Statistics:")
    print(f"  Mean:   {peak_positions.mean():.2f}")
    print(f"  Median: {np.median(peak_positions):.2f}")
    print(f"  Std:    {peak_positions.std():.2f}")
    print(f"  Min:    {peak_positions.min()}")
    print(f"  Max:    {peak_positions.max()}")
    print(f"  Range:  {peak_positions.max() - peak_positions.min()}")
    print()

    # Assessment
    current_start_ML = 50
    mean_peak = peak_positions.mean()
    std_peak = peak_positions.std()

    print("Assessment:")
    print(f"  Current start_ML: {current_start_ML}")
    print(f"  Mean peak position: {mean_peak:.1f}")
    print(
        f"  Distance from start_ML: {abs(mean_peak - current_start_ML):.1f} points")
    print()

    if std_peak < 2:
        print("✅ EXCELLENT: Peak positions are very consistent (std < 2)")
        print("   → Fixed start_ML=50 is SAFE")
    elif std_peak < 5:
        print("✅ GOOD: Peak positions are consistent (std < 5)")
        print("   → Fixed start_ML=50 is acceptable")
    elif std_peak < 10:
        print("⚠️  WARNING: Peak positions vary moderately (std < 10)")
        print("   → Consider adaptive truncation for robustness")
    else:
        print("🔴 CRITICAL: Peak positions vary significantly (std ≥ 10)")
        print("   → Fixed start_ML will cause misalignment!")
        print("   → MUST implement adaptive truncation")

    print()

    if abs(mean_peak - current_start_ML) > 10:
        print("🔴 CRITICAL: Mean peak position differs from start_ML by >10 points!")
        print(
            f"   → Should use start_ML={int(mean_peak)} instead of {current_start_ML}")
    elif abs(mean_peak - current_start_ML) > 5:
        print("⚠️  WARNING: Mean peak differs from start_ML by >5 points")
        print(f"   → Consider adjusting start_ML to {int(mean_peak)}")
    else:
        print("✅ GOOD: Mean peak position close to start_ML")

    print()

    # Peak intensity statistics
    print("Peak Intensity Statistics:")
    print(f"  Mean:   {peak_intensities.mean():.4e}")
    print(f"  Std:    {peak_intensities.std():.4e}")
    print(f"  Min:    {peak_intensities.min():.4e}")
    print(f"  Max:    {peak_intensities.max():.4e}")
    print()

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram of peak positions
    axes[0].hist(peak_positions, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(current_start_ML, color='r', linestyle='--',
                    linewidth=2, label=f'start_ML={current_start_ML}')
    axes[0].axvline(mean_peak, color='g', linestyle='--',
                    linewidth=2, label=f'Mean={mean_peak:.1f}')
    axes[0].set_xlabel('Peak Position (index)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Distribution of Peak Positions')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Peak position vs sample index
    axes[1].scatter(range(len(peak_positions)), peak_positions,
                    alpha=0.3, s=10)
    axes[1].axhline(current_start_ML, color='r', linestyle='--',
                    linewidth=2, label=f'start_ML={current_start_ML}')
    axes[1].axhline(mean_peak, color='g', linestyle='--',
                    linewidth=2, label=f'Mean={mean_peak:.1f}')
    axes[1].fill_between([0, len(peak_positions)],
                         mean_peak - std_peak,
                         mean_peak + std_peak,
                         alpha=0.2, color='green',
                         label=f'±1 std')
    axes[1].set_xlabel('Sample Index')
    axes[1].set_ylabel('Peak Position (index)')
    axes[1].set_title('Peak Position Variation Across Samples')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('peak_position_verification.png', dpi=150)
    print(f"📊 Plot saved to: peak_position_verification.png")
    print()

    # Sample curves
    print("Sample Curves (first 3):")
    fig, axes = plt.subplots(3, 1, figsize=(12, 9))

    for i in range(min(3, n_check)):
        curve, _ = xrd_deprecated.compute_curve_and_profile(
            array=X[i],
            dl=dl,
            m1=700,
            m10=20
        )

        axes[i].plot(curve.Y_R_vseZ, linewidth=1)
        axes[i].axvline(peak_positions[i], color='g',
                        linestyle='--', label=f'Peak at {peak_positions[i]}')
        axes[i].axvline(current_start_ML, color='r',
                        linestyle='--', label=f'start_ML={current_start_ML}')
        axes[i].set_ylabel('Intensity')
        axes[i].set_title(f'Sample {i}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        axes[i].set_yscale('log')

    axes[-1].set_xlabel('Point Index')
    plt.tight_layout()
    plt.savefig('sample_curves_with_peaks.png', dpi=150)
    print(f"📊 Sample curves saved to: sample_curves_with_peaks.png")

    print()
    print("=" * 70)
    print("VERIFICATION COMPLETE")
    print("=" * 70)

    return peak_positions, peak_intensities


if __name__ == "__main__":
    # Configuration
    DATASET_PATH = "datasets/dataset_1000_dl100_balanced.pkl"  # Test dataset
    # DATASET_PATH = "datasets/dataset_100000_dl400.pkl"  # Full dataset
    N_SAMPLES = 1000  # Check all samples in test dataset

    # Run verification
    peak_pos, peak_int = verify_peak_positions(DATASET_PATH, N_SAMPLES)
