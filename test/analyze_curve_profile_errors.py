#!/usr/bin/env python3
"""
Analyze CURVE and PROFILE reconstruction errors, not just parameter errors.

Key insight: Small parameter error doesn't guarantee good curve/profile match!
Some parameters are more sensitive than others.
"""

import torch
import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm

import xrd
import helpers as h
from model_common import (
    XRDRegressor, NormalizedXRDDataset, PARAM_NAMES,
    RANGES, load_dataset
)

mpl.rcParams['figure.dpi'] = 100


def calculate_curve_error(true_params: np.ndarray, pred_params: np.ndarray, dl: float = 100e-8):
    """
    Calculate curve reconstruction error.

    Returns:
        curve_mae: Mean absolute error on curve
        curve_rmse: Root mean squared error on curve
        curve_r2: R² score (correlation)
    """
    true_curve, _ = xrd.compute_curve_and_profile(true_params.tolist(), dl=dl)
    pred_curve, _ = xrd.compute_curve_and_profile(pred_params.tolist(), dl=dl)

    # Use log-space for comparison (more meaningful for XRD)
    true_y = np.log10(true_curve.Y_R_vseZ + 1e-10)
    pred_y = np.log10(pred_curve.Y_R_vseZ + 1e-10)

    curve_mae = np.mean(np.abs(pred_y - true_y))
    curve_rmse = np.sqrt(np.mean((pred_y - true_y)**2))

    # R² score
    ss_res = np.sum((true_y - pred_y)**2)
    ss_tot = np.sum((true_y - np.mean(true_y))**2)
    curve_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return curve_mae, curve_rmse, curve_r2


def calculate_profile_error(true_params: np.ndarray, pred_params: np.ndarray, dl: float = 100e-8):
    """Calculate profile reconstruction error."""
    _, true_profile = xrd.compute_curve_and_profile(true_params.tolist(), dl=dl)
    _, pred_profile = xrd.compute_curve_and_profile(pred_params.tolist(), dl=dl)

    true_y = true_profile.total_Y
    pred_y = pred_profile.total_Y

    profile_mae = np.mean(np.abs(pred_y - true_y))
    profile_rmse = np.sqrt(np.mean((pred_y - true_y)**2))

    # R² score
    ss_res = np.sum((true_y - pred_y)**2)
    ss_tot = np.sum((true_y - np.mean(true_y))**2)
    profile_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return profile_mae, profile_rmse, profile_r2


def analyze_full_dataset_reconstruction(model_path: str, X: torch.Tensor, Y: torch.Tensor,
                                        use_full: bool = False, n_samples: int = 1000):
    """
    Analyze curve and profile reconstruction errors on dataset samples.

    Args:
        n_samples: Number of samples to analyze (curve generation is slow)
    """
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    print(f"\n{'='*100}")
    print(f"🔬 CURVE & PROFILE RECONSTRUCTION ANALYSIS")
    print(f"   Model: {Path(model_path).name}")
    print(f"{'='*100}")

    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    epoch = checkpoint.get('epoch', '?')
    val_loss = checkpoint.get('val_loss', '?')

    print(f"   Epoch: {epoch} | Val Loss: {val_loss}")

    model = XRDRegressor(n_out=7).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # Get predictions
    dataset = NormalizedXRDDataset(X, Y, log_space=True, train=False)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)

    predictions = []
    with torch.no_grad():
        for batch_y, batch_x in loader:
            batch_y = batch_y.to(device)
            pred = model(batch_y)
            predictions.append(pred.cpu())

    predictions = torch.cat(predictions, dim=0).numpy()
    X_np = X.numpy()

    # Subsample for curve generation (too slow for 10k)
    np.random.seed(42)
    indices = np.random.choice(len(X_np), size=min(n_samples, len(X_np)), replace=False)

    print(f"\n🎯 Analyzing {len(indices)} random samples...")

    # Calculate reconstruction errors
    curve_errors = {
        'mae': [],
        'rmse': [],
        'r2': []
    }
    profile_errors = {
        'mae': [],
        'rmse': [],
        'r2': []
    }
    param_errors = []

    for idx in tqdm(indices, desc="Computing curves"):
        true_params = X_np[idx]
        pred_params = predictions[idx]

        # Parameter error
        param_mae = np.mean(np.abs(pred_params - true_params))
        param_errors.append(param_mae)

        # Curve error
        c_mae, c_rmse, c_r2 = calculate_curve_error(true_params, pred_params)
        curve_errors['mae'].append(c_mae)
        curve_errors['rmse'].append(c_rmse)
        curve_errors['r2'].append(c_r2)

        # Profile error
        p_mae, p_rmse, p_r2 = calculate_profile_error(true_params, pred_params)
        profile_errors['mae'].append(p_mae)
        profile_errors['rmse'].append(p_rmse)
        profile_errors['r2'].append(p_r2)

    # Convert to arrays
    param_errors = np.array(param_errors)
    for key in curve_errors:
        curve_errors[key] = np.array(curve_errors[key])
        profile_errors[key] = np.array(profile_errors[key])

    # Print statistics
    print(f"\n📊 RECONSTRUCTION ERROR STATISTICS:")
    print("-"*100)
    print(f"{'Metric':<25} {'Mean':<15} {'Median':<15} {'Std':<15} {'Min':<15} {'Max':<15}")
    print("-"*100)

    # Parameter error
    print(f"{'Parameter MAE':<25} {np.mean(param_errors):<15.6e} {np.median(param_errors):<15.6e} "
          f"{np.std(param_errors):<15.6e} {np.min(param_errors):<15.6e} {np.max(param_errors):<15.6e}")

    # Curve errors
    for metric, values in curve_errors.items():
        name = f"Curve {metric.upper()}"
        print(f"{name:<25} {np.mean(values):<15.6f} {np.median(values):<15.6f} "
              f"{np.std(values):<15.6f} {np.min(values):<15.6f} {np.max(values):<15.6f}")

    # Profile errors
    for metric, values in profile_errors.items():
        name = f"Profile {metric.upper()}"
        print(f"{name:<25} {np.mean(values):<15.6f} {np.median(values):<15.6f} "
              f"{np.std(values):<15.6f} {np.min(values):<15.6f} {np.max(values):<15.6f}")

    print("-"*100)

    # Correlation analysis
    print(f"\n🔗 CORRELATION: Parameter Error vs Reconstruction Error")
    print("-"*100)

    from scipy.stats import pearsonr, spearmanr

    for error_type, errors_dict in [('Curve', curve_errors), ('Profile', profile_errors)]:
        for metric in ['mae', 'rmse']:
            r_pearson, p_pearson = pearsonr(param_errors, errors_dict[metric])
            r_spearman, p_spearman = spearmanr(param_errors, errors_dict[metric])

            print(f"{error_type} {metric.upper()} vs Param MAE:")
            print(f"  Pearson r = {r_pearson:.4f} (p={p_pearson:.4e})")
            print(f"  Spearman ρ = {r_spearman:.4f} (p={p_spearman:.4e})")

            if abs(r_pearson) < 0.5:
                print(f"  ⚠️  WEAK correlation - parameter error doesn't predict reconstruction quality!")

    print("-"*100)

    # Find best and worst cases
    print(f"\n📊 BEST & WORST CASES:")
    print("-"*100)

    for metric_name, (error_dict, error_type) in [
        ('Curve MAE', (curve_errors, 'curve')),
        ('Profile MAE', (profile_errors, 'profile'))
    ]:
        errors = error_dict['mae']
        best_idx = indices[np.argmin(errors)]
        worst_idx = indices[np.argmax(errors)]

        print(f"\n{metric_name}:")
        print(f"  Best:  Sample {best_idx:5d} | Error: {errors[np.argmin(errors)]:.6f} | "
              f"Param MAE: {param_errors[np.argmin(errors)]:.6e}")
        print(f"  Worst: Sample {worst_idx:5d} | Error: {errors[np.argmax(errors)]:.6f} | "
              f"Param MAE: {param_errors[np.argmax(errors)]:.6e}")

    print("="*100)

    # Visualizations
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Row 1: Histograms
    axes[0, 0].hist(param_errors, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Parameter MAE')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Parameter Error Distribution')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].hist(curve_errors['mae'], bins=50, alpha=0.7, color='orange', edgecolor='black')
    axes[0, 1].set_xlabel('Curve MAE (log-space)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Curve Reconstruction Error')
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].hist(profile_errors['mae'], bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[0, 2].set_xlabel('Profile MAE')
    axes[0, 2].set_ylabel('Count')
    axes[0, 2].set_title('Profile Reconstruction Error')
    axes[0, 2].grid(True, alpha=0.3)

    # Row 2: Scatter plots (correlation)
    axes[1, 0].scatter(param_errors, curve_errors['mae'], alpha=0.3, s=10)
    axes[1, 0].set_xlabel('Parameter MAE')
    axes[1, 0].set_ylabel('Curve MAE')
    axes[1, 0].set_title('Parameter Error vs Curve Error')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].scatter(param_errors, profile_errors['mae'], alpha=0.3, s=10, color='green')
    axes[1, 1].set_xlabel('Parameter MAE')
    axes[1, 1].set_ylabel('Profile MAE')
    axes[1, 1].set_title('Parameter Error vs Profile Error')
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].scatter(curve_errors['mae'], profile_errors['mae'], alpha=0.3, s=10, color='purple')
    axes[1, 2].set_xlabel('Curve MAE')
    axes[1, 2].set_ylabel('Profile MAE')
    axes[1, 2].set_title('Curve Error vs Profile Error')
    axes[1, 2].grid(True, alpha=0.3)

    plt.suptitle(f'Reconstruction Error Analysis - {Path(model_path).name}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    filename = f'reconstruction_errors_{Path(model_path).stem}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\n✓ Saved: {filename}")

    return {
        'param_errors': param_errors,
        'curve_errors': curve_errors,
        'profile_errors': profile_errors,
        'indices': indices
    }


def main():
    MODELS = {
        'ideal': 'checkpoints_save/dataset_10000_dl100_7d_v3_unweighted_ideal.pt',
        'final': 'checkpoints/dataset_10000_dl100_7d_v3_unweighted.pt',
    }

    # Load dataset (cropped for unweighted models)
    print("📦 Loading dataset...")
    X, Y = load_dataset('datasets/dataset_10000_dl100_7d.pkl', use_full_curve=False)

    results = {}
    for name, path in MODELS.items():
        results[name] = analyze_full_dataset_reconstruction(
            path, X, Y, use_full=False, n_samples=1000
        )

    # Comparative analysis
    print(f"\n{'='*100}")
    print(f"⚖️  COMPARATIVE ANALYSIS: IDEAL vs FINAL")
    print(f"{'='*100}")

    print(f"\n{'Metric':<30} {'Ideal':<20} {'Final':<20} {'Winner':<15}")
    print("-"*100)

    metrics_to_compare = [
        ('Param MAE', 'param_errors'),
        ('Curve MAE', ('curve_errors', 'mae')),
        ('Curve RMSE', ('curve_errors', 'rmse')),
        ('Curve R²', ('curve_errors', 'r2')),
        ('Profile MAE', ('profile_errors', 'mae')),
        ('Profile RMSE', ('profile_errors', 'rmse')),
        ('Profile R²', ('profile_errors', 'r2')),
    ]

    for metric_name, key in metrics_to_compare:
        if isinstance(key, tuple):
            ideal_val = np.mean(results['ideal'][key[0]][key[1]])
            final_val = np.mean(results['final'][key[0]][key[1]])
        else:
            ideal_val = np.mean(results['ideal'][key])
            final_val = np.mean(results['final'][key])

        # For R², higher is better; for others, lower is better
        if 'R²' in metric_name or 'r2' in metric_name:
            winner = 'ideal' if ideal_val > final_val else 'final'
        else:
            winner = 'ideal' if ideal_val < final_val else 'final'

        print(f"{metric_name:<30} {ideal_val:<20.6f} {final_val:<20.6f} {winner:<15}")

    print("-"*100)

    print(f"\n💡 INTERPRETATION:")
    print("-"*100)
    print("If 'ideal' wins on dataset metrics but 'final' has lower val_loss:")
    print("  → Val loss optimized for dataset distribution")
    print("  → 'Ideal' might be better for specific cases/experiments")
    print("  → Consider ensemble or sample-specific model selection")
    print("")
    print("If parameter error is low but curve/profile error is high:")
    print("  → Sensitive parameter region (small param change → large curve change)")
    print("  → Need tighter parameter prediction tolerances")
    print("="*100)


if __name__ == "__main__":
    main()
