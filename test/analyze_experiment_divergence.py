#!/usr/bin/env python3
"""
Analyze why model performs well on specific experiment at some epochs
but worse at later epochs despite better overall validation loss.

This investigates:
1. Dataset coverage around experiment parameters
2. Per-parameter error distributions
3. Which parameter combinations are problematic
4. Correlation between val_loss and experiment-specific error
"""

import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

import xrd
import helpers as h
from model_common import (
    XRDRegressor, NormalizedXRDDataset, PARAM_NAMES,
    RANGES, load_dataset
)

mpl.rcParams['figure.dpi'] = 100

# Your experiment parameters
EXPERIMENT_PARAMS = np.array([0.008094, 0.000943, 5200e-8, 3500e-8, 0.00255, 3000e-8, -50e-8])

# Models to compare
MODELS = {
    'ideal': 'checkpoints_save/dataset_10000_dl100_7d_v3_unweighted_ideal.pt',
    'final': 'checkpoints/dataset_10000_dl100_7d_v3_unweighted.pt',
}


def load_and_predict_single(model_path: str, input_curve: np.ndarray, use_full: bool = False):
    """Load model and predict for single curve."""
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    epoch = checkpoint.get('epoch', '?')
    val_loss = checkpoint.get('val_loss', '?')

    # Create model
    model = XRDRegressor(n_out=7).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # Prepare input
    # Need to normalize like dataset does
    if use_full:
        curve = input_curve  # 701 points
    else:
        curve = input_curve[50:701]  # Crop to 651 points

    # Convert to tensor
    Y = torch.tensor(curve, dtype=torch.float32).unsqueeze(0)  # [1, L]

    # Create dummy X for dataset (we only need Y normalization)
    X = torch.zeros((1, 7))

    # Apply same normalization as training
    dataset = NormalizedXRDDataset(X, Y, log_space=True, train=False)
    normalized_Y = dataset.Yn[0].unsqueeze(0).to(device)  # [1, 1, L]

    # Predict
    with torch.no_grad():
        prediction = model(normalized_Y).cpu().numpy()[0]

    return prediction, epoch, val_loss


def analyze_dataset_coverage(X: torch.Tensor, experiment_params: np.ndarray):
    """Analyze how well dataset covers region around experiment parameters."""
    X_np = X.numpy()

    print("\n" + "="*100)
    print("📍 DATASET COVERAGE ANALYSIS")
    print("="*100)

    # For each parameter, find closest samples
    distances = np.abs(X_np - experiment_params[None, :])

    # Overall distance (Euclidean in normalized space)
    # Normalize by parameter ranges first
    param_ranges = np.array([RANGES[p][1] - RANGES[p][0] for p in PARAM_NAMES])
    normalized_distances = distances / param_ranges[None, :]
    euclidean_distances = np.sqrt(np.sum(normalized_distances**2, axis=1))

    # Find closest samples
    closest_indices = np.argsort(euclidean_distances)[:20]

    print(f"\nExperiment parameters: {h.fparam(arr=experiment_params)}")
    print(f"\nClosest 10 samples in dataset:")
    print("-"*100)
    for i, idx in enumerate(closest_indices[:10], 1):
        dist = euclidean_distances[idx]
        print(f"{i:2d}. Sample {idx:5d} | Distance: {dist:.4f} | {h.fparam(arr=X_np[idx])}")

    # Parameter-wise coverage
    print(f"\n\nPer-Parameter Distance to Closest Sample:")
    print("-"*100)
    print(f"{'Parameter':<10} {'Experiment':<15} {'Closest':<15} {'Distance':<15} {'% of Range':<15}")
    print("-"*100)

    for i, param in enumerate(PARAM_NAMES):
        exp_val = experiment_params[i]
        param_distances = distances[:, i]
        closest_idx = np.argmin(param_distances)
        closest_val = X_np[closest_idx, i]
        min_dist = param_distances[closest_idx]
        param_range = RANGES[param][1] - RANGES[param][0]
        dist_pct = (min_dist / param_range) * 100

        print(f"{param:<10} {exp_val:<15.6e} {closest_val:<15.6e} {min_dist:<15.6e} {dist_pct:<15.2f}%")

    # Density analysis: how many samples within various radii?
    print(f"\n\nDataset Density Around Experiment Parameters:")
    print("-"*100)
    radii = [0.1, 0.2, 0.5, 1.0, 2.0]
    for radius in radii:
        count = np.sum(euclidean_distances < radius)
        percentage = (count / len(X_np)) * 100
        print(f"Within radius {radius:4.1f}: {count:5d} samples ({percentage:5.2f}%)")

    # Check if experiment parameters are near edges
    print(f"\n\nExperiment Parameters vs Range Edges:")
    print("-"*100)
    print(f"{'Parameter':<10} {'Min':<12} {'Experiment':<12} {'Max':<12} {'Position':<20}")
    print("-"*100)
    for i, param in enumerate(PARAM_NAMES):
        exp_val = experiment_params[i]
        min_val = RANGES[param][0]
        max_val = RANGES[param][1]
        position = (exp_val - min_val) / (max_val - min_val)

        # Determine position description
        if position < 0.1:
            pos_desc = "Near MIN edge"
        elif position > 0.9:
            pos_desc = "Near MAX edge"
        elif 0.4 < position < 0.6:
            pos_desc = "Center"
        else:
            pos_desc = "Mid-range"

        print(f"{param:<10} {min_val:<12.6e} {exp_val:<12.6e} {max_val:<12.6e} {pos_desc:<20} ({position*100:.1f}%)")

    print("="*100)

    return closest_indices, euclidean_distances


def analyze_model_errors_on_dataset(model_path: str, X: torch.Tensor, Y: torch.Tensor,
                                    experiment_params: np.ndarray, use_full: bool = False):
    """Analyze model errors across dataset and compare to experiment region."""
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    epoch = checkpoint.get('epoch', '?')
    val_loss = checkpoint.get('val_loss', '?')

    model = XRDRegressor(n_out=7).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # Create dataset
    dataset = NormalizedXRDDataset(X, Y, log_space=True, train=False)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)

    # Run predictions
    predictions = []
    with torch.no_grad():
        for batch_y, batch_x in loader:
            batch_y = batch_y.to(device)
            pred = model(batch_y)
            predictions.append(pred.cpu())

    predictions = torch.cat(predictions, dim=0).numpy()
    X_np = X.numpy()

    # Calculate errors
    abs_errors = np.abs(predictions - X_np)
    rel_errors = (predictions - X_np) / (np.abs(X_np) + 1e-12) * 100

    # Mean error per sample
    mean_abs_errors = np.mean(abs_errors, axis=1)

    print(f"\n{'='*100}")
    print(f"📊 MODEL ERROR ANALYSIS: {Path(model_path).name}")
    print(f"   Epoch: {epoch} | Val Loss: {val_loss}")
    print(f"{'='*100}")

    # Overall statistics
    print(f"\n📈 Overall Error Statistics:")
    print("-"*100)
    print(f"Mean MAE: {np.mean(mean_abs_errors):.6e}")
    print(f"Median MAE: {np.median(mean_abs_errors):.6e}")
    print(f"Std MAE: {np.std(mean_abs_errors):.6e}")
    print(f"Min MAE: {np.min(mean_abs_errors):.6e} (sample {np.argmin(mean_abs_errors)})")
    print(f"Max MAE: {np.max(mean_abs_errors):.6e} (sample {np.argmax(mean_abs_errors)})")

    # Per-parameter errors
    print(f"\n📊 Per-Parameter MAE:")
    print("-"*100)
    print(f"{'Parameter':<10} {'MAE':<15} {'Median':<15} {'Std':<15} {'Max':<15}")
    print("-"*100)
    for i, param in enumerate(PARAM_NAMES):
        mae = np.mean(abs_errors[:, i])
        median = np.median(abs_errors[:, i])
        std = np.std(abs_errors[:, i])
        max_err = np.max(abs_errors[:, i])
        print(f"{param:<10} {mae:<15.6e} {median:<15.6e} {std:<15.6e} {max_err:<15.6e}")

    # Analyze errors in region around experiment
    param_ranges = np.array([RANGES[p][1] - RANGES[p][0] for p in PARAM_NAMES])
    normalized_distances = np.abs(X_np - experiment_params[None, :]) / param_ranges[None, :]
    euclidean_distances = np.sqrt(np.sum(normalized_distances**2, axis=1))

    # Compare errors in different regions
    print(f"\n📍 Error Comparison: Near Experiment vs Overall:")
    print("-"*100)
    radii = [0.5, 1.0, 2.0]
    for radius in radii:
        near_mask = euclidean_distances < radius
        if np.sum(near_mask) > 0:
            near_mae = np.mean(mean_abs_errors[near_mask])
            overall_mae = np.mean(mean_abs_errors)
            count = np.sum(near_mask)
            print(f"Within radius {radius:4.1f}: {count:4d} samples | "
                  f"MAE = {near_mae:.6e} (overall: {overall_mae:.6e}) | "
                  f"Ratio: {near_mae/overall_mae:.2f}x")

    print("="*100)

    return predictions, abs_errors, mean_abs_errors, euclidean_distances


def compare_models_on_experiment(experiment_params: np.ndarray, models: dict):
    """Compare how different model checkpoints perform on experiment."""
    print("\n" + "="*100)
    print("🔬 EXPERIMENT-SPECIFIC PREDICTIONS")
    print("="*100)
    print(f"\nExperiment: {h.fparam(arr=experiment_params)}")

    # Generate experimental curve
    curve, profile = xrd.compute_curve_and_profile(experiment_params.tolist(), dl=100e-8)

    print("\n" + "-"*100)
    print(f"{'Model':<25} {'Epoch':<10} {'Val Loss':<15} {'Experiment MAE':<20}")
    print("-"*100)

    results = {}
    for name, path in models.items():
        # Determine if full curve
        use_full = 'full' in path.lower()

        if use_full:
            input_curve = curve.Y_R_vseZ
        else:
            input_curve = curve.ML_Y

        pred, epoch, val_loss = load_and_predict_single(path, input_curve, use_full)

        # Calculate error
        mae = np.mean(np.abs(pred - experiment_params))

        print(f"{name:<25} {epoch:<10} {val_loss:<15.6f} {mae:<20.6e}")

        results[name] = {
            'prediction': pred,
            'epoch': epoch,
            'val_loss': val_loss,
            'mae': mae
        }

    print("-"*100)

    # Detailed comparison
    print(f"\n📊 Detailed Parameter-wise Comparison:")
    print("-"*100)
    print(f"{'Parameter':<10}", end='')
    for name in models.keys():
        print(f" | {name:<20}", end='')
    print()
    print("-"*100)

    for i, param in enumerate(PARAM_NAMES):
        true_val = experiment_params[i]
        print(f"{param:<10}", end='')
        for name in models.keys():
            pred_val = results[name]['prediction'][i]
            error = pred_val - true_val
            rel_error = (error / (abs(true_val) + 1e-12)) * 100
            print(f" | {pred_val:.4e} ({rel_error:+6.1f}%)", end='')
        print()

    print("-"*100)

    # Visual comparison
    fig, axes = plt.subplots(2, len(models), figsize=(7*len(models), 10))
    if len(models) == 1:
        axes = axes[:, None]

    for col, (name, result) in enumerate(results.items()):
        pred_params = result['prediction']
        pred_curve, pred_profile = xrd.compute_curve_and_profile(pred_params.tolist(), dl=100e-8)

        # Rocking curve
        ax = axes[0, col]
        ax.plot(curve.X_DeltaTeta, curve.Y_R_vseZ, 'k-', linewidth=2, label='Experiment')
        ax.plot(pred_curve.X_DeltaTeta, pred_curve.Y_R_vseZ, 'r--', linewidth=2, label='Prediction')
        ax.set_xlabel('ΔΘ (arcsec)')
        ax.set_ylabel('Intensity')
        ax.set_title(f'{name}\nEpoch {result["epoch"]} | Val Loss: {result["val_loss"]:.6f}\nMAE: {result["mae"]:.6e}')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Deformation profile
        ax = axes[1, col]
        ax.plot(profile.X, profile.total_Y, 'k-', linewidth=2, label='Experiment')
        ax.plot(pred_profile.X, pred_profile.total_Y, 'r--', linewidth=2, label='Prediction')
        ax.set_xlabel('Depth (m)')
        ax.set_ylabel('Deformation')
        ax.set_title('Deformation Profile')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('experiment_model_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n✓ Saved: experiment_model_comparison.png")

    return results


def analyze_error_by_parameter_bins(X: torch.Tensor, abs_errors: np.ndarray,
                                     experiment_params: np.ndarray):
    """Analyze which parameter bins have highest errors."""
    X_np = X.numpy()

    print("\n" + "="*100)
    print("📊 ERROR ANALYSIS BY PARAMETER BINS")
    print("="*100)

    for param_idx, param_name in enumerate(PARAM_NAMES):
        print(f"\n{param_name}:")
        print("-"*100)

        # Create 3 bins: LOW, MEDIUM, HIGH
        param_values = X_np[:, param_idx]
        param_min = RANGES[param_name][0]
        param_max = RANGES[param_name][1]

        # Define bin edges
        bin_edges = [param_min,
                     param_min + (param_max - param_min) / 3,
                     param_min + 2 * (param_max - param_min) / 3,
                     param_max]

        bin_names = ['LOW', 'MEDIUM', 'HIGH']

        # Assign samples to bins
        bin_indices = np.digitize(param_values, bin_edges[1:-1])

        # Calculate errors per bin
        for bin_idx, bin_name in enumerate(bin_names):
            mask = bin_indices == bin_idx
            count = np.sum(mask)

            if count > 0:
                bin_mae = np.mean(abs_errors[mask, param_idx])
                overall_mae = np.mean(abs_errors[:, param_idx])

                # Check if experiment falls in this bin
                exp_val = experiment_params[param_idx]
                exp_in_bin = bin_edges[bin_idx] <= exp_val < bin_edges[bin_idx + 1]
                marker = " ← EXPERIMENT HERE" if exp_in_bin else ""

                print(f"  {bin_name:<10} [{bin_edges[bin_idx]:.4e}, {bin_edges[bin_idx+1]:.4e}): "
                      f"{count:4d} samples | MAE = {bin_mae:.6e} ({bin_mae/overall_mae:.2f}x overall){marker}")

    print("="*100)


def main():
    # Load dataset
    print("📦 Loading dataset...")
    X, Y = load_dataset('datasets/dataset_10000_dl100_7d.pkl', use_full_curve=False)

    # 1. Analyze dataset coverage
    closest_indices, euclidean_distances = analyze_dataset_coverage(X, EXPERIMENT_PARAMS)

    # 2. Compare models on experiment
    results = compare_models_on_experiment(EXPERIMENT_PARAMS, MODELS)

    # 3. Analyze errors on full dataset for each model
    for name, path in MODELS.items():
        use_full = 'full' in path.lower()

        # Load appropriate dataset
        X_eval, Y_eval = load_dataset('datasets/dataset_10000_dl100_7d.pkl', use_full_curve=use_full)

        predictions, abs_errors, mean_abs_errors, dists = analyze_model_errors_on_dataset(
            path, X_eval, Y_eval, EXPERIMENT_PARAMS, use_full
        )

        # 4. Analyze by parameter bins
        analyze_error_by_parameter_bins(X_eval, abs_errors, EXPERIMENT_PARAMS)

    # Summary
    print("\n" + "="*100)
    print("💡 SUMMARY & RECOMMENDATIONS")
    print("="*100)

    print("\n1. COVERAGE:")
    print("   Check if experiment parameters fall in sparse regions of parameter space")

    print("\n2. MODEL COMPARISON:")
    print("   'ideal' epoch: Better for THIS experiment")
    print("   'final' epoch: Better for dataset OVERALL")
    print("   → This is EXPECTED when experiment is in sparse/edge region")

    print("\n3. SOLUTION:")
    print("   a) Augment dataset with more samples near experiment parameters")
    print("   b) Use ensemble of models from different epochs")
    print("   c) Fine-tune final model on experiment region")
    print("   d) Use 'ideal' checkpoint for this specific case")

    print("\n4. DIAGNOSIS:")
    if euclidean_distances[0] > 1.0:
        print("   ⚠️  SPARSE REGION: Closest sample is far (distance > 1.0)")
        print("   → Model extrapolating, not interpolating")
    else:
        print("   ✓ GOOD COVERAGE: Experiment has nearby training samples")

    print("="*100)


if __name__ == "__main__":
    main()
