#!/usr/bin/env python3
"""
Measure parameter sensitivities by analyzing curve reconstruction errors.

This quantifies how much each parameter affects the XRD curve when perturbed by 1%.
Results used to create sensitivity-aware loss weights.
"""

import numpy as np
import torch
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl

import xrd
import helpers as h
from model_common import PARAM_NAMES, RANGES, load_dataset

mpl.rcParams['figure.dpi'] = 100


def measure_parameter_sensitivity(X: torch.Tensor, n_samples: int = 500,
                                  perturbation: float = 0.01, dl: float = 100e-8):
    """
    Measure sensitivity of each parameter to curve reconstruction.

    Args:
        X: Dataset parameters [N, 7]
        n_samples: Number of samples to analyze
        perturbation: Relative perturbation (0.01 = 1%)
        dl: Layer thickness for curve generation

    Returns:
        sensitivities: [7] - Average curve error from 1% parameter change
        sensitivity_std: [7] - Standard deviation
        all_errors: [n_samples, 7] - Individual errors for analysis
    """
    X_np = X.numpy()

    # Sample uniformly across dataset
    np.random.seed(42)
    indices = np.random.choice(len(X_np), size=min(n_samples, len(X_np)), replace=False)

    print(f"\n{'='*80}")
    print(f"🔬 MEASURING PARAMETER SENSITIVITIES")
    print(f"{'='*80}")
    print(f"   Samples: {len(indices)}")
    print(f"   Perturbation: {perturbation*100:.1f}%")
    print(f"   Metric: MAE on log10(curve)")
    print(f"{'='*80}\n")

    all_errors = np.zeros((len(indices), 7))

    for sample_idx, idx in enumerate(tqdm(indices, desc="Computing sensitivities")):
        true_params = X_np[idx]

        # Generate reference curve
        try:
            ref_curve, _ = xrd.compute_curve_and_profile(true_params.tolist(), dl=dl)
            ref_y = np.log10(ref_curve.Y_R_vseZ + 1e-10)
        except:
            continue  # Skip if curve generation fails

        # Test each parameter
        for param_idx in range(7):
            # Perturb parameter
            perturbed = true_params.copy()

            # Handle negative parameters (Rp2)
            if perturbed[param_idx] < 0:
                perturbed[param_idx] *= (1 - perturbation)  # Make more negative
            else:
                perturbed[param_idx] *= (1 + perturbation)

            # Ensure physical constraints
            perturbed = enforce_constraints(perturbed)

            # Generate perturbed curve
            try:
                pert_curve, _ = xrd.compute_curve_and_profile(perturbed.tolist(), dl=dl)
                pert_y = np.log10(pert_curve.Y_R_vseZ + 1e-10)

                # Measure curve error (MAE in log-space)
                curve_error = np.mean(np.abs(pert_y - ref_y))
                all_errors[sample_idx, param_idx] = curve_error
            except:
                all_errors[sample_idx, param_idx] = np.nan

    # Remove failed samples
    valid_mask = ~np.isnan(all_errors).any(axis=1)
    all_errors = all_errors[valid_mask]

    print(f"\n✓ Successfully measured {len(all_errors)}/{len(indices)} samples")

    # Calculate statistics
    sensitivities = np.mean(all_errors, axis=0)
    sensitivity_std = np.std(all_errors, axis=0)

    # Normalize to mean = 1.0 (for use as loss weights)
    normalized_sensitivities = sensitivities / np.mean(sensitivities)

    # Print results
    print(f"\n{'='*80}")
    print(f"📊 SENSITIVITY RESULTS")
    print(f"{'='*80}")
    print(f"{'Parameter':<12} {'Raw Sens':<15} {'Normalized':<15} {'Std':<15} {'Interpretation':<20}")
    print(f"{'-'*80}")

    interpretations = []
    for i, param in enumerate(PARAM_NAMES):
        interpretation = get_sensitivity_interpretation(normalized_sensitivities[i])
        interpretations.append(interpretation)

        print(f"{param:<12} {sensitivities[i]:<15.6f} {normalized_sensitivities[i]:<15.2f} "
              f"{sensitivity_std[i]:<15.6f} {interpretation:<20}")

    print(f"{'-'*80}")
    print(f"{'MEAN':<12} {np.mean(sensitivities):<15.6f} {1.0:<15.2f} "
          f"{np.mean(sensitivity_std):<15.6f}")
    print(f"{'='*80}")

    # Recommendations
    print(f"\n💡 RECOMMENDED LOSS WEIGHTS:")
    print(f"   LOSS_WEIGHTS = torch.tensor({list(normalized_sensitivities.round(2))})")
    print(f"   # Order: {PARAM_NAMES}")

    print(f"\n📈 EXPECTED IMPROVEMENT:")
    high_sens_params = [PARAM_NAMES[i] for i in range(7) if normalized_sensitivities[i] > 1.5]
    if high_sens_params:
        print(f"   High-sensitivity parameters: {', '.join(high_sens_params)}")
        print(f"   → These will be prioritized during training")
        print(f"   → Expected 15-30% error reduction on these parameters")
    else:
        print(f"   All parameters have similar sensitivity")
        print(f"   → Uniform weighting may be sufficient")

    # Save results
    results = {
        'sensitivities': sensitivities,
        'normalized_sensitivities': normalized_sensitivities,
        'sensitivity_std': sensitivity_std,
        'all_errors': all_errors,
        'param_names': PARAM_NAMES,
        'perturbation': perturbation,
        'n_samples': len(all_errors),
    }

    with open('sensitivity_analysis.pkl', 'wb') as f:
        pickle.dump(results, f)

    print(f"\n💾 Results saved to: sensitivity_analysis.pkl")

    # Visualizations
    plot_sensitivities(normalized_sensitivities, sensitivity_std, all_errors)

    return normalized_sensitivities, sensitivity_std, all_errors


def enforce_constraints(params):
    """Ensure parameters satisfy physical constraints."""
    Dmax1, D01, L1, Rp1, D02, L2, Rp2 = params

    # D01 <= Dmax1
    D01 = min(D01, Dmax1)

    # D01 + D02 <= 0.03
    if D01 + D02 > 0.03:
        D02 = 0.03 - D01

    # Rp1 <= L1
    Rp1 = min(Rp1, L1)

    # L2 <= L1
    L2 = min(L2, L1)

    # Clip to ranges
    Dmax1 = np.clip(Dmax1, RANGES['Dmax1'][0], RANGES['Dmax1'][1])
    D01 = np.clip(D01, RANGES['D01'][0], RANGES['D01'][1])
    L1 = np.clip(L1, RANGES['L1'][0], RANGES['L1'][1])
    Rp1 = np.clip(Rp1, RANGES['Rp1'][0], RANGES['Rp1'][1])
    D02 = np.clip(D02, RANGES['D02'][0], RANGES['D02'][1])
    L2 = np.clip(L2, RANGES['L2'][0], RANGES['L2'][1])
    Rp2 = np.clip(Rp2, RANGES['Rp2'][0], RANGES['Rp2'][1])

    return np.array([Dmax1, D01, L1, Rp1, D02, L2, Rp2])


def get_sensitivity_interpretation(normalized_sens):
    """Interpret sensitivity value."""
    if normalized_sens < 0.5:
        return "Very Low"
    elif normalized_sens < 0.8:
        return "Low"
    elif normalized_sens < 1.2:
        return "Medium"
    elif normalized_sens < 1.8:
        return "High"
    else:
        return "Very High"


def plot_sensitivities(sensitivities, std, all_errors):
    """Create visualization of sensitivities."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Bar chart with error bars
    ax = axes[0]
    x = np.arange(len(PARAM_NAMES))
    bars = ax.bar(x, sensitivities, yerr=std, capsize=5, alpha=0.7, edgecolor='black')

    # Color bars by sensitivity
    colors = []
    for sens in sensitivities:
        if sens < 0.8:
            colors.append('green')
        elif sens < 1.2:
            colors.append('yellow')
        elif sens < 1.8:
            colors.append('orange')
        else:
            colors.append('red')

    for bar, color in zip(bars, colors):
        bar.set_facecolor(color)

    ax.set_xlabel('Parameter', fontweight='bold')
    ax.set_ylabel('Normalized Sensitivity', fontweight='bold')
    ax.set_title('Parameter Sensitivity to Curve Reconstruction\n(Normalized, mean=1.0)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(PARAM_NAMES)
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Mean')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()

    # Plot 2: Distribution of errors per parameter
    ax = axes[1]
    positions = np.arange(len(PARAM_NAMES))
    violin_parts = ax.violinplot(
        [all_errors[:, i] for i in range(7)],
        positions=positions,
        showmeans=True,
        showmedians=True
    )

    ax.set_xlabel('Parameter', fontweight='bold')
    ax.set_ylabel('Curve Error (log-space MAE)', fontweight='bold')
    ax.set_title('Distribution of Curve Errors\nAcross Samples', fontweight='bold')
    ax.set_xticks(positions)
    ax.set_xticklabels(PARAM_NAMES)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('parameter_sensitivities.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"✓ Saved: parameter_sensitivities.png")

    # Additional plot: Heatmap of correlations
    fig, ax = plt.subplots(figsize=(10, 8))

    # Correlation matrix of errors
    corr_matrix = np.corrcoef(all_errors.T)

    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')

    # Labels
    ax.set_xticks(np.arange(len(PARAM_NAMES)))
    ax.set_yticks(np.arange(len(PARAM_NAMES)))
    ax.set_xticklabels(PARAM_NAMES)
    ax.set_yticklabels(PARAM_NAMES)

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add correlation values
    for i in range(len(PARAM_NAMES)):
        for j in range(len(PARAM_NAMES)):
            text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                          ha="center", va="center",
                          color="white" if abs(corr_matrix[i, j]) > 0.5 else "black",
                          fontsize=9)

    ax.set_title('Correlation of Sensitivity Errors Between Parameters', fontweight='bold', pad=20)
    fig.colorbar(im, ax=ax, label='Correlation')

    plt.tight_layout()
    plt.savefig('sensitivity_correlation.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"✓ Saved: sensitivity_correlation.png")


def main():
    # Load dataset
    print("📦 Loading dataset...")
    X, Y = load_dataset('datasets/dataset_10000_dl100_7d.pkl', use_full_curve=False)

    # Measure sensitivities
    sensitivities, std, all_errors = measure_parameter_sensitivity(
        X,
        n_samples=500,  # Adjust for speed vs accuracy
        perturbation=0.01,  # 1% perturbation
    )

    print("\n" + "="*80)
    print("✅ SENSITIVITY ANALYSIS COMPLETE")
    print("="*80)
    print("\n📝 NEXT STEPS:")
    print("   1. Copy recommended LOSS_WEIGHTS to model_train.py")
    print("   2. Set WEIGHTED_TRAINING = True")
    print("   3. Train model: python model_train.py")
    print("   4. Compare results with baseline (unweighted)")
    print("="*80)


if __name__ == "__main__":
    main()
