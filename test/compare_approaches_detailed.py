#!/usr/bin/env python3
"""
Compare Per-Parameter Errors Across All Approaches
===================================================

Runs model_evaluate logic on all 6 trained approaches and compares:
- Per-parameter MAE (absolute)
- Per-parameter MAE (% of range)
- Overall validation loss
- Parameter rankings (which approach is best for each param)
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd

from model_common import (
    XRDRegressor, NormalizedXRDDataset, load_dataset,
    denorm_params, PARAM_NAMES, RANGES
)

# Try to load multi-task and hierarchical models too
from train_multitask import MultiTaskXRDRegressor
from train_hierarchical import HierarchicalXRDRegressor
from train_attention import AttentionXRDRegressor


APPROACHES = [
    {
        'name': 'Baseline',
        'checkpoint': 'checkpoints/approach_baseline.pt',
        'model_class': XRDRegressor,
        'description': 'Standard unweighted'
    },
    {
        'name': 'Augmented Sampling',
        'checkpoint': 'checkpoints/approach_augmented_sampling.pt',
        'model_class': XRDRegressor,
        'description': 'Edge region augmentation'
    },
    {
        'name': 'Curve Loss',
        'checkpoint': 'checkpoints/approach_curve_loss.pt',
        'model_class': XRDRegressor,
        'description': 'Surrogate curve reconstruction'
    },
    {
        'name': 'Sensitivity Weights',
        'checkpoint': 'checkpoints/approach_sensitivity_weights.pt',
        'model_class': XRDRegressor,
        'description': 'Weighted loss'
    },
    {
        'name': 'Multi-Task',
        'checkpoint': 'checkpoints/approach_multitask.pt',
        'model_class': MultiTaskXRDRegressor,
        'description': 'Params + residuals'
    },
    {
        'name': 'Hierarchical',
        'checkpoint': 'checkpoints/approach_hierarchical.pt',
        'model_class': HierarchicalXRDRegressor,
        'description': 'Coarse-to-fine'
    },
    {
        'name': 'Attention',
        'checkpoint': 'checkpoints/approach_attention.pt',
        'model_class': AttentionXRDRegressor,
        'description': 'Multi-head attention'
    },
]


@torch.no_grad()
def evaluate_approach(approach_config, dataset_path, use_log_space=True):
    """Evaluate single approach and return per-parameter MAE."""

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # Load dataset
    X, Y = load_dataset(dataset_path, use_full_curve=False)

    # Load model
    checkpoint_path = approach_config['checkpoint']
    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return None

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Instantiate correct model class
    model_class = approach_config['model_class']
    if model_class == XRDRegressor:
        model = model_class(n_out=7).to(device)
    elif model_class == MultiTaskXRDRegressor:
        model = model_class(n_params=7, curve_length=651).to(device)
    elif model_class == HierarchicalXRDRegressor:
        model = model_class(n_params=7, sensitive_indices=[3, 5, 6]).to(device)
    elif model_class == AttentionXRDRegressor:
        model = model_class(n_params=7, num_heads=4).to(device)
    else:
        raise ValueError(f"Unknown model class: {model_class}")

    model.load_state_dict(ckpt['model'])
    model.eval()

    # Create dataset (validation split)
    ds = NormalizedXRDDataset(X, Y, log_space=use_log_space, train=False)

    # Use same val split as training (last 20%)
    train_size = int(0.8 * len(ds))
    val_indices = list(range(train_size, len(ds)))
    val_dataset = torch.utils.data.Subset(ds, val_indices)

    dl = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Run predictions
    preds = []
    for y, _ in dl:
        y = y.to(device)

        # Handle different model outputs
        if isinstance(model, MultiTaskXRDRegressor):
            p, _ = model(y)  # Returns (params, residual)
        elif isinstance(model, HierarchicalXRDRegressor):
            p = model(y, return_coarse=False)  # Returns refined params
        else:
            p = model(y)

        preds.append(p.cpu())

    P = torch.cat(preds, dim=0)
    Theta_hat = denorm_params(P)

    # Get true values for validation set
    X_val = X[val_indices]

    # Calculate per-parameter MAE
    abs_err = torch.abs(Theta_hat - X_val)
    mae = abs_err.mean(dim=0).numpy()

    # MAE as percentage of range
    rng = np.array([float(RANGES[k][1] - RANGES[k][0]) for k in PARAM_NAMES])
    mae_pct_range = (mae / rng) * 100.0

    # Calculate validation loss (normalized parameter MSE)
    val_loss = ckpt.get('val_loss', None)

    return {
        'name': approach_config['name'],
        'mae': mae,
        'mae_pct_range': mae_pct_range,
        'val_loss': val_loss,
        'description': approach_config['description']
    }


def main():
    dataset_path = 'datasets/dataset_1000_dl100_7d.pkl'

    print("=" * 80)
    print("DETAILED COMPARISON: Per-Parameter Errors Across All Approaches")
    print("=" * 80)
    print(f"\nDataset: {dataset_path}")
    print(f"Evaluating {len(APPROACHES)} approaches...\n")

    # Evaluate all approaches
    results = []
    for approach in APPROACHES:
        print(f"Evaluating: {approach['name']}")
        result = evaluate_approach(approach, dataset_path)
        if result is not None:
            results.append(result)

    if not results:
        print("❌ No results to compare!")
        return

    print(f"\n✓ Successfully evaluated {len(results)} approaches\n")

    # ========== CREATE COMPARISON TABLES ==========

    # Table 1: Overall validation loss
    print("=" * 80)
    print("TABLE 1: Overall Validation Loss")
    print("=" * 80)
    print(f"{'Rank':<6} {'Approach':<25} {'Val Loss':<12} {'Description':<30}")
    print("-" * 80)

    sorted_results = sorted(results, key=lambda x: x['val_loss'])
    for rank, r in enumerate(sorted_results, 1):
        marker = "🏆" if rank == 1 else "  "
        print(f"{marker} {rank:<4} {r['name']:<25} {r['val_loss']:<12.6f} {r['description']:<30}")

    # Table 2: Per-parameter MAE (absolute)
    print("\n" + "=" * 80)
    print("TABLE 2: Per-Parameter MAE (Absolute)")
    print("=" * 80)

    # Create DataFrame
    mae_data = {r['name']: r['mae'] for r in results}
    df_mae = pd.DataFrame(mae_data, index=PARAM_NAMES)

    print(df_mae.to_string())

    # Table 3: Per-parameter MAE (% of range)
    print("\n" + "=" * 80)
    print("TABLE 3: Per-Parameter MAE (% of Parameter Range)")
    print("=" * 80)

    mae_pct_data = {r['name']: r['mae_pct_range'] for r in results}
    df_mae_pct = pd.DataFrame(mae_pct_data, index=PARAM_NAMES)

    print(df_mae_pct.to_string())

    # Table 4: Best approach for each parameter
    print("\n" + "=" * 80)
    print("TABLE 4: Best Approach for Each Parameter")
    print("=" * 80)
    print(f"{'Parameter':<10} {'Best Approach':<25} {'MAE':<15} {'% of Range':<15}")
    print("-" * 80)

    for i, param in enumerate(PARAM_NAMES):
        # Find approach with lowest MAE for this parameter
        best_result = min(results, key=lambda r: r['mae'][i])
        best_mae = best_result['mae'][i]
        best_pct = best_result['mae_pct_range'][i]

        print(f"{param:<10} {best_result['name']:<25} {best_mae:<15.6f} {best_pct:<15.2f}%")

    # Table 5: Parameter rankings (how many params each approach wins on)
    print("\n" + "=" * 80)
    print("TABLE 5: Parameter Win Count")
    print("=" * 80)
    print(f"{'Approach':<25} {'Params Won':<15} {'Parameters':<40}")
    print("-" * 80)

    win_count = {r['name']: 0 for r in results}
    win_params = {r['name']: [] for r in results}

    for i, param in enumerate(PARAM_NAMES):
        best_result = min(results, key=lambda r: r['mae'][i])
        win_count[best_result['name']] += 1
        win_params[best_result['name']].append(param)

    sorted_wins = sorted(win_count.items(), key=lambda x: x[1], reverse=True)
    for name, count in sorted_wins:
        params_str = ', '.join(win_params[name]) if win_params[name] else 'None'
        marker = "🏆" if count == max(win_count.values()) else "  "
        print(f"{marker} {name:<25} {count:<15} {params_str:<40}")

    # Table 6: Relative improvement vs baseline
    print("\n" + "=" * 80)
    print("TABLE 6: Improvement vs Baseline (Per Parameter)")
    print("=" * 80)

    baseline_result = next((r for r in results if r['name'] == 'Baseline'), None)
    if baseline_result:
        improvement_data = {}
        for r in results:
            if r['name'] == 'Baseline':
                continue
            # Calculate % improvement (negative = worse)
            improvement = ((baseline_result['mae'] - r['mae']) / baseline_result['mae']) * 100
            improvement_data[r['name']] = improvement

        df_improvement = pd.DataFrame(improvement_data, index=PARAM_NAMES)
        print(df_improvement.to_string())
        print("\nNote: Positive = improvement, Negative = worse than baseline")

    # Save to CSV
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)

    df_mae.to_csv(output_dir / 'comparison_mae_absolute.csv')
    df_mae_pct.to_csv(output_dir / 'comparison_mae_percentage.csv')
    if baseline_result:
        df_improvement.to_csv(output_dir / 'comparison_vs_baseline.csv')

    print(f"\n✓ Results saved to {output_dir}/")
    print("  - comparison_mae_absolute.csv")
    print("  - comparison_mae_percentage.csv")
    print("  - comparison_vs_baseline.csv")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    best_overall = sorted_results[0]
    print(f"🏆 Best overall (val loss): {best_overall['name']} ({best_overall['val_loss']:.6f})")

    best_param_winner = sorted_wins[0]
    print(f"🎯 Best per-parameter: {best_param_winner[0]} (wins on {best_param_winner[1]}/7 params)")

    # Check if they're the same
    if best_overall['name'] == best_param_winner[0]:
        print(f"\n✅ CONSENSUS: {best_overall['name']} is best on both metrics!")
    else:
        print(f"\n⚠️  SPLIT: {best_overall['name']} has lowest val loss, "
              f"but {best_param_winner[0]} wins most parameters")


if __name__ == "__main__":
    main()
