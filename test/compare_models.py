#!/usr/bin/env python3
"""
Ultimate Model Comparison Script
Compares multiple trained models on the same dataset with comprehensive metrics.
"""

import torch
import numpy as np
import pickle
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple
import sys

# Import from model_common
from model_common import (
    XRDRegressor, NormalizedXRDDataset, PARAM_NAMES,
    RANGES, load_dataset
)


@dataclass
class ModelConfig:
    """Configuration for a model to evaluate."""
    name: str
    checkpoint_path: str
    use_full_curve: bool
    description: str


def load_model_and_predict(
    config: ModelConfig,
    X: torch.Tensor,
    Y: torch.Tensor,
    device: torch.device
) -> Tuple[torch.Tensor, Dict]:
    """
    Load a model and run predictions.

    Returns:
        predictions: Tensor of shape (N, 7)
        metadata: Dict with model info (epoch, val_loss, etc.)
    """
    print(f"\n{'='*70}")
    print(f"🔮 Evaluating: {config.name}")
    print(f"   Description: {config.description}")
    print(f"   Checkpoint: {config.checkpoint_path}")
    print(f"   Use full curve: {config.use_full_curve}")

    # Load checkpoint
    checkpoint = torch.load(config.checkpoint_path, map_location=device)
    epoch = checkpoint.get('epoch', 'unknown')
    val_loss = checkpoint.get('val_loss', 'unknown')

    print(f"   Epoch: {epoch}, Val Loss: {val_loss}")

    # Create model
    model = XRDRegressor(n_out=7).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # Create dataset (no augmentation for evaluation)
    dataset = NormalizedXRDDataset(X, Y, log_space=True, train=False)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=256, shuffle=False
    )

    # Run predictions
    predictions = []
    with torch.no_grad():
        for batch_y, batch_x in loader:
            batch_y = batch_y.to(device)
            pred = model(batch_y)
            predictions.append(pred.cpu())

    predictions = torch.cat(predictions, dim=0)

    metadata = {
        'epoch': epoch,
        'val_loss': val_loss,
        'input_dim': Y.shape[1],
    }

    return predictions, metadata


def calculate_metrics(
    X_true: torch.Tensor,
    X_pred: torch.Tensor
) -> Dict[str, np.ndarray]:
    """
    Calculate comprehensive error metrics.

    Returns dict with:
        - mae: Mean Absolute Error per parameter (7,)
        - mse: Mean Squared Error per parameter (7,)
        - rmse: Root Mean Squared Error per parameter (7,)
        - mape: Mean Absolute Percentage Error per parameter (7,)
        - abs_errors: Absolute errors per sample per parameter (N, 7)
        - rel_errors: Relative errors in % per sample per parameter (N, 7)
    """
    X_true_np = X_true.numpy()
    X_pred_np = X_pred.numpy()

    abs_errors = np.abs(X_pred_np - X_true_np)
    rel_errors = (X_pred_np - X_true_np) / (np.abs(X_true_np) + 1e-12) * 100

    mae = np.mean(abs_errors, axis=0)
    mse = np.mean((X_pred_np - X_true_np)**2, axis=0)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs(rel_errors), axis=0)

    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'mape': mape,
        'abs_errors': abs_errors,
        'rel_errors': rel_errors,
    }


def print_comparison_table(
    configs: List[ModelConfig],
    all_metrics: Dict[str, Dict],
    all_metadata: Dict[str, Dict]
):
    """Print comprehensive comparison table."""

    print("\n" + "="*120)
    print("📊 MODEL COMPARISON SUMMARY")
    print("="*120)

    # Training info table
    print("\n🔧 TRAINING INFO:")
    print("-"*120)
    print(f"{'Model':<30} {'Val Loss':<15} {'Epoch':<10} {'Input Dim':<12} {'Description':<30}")
    print("-"*120)
    for config in configs:
        meta = all_metadata[config.name]
        print(f"{config.name:<30} {meta['val_loss']:<15.6f} {meta['epoch']:<10} "
              f"{meta['input_dim']:<12} {config.description:<30}")
    print("-"*120)

    # MAE comparison
    print("\n📏 MEAN ABSOLUTE ERROR (MAE) COMPARISON:")
    print("-"*120)
    header = f"{'Parameter':<12}"
    for config in configs:
        header += f"{config.name:<20}"
    header += "{'Best Model':<25}"
    print(header)
    print("-"*120)

    for i, param_name in enumerate(PARAM_NAMES):
        row = f"{param_name:<12}"
        param_maes = []
        for config in configs:
            mae_val = all_metrics[config.name]['mae'][i]
            param_maes.append((mae_val, config.name))
            row += f"{mae_val:.6e}  "

        # Find best (lowest MAE)
        best_mae, best_model = min(param_maes)
        row += f"  ✓ {best_model}"
        print(row)

    print("-"*120)

    # MAPE comparison (Mean Absolute Percentage Error)
    print("\n📊 MEAN ABSOLUTE PERCENTAGE ERROR (MAPE) COMPARISON:")
    print("-"*120)
    header = f"{'Parameter':<12}"
    for config in configs:
        header += f"{config.name:<20}"
    header += "{'Best Model':<25}"
    print(header)
    print("-"*120)

    for i, param_name in enumerate(PARAM_NAMES):
        row = f"{param_name:<12}"
        param_mapes = []
        for config in configs:
            mape_val = all_metrics[config.name]['mape'][i]
            param_mapes.append((mape_val, config.name))
            row += f"{mape_val:>8.2f}%         "

        # Find best (lowest MAPE)
        best_mape, best_model = min(param_mapes)
        row += f"  ✓ {best_model}"
        print(row)

    print("-"*120)

    # Overall performance ranking
    print("\n🏆 OVERALL RANKING:")
    print("-"*120)
    rankings = []
    for config in configs:
        metrics = all_metrics[config.name]
        # Overall score: average MAPE across all parameters
        avg_mape = np.mean(metrics['mape'])
        # Also consider worst parameter (max MAPE)
        max_mape = np.max(metrics['mape'])
        rankings.append((config.name, avg_mape, max_mape, all_metadata[config.name]['val_loss']))

    rankings.sort(key=lambda x: x[1])  # Sort by avg MAPE

    print(f"{'Rank':<6} {'Model':<30} {'Avg MAPE':<15} {'Max MAPE':<15} {'Val Loss':<15}")
    print("-"*120)
    for rank, (name, avg_mape, max_mape, val_loss) in enumerate(rankings, 1):
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
        print(f"{medal} {rank:<3} {name:<30} {avg_mape:>8.2f}%      {max_mape:>8.2f}%      {val_loss:<15.6f}")

    print("-"*120)

    # Win count per model
    print("\n🎯 PARAMETER-WISE WINS:")
    print("-"*120)
    win_counts = {config.name: 0 for config in configs}

    for i, param_name in enumerate(PARAM_NAMES):
        param_maes = [(all_metrics[config.name]['mae'][i], config.name) for config in configs]
        _, best_model = min(param_maes)
        win_counts[best_model] += 1

    sorted_wins = sorted(win_counts.items(), key=lambda x: x[1], reverse=True)
    for model, wins in sorted_wins:
        print(f"{model:<30} {wins}/7 parameters")

    print("-"*120)


def statistical_comparison(
    configs: List[ModelConfig],
    X_true: torch.Tensor,
    all_predictions: Dict[str, torch.Tensor]
):
    """Perform statistical significance tests between models."""
    from scipy import stats

    print("\n" + "="*120)
    print("📈 STATISTICAL SIGNIFICANCE TESTS (Paired Tests)")
    print("="*120)
    print("Comparing each pair of models using Wilcoxon signed-rank test")
    print("(Tests if differences in errors are statistically significant)")
    print("-"*120)

    X_true_np = X_true.numpy()

    # Calculate mean absolute errors per sample for each model
    model_errors = {}
    for config in configs:
        pred_np = all_predictions[config.name].numpy()
        # Mean absolute error per sample (averaged across all 7 parameters)
        sample_errors = np.mean(np.abs(pred_np - X_true_np), axis=1)
        model_errors[config.name] = sample_errors

    # Pairwise comparisons
    model_names = [c.name for c in configs]
    for i, model1 in enumerate(model_names):
        for model2 in model_names[i+1:]:
            errors1 = model_errors[model1]
            errors2 = model_errors[model2]

            # Wilcoxon signed-rank test (paired non-parametric test)
            statistic, p_value = stats.wilcoxon(errors1, errors2)

            # Calculate mean difference
            mean_diff = np.mean(errors1 - errors2)

            # Determine significance
            if p_value < 0.001:
                sig = "***"
            elif p_value < 0.01:
                sig = "**"
            elif p_value < 0.05:
                sig = "*"
            else:
                sig = "n.s."

            better_model = model2 if mean_diff > 0 else model1

            print(f"\n{model1} vs {model2}:")
            print(f"  Mean difference: {mean_diff:.6e}")
            print(f"  p-value: {p_value:.6f} {sig}")
            print(f"  Better model: {better_model}")

    print("\n" + "-"*120)
    print("Significance levels: *** p<0.001, ** p<0.01, * p<0.05, n.s. = not significant")
    print("="*120)


def save_results(
    configs: List[ModelConfig],
    all_metrics: Dict[str, Dict],
    all_predictions: Dict[str, torch.Tensor],
    all_metadata: Dict[str, Dict],
    X_true: torch.Tensor,
    output_path: str = "comparison_results.pkl"
):
    """Save all results for further analysis."""
    results = {
        'configs': configs,
        'metrics': all_metrics,
        'predictions': all_predictions,
        'metadata': all_metadata,
        'X_true': X_true,
    }

    with open(output_path, 'wb') as f:
        pickle.dump(results, f)

    print(f"\n💾 Results saved to {output_path}")


def main():
    # Device setup
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("✓ Using CUDA")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("✓ Using MPS (Apple Silicon)")
    else:
        device = torch.device('cpu')
        print("✓ Using CPU")

    # Define models to compare
    DATASET_NAME = "dataset_10000_dl100_7d"

    configs = [
        ModelConfig(
            name="v3_unweighted_full",
            checkpoint_path=f"checkpoints/{DATASET_NAME}_v3_unweighted_full.pt",
            use_full_curve=True,
            description="Unweighted + Full curve"
        ),
        ModelConfig(
            name="v3_full",
            checkpoint_path=f"checkpoints/{DATASET_NAME}_v3_full.pt",
            use_full_curve=True,
            description="Weighted + Full curve"
        ),
        ModelConfig(
            name="v3_unweighted",
            checkpoint_path=f"checkpoints/{DATASET_NAME}_v3_unweighted.pt",
            use_full_curve=False,
            description="Unweighted + Cropped"
        ),
        ModelConfig(
            name="v3",
            checkpoint_path=f"checkpoints/{DATASET_NAME}_v3.pt",
            use_full_curve=False,
            description="Weighted + Cropped"
        ),
    ]

    # Load dataset
    print(f"\n📦 Loading dataset: datasets/{DATASET_NAME}.pkl")
    X_full, Y_full = load_dataset(f"datasets/{DATASET_NAME}.pkl", use_full_curve=True)
    X_crop, Y_crop = load_dataset(f"datasets/{DATASET_NAME}.pkl", use_full_curve=False)

    print(f"   Full: X={X_full.shape}, Y={Y_full.shape}")
    print(f"   Crop: X={X_crop.shape}, Y={Y_crop.shape}")

    # Run predictions for all models
    all_predictions = {}
    all_metadata = {}
    all_metrics = {}

    for config in configs:
        # Choose appropriate dataset
        X = X_full
        Y = Y_full if config.use_full_curve else Y_crop

        # Load and predict
        predictions, metadata = load_model_and_predict(config, X, Y, device)
        all_predictions[config.name] = predictions
        all_metadata[config.name] = metadata

        # Calculate metrics
        metrics = calculate_metrics(X, predictions)
        all_metrics[config.name] = metrics

        print(f"   ✓ MAE range: {metrics['mae'].min():.6e} - {metrics['mae'].max():.6e}")
        print(f"   ✓ MAPE range: {metrics['mape'].min():.2f}% - {metrics['mape'].max():.2f}%")

    # Print comparison tables
    print_comparison_table(configs, all_metrics, all_metadata)

    # Statistical comparison
    statistical_comparison(configs, X_full, all_predictions)

    # Save results
    save_results(configs, all_metrics, all_predictions, all_metadata, X_full)

    print("\n" + "="*120)
    print("✅ COMPARISON COMPLETED!")
    print("="*120)
    print("\n💡 RECOMMENDATIONS:")
    print("   1. Check the 'OVERALL RANKING' table for best performing model")
    print("   2. Review 'PARAMETER-WISE WINS' to see which model handles each parameter best")
    print("   3. Check statistical significance tests to ensure differences are real")
    print("   4. Use j_compare_models.ipynb for visual comparison of predictions")
    print("="*120)


if __name__ == "__main__":
    main()
