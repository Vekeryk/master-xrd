"""
Обчислення детальних метрик моделі
"""
import sys
from pathlib import Path
import torch
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model_common import (
    XRDRegressor,
    NormalizedXRDDataset,
    load_dataset,
    denorm_params,
    get_device,
    PARAM_NAMES,
    RANGES
)


@torch.no_grad()
def compute_metrics(data_path, model_path, use_log_space=True, use_full_curve=False, batch_size=32):
    """Обчислення детальних метрик"""

    device = get_device()
    print(f"Using device: {device}")

    # Load data
    print(f"\nЗавантаження даних з {data_path}...")
    X, Y = load_dataset(Path(data_path), use_full_curve=use_full_curve)
    print(f"Завантажено {len(X)} зразків")

    # Load model
    print(f"\nЗавантаження моделі з {model_path}...")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    model = XRDRegressor().to(device)
    state_dict = ckpt["model"]
    if "hann_window" in state_dict:
        state_dict.pop("hann_window")

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Create dataset
    ds = NormalizedXRDDataset(X, Y, log_space=use_log_space, train=False)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size)

    # Run predictions
    print("\nВиконання передбачень...")
    preds = []

    for i, (y, _) in enumerate(dl):
        if i % 100 == 0:
            print(f"  Batch {i}/{len(dl)}")
        p = model(y.to(device))
        preds.append(p.cpu())

    P = torch.cat(preds, dim=0)
    Theta_hat = denorm_params(P)

    # Calculate detailed metrics
    print("\nОбчислення метрик...")

    abs_err = torch.abs(Theta_hat - X)  # [N, P]
    squared_err = (Theta_hat - X) ** 2

    # Basic metrics
    mae = abs_err.mean(dim=0)
    rmse = torch.sqrt(squared_err.mean(dim=0))
    max_err = abs_err.max(dim=0)[0]
    median_err = abs_err.median(dim=0)[0]
    std_err = abs_err.std(dim=0)

    # R² score
    ss_res = squared_err.sum(dim=0)
    ss_tot = ((X - X.mean(dim=0)) ** 2).sum(dim=0)
    r2 = 1 - (ss_res / (ss_tot + 1e-12))

    # Percentage metrics
    rng = torch.tensor([float(RANGES[k][1] - RANGES[k][0]) for k in PARAM_NAMES])
    mean_true = X.mean(dim=0)

    mae_pct_range = (mae / rng) * 100.0
    mae_pct_mean = (mae / (mean_true.abs() + 1e-12)) * 100.0
    rmse_pct_range = (rmse / rng) * 100.0

    # Print results
    print("\n" + "="*100)
    print("ДЕТАЛЬНІ МЕТРИКИ ЯКОСТІ МОДЕЛІ")
    print("="*100)
    print(f"{'Параметр':<10} {'MAE':<12} {'RMSE':<12} {'MaxErr':<12} {'Median':<12} {'Std':<12} {'R²':<8}")
    print("-"*100)

    for j, k in enumerate(PARAM_NAMES):
        print(f"{k:<10} {mae[j].item():<12.3e} {rmse[j].item():<12.3e} "
              f"{max_err[j].item():<12.3e} {median_err[j].item():<12.3e} "
              f"{std_err[j].item():<12.3e} {r2[j].item():<8.4f}")

    print("\n" + "="*100)
    print("ВІДНОСНІ МЕТРИКИ (% від діапазону та середнього)")
    print("="*100)
    print(f"{'Параметр':<10} {'MAE % rng':<12} {'RMSE % rng':<12} {'MAE % mean':<12}")
    print("-"*100)

    for j, k in enumerate(PARAM_NAMES):
        print(f"{k:<10} {mae_pct_range[j].item():<12.2f} "
              f"{rmse_pct_range[j].item():<12.2f} {mae_pct_mean[j].item():<12.2f}")

    print("="*100)

    # Return metrics as dict
    metrics = {}
    for j, k in enumerate(PARAM_NAMES):
        metrics[k] = {
            'mae': mae[j].item(),
            'rmse': rmse[j].item(),
            'max_err': max_err[j].item(),
            'median_err': median_err[j].item(),
            'std_err': std_err[j].item(),
            'r2': r2[j].item(),
            'mae_pct_range': mae_pct_range[j].item(),
            'rmse_pct_range': rmse_pct_range[j].item(),
            'mae_pct_mean': mae_pct_mean[j].item()
        }

    return metrics


if __name__ == '__main__':
    DATA_PATH = "datasets/dataset_100000_dl100_7d.pkl"
    MODEL_PATH = "checkpoints/100000_log_best_params.pt"

    metrics = compute_metrics(
        data_path=DATA_PATH,
        model_path=MODEL_PATH,
        use_log_space=True,
        use_full_curve=False,
        batch_size=32
    )

    # Save metrics
    import pickle
    output_file = Path(__file__).parent.parent / 'content' / 'detailed_metrics.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(metrics, f)
    print(f"\n✓ Метрики збережено у {output_file}")
