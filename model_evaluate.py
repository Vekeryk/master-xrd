"""
Evaluate XRD CNN Model
======================
Evaluate trained model on dataset and show detailed metrics.

Usage:
    python evaluate.py

The script will:
- Load trained model from checkpoint
- Evaluate on full dataset
- Show MAE metrics per parameter
- Display random prediction examples
"""

from pathlib import Path
import torch
from tqdm import tqdm
from model_common import (
    XRDRegressor,
    NormalizedXRDDataset,
    load_dataset,
    denorm_params,
    get_device,
    PARAM_NAMES,
    RANGES
)


# =============================================================================
# EVALUATION FUNCTION
# =============================================================================

@torch.no_grad()
def evaluate(data_path, model_path, batch_size, use_log_space, show_examples):
    """
    Evaluate trained model and display metrics.

    Args:
        data_path: Path to dataset pickle file
        model_path: Path to trained model checkpoint
        batch_size: Batch size for evaluation
        use_log_space: Apply log10 transformation (must match training)
        show_examples: Number of random examples to display (0 to disable)
    """
    print("=" * 70)
    print("XRD CNN MODEL EVALUATION")
    print("=" * 70)

    # Setup
    device = get_device()
    X, Y = load_dataset(Path(data_path))

    # Load model
    print(f"\n📦 Loading model from: {model_path}")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model = XRDRegressor().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    if "epoch" in ckpt:
        print(
            f"   Checkpoint: epoch {ckpt['epoch']}, val_loss {ckpt.get('val_loss', 'N/A')}")

    # Create dataset
    ds = NormalizedXRDDataset(X, Y, log_space=use_log_space, train=False)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size)

    # Run predictions
    print(f"\n🔮 Running predictions on {len(ds)} samples...")
    preds = []

    for y, _ in tqdm(dl, desc="Evaluating", unit="batch"):
        p = model(y.to(device))
        preds.append(p.cpu())

    P = torch.cat(preds, dim=0)
    Theta_hat = denorm_params(P)

    # Calculate metrics
    abs_err = torch.abs(Theta_hat - X)  # [N, P]
    mae = abs_err.mean(dim=0)

    # MAE as percentage of parameter range
    rng = torch.tensor([float(RANGES[k][1] - RANGES[k][0])
                       for k in PARAM_NAMES])
    mae_pct_range = (mae / rng) * 100.0

    # MAE as percentage of mean true value
    mean_true = X.mean(dim=0)
    mae_pct_true = (mae / (mean_true + 1e-12)) * 100.0

    # === DISPLAY METRICS ===
    print("\n" + "=" * 70)
    print("📊 MAE BY PARAMETER")
    print("=" * 70)
    print(f"{'Parameter':<10} {'MAE (abs)':<15} {'% of range':<15} {'% of mean':<15}")
    print("-" * 70)

    for j, k in enumerate(PARAM_NAMES):
        print(
            f"{k:<10} {mae[j].item():<15.6e} {mae_pct_range[j].item():<15.2f} {mae_pct_true[j].item():<15.2f}")

    print("=" * 70)

    # === DISPLAY EXAMPLES ===
    if show_examples > 0:
        print(f"\n📋 RANDOM PREDICTION EXAMPLES (n={show_examples})")
        print("=" * 70)

        idx = torch.randint(0, X.size(0), (show_examples,))

        output_to_copy = []

        for i in idx.tolist():
            t = X[i]  # true
            p = Theta_hat[i]  # predicted
            e = (p - t)  # error

            # Format with appropriate precision
            t_str = ", ".join(
                f"{v:.6f}" if j in [0, 1, 4] else f"{v:.3e}"
                for j, v in enumerate(t.tolist())
            )
            p_str = ", ".join(
                f"{v:.6f}" if j in [0, 1, 4] else f"{v:.3e}"
                for j, v in enumerate(p.tolist())
            )
            e_str = ", ".join(f"{v:+.3e}" for v in e.tolist())

            print(f"\nSample #{i:05d}:")
            print(f"  True:      [{t_str}]")
            print(f"  Predicted: [{p_str}]")
            print(f"  Error:     [{e_str}]")

            output_to_copy.append(f"  ([{t_str}], [{p_str}]),")

        # Copyable output for further analysis
        if output_to_copy:
            print("\n" + "=" * 70)
            print("📋 COPYABLE OUTPUT (for error analysis)")
            print("=" * 70)
            for line in output_to_copy:
                print(line)

    print("\n" + "=" * 70)
    print("✅ Evaluation completed!")
    print("=" * 70)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # =============================================================================
    # CONFIGURATION
    # =============================================================================
    #
    # Evaluate Ziegler-inspired model (v3) with:
    # - K=15 kernel size (from Ziegler et al.)
    # - Progressive channel expansion: 32→48→64→96→128→128
    # - Attention pooling (preserves spatial info)
    # - 6 residual blocks with dilations up to 32
    # - Physics-constrained loss
    #
    # v2 Results (100k samples):
    # - Rp2: 12.36%, L2: 5.86%, Val loss: 0.01301
    #
    # Expected v3 Results:
    # - Rp2: 7-9%, L2: 3.5-4.5%
    # =============================================================================

    # Dataset selection (must match training dataset)
    # Full evaluation (compare with v2)
    DATA_PATH = "datasets/dataset_10000_dl100_7d.pkl"
    # DATA_PATH = "datasets/dataset_10000_dl100_jit.pkl"  # For quick testing v3
    # DATA_PATH = "datasets/dataset_1000_dl100_jit.pkl"   # For debugging

    DATASET_NAME = DATA_PATH.split('/')[-1].replace('.pkl', '')

    # Model path - use v3 for Ziegler-inspired model
    MODEL_PATH = f"checkpoints/{DATASET_NAME}_v3.pt"
    # MODEL_PATH = f"checkpoints/{DATASET_NAME}_v2.pt"  # v2 physics-informed (for comparison)
    # MODEL_PATH = f"checkpoints/{DATASET_NAME}.pt"  # Old baseline model (for comparison)

    BATCH_SIZE = 256
    USE_LOG_SPACE = True  # Must match training setting

    SHOW_EXAMPLES = 10  # Number of random examples to display (0 to disable)

    # Run evaluation
    evaluate(
        data_path=DATA_PATH,
        model_path=MODEL_PATH,
        batch_size=BATCH_SIZE,
        use_log_space=USE_LOG_SPACE,
        show_examples=SHOW_EXAMPLES
    )
