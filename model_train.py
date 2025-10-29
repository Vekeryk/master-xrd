"""
Train XRD CNN Model
===================
Train a 1D CNN regressor on pickled XRD dataset.

Usage:
    python train.py

The script will:
- Load dataset from hardcoded path
- Train model with hardcoded hyperparameters
- Save best model to checkpoints/
- Display training progress
"""

import time
from pathlib import Path
import torch
import torch.nn.functional as F
from model_common import (
    XRDRegressor,
    PickleXRDDataset,
    load_dataset,
    set_seed,
    get_device,
    PARAM_NAMES
)


# =============================================================================
# TRAINING FUNCTION
# =============================================================================

def train(
    data_path,
    model_path,
    epochs,
    batch_size,
    learning_rate,
    weight_decay,
    val_split,
    max_val_samples,
    use_log_space,
    seed
):
    """
    Main training loop.

    Args:
        data_path: Path to dataset pickle file
        model_path: Path to save model checkpoint
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Initial learning rate
        weight_decay: L2 regularization weight
        val_split: Fraction of data for validation (0.0 to 1.0)
        max_val_samples: Maximum number of validation samples
        use_log_space: Apply log10 transformation to curves
        seed: Random seed for reproducibility
    """
    print("=" * 70)
    print("XRD CNN TRAINING")
    print("=" * 70)

    # Setup
    set_seed(seed)
    device = get_device()
    X, Y = load_dataset(Path(data_path))
    n = X.size(0)

    # Weighted loss for parameter importance
    # Order: Dmax1, D01, L1, Rp1, D02, L2, Rp2
    # Higher weights for parameters with poor performance (L2: 22.72%, Rp2: 39.25%)
    loss_weights = torch.tensor(
        [1.0, 1.5, 1.0, 1.0, 2.0, 2.5, 3.5],
        device=device
    )

    # Train/val split
    idx = torch.randperm(n)
    n_val = int(val_split * n)
    n_val = max(1, min(n_val, n - 1, max_val_samples))
    tr_idx, va_idx = idx[n_val:], idx[:n_val]
    print(f"\n📊 Data split:")
    print(f"   Train: {len(tr_idx)} samples")
    print(f"   Val:   {len(va_idx)} samples")

    # Create datasets
    ds_tr = PickleXRDDataset(X[tr_idx], Y[tr_idx],
                             log_space=use_log_space, train=True)
    ds_va = PickleXRDDataset(X[va_idx], Y[va_idx],
                             log_space=use_log_space, train=False)

    # Create dataloaders
    dl_tr = torch.utils.data.DataLoader(
        ds_tr, batch_size=batch_size, shuffle=True)
    dl_va = torch.utils.data.DataLoader(
        ds_va, batch_size=batch_size, shuffle=False)

    # Create model
    model = XRDRegressor().to(device)
    print(f"\n🧠 Model: XRDRegressor")
    print(f"   Parameters to predict: {PARAM_NAMES}")
    print(f"   Output dim: {len(PARAM_NAMES)}")
    print(f"\n⚖️  Weighted Loss Configuration:")
    print(f"   Weights: {loss_weights.cpu().numpy()}")
    print(f"   (Higher weights for L2 and Rp2 to improve their accuracy)")

    # Optimizer and scheduler
    opt = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )

    # Training state
    best_val_loss = float("inf")
    Path("checkpoints").mkdir(exist_ok=True)

    print(f"\n🚀 Starting training for {epochs} epochs...")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Weight decay: {weight_decay}")
    print("-" * 70)

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # === TRAINING ===
        model.train()
        train_loss_sum = 0.0

        for y, t in dl_tr:
            y, t = y.to(device), t.to(device)

            # Forward pass
            p = model(y)

            # Compute weighted loss (higher weight for L2 and Rp2)
            per_param_loss = F.smooth_l1_loss(p, t, reduction='none')
            loss = (loss_weights * per_param_loss).mean()

            # Backward pass
            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss_sum += loss.item() * y.size(0)

        train_loss = train_loss_sum / len(ds_tr)

        # === VALIDATION ===
        model.eval()
        val_loss_sum = 0.0

        with torch.no_grad():
            for y, t in dl_va:
                y, t = y.to(device), t.to(device)
                p = model(y)
                per_param_loss = F.smooth_l1_loss(p, t, reduction='none')
                weighted_loss = (loss_weights * per_param_loss).mean()
                val_loss_sum += weighted_loss.item() * y.size(0)

        val_loss = val_loss_sum / len(ds_va)

        # Update scheduler
        sched.step(val_loss)
        curr_lr = opt.param_groups[0]['lr']

        # Print progress
        print(f"Epoch {epoch:03d}/{epochs} | "
              f"train: {train_loss:.5f} | "
              f"val: {val_loss:.5f} | "
              f"lr: {curr_lr:.2e}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model": model.state_dict(),
                "L": Y.size(1),
                "epoch": epoch,
                "val_loss": val_loss
            }, model_path)
            print(f"   → Saved best model (val_loss: {val_loss:.5f})")

    elapsed = time.time() - start_time

    print("-" * 70)
    print(f"✅ Training completed!")
    print(f"   Total time: {elapsed/60:.2f} minutes")
    print(f"   Best val loss: {best_val_loss:.5f}")
    print(f"   Model saved to: {model_path}")
    print("=" * 70)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Hardcoded configuration values

    # ===========================

    # 100,000 samples, dl=400
    DATA_PATH = "datasets/dataset_100000_dl400.pkl"

    # 10,000 samples, dl=400
    # DATA_PATH = "datasets/dataset_10000_dl400.pkl"

    # 1,000 samples, dl=400
    # DATA_PATH = "datasets/dataset_1000_dl400.pkl"

    # ===========================

    DATASET_NAME = DATA_PATH.split('/')[-1].replace('.pkl', '')
    MODEL_PATH = f"checkpoints/{DATASET_NAME}.pt"

    # EPOCHS = 80
    EPOCHS = 150  # Increased from 80 for better convergence with weighted loss

    BATCH_SIZE = 128
    LEARNING_RATE = 0.0015
    WEIGHT_DECAY = 5e-4

    VAL_SPLIT = 0.2  # 20% validation
    MAX_VAL_SAMPLES = 1000  # Cap validation set size

    USE_LOG_SPACE = True  # Apply log10 to curves before normalization
    SEED = 1234

    # Run training
    train(
        data_path=DATA_PATH,
        model_path=MODEL_PATH,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        val_split=VAL_SPLIT,
        max_val_samples=MAX_VAL_SAMPLES,
        use_log_space=USE_LOG_SPACE,
        seed=SEED
    )
