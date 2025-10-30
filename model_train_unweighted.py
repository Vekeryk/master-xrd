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
    NormalizedXRDDataset,
    load_dataset,
    set_seed,
    get_device,
    PARAM_NAMES,
    physics_constrained_loss
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

    # UNWEIGHTED loss - all parameters equal importance
    # Testing baseline performance on improved 7D stratified dataset
    # Order: Dmax1, D01, L1, Rp1, D02, L2, Rp2
    loss_weights = torch.tensor(
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        device=device
    )
    print(f"\n⚖️  UNWEIGHTED Loss (Clean Baseline)")
    print(f"   All parameters have equal weight: {loss_weights.cpu().tolist()}")

    # Train/val split
    idx = torch.randperm(n)
    n_val = int(val_split * n)

    # Optional: cap validation set size (set max_val_samples=None for percentage-based split)
    if max_val_samples is not None:
        n_val = min(n_val, max_val_samples)

    # Ensure at least 1 sample for validation and at least 1 for training
    n_val = max(1, min(n_val, n - 1))

    tr_idx, va_idx = idx[n_val:], idx[:n_val]
    print(f"\n📊 Data split:")
    print(f"   Train: {len(tr_idx)} samples ({100 * len(tr_idx) / n:.1f}%)")
    print(f"   Val:   {len(va_idx)} samples ({100 * len(va_idx) / n:.1f}%)")

    # Create datasets
    ds_tr = NormalizedXRDDataset(X[tr_idx], Y[tr_idx],
                                 log_space=use_log_space, train=True)
    ds_va = NormalizedXRDDataset(X[va_idx], Y[va_idx],
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
    print(f"\n⚖️  Loss Configuration:")
    print(f"   Weights: {loss_weights.cpu().numpy()}")
    print(f"   UNWEIGHTED - все параметри рівноцінні")
    print(f"   Physics constraints: D01≤Dmax1, D01+D02≤0.03, Rp1≤L1, L2≤L1")

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
        train_constraint_sum = 0.0

        for y, t in dl_tr:
            y, t = y.to(device), t.to(device)

            # Forward pass
            p = model(y)

            # Compute physics-constrained loss
            loss, constraint_penalty = physics_constrained_loss(
                p, t, loss_weights, base_loss_fn=F.smooth_l1_loss
            )

            # Backward pass
            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss_sum += loss.item() * y.size(0)
            train_constraint_sum += constraint_penalty.item() * y.size(0)

        train_loss = train_loss_sum / len(ds_tr)
        train_constraint = train_constraint_sum / len(ds_tr)

        # === VALIDATION ===
        model.eval()
        val_loss_sum = 0.0
        val_constraint_sum = 0.0

        with torch.no_grad():
            for y, t in dl_va:
                y, t = y.to(device), t.to(device)
                p = model(y)
                loss, constraint_penalty = physics_constrained_loss(
                    p, t, loss_weights, base_loss_fn=F.smooth_l1_loss
                )
                val_loss_sum += loss.item() * y.size(0)
                val_constraint_sum += constraint_penalty.item() * y.size(0)

        val_loss = val_loss_sum / len(ds_va)
        val_constraint = val_constraint_sum / len(ds_va)

        # Update scheduler
        sched.step(val_loss)
        curr_lr = opt.param_groups[0]['lr']

        # Print progress (show constraint violations every 10 epochs)
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}/{epochs} | "
                  f"train: {train_loss:.5f} | "
                  f"val: {val_loss:.5f} | "
                  f"constraint: {val_constraint:.4f} | "
                  f"lr: {curr_lr:.2e}")
        else:
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
    # =============================================================================
    # CONFIGURATION - UNWEIGHTED BASELINE
    # =============================================================================
    #
    # Testing model on improved 7D stratified dataset WITHOUT weighted loss
    #
    # Architecture (Ziegler-Inspired):
    # - K=15 kernel size
    # - Progressive channel expansion: 32→48→64→96→128→128
    # - Attention-based pooling
    # - 6 residual blocks with dilations up to 32
    # - Deeper MLP: 128→256→128→7
    # - Physics-constrained loss (D01≤Dmax1, Rp1≤L1, etc.)
    # - **UNWEIGHTED loss: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]**
    #
    # Goal: Evaluate if improved 7D stratified dataset removes need for weighted loss
    # =============================================================================

    # Dataset selection - NEW 7D stratified dataset
    DATA_PATH = "datasets/dataset_10000_dl100_7d_20251030_124511.pkl"

    DATASET_NAME = DATA_PATH.split('/')[-1].replace('.pkl', '')
    MODEL_PATH = f"checkpoints/{DATASET_NAME}_unweighted.pt"  # Clean baseline без weighted loss

    # Training hyperparameters
    EPOCHS = 100  # Full training for larger model
    # EPOCHS = 20  # Quick test of v3 architecture
    BATCH_SIZE = 256  # 128/256/512
    LEARNING_RATE = 0.0015  # 0.0015
    WEIGHT_DECAY = 5e-4

    VAL_SPLIT = 0.05  # 5% validation (50k samples for 1M dataset)
    MAX_VAL_SAMPLES = None  # No cap - use percentage-based split
    # MAX_VAL_SAMPLES = 50000  # Optional: cap at 50k if dataset is huge

    USE_LOG_SPACE = True  # Apply log10 to curves (critical for XRD!)
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
