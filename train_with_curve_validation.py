"""
Train XRD CNN Model with Curve Reconstruction Validation
=========================================================
Train model using parameter loss, but validate using CURVE RECONSTRUCTION quality.

Key innovation: Early stopping based on curve error, not param error!

This addresses the "good params → bad curve" problem by directly optimizing
for what we care about: curve reconstruction quality.

Strategy (Variant 1 - Validation Metric):
- Training: Fast parameter loss (no simulations)
- Validation: Curve reconstruction loss (2 sims per sample)
- Early stopping: Based on curve_error (not param_error)

Usage:
    python train_with_curve_validation.py
"""

import time
from pathlib import Path
import torch
import torch.nn.functional as F
from tqdm import tqdm
from model_common import (
    XRDRegressor,
    NormalizedXRDDataset,
    load_dataset,
    set_seed,
    get_device,
    PARAM_NAMES,
    physics_constrained_loss
)
from xrd_pytorch import CurveReconstructionLoss


# =============================================================================
# TRAINING FUNCTION WITH CURVE VALIDATION
# =============================================================================

def train_with_curve_validation(
    data_path,
    model_path_params,  # Best on params loss
    model_path_curve,   # Best on curve loss
    epochs,
    batch_size,
    learning_rate,
    weight_decay,
    val_split,
    use_log_space,
    use_full_curve,
    loss_weights,
    seed,
    c_rel=1e-3,  # SPSA step size for curve validation
    # Smaller batch for curve validation (2 sims per sample!)
    curve_val_batch_size=4
):
    """
    Train with parameter loss, but validate with curve reconstruction.

    Args:
        model_path_params: Path to save best model by params loss
        model_path_curve: Path to save best model by curve loss
        curve_val_batch_size: Batch size for curve validation (smaller because 2 sims/sample)
    """
    print("=" * 70)
    print("XRD CNN TRAINING WITH CURVE VALIDATION")
    print("=" * 70)

    # Setup
    set_seed(seed)
    device = get_device()
    X, Y = load_dataset(Path(data_path), crop_by_peak=True)
    n = X.size(0)

    # Move loss weights to device
    loss_weights = loss_weights.to(device)

    # Train/val split
    idx = torch.randperm(n)
    n_val = int(val_split * n)
    n_val = max(1, min(n_val, n - 1))

    tr_idx, va_idx = idx[n_val:], idx[:n_val]
    print(f"\n📊 Data split:")
    print(f"   Train: {len(tr_idx)} samples ({100 * len(tr_idx) / n:.1f}%)")
    print(f"   Val:   {len(va_idx)} samples ({100 * len(va_idx) / n:.1f}%)")

    # Get train/val splits
    X_train, Y_train = X[tr_idx], Y[tr_idx]
    X_val, Y_val = X[va_idx], Y[va_idx]

    # Create datasets
    ds_tr = NormalizedXRDDataset(X_train, Y_train,
                                 log_space=use_log_space, train=True)
    ds_va = NormalizedXRDDataset(X_val, Y_val,
                                 log_space=use_log_space, train=False)

    # Create dataloaders
    dl_tr = torch.utils.data.DataLoader(
        ds_tr, batch_size=batch_size, shuffle=True)
    dl_va = torch.utils.data.DataLoader(
        ds_va, batch_size=batch_size, shuffle=False)

    # Special dataloader for curve validation (smaller batch!)
    dl_va_curve = torch.utils.data.DataLoader(
        ds_va, batch_size=curve_val_batch_size, shuffle=False)

    # Create model
    model = XRDRegressor().to(device)

    print(f"\n🧠 Model: XRDRegressor")
    print(f"   Parameters to predict: {PARAM_NAMES}")

    # Check if weighted or unweighted
    is_weighted = not torch.allclose(
        loss_weights, torch.ones_like(loss_weights))

    print(f"\n⚖️  Loss Configuration:")
    if is_weighted:
        print(f"   WEIGHTED loss: {loss_weights.cpu().numpy()}")
    else:
        print(f"   UNWEIGHTED loss: {loss_weights.cpu().numpy()}")
    print(f"   Physics constraints: D01≤Dmax1, D01+D02≤0.03, Rp1≤L1, L2≤L1")

    print(f"\n🎯 Curve Validation:")
    print(f"   SPSA step size (c_rel): {c_rel}")
    print(f"   Curve val batch size: {curve_val_batch_size} (2 sims/sample)")
    print(f"   Total sims per epoch: {len(ds_va)} × 2 = {len(ds_va) * 2}")

    # Optimizer and scheduler
    opt = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # IMPORTANT: Schedule based on CURVE error (not param error!)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )

    # Curve reconstruction loss (for validation only, no gradients)
    curve_loss_fn = CurveReconstructionLoss(
        c_rel=c_rel,
        crop_start=40,
        crop_end=701,
        reduction='mean'
    )

    # Training state
    best_val_loss_params = float("inf")  # Best on params
    best_val_loss_curve = float("inf")   # Best on curve (for early stopping!)
    Path("checkpoints").mkdir(exist_ok=True)

    print(f"\n🚀 Starting training for {epochs} epochs...")
    print(f"   Batch size (train): {batch_size}")
    print(f"   Batch size (curve val): {curve_val_batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Weight decay: {weight_decay}")
    print(f"   Early stopping criterion: CURVE ERROR (not param error!)")
    print("-" * 70)

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # === TRAINING (params loss only, fast) ===
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

        # === VALIDATION (params loss - fast) ===
        model.eval()
        val_loss_params_sum = 0.0
        val_constraint_sum = 0.0

        with torch.no_grad():
            for y, t in dl_va:
                y, t = y.to(device), t.to(device)
                p = model(y)
                loss, constraint_penalty = physics_constrained_loss(
                    p, t, loss_weights, base_loss_fn=F.smooth_l1_loss
                )
                val_loss_params_sum += loss.item() * y.size(0)
                val_constraint_sum += constraint_penalty.item() * y.size(0)

        val_loss_params = val_loss_params_sum / len(ds_va)
        val_constraint = val_constraint_sum / len(ds_va)

        # === CURVE VALIDATION (slow, but critical!) ===
        # This is the KEY innovation: we validate on curve reconstruction quality!
        val_loss_curve_sum = 0.0

        print(f"   Computing curve validation (this may take ~30-60 sec)...",
              end=' ', flush=True)
        curve_val_start = time.time()

        with torch.no_grad():
            for y, t in tqdm(dl_va_curve, desc=f"Epoch {epoch} curve val", leave=False, disable=True):
                y, t = y.to(device), t.to(device)
                p = model(y)

                # CRITICAL: Curve reconstruction loss (2 sims per sample)
                # This directly measures "how well can we reconstruct the curve from predicted params"
                # Different seed each epoch
                curve_loss = curve_loss_fn(y, p, seed=epoch)
                val_loss_curve_sum += curve_loss.item() * y.size(0)

        val_loss_curve = val_loss_curve_sum / len(ds_va)
        curve_val_time = time.time() - curve_val_start
        print(f"done ({curve_val_time:.1f}s)")

        # Update scheduler based on CURVE error (not param error!)
        sched.step(val_loss_curve)
        curr_lr = opt.param_groups[0]['lr']

        # Print progress
        print(f"Epoch {epoch:03d}/{epochs} | "
              f"train: {train_loss:.5f} | "
              f"val_params: {val_loss_params:.5f} | "
              f"val_CURVE: {val_loss_curve:.5f} | "
              f"lr: {curr_lr:.2e}")

        # Save best model by PARAMS loss (for comparison)
        if val_loss_params < best_val_loss_params:
            best_val_loss_params = val_loss_params
            torch.save({
                "model": model.state_dict(),
                "L": Y.size(1),
                "epoch": epoch,
                "val_loss_params": val_loss_params,
                "val_loss_curve": val_loss_curve
            }, model_path_params)
            print(
                f"   → Saved best PARAMS model (params: {val_loss_params:.5f}, curve: {val_loss_curve:.5f})")

        # Save best model by CURVE loss (for early stopping!)
        if val_loss_curve < best_val_loss_curve:
            best_val_loss_curve = val_loss_curve
            torch.save({
                "model": model.state_dict(),
                "L": Y.size(1),
                "epoch": epoch,
                "val_loss_params": val_loss_params,
                "val_loss_curve": val_loss_curve
            }, model_path_curve)
            print(
                f"   → Saved best CURVE model (params: {val_loss_params:.5f}, curve: {val_loss_curve:.5f}) ⭐")

    elapsed = time.time() - start_time

    print("-" * 70)
    print(f"✅ Training completed!")
    print(f"   Total time: {elapsed/60:.2f} minutes")
    print(f"\n📊 Best models:")
    print(
        f"   Best by PARAMS loss: {best_val_loss_params:.5f} → {model_path_params}")
    print(
        f"   Best by CURVE loss:  {best_val_loss_curve:.5f} → {model_path_curve} ⭐")
    print(f"\n💡 Recommendation: Use CURVE model for inference!")
    print("=" * 70)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # =============================================================================
    # CONFIGURATION
    # =============================================================================

    # Dataset
    DATA_PATH = "datasets/dataset_1000_dl100_targeted.pkl"
    DATASET_NAME = Path(DATA_PATH).stem

    # Model paths
    MODEL_PATH_PARAMS = f"checkpoints/{DATASET_NAME}_curve_val_best_params.pt"
    MODEL_PATH_CURVE = f"checkpoints/{DATASET_NAME}_curve_val_best_curve.pt"

    # Training hyperparameters
    EPOCHS = 100
    BATCH_SIZE = 32
    # BATCH_SIZE = 128
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    VAL_SPLIT = 0.2
    SEED = 42

    # Preprocessing
    USE_LOG_SPACE = False   # CRITICAL: Must match model v3.1
    # Use cropped [40:701] = 661 points (MUST match SPSA!)
    USE_FULL_CURVE = False

    # Loss configuration
    WEIGHTED_TRAINING = False  # Empirically proven: unweighted is better!

    if WEIGHTED_TRAINING:
        # Weighted (DEPRECATED - worse performance)
        LOSS_WEIGHTS = torch.tensor([1.0, 1.2, 1.0, 1.0, 1.5, 2.0, 2.5])
    else:
        # Unweighted (RECOMMENDED)
        LOSS_WEIGHTS = torch.ones(7)

    # Curve validation settings
    C_REL = 1e-3  # SPSA step size
    CURVE_VAL_BATCH_SIZE = 4  # Small batch because 2 sims/sample

    print("\n" + "=" * 70)
    print("CONFIGURATION SUMMARY")
    print("=" * 70)
    print(f"Dataset: {DATA_PATH}")
    print(f"Training strategy: Parameter loss (fast)")
    print(f"Validation strategy: Curve reconstruction (2 sims/sample)")
    print(f"Early stopping: Based on CURVE error ⭐")
    print(f"\nHyperparameters:")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size (train): {BATCH_SIZE}")
    print(f"  Batch size (curve val): {CURVE_VAL_BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Weight decay: {WEIGHT_DECAY}")
    print(f"  Val split: {VAL_SPLIT}")
    print(f"  Weighted loss: {WEIGHTED_TRAINING}")
    print(f"  SPSA c_rel: {C_REL}")
    print("=" * 70 + "\n")

    # Run training
    train_with_curve_validation(
        data_path=DATA_PATH,
        model_path_params=MODEL_PATH_PARAMS,
        model_path_curve=MODEL_PATH_CURVE,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        val_split=VAL_SPLIT,
        use_log_space=USE_LOG_SPACE,
        use_full_curve=USE_FULL_CURVE,
        loss_weights=LOSS_WEIGHTS,
        seed=SEED,
        c_rel=C_REL,
        curve_val_batch_size=CURVE_VAL_BATCH_SIZE
    )
