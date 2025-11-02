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
    use_full_curve,
    loss_weights,
    seed,
    augmented_sampling=False,
    augmentation_factor=2,
    focus_params=[5, 6],  # L2, Rp2
    load_checkpoint_path=None  # NEW: Path to pre-trained model for fine-tuning
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
        use_full_curve: If True, use full curve without cropping
        loss_weights: Tensor of weights for each parameter [7]
        seed: Random seed for reproducibility
        augmented_sampling: If True, duplicate edge samples for better coverage
        augmentation_factor: How many times to duplicate edge samples
        focus_params: List of parameter indices to focus on for augmented sampling
        load_checkpoint_path: Path to pre-trained model checkpoint for fine-tuning
    """
    print("=" * 70)
    print("XRD CNN TRAINING")
    print("=" * 70)

    # Setup
    set_seed(seed)
    device = get_device()
    X, Y = load_dataset(Path(data_path), use_full_curve=use_full_curve)
    n = X.size(0)

    # Move loss weights to device
    loss_weights = loss_weights.to(device)

    # Train/val split
    idx = torch.randperm(n)
    n_val = int(val_split * n)

    # Optional: cap validation set size (set max_val_samples=None for percentage-based split)
    if max_val_samples is not None:
        n_val = min(n_val, max_val_samples)

    # Ensure at least 1 sample for validation and at least 1 for training
    n_val = max(1, min(n_val, n - 1))

    tr_idx, va_idx = idx[n_val:], idx[:n_val]
    print(f"\n📊 Data split (before augmentation):")
    print(f"   Train: {len(tr_idx)} samples ({100 * len(tr_idx) / n:.1f}%)")
    print(f"   Val:   {len(va_idx)} samples ({100 * len(va_idx) / n:.1f}%)")

    # Get train/val splits
    X_train, Y_train = X[tr_idx], Y[tr_idx]
    X_val, Y_val = X[va_idx], Y[va_idx]

    # Augmented Sampling: ONLY augment training set (keep validation original)
    if augmented_sampling:
        print(f"\n🔬 Augmented Sampling:")
        print(f"   Focus params: {[PARAM_NAMES[i] for i in focus_params]}")
        print(f"   Augmentation factor: {augmentation_factor}x")

        # Find edge samples in TRAINING set
        from model_common import RANGES
        edge_mask = torch.zeros(len(X_train), dtype=torch.bool)

        for param_idx in focus_params:
            param_values = X_train[:, param_idx]
            param_range = RANGES[PARAM_NAMES[param_idx]]

            threshold_low = param_range[0] + 0.2 * \
                (param_range[1] - param_range[0])
            threshold_high = param_range[1] - \
                0.2 * (param_range[1] - param_range[0])

            edge_mask |= (param_values < threshold_low) | (
                param_values > threshold_high)

        edge_indices = torch.where(edge_mask)[0]
        print(
            f"   Edge samples found: {len(edge_indices)} ({100*len(edge_indices)/len(X_train):.1f}%)")

        # Duplicate edge samples
        X_train_augmented = torch.cat(
            [X_train] + [X_train[edge_indices]] * (augmentation_factor - 1))
        Y_train_augmented = torch.cat(
            [Y_train] + [Y_train[edge_indices]] * (augmentation_factor - 1))

        print(
            f"   Augmented train set: {len(X_train_augmented)} samples (+{len(X_train_augmented) - len(X_train)})")

        X_train, Y_train = X_train_augmented, Y_train_augmented

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

    # Create model
    model = XRDRegressor().to(device)

    # Load pre-trained checkpoint if fine-tuning
    if load_checkpoint_path is not None:
        print(f"\n🔄 Fine-tuning mode: Loading pre-trained model")
        print(f"   Checkpoint: {load_checkpoint_path}")
        checkpoint = torch.load(load_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        print(f"   ✅ Loaded model from epoch {checkpoint.get('epoch', 'N/A')}")
        print(
            f"   ✅ Pre-trained val_loss: {checkpoint.get('val_loss', 'N/A'):.5f}")

    print(f"\n🧠 Model: XRDRegressor")
    print(f"   Parameters to predict: {PARAM_NAMES}")
    print(f"   Output dim: {len(PARAM_NAMES)}")

    # Check if weighted or unweighted
    is_weighted = not torch.allclose(
        loss_weights, torch.ones_like(loss_weights))

    print(f"\n⚖️  Loss Configuration:")
    if is_weighted:
        print(f"   WEIGHTED loss: {loss_weights.cpu().numpy()}")
        print(f"   Higher weights for challenging parameters (L2, Rp2)")
    else:
        print(f"   UNWEIGHTED loss: {loss_weights.cpu().numpy()}")
        print(f"   All parameters have equal importance")
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
    # CONFIGURATION
    # =============================================================================
    #
    # IMPROVEMENTS v3 (Ziegler-Inspired Architecture):
    # - K=15 kernel size (from Ziegler et al., up from K=7)
    # - Progressive channel expansion: 32→48→64→96→128→128 (vs constant 64)
    # - Attention-based pooling (preserves spatial info for Rp2)
    # - 6 residual blocks with dilations up to 32 (RF >100% of curve with K=15)
    # - Deeper MLP: 128→256→128→7
    # - Physics-constrained loss (D01≤Dmax1, Rp1≤L1, etc.)
    #
    # Expected improvements over v2:
    # - Rp2: 12.36% → 7-9% (K=15 + progressive channels)
    # - L2: 5.86% → 3.5-4.5% (better feature extraction)
    #
    # v2 Results (100k samples):
    # - Rp2: 12.36%, L2: 5.86%, Val loss: 0.01301
    # =============================================================================

    # =============================================================================
    # TRAINING MODE FLAGS
    # =============================================================================

    # Fine-tuning: Load pre-trained model and continue training on targeted dataset
    # FINE_TUNING = False  # Set True to enable fine-tuning
    FINE_TUNING = True  # Uncomment to enable fine-tuning

    # Pre-trained model path (only used if FINE_TUNING=True)
    PRETRAINED_MODEL_PATH = "checkpoints/dataset_200000_dl100_unweighted_full_augmented.pt"

    # Weighted loss: Use parameter-specific weights vs equal weights
    # WEIGHTED_TRAINING = False  # Unweighted baseline
    WEIGHTED_TRAINING = False

    # Full curve training (no cropping)
    # FULL_CURVE_TRAINING = True # Enable for full curve training
    FULL_CURVE_TRAINING = True

    # Log-space transformation: Apply log10 to curves before normalization
    USE_LOG_SPACE = True  # ⚠️ CRITICAL for XRD! Model v3 trained with log_space=True
    # USE_LOG_SPACE = False  # Linear space (not recommended for XRD curves)

    # Augmented Sampling: Duplicate edge samples for better coverage of sensitive regions
    # ⚠️ IMPORTANT: Split happens BEFORE augmentation (validation remains clean)
    AUGMENTED_SAMPLING = True  # Set True to enable
    AUGMENTATION_FACTOR = 2     # How many times to duplicate edge samples
    FOCUS_PARAMS = [5, 6]       # L2, Rp2 (most sensitive parameters)

    # =============================================================================
    # LOSS WEIGHTS
    # =============================================================================
    # Order: Dmax1, D01, L1, Rp1, D02, L2, Rp2

    if WEIGHTED_TRAINING:
        # Higher weights for challenging parameters (L2, Rp2)
        LOSS_WEIGHTS = torch.tensor([1.0, 1.2, 1.0, 1.0, 1.5, 2.0, 2.5])
        AUGMENTED_SAMPLING = False
    else:
        # Unweighted: all parameters equal importance
        LOSS_WEIGHTS = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    # =============================================================================
    # DATASET AND MODEL PATH
    # =============================================================================

    # Dataset selection
    if FINE_TUNING:
        # Use targeted dataset for fine-tuning
        DATA_PATH = "datasets/dataset_10000_dl100_targeted_std15.0.pkl"
        DATASET_NAME = PRETRAINED_MODEL_PATH.split('/')[-1].split('_dl100')[0]
    else:
        # Use general dataset for training from scratch
        # DATA_PATH = "datasets/dataset_1000_dl100_7d.pkl"   # For debugging
        DATA_PATH = "datasets/dataset_200000_dl100_7d.pkl"  # For quick testing
        # DATA_PATH = "datasets/dataset_100000_dl100_7d.pkl"  # For mid
        DATASET_NAME = DATA_PATH.split(
            '/')[-1].replace('.pkl', '').replace('_7d', '').replace('_targeted_std15.0', '')

    # Build model path with suffixes based on training mode
    model_suffix = ""
    if not WEIGHTED_TRAINING:
        model_suffix += "_unweighted"
    if FULL_CURVE_TRAINING:
        model_suffix += "_full"
    if AUGMENTED_SAMPLING:
        model_suffix += "_augmented"
    if FINE_TUNING:
        model_suffix += "_finetuned"

    MODEL_PATH = f"checkpoints/{DATASET_NAME}{model_suffix}.pt"

    # Training hyperparameters
    if FINE_TUNING:
        # Fine-tuning: fewer epochs, lower learning rate
        EPOCHS = 50
        LEARNING_RATE = 1e-4  # 10× lower than normal training
    else:
        # Normal training from scratch
        EPOCHS = 100  # Full training for larger model
        # EPOCHS = 20  # Quick test of v3 architecture
        LEARNING_RATE = 0.0015  # 0.0015

    BATCH_SIZE = 256  # 128/256/512
    WEIGHT_DECAY = 5e-4

    VAL_SPLIT = 0.05  # 5% validation (50k samples for 1M dataset)
    MAX_VAL_SAMPLES = None  # No cap - use percentage-based split
    # MAX_VAL_SAMPLES = 50000  # Optional: cap at 50k if dataset is huge

    SEED = 1234

    # =============================================================================
    # RUN TRAINING
    # =============================================================================

    print(f"\n{'='*70}")
    print(f"TRAINING CONFIGURATION SUMMARY")
    print(f"{'='*70}")
    print(f"Fine-tuning: {FINE_TUNING}")
    if FINE_TUNING:
        print(f"   Pre-trained model: {PRETRAINED_MODEL_PATH}")
    print(f"Dataset: {DATA_PATH}")
    print(f"Model: {MODEL_PATH}")
    print(f"Weighted loss: {WEIGHTED_TRAINING}")
    print(f"Full curve: {FULL_CURVE_TRAINING}")
    print(f"Log-space: {USE_LOG_SPACE}")
    print(f"Augmented sampling: {AUGMENTED_SAMPLING}")
    if AUGMENTED_SAMPLING:
        print(f"   Factor: {AUGMENTATION_FACTOR}x")
        print(f"   Focus params: {[PARAM_NAMES[i] for i in FOCUS_PARAMS]}")
    print(f"Loss weights: {LOSS_WEIGHTS.tolist()}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Epochs: {EPOCHS}")
    print(f"{'='*70}\n")

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
        use_full_curve=FULL_CURVE_TRAINING,
        loss_weights=LOSS_WEIGHTS,
        seed=SEED,
        augmented_sampling=AUGMENTED_SAMPLING,
        augmentation_factor=AUGMENTATION_FACTOR,
        focus_params=FOCUS_PARAMS,
        load_checkpoint_path=PRETRAINED_MODEL_PATH if FINE_TUNING else None
    )
