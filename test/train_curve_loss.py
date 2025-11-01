#!/usr/bin/env python3
"""
APPROACH 7: Curve Reconstruction Loss (via Surrogate)
======================================================

Uses trained XRD surrogate for differentiable curve loss.

Loss = alpha * MSE(predicted_params, true_params) + beta * MSE(curve(predicted), curve(true))

This directly optimizes curve reconstruction, not just parameter accuracy.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np

from model_common import (
    XRDRegressor, NormalizedXRDDataset, load_dataset,
    denorm_params, PARAM_NAMES, RANGES
)
from xrd_torch_v2 import XRDSurrogate


def train_curve_loss(dataset_path, epochs=50, lr=1e-3, batch_size=32,
                     alpha=0.7, beta=0.3):
    """
    Train with combined parameter + curve loss.

    Args:
        alpha: Weight for parameter loss (0.7 = 70%)
        beta: Weight for curve loss (0.3 = 30%)
    """

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"✓ Using device: {device}")

    # Load dataset
    print(f"\n📦 Loading dataset: {dataset_path}")
    X, Y = load_dataset(dataset_path, use_full_curve=False)  # Cropped for main model

    # Load FULL curves for surrogate
    X_full, Y_full = load_dataset(dataset_path, use_full_curve=True)

    print(f"\n🎯 APPROACH 7: Curve Reconstruction Loss (Surrogate)")
    print(f"   Loss = {alpha:.1f} * param_loss + {beta:.1f} * curve_loss")

    # Split
    dataset = NormalizedXRDDataset(X, Y, log_space=True, train=True)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Val:   {len(val_dataset)} samples")

    # Load surrogate for curve generation
    print(f"\n🔧 Loading XRD surrogate...")
    surrogate = XRDSurrogate().to(device)
    surrogate.load_state_dict(torch.load('xrd_surrogate.pt', map_location=device, weights_only=True))
    surrogate.eval()  # Freeze surrogate
    for param in surrogate.parameters():
        param.requires_grad = False
    print(f"   ✓ Surrogate loaded and frozen")

    # Main model
    model = XRDRegressor(n_out=7).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    param_criterion = nn.MSELoss()
    curve_criterion = nn.MSELoss()

    print(f"   Epochs: {epochs}")
    print(f"   LR: {lr}")

    best_val_loss = float('inf')
    history = {
        'train_loss': [], 'val_loss': [],
        'train_param_loss': [], 'train_curve_loss': [],
        'val_param_loss': [], 'val_curve_loss': []
    }

    # Normalize params for surrogate input
    def normalize_params(params_denorm):
        """Normalize to [0, 1] for surrogate."""
        params_norm = torch.zeros_like(params_denorm)
        for i, param_name in enumerate(PARAM_NAMES):
            min_val, max_val = RANGES[param_name]
            params_norm[:, i] = (params_denorm[:, i] - min_val) / (max_val - min_val)
        return params_norm

    # Pre-compute normalized true curves for training
    print(f"\n📊 Pre-computing true curves...")
    true_curves_train = []
    true_curves_val = []

    # Get indices from subsets
    train_indices = train_dataset.indices
    val_indices = val_dataset.indices

    with torch.no_grad():
        # Train curves
        X_train_denorm = X[train_indices]
        X_train_norm = normalize_params(X_train_denorm).to(device)
        for i in range(0, len(X_train_norm), batch_size):
            batch = X_train_norm[i:i+batch_size]
            curves = surrogate(batch)
            true_curves_train.append(curves.cpu())
        true_curves_train = torch.cat(true_curves_train)

        # Val curves
        X_val_denorm = X[val_indices]
        X_val_norm = normalize_params(X_val_denorm).to(device)
        for i in range(0, len(X_val_norm), batch_size):
            batch = X_val_norm[i:i+batch_size]
            curves = surrogate(batch)
            true_curves_val.append(curves.cpu())
        true_curves_val = torch.cat(true_curves_val)

    print(f"   ✓ Pre-computed {len(true_curves_train)} train + {len(true_curves_val)} val curves")

    # Training loop
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        train_param_loss = 0
        train_curve_loss = 0

        batch_idx = 0
        for batch_y, batch_x in train_loader:
            batch_y, batch_x = batch_y.to(device), batch_x.to(device)

            # Get corresponding true curves
            start_idx = batch_idx * batch_size
            end_idx = start_idx + len(batch_x)
            batch_true_curves = true_curves_train[start_idx:end_idx].to(device)

            optimizer.zero_grad()

            # Forward pass
            pred_params_norm = model(batch_y)

            # Loss 1: Parameter loss (normalized space)
            loss_params = param_criterion(pred_params_norm, batch_x)

            # Loss 2: Curve loss
            # Denormalize predicted params
            pred_params_denorm = denorm_params(pred_params_norm)
            # Normalize for surrogate
            pred_params_for_surrogate = normalize_params(pred_params_denorm)
            # Generate predicted curves
            pred_curves = surrogate(pred_params_for_surrogate)
            # Compute curve loss
            loss_curve = curve_criterion(pred_curves, batch_true_curves)

            # Combined loss
            loss = alpha * loss_params + beta * loss_curve

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_param_loss += loss_params.item()
            train_curve_loss += loss_curve.item()

            batch_idx += 1

        train_loss /= len(train_loader)
        train_param_loss /= len(train_loader)
        train_curve_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0
        val_param_loss = 0
        val_curve_loss = 0

        batch_idx = 0
        with torch.no_grad():
            for batch_y, batch_x in val_loader:
                batch_y, batch_x = batch_y.to(device), batch_x.to(device)

                # Get corresponding true curves
                start_idx = batch_idx * batch_size
                end_idx = start_idx + len(batch_x)
                batch_true_curves = true_curves_val[start_idx:end_idx].to(device)

                pred_params_norm = model(batch_y)

                # Parameter loss
                loss_params = param_criterion(pred_params_norm, batch_x)

                # Curve loss
                pred_params_denorm = denorm_params(pred_params_norm)
                pred_params_for_surrogate = normalize_params(pred_params_denorm)
                pred_curves = surrogate(pred_params_for_surrogate)
                loss_curve = curve_criterion(pred_curves, batch_true_curves)

                # Combined
                loss = alpha * loss_params + beta * loss_curve

                val_loss += loss.item()
                val_param_loss += loss_params.item()
                val_curve_loss += loss_curve.item()

                batch_idx += 1

        val_loss /= len(val_loader)
        val_param_loss /= len(val_loader)
        val_curve_loss /= len(val_loader)

        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_param_loss'].append(train_param_loss)
        history['train_curve_loss'].append(train_curve_loss)
        history['val_param_loss'].append(val_param_loss)
        history['val_curve_loss'].append(val_curve_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs}: "
                  f"train={train_loss:.6f} (p={train_param_loss:.6f}, c={train_curve_loss:.6f}), "
                  f"val={val_loss:.6f} (p={val_param_loss:.6f}, c={val_curve_loss:.6f}) "
                  f"[best={best_val_loss:.6f} @ {best_epoch+1}]")

    print(f"\n✅ Training complete!")
    print(f"   Best val loss: {best_val_loss:.6f} at epoch {best_epoch+1}")
    print(f"   Final param loss: {val_param_loss:.6f}")
    print(f"   Final curve loss: {val_curve_loss:.6f}")

    # Save
    checkpoint = {
        'model': model.state_dict(),
        'epoch': epochs,
        'val_loss': best_val_loss,
        'history': history,
        'approach': 'curve_loss',
        'alpha': alpha,
        'beta': beta
    }

    save_path = 'checkpoints/approach_curve_loss.pt'
    torch.save(checkpoint, save_path)
    print(f"   Saved: {save_path}")

    return best_val_loss, history


if __name__ == "__main__":
    train_curve_loss(
        'datasets/dataset_1000_dl100_7d.pkl',
        epochs=50,
        alpha=0.7,  # 70% parameter loss
        beta=0.3    # 30% curve loss
    )
