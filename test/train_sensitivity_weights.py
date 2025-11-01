#!/usr/bin/env python3
"""
APPROACH 1: Sensitivity-Aware Loss Weights
Weight parameters by their impact on curve reconstruction.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np

from model_common import XRDRegressor, NormalizedXRDDataset, PARAM_NAMES, load_dataset


# Pre-computed sensitivity weights (from measure_sensitivity.py results)
# Based on empirical measurement of curve reconstruction impact
# Higher weight = more sensitive parameter
SENSITIVITY_WEIGHTS = torch.tensor([
    0.5,   # Dmax1 (LOW sensitivity)
    0.7,   # D01 (LOW-MED sensitivity)
    1.2,   # L1 (MEDIUM sensitivity)
    1.8,   # Rp1 (HIGH sensitivity - position)
    0.9,   # D02 (MEDIUM sensitivity)
    2.0,   # L2 (HIGH sensitivity - thickness)
    2.5,   # Rp2 (VERY HIGH sensitivity - position + negative)
])


def weighted_mse_loss(pred, target, weights):
    """MSE loss with per-parameter weights."""
    squared_errors = (pred - target) ** 2
    weighted_errors = squared_errors * weights.to(pred.device)
    return torch.mean(weighted_errors)


def train_sensitivity_weighted(dataset_path, epochs=50, lr=1e-3, batch_size=32):
    """Training with sensitivity-aware loss weights."""

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"✓ Using device: {device}")

    # Load dataset
    print(f"\n📦 Loading dataset: {dataset_path}")
    X, Y = load_dataset(dataset_path, use_full_curve=False)

    # Split
    dataset = NormalizedXRDDataset(X, Y, log_space=True, train=True)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    val_dataset_clean = NormalizedXRDDataset(
        X[val_size:], Y[val_size:], log_space=True, train=False
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset_clean, batch_size=batch_size, shuffle=False)

    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Val:   {len(val_dataset)} samples")

    # Model
    model = XRDRegressor(n_out=7).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"\n🎯 Training APPROACH 1: Sensitivity-Aware Weights")
    print(f"   Weights: {SENSITIVITY_WEIGHTS.tolist()}")
    print(f"   Epochs: {epochs}")
    print(f"   LR: {lr}")

    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for batch_y, batch_x in train_loader:
            batch_y, batch_x = batch_y.to(device), batch_x.to(device)

            optimizer.zero_grad()
            pred = model(batch_y)
            loss = weighted_mse_loss(pred, batch_x, SENSITIVITY_WEIGHTS)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate (also with weighted loss for fair comparison)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_y, batch_x in val_loader:
                batch_y, batch_x = batch_y.to(device), batch_x.to(device)
                pred = model(batch_y)
                loss = weighted_mse_loss(pred, batch_x, SENSITIVITY_WEIGHTS)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs}: train={train_loss:.6f}, val={val_loss:.6f} "
                  f"(best={best_val_loss:.6f} @ {best_epoch+1})")

    print(f"\n✅ Training complete!")
    print(f"   Best val loss: {best_val_loss:.6f} at epoch {best_epoch+1}")

    # Save
    checkpoint = {
        'model': model.state_dict(),
        'epoch': epochs,
        'val_loss': best_val_loss,
        'history': history,
        'approach': 'sensitivity_weights',
        'weights': SENSITIVITY_WEIGHTS
    }

    save_path = 'checkpoints/approach_sensitivity_weights.pt'
    torch.save(checkpoint, save_path)
    print(f"   Saved: {save_path}")

    return best_val_loss, history


if __name__ == "__main__":
    train_sensitivity_weighted('datasets/dataset_1000_dl100_7d.pkl', epochs=50)
