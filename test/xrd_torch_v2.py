#!/usr/bin/env python3
"""
Simplified Differentiable XRD (PyTorch)
========================================

PRAGMATIC APPROACH:
Instead of porting 1000+ lines of complex dynamical diffraction theory,
we create a fast differentiable SURROGATE MODEL:

1. Train a small CNN to approximate xrd.compute_curve_and_profile()
2. Use this surrogate in training with curve loss
3. Much faster and fully differentiable

This is Option 4 from our earlier discussion - "Surrogate Model"
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
import pickle
from pathlib import Path


class XRDSurrogate(nn.Module):
    """
    Fast differentiable surrogate for XRD curve generation.

    Architecture: params [7] → curve [701]
    """

    def __init__(self, hidden_dims=[128, 256, 512, 1024, 701]):
        super().__init__()

        # Encoder: params → hidden
        layers = []
        prev_dim = 7
        for hidden_dim in hidden_dims[:-1]:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim

        # Output: hidden → curve
        layers.append(nn.Linear(prev_dim, hidden_dims[-1]))

        self.network = nn.Sequential(*layers)

    def forward(self, params):
        """
        Args:
            params: [B, 7] or [7] normalized parameters

        Returns:
            curve: [B, 701] or [701] XRD curve (log space, normalized)
        """
        was_1d = params.dim() == 1
        if was_1d:
            params = params.unsqueeze(0)

        curve = self.network(params)

        if was_1d:
            curve = curve.squeeze(0)

        return curve


def train_surrogate(dataset_path='datasets/dataset_10000_dl100_7d.pkl',
                   epochs=100,
                   batch_size=64,
                   lr=1e-3):
    """
    Train surrogate model to approximate xrd.compute_curve_and_profile().

    This is a ONE-TIME training to create the differentiable approximation.
    """
    import xrd
    from model_common import load_dataset, PARAM_NAMES, RANGES

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Training XRD Surrogate on {device}")

    # Load dataset
    print(f"\nLoading {dataset_path}...")
    X, Y = load_dataset(dataset_path, use_full_curve=True)  # Full curve!

    print(f"Dataset: {len(X)} samples, {Y.shape[1]} curve points")

    # Normalize parameters to [0, 1]
    X_norm = torch.zeros_like(X)
    for i, param in enumerate(PARAM_NAMES):
        min_val, max_val = RANGES[param]
        X_norm[:, i] = (X[:, i] - min_val) / (max_val - min_val)

    # Log-space normalize curves
    Y_log = torch.log10(Y + 1e-10)
    Y_norm = (Y_log - Y_log.min(dim=1, keepdim=True)[0]) / \
             (Y_log.max(dim=1, keepdim=True)[0] - Y_log.min(dim=1, keepdim=True)[0] + 1e-10)

    # Split
    train_size = int(0.8 * len(X))
    indices = torch.randperm(len(X))
    train_idx = indices[:train_size]
    val_idx = indices[train_size:]

    X_train, Y_train = X_norm[train_idx], Y_norm[train_idx]
    X_val, Y_val = X_norm[val_idx], Y_norm[val_idx]

    # Model
    model = XRDSurrogate().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')

    print(f"\nTraining for {epochs} epochs...")

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0

        for i in range(0, len(X_train), batch_size):
            batch_x = X_train[i:i+batch_size].to(device)
            batch_y = Y_train[i:i+batch_size].to(device)

            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= (len(X_train) // batch_size)

        # Validate
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for i in range(0, len(X_val), batch_size):
                batch_x = X_val[i:i+batch_size].to(device)
                batch_y = Y_val[i:i+batch_size].to(device)

                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                val_loss += loss.item()

        val_loss /= (len(X_val) // batch_size)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'xrd_surrogate.pt')

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs}: train={train_loss:.6f}, val={val_loss:.6f} (best={best_val_loss:.6f})")

    print(f"\n✅ Training complete! Best val loss: {best_val_loss:.6f}")
    print(f"   Saved: xrd_surrogate.pt")

    return model


def test_surrogate():
    """Test surrogate against original XRD."""
    import xrd
    from model_common import PARAM_NAMES, RANGES

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # Test parameters
    test_params_raw = [0.008094, 0.000943, 5200e-8, 3500e-8, 0.00255, 3000e-8, -50e-8]

    print("="*80)
    print("Testing XRD Surrogate vs Original")
    print("="*80)

    # Original
    print("\n1. Running original XRD...")
    curve_orig, _ = xrd.compute_curve_and_profile(test_params_raw, dl=100e-8)
    Y_orig = curve_orig.Y_R_vseZ  # Full curve

    # Surrogate
    print("\n2. Running surrogate...")

    # Normalize params
    test_params_norm = []
    for i, (param, val) in enumerate(zip(PARAM_NAMES, test_params_raw)):
        min_val, max_val = RANGES[param]
        norm_val = (val - min_val) / (max_val - min_val)
        test_params_norm.append(norm_val)

    test_params_torch = torch.tensor(test_params_norm, dtype=torch.float32).to(device)

    # Load model
    model = XRDSurrogate().to(device)
    model.load_state_dict(torch.load('xrd_surrogate.pt', map_location=device, weights_only=True))
    model.eval()

    with torch.no_grad():
        Y_surr_norm = model(test_params_torch).cpu().numpy()

    # Denormalize surrogate output
    # (Need to match original curve's scale)
    Y_orig_log = np.log10(Y_orig + 1e-10)
    Y_surr_log = Y_surr_norm * (Y_orig_log.max() - Y_orig_log.min()) + Y_orig_log.min()
    Y_surr = 10 ** Y_surr_log

    # Compare
    print("\n3. Comparison:")
    print(f"   Original range: [{Y_orig.min():.2e}, {Y_orig.max():.2e}]")
    print(f"   Surrogate range: [{Y_surr.min():.2e}, {Y_surr.max():.2e}]")

    # MSE in log space
    mse_log = np.mean((np.log10(Y_orig + 1e-10) - np.log10(Y_surr + 1e-10))**2)
    print(f"\n   MSE (log space): {mse_log:.6f}")

    if mse_log < 0.01:
        print("\n✅ PASS: Surrogate closely approximates original!")
    else:
        print("\n⚠️  Surrogate has some differences (may need more training)")

    return mse_log


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        print("\n" + "="*80)
        print("TRAINING XRD SURROGATE MODEL")
        print("="*80)
        print("\nThis creates a fast, differentiable approximation of XRD physics.")
        print("Once trained, use this for curve reconstruction loss in training.\n")

        train_surrogate(
            dataset_path='datasets/dataset_10000_dl100_7d.pkl',
            epochs=100
        )
    else:
        print("\n" + "="*80)
        print("XRD SURROGATE - USAGE")
        print("="*80)
        print("\nSTEP 1: Train surrogate (one-time):")
        print("   python xrd_torch_v2.py train")
        print("\nSTEP 2: Use in training:")
        print("   from xrd_torch_v2 import XRDSurrogate")
        print("   surrogate = XRDSurrogate()")
        print("   surrogate.load_state_dict(torch.load('xrd_surrogate.pt'))")
        print("   pred_curve = surrogate(predicted_params)")
        print("   loss = MSE(pred_curve, true_curve)")
        print("\n" + "="*80)
