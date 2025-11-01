#!/usr/bin/env python3
"""
Training with Curve Reconstruction Validation
==============================================

Trains with standard parameter MSE loss (fast, differentiable),
but validates using BOTH parameter and curve reconstruction errors.

This tests whether parameter loss is a good proxy for curve loss.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from tqdm import tqdm
import time

from model_common import (
    XRDRegressor, NormalizedXRDDataset, load_dataset,
    denorm_params, PARAM_NAMES, RANGES
)
import xrd


def compute_curve_error(pred_params_np, true_params_np, dl=100e-8):
    """
    Compute curve reconstruction MSE.

    Args:
        pred_params_np: Predicted params [7] (NumPy, denormalized)
        true_params_np: True params [7] (NumPy, denormalized)
        dl: Layer thickness

    Returns:
        curve_mse: Mean squared error in log space
    """
    # Generate curves
    pred_curve, _ = xrd.compute_curve_and_profile(pred_params_np.tolist(), dl=dl)
    true_curve, _ = xrd.compute_curve_and_profile(true_params_np.tolist(), dl=dl)

    # Use Y_R_vseZ (full curve) then apply same cropping as ML_Y
    pred_y_full = pred_curve.Y_R_vseZ
    true_y_full = true_curve.Y_R_vseZ

    # Apply cropping [50:701] to match training
    pred_y = pred_y_full[50:701]
    true_y = true_y_full[50:701]

    # Log space (same preprocessing as model training)
    pred_y_log = np.log10(pred_y + 1e-10)
    true_y_log = np.log10(true_y + 1e-10)

    # MSE in log space (no normalization - we want to capture differences)
    mse = np.mean((pred_y_log - true_y_log) ** 2)

    return float(mse)


def validate_with_curves(model, val_loader, device, max_samples=200):
    """
    Validate with both parameter and curve errors.

    Args:
        max_samples: Limit samples for speed (curve gen is slow)

    Returns:
        param_loss: Parameter MSE (normalized)
        curve_loss: Curve reconstruction MSE (log space)
        param_mae: Per-parameter MAE (denormalized)
        time_per_sample: Seconds per curve generation
    """
    model.eval()

    param_losses = []
    curve_errors = []
    param_maes = {p: [] for p in PARAM_NAMES}

    start_time = time.time()
    samples_processed = 0

    with torch.no_grad():
        for batch_y, batch_x in val_loader:
            batch_y, batch_x = batch_y.to(device), batch_x.to(device)

            pred = model(batch_y)

            # Parameter loss (normalized)
            param_loss = nn.MSELoss()(pred, batch_x)
            param_losses.append(param_loss.item())

            # Denormalize for curve generation
            pred_denorm = denorm_params(pred.cpu())
            true_denorm = denorm_params(batch_x.cpu())

            # Per-parameter MAE
            for i, param in enumerate(PARAM_NAMES):
                mae = torch.abs(pred_denorm[:, i] - true_denorm[:, i]).mean().item()
                param_maes[param].append(mae)

            # Curve errors (slow - limit samples)
            for i in range(len(batch_y)):
                if samples_processed >= max_samples:
                    break

                pred_params_np = pred_denorm[i].numpy()
                true_params_np = true_denorm[i].numpy()

                try:
                    curve_err = compute_curve_error(pred_params_np, true_params_np)
                    curve_errors.append(curve_err)
                except Exception as e:
                    # Skip if curve generation fails
                    print(f"Warning: Curve generation failed: {e}")
                    continue

                samples_processed += 1

            if samples_processed >= max_samples:
                break

    elapsed = time.time() - start_time
    time_per_sample = elapsed / samples_processed if samples_processed > 0 else 0

    # Aggregate
    avg_param_loss = np.mean(param_losses)
    avg_curve_loss = np.mean(curve_errors) if curve_errors else float('nan')
    avg_param_mae = {p: np.mean(param_maes[p]) for p in PARAM_NAMES}

    return avg_param_loss, avg_curve_loss, avg_param_mae, time_per_sample


def train_with_curve_validation(dataset_path, epochs=50, lr=1e-3, batch_size=32,
                                curve_val_interval=5, curve_val_samples=100):
    """
    Train with parameter loss, validate with curve loss.

    Args:
        curve_val_interval: Compute curve validation every N epochs (slow!)
        curve_val_samples: Max samples for curve validation
    """

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"✓ Using device: {device}")

    # Load dataset
    print(f"\n📦 Loading dataset: {dataset_path}")
    X, Y = load_dataset(dataset_path, use_full_curve=False)

    print(f"\n🎯 Training with Curve Validation")
    print(f"   Training loss: Parameter MSE (fast, differentiable)")
    print(f"   Validation: Both param + curve MSE")
    print(f"   Curve validation: Every {curve_val_interval} epochs, {curve_val_samples} samples")

    # Split
    dataset = NormalizedXRDDataset(X, Y, log_space=True, train=True)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Val:   {len(val_dataset)} samples")

    # Model
    model = XRDRegressor(n_out=7).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    print(f"   Epochs: {epochs}")
    print(f"   LR: {lr}")

    best_param_loss = float('inf')
    best_curve_loss = float('inf')
    history = {
        'train_param_loss': [],
        'val_param_loss': [],
        'val_curve_loss': [],
        'curve_val_epochs': []
    }

    for epoch in range(epochs):
        # ========== TRAIN ==========
        model.train()
        train_loss = 0

        for batch_y, batch_x in train_loader:
            batch_y, batch_x = batch_y.to(device), batch_x.to(device)

            optimizer.zero_grad()
            pred = model(batch_y)
            loss = criterion(pred, batch_x)  # Parameter MSE
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        history['train_param_loss'].append(train_loss)

        # ========== VALIDATE (Param Loss) ==========
        model.eval()
        val_param_loss = 0

        with torch.no_grad():
            for batch_y, batch_x in val_loader:
                batch_y, batch_x = batch_y.to(device), batch_x.to(device)
                pred = model(batch_y)
                loss = criterion(pred, batch_x)
                val_param_loss += loss.item()

        val_param_loss /= len(val_loader)
        history['val_param_loss'].append(val_param_loss)

        if val_param_loss < best_param_loss:
            best_param_loss = val_param_loss
            best_param_epoch = epoch

        # ========== VALIDATE (Curve Loss - Slow!) ==========
        val_curve_loss = None
        if (epoch + 1) % curve_val_interval == 0 or epoch == epochs - 1:
            print(f"\n   Computing curve validation (slow!)...")
            _, val_curve_loss, param_mae, time_per_sample = validate_with_curves(
                model, val_loader, device, max_samples=curve_val_samples
            )

            history['val_curve_loss'].append(val_curve_loss)
            history['curve_val_epochs'].append(epoch)

            if val_curve_loss < best_curve_loss:
                best_curve_loss = val_curve_loss
                best_curve_epoch = epoch

            print(f"   Curve validation: {val_curve_loss:.9f} "
                  f"({time_per_sample:.3f}s/sample, {curve_val_samples} samples)")

        # Print progress
        if (epoch + 1) % 10 == 0 or val_curve_loss is not None:
            curve_str = f", curve={val_curve_loss:.9f}" if val_curve_loss is not None else ""
            print(f"Epoch {epoch+1:3d}/{epochs}: "
                  f"train_param={train_loss:.6f}, "
                  f"val_param={val_param_loss:.6f}{curve_str} "
                  f"(best_param={best_param_loss:.6f} @ {best_param_epoch+1})")

    print(f"\n✅ Training complete!")
    print(f"   Best param loss: {best_param_loss:.6f} at epoch {best_param_epoch+1}")
    if len(history['val_curve_loss']) > 0:
        print(f"   Best curve loss: {best_curve_loss:.6f} at epoch {best_curve_epoch+1}")

        # Critical analysis
        print(f"\n📊 CRITICAL ANALYSIS:")
        print(f"   Does best param loss = best curve loss?")
        if best_param_epoch == best_curve_epoch:
            print(f"   ✅ YES! Both optimized at epoch {best_param_epoch+1}")
            print(f"      → Parameter MSE is a GOOD proxy for curve MSE")
        else:
            print(f"   ❌ NO! Param best @ {best_param_epoch+1}, Curve best @ {best_curve_epoch+1}")
            print(f"      → Parameter MSE may NOT fully optimize curve reconstruction")
            print(f"      → Consider implementing differentiable curve loss")

    # Save
    checkpoint = {
        'model': model.state_dict(),
        'epoch': epochs,
        'val_loss': best_param_loss,
        'val_curve_loss': best_curve_loss if len(history['val_curve_loss']) > 0 else None,
        'history': history,
        'approach': 'curve_validation'
    }

    save_path = 'checkpoints/approach_curve_validation.pt'
    torch.save(checkpoint, save_path)
    print(f"   Saved: {save_path}")

    return best_param_loss, best_curve_loss, history


if __name__ == "__main__":
    train_with_curve_validation(
        'datasets/dataset_1000_dl100_7d.pkl',
        epochs=50,
        curve_val_interval=10,  # Check curve loss every 10 epochs
        curve_val_samples=100   # Use 100 samples (30-60s per validation)
    )
