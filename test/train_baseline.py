#!/usr/bin/env python3
"""
BASELINE: Standard training (unweighted, no tricks)
For comparison with enhanced approaches.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import pickle
from datetime import datetime

from model_common import XRDRegressor, NormalizedXRDDataset, PARAM_NAMES, load_dataset


def train_baseline(dataset_path, epochs=50, lr=1e-3, batch_size=32):
    """Standard training - no enhancements."""

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"✓ Using device: {device}")

    # Load dataset
    print(f"\n📦 Loading dataset: {dataset_path}")
    X, Y = load_dataset(dataset_path, use_full_curve=False)

    # Split train/val
    dataset = NormalizedXRDDataset(X, Y, log_space=True, train=True)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Val dataset without augmentation
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
    criterion = nn.MSELoss()

    print(f"\n🎯 Training BASELINE (unweighted, standard)")
    print(f"   Epochs: {epochs}")
    print(f"   LR: {lr}")
    print(f"   Batch size: {batch_size}")

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
            loss = criterion(pred, batch_x)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_y, batch_x in val_loader:
                batch_y, batch_x = batch_y.to(device), batch_x.to(device)
                pred = model(batch_y)
                loss = criterion(pred, batch_x)
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
        'approach': 'baseline'
    }

    save_path = 'checkpoints/approach_baseline.pt'
    torch.save(checkpoint, save_path)
    print(f"   Saved: {save_path}")

    return best_val_loss, history


if __name__ == "__main__":
    train_baseline('datasets/dataset_1000_dl100_7d.pkl', epochs=50)
