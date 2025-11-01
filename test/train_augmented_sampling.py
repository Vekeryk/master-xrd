#!/usr/bin/env python3
"""
APPROACH 2: Higher Resolution Sampling for Rp2/L2
Augment dataset with extra samples in sensitive parameter regions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, ConcatDataset
import numpy as np

from model_common import XRDRegressor, NormalizedXRDDataset, PARAM_NAMES, RANGES, load_dataset


def augment_sensitive_regions(X, Y, focus_param_indices=[5, 6], augmentation_factor=2):
    """
    Augment dataset with more samples where sensitive parameters are at edges.

    Args:
        X: Parameters [N, 7]
        Y: Curves [N, L]
        focus_param_indices: [5, 6] = [L2, Rp2] (most sensitive)
        augmentation_factor: How many extra copies for edge samples
    """
    X_np = X.numpy() if isinstance(X, torch.Tensor) else X
    Y_np = Y.numpy() if isinstance(Y, torch.Tensor) else Y

    # Find samples where sensitive parameters are near edges (top/bottom 20%)
    edge_mask = np.zeros(len(X_np), dtype=bool)

    for param_idx in focus_param_indices:
        param_values = X_np[:, param_idx]
        param_range = RANGES[PARAM_NAMES[param_idx]]

        # Bottom 20%
        threshold_low = param_range[0] + 0.2 * (param_range[1] - param_range[0])
        # Top 20%
        threshold_high = param_range[1] - 0.2 * (param_range[1] - param_range[0])

        edge_mask |= (param_values < threshold_low) | (param_values > threshold_high)

    # Augment edge samples
    edge_indices = np.where(edge_mask)[0]

    if len(edge_indices) == 0:
        print("   No edge samples found for augmentation")
        return X, Y

    # Duplicate edge samples
    X_aug = [X_np]
    Y_aug = [Y_np]

    for _ in range(augmentation_factor - 1):
        X_aug.append(X_np[edge_indices])
        Y_aug.append(Y_np[edge_indices])

    X_augmented = np.concatenate(X_aug, axis=0)
    Y_augmented = np.concatenate(Y_aug, axis=0)

    print(f"   Original: {len(X_np)} samples")
    print(f"   Edge samples: {len(edge_indices)} ({100*len(edge_indices)/len(X_np):.1f}%)")
    print(f"   Augmented: {len(X_augmented)} samples (+{len(X_augmented)-len(X_np)})")

    return torch.tensor(X_augmented, dtype=torch.float32), torch.tensor(Y_augmented, dtype=torch.float32)


def train_augmented_sampling(dataset_path, epochs=50, lr=1e-3, batch_size=32):
    """Training with augmented sampling for sensitive regions."""

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"✓ Using device: {device}")

    # Load dataset
    print(f"\n📦 Loading dataset: {dataset_path}")
    X, Y = load_dataset(dataset_path, use_full_curve=False)

    print(f"\n🎯 APPROACH 2: Augmenting sensitive regions (Rp2, L2)")
    X_aug, Y_aug = augment_sensitive_regions(X, Y, focus_param_indices=[5, 6], augmentation_factor=2)

    # Split
    dataset = NormalizedXRDDataset(X_aug, Y_aug, log_space=True, train=True)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Val on original (not augmented)
    val_dataset_clean = NormalizedXRDDataset(
        X[-val_size:], Y[-val_size:], log_space=True, train=False
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset_clean, batch_size=batch_size, shuffle=False)

    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Val:   {len(val_dataset)} samples")

    # Model
    model = XRDRegressor(n_out=7).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

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
        'approach': 'augmented_sampling'
    }

    save_path = 'checkpoints/approach_augmented_sampling.pt'
    torch.save(checkpoint, save_path)
    print(f"   Saved: {save_path}")

    return best_val_loss, history


if __name__ == "__main__":
    train_augmented_sampling('datasets/dataset_1000_dl100_7d.pkl', epochs=50)
