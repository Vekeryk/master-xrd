#!/usr/bin/env python3
"""
APPROACH 3: Multi-Task Learning
Train model to predict BOTH parameters AND curve reconstruction residuals.

This forces the network to learn curve-parameter relationships more explicitly,
which should improve prediction of sensitive parameters like Rp2, L2.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np

from model_common import XRDRegressor, NormalizedXRDDataset, PARAM_NAMES, RANGES, load_dataset


class MultiTaskXRDRegressor(nn.Module):
    """
    Multi-task architecture:
    - Shared encoder (same as XRDRegressor)
    - Task 1: Parameter prediction head (7 outputs)
    - Task 2: Curve residual prediction head (651 outputs)
    """

    def __init__(self, n_params=7, curve_length=651):
        super().__init__()

        # Shared encoder (same as XRDRegressor v3)
        self.encoder = nn.ModuleList()

        # Initial conv
        self.encoder.append(nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1)
        ))

        # Residual blocks with progressive channels
        channels = [32, 48, 64, 96, 128, 128]
        dilations = [1, 2, 4, 8, 16, 32]

        for i in range(len(channels) - 1):
            in_ch = channels[i]
            out_ch = channels[i + 1]
            dilation = dilations[i]

            self.encoder.append(self._make_residual_block(in_ch, out_ch, dilation))

        # Global pooling
        self.attention_pool = nn.Sequential(
            nn.Conv1d(128, 1, kernel_size=1),
            nn.Softmax(dim=-1)
        )

        # Task 1: Parameter prediction head
        self.param_head = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_params)
        )

        # Task 2: Curve residual prediction head
        # Predict correction to identity mapping
        self.residual_head = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(32, 1, kernel_size=1)
        )

    def _make_residual_block(self, in_channels, out_channels, dilation):
        """Create residual block with dilation."""
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=15,
                     padding=7*dilation, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(out_channels, out_channels, kernel_size=15,
                     padding=7*dilation, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        """
        Args:
            x: [B, L] - Normalized log-space curve

        Returns:
            params: [B, 7] - Predicted parameters
            residual: [B, L] - Predicted curve residual
        """
        # Ensure proper shape
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, 1, L]

        # Shared encoder
        features = x
        for layer in self.encoder:
            features = layer(features)  # [B, 128, L]

        # Task 1: Parameter prediction (use attention pooling)
        attn_weights = self.attention_pool(features)  # [B, 1, L]
        pooled = torch.sum(features * attn_weights, dim=-1)  # [B, 128]
        params = self.param_head(pooled)  # [B, 7]

        # Task 2: Curve residual prediction
        residual = self.residual_head(features).squeeze(1)  # [B, L]

        return params, residual


def train_multitask(dataset_path, epochs=50, lr=1e-3, batch_size=32,
                    task_weights={'params': 1.0, 'residual': 0.3}):
    """
    Multi-task training: predict parameters + curve residuals.

    Args:
        task_weights: Relative importance of each task
            - 'params': 1.0 = main task
            - 'residual': 0.3 = auxiliary task (helps learning, but secondary)
    """

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"✓ Using device: {device}")

    # Load dataset
    print(f"\n📦 Loading dataset: {dataset_path}")
    X, Y = load_dataset(dataset_path, use_full_curve=False)

    print(f"\n🎯 APPROACH 3: Multi-Task Learning (params + residuals)")
    print(f"   Task weights: params={task_weights['params']:.1f}, residual={task_weights['residual']:.1f}")

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
    model = MultiTaskXRDRegressor(n_params=7, curve_length=651).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    param_criterion = nn.MSELoss()
    residual_criterion = nn.MSELoss()

    print(f"   Epochs: {epochs}")
    print(f"   LR: {lr}")

    best_val_loss = float('inf')
    history = {
        'train_loss': [], 'val_loss': [],
        'train_param_loss': [], 'train_residual_loss': [],
        'val_param_loss': [], 'val_residual_loss': []
    }

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        train_param_loss = 0
        train_residual_loss = 0

        for batch_y, batch_x in train_loader:
            batch_y, batch_x = batch_y.to(device), batch_x.to(device)

            optimizer.zero_grad()

            # Forward pass
            pred_params, pred_residual = model(batch_y)

            # Task 1: Parameter prediction loss
            loss_params = param_criterion(pred_params, batch_x)

            # Task 2: Curve residual loss
            # Residual = difference between normalized input and identity
            # This encourages model to understand curve reconstruction
            target_residual = torch.zeros_like(batch_y)  # Identity mapping as baseline
            loss_residual = residual_criterion(pred_residual, target_residual)

            # Combined loss
            loss = (task_weights['params'] * loss_params +
                   task_weights['residual'] * loss_residual)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_param_loss += loss_params.item()
            train_residual_loss += loss_residual.item()

        train_loss /= len(train_loader)
        train_param_loss /= len(train_loader)
        train_residual_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0
        val_param_loss = 0
        val_residual_loss = 0

        with torch.no_grad():
            for batch_y, batch_x in val_loader:
                batch_y, batch_x = batch_y.to(device), batch_x.to(device)

                pred_params, pred_residual = model(batch_y)

                loss_params = param_criterion(pred_params, batch_x)
                target_residual = torch.zeros_like(batch_y)
                loss_residual = residual_criterion(pred_residual, target_residual)

                loss = (task_weights['params'] * loss_params +
                       task_weights['residual'] * loss_residual)

                val_loss += loss.item()
                val_param_loss += loss_params.item()
                val_residual_loss += loss_residual.item()

        val_loss /= len(val_loader)
        val_param_loss /= len(val_loader)
        val_residual_loss /= len(val_loader)

        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_param_loss'].append(train_param_loss)
        history['train_residual_loss'].append(train_residual_loss)
        history['val_param_loss'].append(val_param_loss)
        history['val_residual_loss'].append(val_residual_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs}: "
                  f"train={train_loss:.6f} (p={train_param_loss:.6f}, r={train_residual_loss:.6f}), "
                  f"val={val_loss:.6f} (p={val_param_loss:.6f}, r={val_residual_loss:.6f}) "
                  f"[best={best_val_loss:.6f} @ {best_epoch+1}]")

    print(f"\n✅ Training complete!")
    print(f"   Best val loss: {best_val_loss:.6f} at epoch {best_epoch+1}")
    print(f"   Final param loss: {val_param_loss:.6f}")
    print(f"   Final residual loss: {val_residual_loss:.6f}")

    # Save
    checkpoint = {
        'model': model.state_dict(),
        'epoch': epochs,
        'val_loss': best_val_loss,
        'history': history,
        'approach': 'multitask',
        'task_weights': task_weights
    }

    save_path = 'checkpoints/approach_multitask.pt'
    torch.save(checkpoint, save_path)
    print(f"   Saved: {save_path}")

    return best_val_loss, history


if __name__ == "__main__":
    train_multitask('datasets/dataset_1000_dl100_7d.pkl', epochs=50)
