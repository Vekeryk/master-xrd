#!/usr/bin/env python3
"""
APPROACH 4: Hierarchical Coarse-to-Fine Training
Two-stage approach:
  Stage 1: Predict all 7 parameters coarsely (standard training)
  Stage 2: Refine sensitive parameters (Rp1, L2, Rp2) with specialized head

This focuses learning capacity on the most challenging parameters.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np

from model_common import XRDRegressor, NormalizedXRDDataset, PARAM_NAMES, RANGES, load_dataset


class HierarchicalXRDRegressor(nn.Module):
    """
    Hierarchical architecture:
    - Shared encoder
    - Coarse head: Predicts all 7 parameters
    - Refinement head: Refines sensitive parameters (Rp1, L2, Rp2)
    """

    def __init__(self, n_params=7, sensitive_indices=[3, 5, 6]):
        super().__init__()
        self.sensitive_indices = sensitive_indices  # [3, 5, 6] = [Rp1, L2, Rp2]

        # Shared encoder (same as XRDRegressor v3)
        self.encoder = nn.ModuleList()

        # Initial conv
        self.encoder.append(nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1)
        ))

        # Residual blocks
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

        # Coarse head: Predict all parameters
        self.coarse_head = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_params)
        )

        # Refinement head: Refine sensitive parameters
        # Takes coarse predictions + features as input
        self.refinement_head = nn.Sequential(
            nn.Linear(128 + n_params, 128),  # Features + coarse params
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, len(sensitive_indices))  # Only sensitive params
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

    def forward(self, x, return_coarse=False):
        """
        Args:
            x: [B, L] - Normalized log-space curve
            return_coarse: If True, return (coarse, refined), else just refined

        Returns:
            refined_params: [B, 7] - Refined predictions
            (or tuple if return_coarse=True)
        """
        # Ensure proper shape
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, 1, L]

        # Shared encoder
        features = x
        for layer in self.encoder:
            features = layer(features)  # [B, 128, L]

        # Attention pooling
        attn_weights = self.attention_pool(features)  # [B, 1, L]
        pooled = torch.sum(features * attn_weights, dim=-1)  # [B, 128]

        # Stage 1: Coarse prediction
        coarse_params = self.coarse_head(pooled)  # [B, 7]

        # Stage 2: Refinement
        # Concatenate features + coarse predictions
        refinement_input = torch.cat([pooled, coarse_params], dim=-1)  # [B, 128+7]
        sensitive_refinements = self.refinement_head(refinement_input)  # [B, 3]

        # Combine: Use refined values for sensitive params, coarse for others
        refined_params = coarse_params.clone()
        refined_params[:, self.sensitive_indices] = sensitive_refinements

        if return_coarse:
            return coarse_params, refined_params
        else:
            return refined_params


def train_hierarchical(dataset_path, epochs=50, lr=1e-3, batch_size=32,
                      stage1_epochs=None, stage2_epochs=None):
    """
    Hierarchical two-stage training.

    Args:
        stage1_epochs: Epochs for stage 1 (coarse). If None, use epochs/2
        stage2_epochs: Epochs for stage 2 (refine). If None, use epochs/2
    """

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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"\n🎯 APPROACH 4: Hierarchical Coarse-to-Fine Training")
    print(f"   Sensitive params: {[PARAM_NAMES[i] for i in [3, 5, 6]]}")
    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Val:   {len(val_dataset)} samples")

    # Determine stage epochs
    if stage1_epochs is None:
        stage1_epochs = epochs // 2
    if stage2_epochs is None:
        stage2_epochs = epochs - stage1_epochs

    print(f"   Stage 1 (coarse): {stage1_epochs} epochs")
    print(f"   Stage 2 (refine): {stage2_epochs} epochs")
    print(f"   LR: {lr}")

    # Model
    model = HierarchicalXRDRegressor(n_params=7, sensitive_indices=[3, 5, 6]).to(device)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    history = {
        'train_loss': [], 'val_loss': [],
        'train_coarse_loss': [], 'train_refine_loss': [],
        'val_coarse_loss': [], 'val_refine_loss': []
    }

    # ========== STAGE 1: Train coarse head ==========
    print(f"\n{'='*80}")
    print(f"STAGE 1: Coarse Training (all parameters)")
    print(f"{'='*80}")

    # Freeze refinement head, train only coarse
    for param in model.refinement_head.parameters():
        param.requires_grad = False

    optimizer_stage1 = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )

    for epoch in range(stage1_epochs):
        # Train
        model.train()
        train_loss = 0

        for batch_y, batch_x in train_loader:
            batch_y, batch_x = batch_y.to(device), batch_x.to(device)

            optimizer_stage1.zero_grad()

            # Only use coarse predictions in stage 1
            coarse_pred, _ = model(batch_y, return_coarse=True)
            loss = criterion(coarse_pred, batch_x)

            loss.backward()
            optimizer_stage1.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch_y, batch_x in val_loader:
                batch_y, batch_x = batch_y.to(device), batch_x.to(device)
                coarse_pred, _ = model(batch_y, return_coarse=True)
                loss = criterion(coarse_pred, batch_x)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_coarse_loss'].append(train_loss)
        history['val_coarse_loss'].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch

        if (epoch + 1) % 10 == 0:
            print(f"Stage 1 - Epoch {epoch+1:3d}/{stage1_epochs}: "
                  f"train={train_loss:.6f}, val={val_loss:.6f} "
                  f"(best={best_val_loss:.6f} @ {best_epoch+1})")

    print(f"\n✓ Stage 1 complete! Best coarse val loss: {best_val_loss:.6f}")

    # ========== STAGE 2: Train refinement head ==========
    print(f"\n{'='*80}")
    print(f"STAGE 2: Refinement Training (sensitive parameters)")
    print(f"{'='*80}")

    # Unfreeze refinement head, optionally freeze encoder
    for param in model.refinement_head.parameters():
        param.requires_grad = True

    # Fine-tune entire network or just refinement?
    # Option 1: Fine-tune all (slower but better)
    # Option 2: Freeze encoder, train only refinement (faster)
    # Using Option 1 for better performance

    optimizer_stage2 = optim.Adam(model.parameters(), lr=lr * 0.5)  # Lower LR for refinement

    best_val_loss_stage2 = float('inf')

    for epoch in range(stage2_epochs):
        # Train
        model.train()
        train_loss = 0
        train_coarse_loss = 0
        train_refine_loss = 0

        for batch_y, batch_x in train_loader:
            batch_y, batch_x = batch_y.to(device), batch_x.to(device)

            optimizer_stage2.zero_grad()

            # Get both coarse and refined predictions
            coarse_pred, refined_pred = model(batch_y, return_coarse=True)

            # Loss on refined predictions (primary)
            loss_refine = criterion(refined_pred, batch_x)

            # Optional: Also supervise coarse predictions (regularization)
            loss_coarse = criterion(coarse_pred, batch_x)

            # Combined loss (emphasize refinement)
            loss = 0.8 * loss_refine + 0.2 * loss_coarse

            loss.backward()
            optimizer_stage2.step()

            train_loss += loss.item()
            train_coarse_loss += loss_coarse.item()
            train_refine_loss += loss_refine.item()

        train_loss /= len(train_loader)
        train_coarse_loss /= len(train_loader)
        train_refine_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0
        val_coarse_loss = 0
        val_refine_loss = 0

        with torch.no_grad():
            for batch_y, batch_x in val_loader:
                batch_y, batch_x = batch_y.to(device), batch_x.to(device)

                coarse_pred, refined_pred = model(batch_y, return_coarse=True)

                loss_refine = criterion(refined_pred, batch_x)
                loss_coarse = criterion(coarse_pred, batch_x)
                loss = 0.8 * loss_refine + 0.2 * loss_coarse

                val_loss += loss.item()
                val_coarse_loss += loss_coarse.item()
                val_refine_loss += loss_refine.item()

        val_loss /= len(val_loader)
        val_coarse_loss /= len(val_loader)
        val_refine_loss /= len(val_loader)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_refine_loss'].append(train_refine_loss)
        history['val_refine_loss'].append(val_refine_loss)

        if val_loss < best_val_loss_stage2:
            best_val_loss_stage2 = val_loss
            best_epoch_stage2 = epoch

        if (epoch + 1) % 10 == 0:
            print(f"Stage 2 - Epoch {epoch+1:3d}/{stage2_epochs}: "
                  f"train={train_loss:.6f} (c={train_coarse_loss:.6f}, r={train_refine_loss:.6f}), "
                  f"val={val_loss:.6f} (c={val_coarse_loss:.6f}, r={val_refine_loss:.6f}) "
                  f"[best={best_val_loss_stage2:.6f} @ {best_epoch_stage2+1}]")

    print(f"\n✅ Training complete!")
    print(f"   Stage 1 best: {best_val_loss:.6f}")
    print(f"   Stage 2 best: {best_val_loss_stage2:.6f}")
    print(f"   Improvement: {(1 - best_val_loss_stage2/best_val_loss)*100:.1f}%")

    # Save
    checkpoint = {
        'model': model.state_dict(),
        'epoch': stage1_epochs + stage2_epochs,
        'val_loss': best_val_loss_stage2,
        'history': history,
        'approach': 'hierarchical',
        'stage1_epochs': stage1_epochs,
        'stage2_epochs': stage2_epochs
    }

    save_path = 'checkpoints/approach_hierarchical.pt'
    torch.save(checkpoint, save_path)
    print(f"   Saved: {save_path}")

    return best_val_loss_stage2, history


if __name__ == "__main__":
    train_hierarchical('datasets/dataset_1000_dl100_7d.pkl', epochs=50)
