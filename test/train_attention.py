#!/usr/bin/env python3
"""
APPROACH 5: Attention Mechanisms for Position Parameters
Enhanced architecture with multi-head self-attention to better capture
spatial dependencies critical for position parameters (Rp1, Rp2).

Key idea: Position parameters require understanding WHERE features occur in the curve,
not just WHAT features are present. Attention helps model focus on relevant regions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import math

from model_common import XRDRegressor, NormalizedXRDDataset, PARAM_NAMES, RANGES, load_dataset


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention for 1D sequences."""

    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: [B, L, D] - Input features

        Returns:
            out: [B, L, D] - Attended features
        """
        B, L, D = x.shape

        # Project to Q, K, V
        qkv = self.qkv_proj(x)  # [B, L, 3*D]
        qkv = qkv.reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, L, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # [B, num_heads, L, L]
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)  # [B, num_heads, L, head_dim]
        out = out.transpose(1, 2).reshape(B, L, D)  # [B, L, D]
        out = self.out_proj(out)

        return out


class AttentionXRDRegressor(nn.Module):
    """
    Enhanced XRD regressor with multi-head attention.

    Architecture:
    1. Conv encoder (local features)
    2. Multi-head attention (global dependencies)
    3. Position-aware prediction heads
    """

    def __init__(self, n_params=7, num_heads=4):
        super().__init__()

        # Stage 1: Convolutional encoder (same as v3)
        self.conv_encoder = nn.ModuleList()

        self.conv_encoder.append(nn.Sequential(
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
            self.conv_encoder.append(self._make_residual_block(in_ch, out_ch, dilation))

        # Stage 2: Multi-head attention
        # Convert [B, 128, L] → [B, L, 128] for attention
        self.attention_layers = nn.ModuleList([
            MultiHeadAttention(128, num_heads=num_heads, dropout=0.1),
            MultiHeadAttention(128, num_heads=num_heads, dropout=0.1)
        ])

        self.attention_norm = nn.ModuleList([
            nn.LayerNorm(128),
            nn.LayerNorm(128)
        ])

        # Stage 3: Pooling and prediction
        # Attention-based pooling
        self.attention_pool = nn.Sequential(
            nn.Conv1d(128, 1, kernel_size=1),
            nn.Softmax(dim=-1)
        )

        # Separate heads for amplitude/thickness vs position parameters
        # Amplitude/thickness: Dmax1, D01, L1, D02, L2 (indices 0,1,2,4,5)
        # Position: Rp1, Rp2 (indices 3, 6)

        # Shared MLP
        self.shared_mlp = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # Amplitude/thickness head (5 params)
        self.amplitude_head = nn.Linear(128, 5)

        # Position head (2 params) - more capacity
        self.position_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2)
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
        """
        # Ensure proper shape
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, 1, L]

        # Stage 1: Convolutional encoder
        features = x
        for layer in self.conv_encoder:
            features = layer(features)  # [B, 128, L]

        # Stage 2: Attention
        # Convert to [B, L, 128] for attention
        attn_features = features.transpose(1, 2)  # [B, L, 128]

        for attn_layer, norm_layer in zip(self.attention_layers, self.attention_norm):
            # Residual connection + attention
            attn_out = attn_layer(attn_features)
            attn_features = norm_layer(attn_features + attn_out)  # [B, L, 128]

        # Convert back to [B, 128, L]
        features = attn_features.transpose(1, 2)  # [B, 128, L]

        # Stage 3: Pooling
        attn_weights = self.attention_pool(features)  # [B, 1, L]
        pooled = torch.sum(features * attn_weights, dim=-1)  # [B, 128]

        # Shared representation
        shared = self.shared_mlp(pooled)  # [B, 128]

        # Separate heads
        amplitude_params = self.amplitude_head(shared)  # [B, 5]
        position_params = self.position_head(shared)    # [B, 2]

        # Combine: [Dmax1, D01, L1, Rp1, D02, L2, Rp2]
        # Indices: [0, 1, 2, 3, 4, 5, 6]
        # Amplitude: [0, 1, 2, 4, 5] → [Dmax1, D01, L1, D02, L2]
        # Position: [3, 6] → [Rp1, Rp2]

        params = torch.zeros(x.size(0), 7, device=x.device)
        params[:, [0, 1, 2, 4, 5]] = amplitude_params
        params[:, [3, 6]] = position_params

        return params


def train_attention(dataset_path, epochs=50, lr=1e-3, batch_size=32, num_heads=4):
    """
    Training with attention mechanisms for position parameters.

    Args:
        num_heads: Number of attention heads (4 or 8 recommended)
    """

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"✓ Using device: {device}")

    # Load dataset
    print(f"\n📦 Loading dataset: {dataset_path}")
    X, Y = load_dataset(dataset_path, use_full_curve=False)

    print(f"\n🎯 APPROACH 5: Attention Mechanisms for Position Parameters")
    print(f"   Num attention heads: {num_heads}")
    print(f"   Attention layers: 2")
    print(f"   Separate heads: Amplitude/thickness (5) + Position (2)")

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
    model = AttentionXRDRegressor(n_params=7, num_heads=num_heads).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    print(f"   Epochs: {epochs}")
    print(f"   LR: {lr}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Trainable parameters: {total_params:,}")

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
        'approach': 'attention',
        'num_heads': num_heads
    }

    save_path = 'checkpoints/approach_attention.pt'
    torch.save(checkpoint, save_path)
    print(f"   Saved: {save_path}")

    return best_val_loss, history


if __name__ == "__main__":
    train_attention('datasets/dataset_1000_dl100_7d.pkl', epochs=50, num_heads=4)
