"""
XRD Model Architecture and Utilities
=====================================
Shared components for training and evaluation:
- Model architecture (XRDRegressor)
- Dataset class (PickleXRDDataset)
- Parameter ranges and normalization
- Helper functions
"""

import pickle
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# PARAMETER RANGES AND CONFIGURATION
# =============================================================================

# Parameter ranges for normalization
# Order matters! Must match X columns in dataset
RANGES = {
    "Dmax1": (0.002, 0.030),
    "D01": (0.0010, 0.030),
    "L1": (1000e-8, 7000e-8),
    "Rp1": (0.0, 7000e-8),
    "D02": (0.0020, 0.0300),
    "L2": (1000e-8, 7000e-8),
    "Rp2": (-6000e-8, 0.0),
}

PARAM_NAMES = list(RANGES.keys())


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def set_seed(seed: int = 1234):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device():
    """Get best available device (MPS > CUDA > CPU)"""
    if torch.backends.mps.is_available():
        print("✓ Using MPS (Apple Silicon)")
        return torch.device("mps")

    if torch.cuda.is_available():
        print("✓ Using CUDA")
        return torch.device("cuda")

    print("✓ Using CPU")
    return torch.device("cpu")


def load_dataset(path: Path):
    """
    Load pickled dataset.

    Expected format: {"X": tensor/ndarray [N,P], "Y": tensor/ndarray [N,L]}
    Where:
        X = parameters to predict [Dmax1, D01, L1, Rp1, D02, L2, Rp2]
        Y = XRD rocking curves
    """
    with open(path, "rb") as f:
        data = pickle.load(f)

    X = torch.as_tensor(data["X"]).float()  # [N, P]
    Y = torch.as_tensor(data["Y"]).float()  # [N, L]

    assert X.ndim == 2, f"X must be 2D [N,P], got {tuple(X.shape)}"
    assert Y.ndim == 2, f"Y must be 2D [N,L], got {tuple(Y.shape)}"
    assert X.size(1) == len(PARAM_NAMES), (
        f"X has {X.size(1)} columns but RANGES defines {len(PARAM_NAMES)} params.\n"
        f"PARAM_NAMES={PARAM_NAMES}\n"
        "Update RANGES or your dataset so they match in size & order."
    )

    print(f"✓ Loaded dataset from {path}")
    print(f"  X shape: {tuple(X.shape)}  Y shape: {tuple(Y.shape)}")

    return X, Y


@torch.no_grad()
def denorm_params(p_norm: torch.Tensor) -> torch.Tensor:
    """
    Denormalize predictions from [0,1] to physical parameter ranges.

    Args:
        p_norm: Normalized predictions [B, P] in range [0, 1]

    Returns:
        Physical parameters [B, P]
    """
    B, P = p_norm.size(0), p_norm.size(1)
    assert P == len(PARAM_NAMES), "Prediction width must match PARAM_NAMES"

    outs = []
    for j, name in enumerate(PARAM_NAMES):
        lo, hi = RANGES[name]
        lo = float(lo)
        hi = float(hi)
        phys = lo + (hi - lo) * p_norm[..., j]
        outs.append(phys.unsqueeze(-1))

    return torch.cat(outs, dim=-1)


def physics_constrained_loss(predictions, targets, loss_weights, base_loss_fn=None):
    """
    Physics-informed loss function for XRD parameter regression.

    Combines:
    1. Base prediction loss (weighted by parameter importance)
    2. Physical constraint violations (penalty terms)

    Physical constraints for deformation profile parameters:
    - D01 ≤ Dmax1 (surface deformation ≤ maximum deformation)
    - D01 + D02 ≤ 0.03 (total deformation physical limit)
    - Rp1 ≤ L1 (deformation peak position ≤ layer thickness)
    - L2 ≤ L1 (decaying layer thickness ≤ main layer)

    Args:
        predictions: Model predictions [B, 7] in normalized [0,1] space
                    Order: [Dmax1, D01, L1, Rp1, D02, L2, Rp2]
        targets: Ground truth [B, 7] in normalized [0,1] space
        loss_weights: Per-parameter weights [7] for importance weighting
        base_loss_fn: Base loss function (default: smooth_l1_loss)

    Returns:
        total_loss: Combined loss (base + constraints)
        constraint_penalty: Constraint violation magnitude (for monitoring)
    """
    if base_loss_fn is None:
        base_loss_fn = F.smooth_l1_loss

    # Base prediction loss (weighted by parameter importance)
    per_param_loss = base_loss_fn(predictions, targets, reduction='none')
    main_loss = (loss_weights * per_param_loss).mean()

    # Physics constraint penalties
    constraint_penalty = 0.0

    # Constraint 1: D01 ≤ Dmax1 (idx 1 ≤ idx 0)
    # ReLU activates only when constraint is violated
    constraint_penalty += F.relu(predictions[:, 1] - predictions[:, 0]).mean()

    # Constraint 2: D01 + D02 ≤ 0.03
    # Need to convert from normalized [0,1] to physical scale
    # D01: [0.001, 0.030] normalized to [0,1]
    # D02: [0.002, 0.030] normalized to [0,1]
    D01_phys = 0.001 + (0.030 - 0.001) * predictions[:, 1]
    D02_phys = 0.002 + (0.030 - 0.002) * predictions[:, 4]
    # Higher weight
    constraint_penalty += F.relu(D01_phys + D02_phys - 0.03).mean() * 10.0

    # Constraint 3: Rp1 ≤ L1 (idx 3 ≤ idx 2)
    # Both normalized to same range [0, 7000e-8], so can compare directly
    constraint_penalty += F.relu(predictions[:, 3] - predictions[:, 2]).mean()

    # Constraint 4: L2 ≤ L1 (idx 5 ≤ idx 2)
    # Both normalized to same range [1000e-8, 7000e-8], so can compare directly
    constraint_penalty += F.relu(predictions[:, 5] - predictions[:, 2]).mean()

    # Total loss: base + constraint penalty
    # Constraint weight 0.1 balances accuracy vs physical validity
    total_loss = main_loss + 0.1 * constraint_penalty

    return total_loss, constraint_penalty


# =============================================================================
# DATASET CLASS
# =============================================================================

class PickleXRDDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for XRD rocking curves.

    Performs:
    - Log-space normalization of curves (optional)
    - Per-sample normalization to [0, 1]
    - Target parameter normalization using RANGES
    """

    def __init__(self, X: torch.Tensor, Y: torch.Tensor, log_space: bool = True, train: bool = True):
        """
        Args:
            X: Parameters [N, P]
            Y: XRD curves [N, L]
            log_space: Apply log10 transformation before normalization
            train: Training mode flag (currently unused, for future augmentation)
        """
        self.X = X.clone()
        self.Y = Y.clone()
        self.train = train
        self.log_space = log_space
        self.param_names = PARAM_NAMES

        # Normalize inputs (curves) per-sample to [0,1] (optionally after log10)
        if self.log_space:
            Y_safe = self.Y + 1e-10  # prevent log(0)
            Yp = torch.log10(Y_safe)
            # Normalize to [0, 1]
            Yp = (Yp - Yp.amin(dim=1, keepdim=True)) / \
                (Yp.amax(dim=1, keepdim=True) -
                 Yp.amin(dim=1, keepdim=True) + 1e-12)
        else:
            Yp = self.Y / (self.Y.amax(dim=1, keepdim=True) + 1e-12)

        # Additional normalization step
        # Yp = (Yp - Yp.amin(dim=1, keepdim=True)) / \
        #     (Yp.amax(dim=1, keepdim=True) - Yp.amin(dim=1, keepdim=True) + 1e-12)

        self.Yn = Yp.unsqueeze(1)  # [N, 1, L] for Conv1d

        # Normalize targets to [0,1] using RANGES
        N, P = self.X.size(0), self.X.size(1)
        self.Tn = torch.empty(N, P, dtype=torch.float32)
        eps = 1e-12

        for j, name in enumerate(self.param_names):
            lo, hi = RANGES[name]
            lo = float(lo)
            hi = float(hi)
            rng = max(hi - lo, eps)
            self.Tn[:, j] = ((self.X[:, j] - lo) / rng).clamp(0.0, 1.0)

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        """
        Returns:
            Normalized curve [1, L]
            Normalized targets [P]
        """
        return self.Yn[idx], self.Tn[idx]


# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

class ResidualBlock(nn.Module):
    """Residual block with dilated convolutions for 1D CNN"""

    def __init__(self, c: int, dilation: int = 1, kernel_size: int = 15):
        """
        Args:
            c: Number of channels
            dilation: Dilation factor for convolutions
            kernel_size: Kernel size (default: 15, recommended by Ziegler et al.)
        """
        super().__init__()
        # Proper padding to maintain dimension
        pad = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(c, c, kernel_size=kernel_size,
                               padding=pad, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(c)
        self.conv2 = nn.Conv1d(c, c, kernel_size=kernel_size,
                               padding=pad, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(c)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = self.act(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return self.act(x + h)


class AttentionPool1d(nn.Module):
    """
    Attention-based pooling for 1D signals.

    Instead of Global Average Pooling which loses spatial information,
    this learns to weight different positions based on their importance.
    Critical for XRD: allows model to focus on specific regions of the curve
    that encode parameters like Rp2 (position of deformation maximum).
    """

    def __init__(self, channels: int):
        super().__init__()
        # Learn attention weights from features
        self.attention = nn.Sequential(
            nn.Conv1d(channels, channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels // 4, 1, kernel_size=1),
        )

    def forward(self, x):
        """
        Args:
            x: [B, C, L] - features from CNN backbone

        Returns:
            [B, C] - attention-weighted pooled features
        """
        # Compute attention weights for each position
        attn_logits = self.attention(x)  # [B, 1, L]
        attn_weights = F.softmax(attn_logits, dim=-1)  # [B, 1, L]

        # Weighted sum across spatial dimension
        pooled = (x * attn_weights).sum(dim=-1)  # [B, C]

        return pooled


class XRDRegressor(nn.Module):
    """
    Physics-informed 1D CNN Regressor for XRD rocking curve analysis (v3).

    Improvements over v2:
    - K=15 kernel size (from Ziegler et al.): better feature extraction
    - Progressive channel expansion: 32→48→64→96→128→128 (vs constant 64)
    - Larger receptive field: 6 blocks with dilations up to 32 + larger kernels
    - Attention-based pooling: preserves spatial info for position parameters (Rp2)
    - Deeper MLP head: better parameter disentanglement

    Architecture:
    - Stem: Conv1d(1→32) + BN + ReLU
    - 6 Residual blocks with progressive channels and dilations:
      - Block 1: 32 ch, dilation=1  (local features)
      - Block 2: 48 ch, dilation=2  (short-range)
      - Block 3: 64 ch, dilation=4  (medium-range)
      - Block 4: 96 ch, dilation=8  (long-range)
      - Block 5: 128 ch, dilation=16 (very long-range)
      - Block 6: 128 ch, dilation=32 (global features)
    - Attention pooling (instead of GAP)
    - MLP head: 128→256→128→7 with Sigmoid output
    """

    def __init__(self, n_out: int | None = None, kernel_size: int = 15):
        """
        Args:
            n_out: Number of output parameters (default: len(PARAM_NAMES))
            kernel_size: Kernel size for residual blocks (default: 15)
        """
        super().__init__()
        if n_out is None:
            n_out = len(PARAM_NAMES)

        # Progressive channel expansion: 32→48→64→96→128→128
        channels = [32, 48, 64, 96, 128, 128]
        dilations = [1, 2, 4, 8, 16, 32]

        # Stem: initial feature extraction
        self.stem = nn.Sequential(
            nn.Conv1d(1, channels[0], kernel_size=9, padding=4),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(inplace=True),
        )

        # Residual blocks with progressive channels and increasing dilation
        # Receptive field: ~900 points with K=15 (>100% of 650-point curve)
        # Critical for capturing long-range interference patterns (L2, Rp2)
        self.block1 = ResidualBlock(
            channels[0], dilation=dilations[0], kernel_size=kernel_size)
        self.trans1 = nn.Conv1d(
            channels[0], channels[1], kernel_size=1)  # Channel transition

        self.block2 = ResidualBlock(
            channels[1], dilation=dilations[1], kernel_size=kernel_size)
        self.trans2 = nn.Conv1d(channels[1], channels[2], kernel_size=1)

        self.block3 = ResidualBlock(
            channels[2], dilation=dilations[2], kernel_size=kernel_size)
        self.trans3 = nn.Conv1d(channels[2], channels[3], kernel_size=1)

        self.block4 = ResidualBlock(
            channels[3], dilation=dilations[3], kernel_size=kernel_size)
        self.trans4 = nn.Conv1d(channels[3], channels[4], kernel_size=1)

        self.block5 = ResidualBlock(
            channels[4], dilation=dilations[4], kernel_size=kernel_size)
        self.trans5 = nn.Conv1d(channels[4], channels[5], kernel_size=1)

        self.block6 = ResidualBlock(
            channels[5], dilation=dilations[5], kernel_size=kernel_size)

        # Attention pooling: learns where to look on the curve
        # Essential for position parameters (Rp1, Rp2)
        self.pool = AttentionPool1d(channels[5])

        # MLP head: maps features to physical parameters
        # Deeper network for better parameter disentanglement
        self.head = nn.Sequential(
            nn.Linear(channels[5], 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),  # Regularization
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, n_out),
            nn.Sigmoid(),  # outputs in [0,1]
        )

    def forward(self, x):
        """
        Args:
            x: Input curves [B, 1, L] (L=650 for XRD rocking curves)

        Returns:
            Normalized parameters [B, P] in range [0, 1]
            Order: [Dmax1, D01, L1, Rp1, D02, L2, Rp2]
        """
        x = self.stem(x)       # [B, 32, L]

        # Progressive channel expansion through residual blocks
        x = self.block1(x)     # [B, 32, L]
        x = self.trans1(x)     # [B, 48, L]

        x = self.block2(x)     # [B, 48, L]
        x = self.trans2(x)     # [B, 64, L]

        x = self.block3(x)     # [B, 64, L]
        x = self.trans3(x)     # [B, 96, L]

        x = self.block4(x)     # [B, 96, L]
        x = self.trans4(x)     # [B, 128, L]

        x = self.block5(x)     # [B, 128, L]
        x = self.trans5(x)     # [B, 128, L]

        x = self.block6(x)     # [B, 128, L] - preserves spatial info

        x = self.pool(x)       # [B, 128] - attention-weighted pooling
        return self.head(x)    # [B, 7]
