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
        Yp = (Yp - Yp.amin(dim=1, keepdim=True)) / \
            (Yp.amax(dim=1, keepdim=True) - Yp.amin(dim=1, keepdim=True) + 1e-12)

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

    def __init__(self, c: int, dilation: int = 1):
        """
        Args:
            c: Number of channels
            dilation: Dilation factor for convolutions
        """
        super().__init__()
        kernel_size = 7
        # Proper padding to maintain dimension
        pad = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(c, c, kernel_size=7,
                               padding=pad, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(c)
        self.conv2 = nn.Conv1d(c, c, kernel_size=7,
                               padding=pad, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(c)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = self.act(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return self.act(x + h)


class XRDRegressor(nn.Module):
    """
    1D CNN Regressor for XRD curve analysis.

    Architecture:
    - Stem: Conv1d + BN + ReLU
    - Residual blocks with increasing dilation
    - Global average pooling
    - MLP head with Sigmoid output (normalized params)
    """

    def __init__(self, n_out: int | None = None):
        """
        Args:
            n_out: Number of output parameters (default: len(PARAM_NAMES))
        """
        super().__init__()
        if n_out is None:
            n_out = len(PARAM_NAMES)

        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=9, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )

        self.blocks = nn.Sequential(
            ResidualBlock(32, dilation=1),
            ResidualBlock(32, dilation=2),
            ResidualBlock(32, dilation=4),
            ResidualBlock(32, dilation=8),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, n_out),
            nn.Sigmoid(),  # outputs in [0,1]
        )

    def forward(self, x):
        """
        Args:
            x: Input curves [B, 1, L]

        Returns:
            Normalized parameters [B, P] in range [0, 1]
        """
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)
