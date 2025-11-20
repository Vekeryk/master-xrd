"""
XRD Model Architecture and Utilities
=====================================
Shared components for training and evaluation:
- Model architecture (XRDRegressor)
- Dataset class (NormalizedXRDDataset)
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

# Parameter ranges for normalization (GRID 5)
# Order matters! Must match X columns in dataset
# ⚠️ ВАЖЛИВО: Ці діапазони МУСИТЬ відповідати грідам у dataset_stratified.py!
# ⚠️ ВАЖЛИВО: L1, Rp1, L2, Rp2 в СМ (бо X зберігається в см)!
# ⚠️ ВАЖЛИВО: max значення скориговані щоб (max-min) було кратне step!
# Якщо не відповідають → denormalization працює неправильно
RANGES = {
    "Dmax1": (0.0010, 0.0310),      # Grid 5: 0.031 покриває 0.030
    # Grid 5: 0.0305 покриває експеримент 0.000943
    "D01": (0.0005, 0.0305),
    "L1": (500e-8, 7000e-8),     # 500 Å = 500e-8 см, 7000 Å = 7000e-8 см ✓
    # 50 Å = 50e-8 см, 5050 Å = 5050e-8 см (покриває 5000)
    "Rp1": (50e-8, 5050e-8),
    "D02": (0.0010, 0.0310),      # Grid 5: 0.031 покриває 0.030
    "L2": (500e-8, 5000e-8),     # 500 Å = 500e-8 см, 5000 Å = 5000e-8 см ✓
    # Grid 5: 0 Å покриває експеримент -50 Å та -500 Å
    "Rp2": (-6500e-8, 0e-8),
}

PARAM_NAMES = list(RANGES.keys())

# Grid steps for dataset generation (defines grid density)
# ⚠️ ВАЖЛИВО: Використовуйте ці значення у dataset.py, dataset_parallel.py, dataset_stratified.py
# Steps визначають густоту сітки, min/max беруться з RANGES
GRID_STEPS = {
    'Dmax1': 0.0025,
    'D01': 0.0025,
    'L1': 500.,      # Ångströms
    'Rp1': 500.,     # Ångströms
    'D02': 0.0025,
    'L2': 500.,      # Ångströms
    'Rp2': 500.,     # Ångströms
}


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


def preprocess_curve(curve, crop_by_peak=True, peak_offset=30, target_length=None):
    """
    Apply noise tail and optionally pad/truncate to target length.

    Args:
        curve: Input curve (numpy array or torch tensor)
        crop_by_peak: If True, crop from peak position
        peak_offset: Offset after peak (default 30)
        target_length: If specified, pad or truncate to this length

    Returns:
        numpy array with noise tail applied and adjusted to target_length
    """
    # Convert to numpy if needed
    if isinstance(curve, torch.Tensor):
        curve_np = curve.cpu().numpy().copy()
    else:
        curve_np = np.array(curve, copy=True)

    # Crop by peak if requested
    if crop_by_peak:
        peak_idx = np.argmax(curve_np)
        crop_start = peak_idx + peak_offset
        curve_np = curve_np[crop_start:]

    # Find last point >= threshold from end
    NOISE_THRESHOLD = 0.00025

    last_high_idx = None
    for j in range(len(curve_np) - 1, -1, -1):
        if curve_np[j] >= NOISE_THRESHOLD:
            last_high_idx = j
            break

    if last_high_idx is not None:
        curve_np[last_high_idx:] = NOISE_THRESHOLD

        w, i = 20, last_high_idx                         # ширина і центр сходинки
        L, R = max(0, i - w), min(len(curve_np) - 1, i + 50)
        eps = 1e-12
        t = np.linspace(0, 1, R - L + 1)
        s = 1 - (1 - t)**3
        yl = np.log10(curve_np[L] + eps)
        fl = np.log10(curve_np[R] + eps)
        curve_np[L:R + 1] = 10**(fl + (yl - fl) * (1 - s))

        # ≈ ±2% noise
        curve_np[L:] *= np.exp(np.random.normal(0, 0.02, len(curve_np[L:])))

    # Pad or truncate to target_length if specified
    if target_length is not None:
        current_length = len(curve_np)

        if current_length < target_length:
            # Pad with constant NOISE_THRESHOLD then apply exponential noise
            pad_len = target_length - current_length
            pad_values = np.full(pad_len, NOISE_THRESHOLD)
            curve_np = np.concatenate([curve_np, pad_values])

            # Apply exponential noise to padded section (±2% like line 129)
            curve_np[current_length:] *= np.exp(
                np.random.normal(0, 0.02, pad_len))

        elif current_length > target_length:
            # Truncate
            curve_np = curve_np[:target_length]

    return curve_np


def load_dataset(path: Path, use_full_curve=False, crop_by_peak=True):
    """
    Load pickled dataset.

    Expected format: {"X": tensor/ndarray [N,P], "Y": tensor/ndarray [N,L]}
    Where:
        X = parameters to predict [Dmax1, D01, L1, Rp1, D02, L2, Rp2]
        Y = XRD rocking curves

    Args:
        path: Path to pickle file
        use_full_curve: If False (default), apply crop_params if available.
                       If True, use full curve without cropping.
        crop_by_peak: If True, crop each curve starting from peak + offset
        peak_offset: Offset after peak to start cropping (default: 0 = from peak)

    Returns:
        X: Parameters tensor [N, P]
        Y: Curves tensor [N, L] (cropped or full depending on settings)
    """
    with open(path, "rb") as f:
        data = pickle.load(f)

    X = torch.as_tensor(data["X"]).float()  # [N, P]
    Y = torch.as_tensor(data["Y"]).float()  # [N, L]

    # Crop by peak for each curve
    if crop_by_peak and not use_full_curve:
        N, L_orig = Y.shape
        print(f"  Before crop: {tuple(Y.shape)}")

        # Find crop points (peak + offset)
        Y_cropped = []
        for i in range(N):
            curve_np = preprocess_curve(Y[i].clone())
            curve_cropped = torch.from_numpy(curve_np).float()

            Y_cropped.append(curve_cropped)

        Y = torch.stack(Y_cropped, dim=0)
        print(f"  After crop:  {tuple(Y.shape)}")

    # Auto-crop if crop_params available and not using full curve
    elif not use_full_curve:
        # crop_info = data["crop_params"]
        # start_ML = crop_info.get("start_ML", 50)
        start_ML = 40
        m1 = 700

        print(f"✓ Applying crop_params: Y[:, {start_ML}:{m1}]")
        print(f"  Before crop: {tuple(Y.shape)}")
        Y = Y[:, start_ML:m1]
        print(f"  After crop:  {tuple(Y.shape)}")

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
    # CANNOT compare normalized directly - different ranges!
    # Dmax1: [0.0010, 0.0310], D01: [0.0005, 0.0305]
    # Must denormalize first
    Dmax1_min, Dmax1_max = RANGES["Dmax1"]
    D01_min, D01_max = RANGES["D01"]
    Dmax1_phys = Dmax1_min + (Dmax1_max - Dmax1_min) * predictions[:, 0]
    D01_phys = D01_min + (D01_max - D01_min) * predictions[:, 1]
    constraint_penalty += F.relu(D01_phys - Dmax1_phys).mean()

    # Constraint 2: D01 + D02 ≤ 0.03
    # Need to convert from normalized [0,1] to physical scale using RANGES
    D01_min, D01_max = RANGES["D01"]
    D02_min, D02_max = RANGES["D02"]
    D01_phys = D01_min + (D01_max - D01_min) * predictions[:, 1]
    D02_phys = D02_min + (D02_max - D02_min) * predictions[:, 4]
    # Higher weight for critical constraint
    constraint_penalty += F.relu(D01_phys + D02_phys - 0.03).mean() * 10.0

    # Constraint 3: Rp1 ≤ L1 (idx 3 ≤ idx 2)
    # CANNOT compare normalized directly - different ranges!
    # L1: [500e-8, 7000e-8], Rp1: [50e-8, 5050e-8]
    # Must denormalize first
    L1_min, L1_max = RANGES["L1"]
    Rp1_min, Rp1_max = RANGES["Rp1"]
    L1_phys = L1_min + (L1_max - L1_min) * predictions[:, 2]
    Rp1_phys = Rp1_min + (Rp1_max - Rp1_min) * predictions[:, 3]
    constraint_penalty += F.relu(Rp1_phys - L1_phys).mean()

    # Constraint 4: L2 ≤ L1 (idx 5 ≤ idx 2)
    # CANNOT compare normalized directly - different ranges!
    # L1: [500e-8, 7000e-8], L2: [500e-8, 5000e-8]
    # Must denormalize first
    L2_min, L2_max = RANGES["L2"]
    L2_phys = L2_min + (L2_max - L2_min) * predictions[:, 5]
    # L1_phys already computed above
    constraint_penalty += F.relu(L2_phys - L1_phys).mean()

    # Total loss: base + constraint penalty
    # Constraint weight 0.1 balances accuracy vs physical validity
    total_loss = main_loss + 0.1 * constraint_penalty

    return total_loss, constraint_penalty


# =============================================================================
# DATASET CLASS
# =============================================================================

class NormalizedXRDDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for XRD rocking curves with automatic normalization.

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

        # Store only intensity channel, add positional channel dynamically in __getitem__
        # This ensures device compatibility (position created on same device as curve)
        self.Yn = Yp.unsqueeze(1)  # [N, 1, L] intensity only
        self.curve_length = Yp.shape[1]  # Store length for positional channel

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
            Normalized curve (intensity only) [1, L]
            Normalized targets [P]

        Note: Positional channel added in model forward() to avoid
        creating torch.linspace for every sample (performance)
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
        # Smooth activation for better gradients
        self.act = nn.SiLU(inplace=True)

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
            nn.SiLU(inplace=True),  # Smooth activation
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
    Physics-informed 1D CNN Regressor for XRD rocking curve analysis (v3 + improvements).

    Improvements over v3:
    - SiLU activation: smoother gradients than ReLU (better for low intensities)
    - Positional channel: helps with position-sensitive parameters (Rp1, Rp2)
    - K=15 kernel size (from Ziegler et al.): better feature extraction
    - Progressive channel expansion: 32→48→64→96→128→128 (vs constant 64)
    - Larger receptive field: 6 blocks with dilations up to 32 + larger kernels
    - Attention-based pooling: preserves spatial info for position parameters
    - Deeper MLP head: better parameter disentanglement

    Architecture:
    - Input: [B, 1, L] intensity only (positional channel added in forward)
    - Stem: Conv1d(2→32) + BN + SiLU (2 channels: intensity + position)
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

        # Stem: initial feature extraction from 2 input channels
        # Channel 0: XRD intensity, Channel 1: normalized position
        self.stem = nn.Sequential(
            nn.Conv1d(2, channels[0], kernel_size=9,
                      padding=4),  # 2 input channels
            nn.BatchNorm1d(channels[0]),
            nn.SiLU(inplace=True),  # Smooth activation
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

        # FFT spectral branch: captures oscillation periodicity
        # Critical for layer thickness parameters (L1, L2)
        # Oscillation period ∝ layer thickness (physical relationship)
        self.fft_mlp = nn.Sequential(
            nn.Linear(50, 64),  # 50 FFT frequency bins → 64 features
            nn.SiLU(),
            nn.Linear(64, 32),  # 64 → 32 spectral features
            nn.SiLU(),
        )

        # Hann window for FFT (reduces spectral leakage)
        # Registered as buffer so it moves to correct device with model
        # Initialize as empty tensor (safer than None - avoids edge cases)
        self.register_buffer('hann_window', torch.empty(0))

        # MLP head: maps combined CNN + FFT features to physical parameters
        # Input: 128 (CNN) + 32 (FFT) = 160 features
        combined_features = channels[5] + 32
        self.head = nn.Sequential(
            nn.Linear(combined_features, 256),
            nn.SiLU(inplace=True),  # Smooth activation for better gradients
            nn.Dropout(0.2),  # Regularization
            nn.Linear(256, 128),
            nn.SiLU(inplace=True),  # Smooth activation
            nn.Dropout(0.2),
            nn.Linear(128, n_out),
            nn.Sigmoid(),  # outputs in [0,1]
        )

    def forward(self, x):
        """
        Args:
            x: Input curves [B, 1, L] (L=700 for XRD rocking curves)
               XRD intensity (log-normalized)

        Returns:
            Normalized parameters [B, P] in range [0, 1]
            Order: [Dmax1, D01, L1, Rp1, D02, L2, Rp2]
        """
        # Add positional channel (create once per batch, not per sample)
        # More efficient than creating in dataset.__getitem__
        B, _, L = x.shape
        position_channel = torch.linspace(
            0, 1, L,
            device=x.device,
            dtype=x.dtype
        ).unsqueeze(0).unsqueeze(0).expand(B, 1, -1)  # [B, 1, L]

        x = torch.cat([x, position_channel], dim=1)  # [B, 2, L]

        # Save intensity channel for FFT branch
        # IMPORTANT: This is log10-normalized intensity (from dataset preprocessing)
        # FFT will extract periodicity from the log-space curve
        x_intensity = x[:, 0, :]  # [B, L] - log10-normalized XRD intensity

        # CNN path: process both channels through stem + residual blocks
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

        # CNN pooling
        cnn_features = self.pool(x)  # [B, 128] - attention-weighted pooling

        # FFT spectral branch: extract frequency features
        # Captures oscillation periodicity for layer thickness (L1, L2)

        # Apply Hann window to reduce spectral leakage
        # In our case L is always 700 (interpolation), but we create dynamically for safety
        L = x_intensity.shape[-1]
        if self.hann_window.numel() != L:
            # Create or recreate window if length changed
            self.hann_window = torch.hann_window(
                L,
                device=x_intensity.device,
                dtype=x_intensity.dtype
            )

        x_windowed = x_intensity * self.hann_window  # [B, L] apply window

        # [B, L//2+1] complex
        fft = torch.fft.rfft(x_windowed, dim=-1, norm='ortho')
        # [B, L//2+1] real, avoid log(0)
        fft_magnitude = torch.abs(fft) + 1e-12

        # Drop DC bin (index 0) - not useful for periodic pattern detection
        fft_magnitude = fft_magnitude[:, 1:]  # [B, L//2] without DC

        # Log normalization of FFT magnitude for scale invariance
        # NOTE: This is NOT duplicate logging! Input is log10(intensity), but
        # FFT magnitude is a different signal (frequency domain) that needs its own log
        # log10(intensity) = preprocessing, log1p(|FFT|) = spectral feature scaling
        fft_magnitude = torch.log1p(
            fft_magnitude / (fft_magnitude.amax(dim=-1, keepdim=True) + 1e-12)
        )  # [B, L//2] normalized log magnitude

        # Take first 50 bins (frequencies 1-50, since DC=0 is dropped)
        # This captures oscillation periods relevant for layer thickness (L1, L2)
        fft_features = fft_magnitude[:, :50]  # [B, 50] - bins 1-50
        fft_features = self.fft_mlp(fft_features)  # [B, 32]

        # Combine CNN and FFT features
        combined = torch.cat([cnn_features, fft_features], dim=1)  # [B, 160]

        return self.head(combined)  # [B, 7]
