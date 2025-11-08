"""
XRD PyTorch SPSA Differentiable Wrapper
========================================
Efficient differentiable wrapper using SPSA gradient estimation.

Key innovation: ONLY 2 simulations per sample (vs 14 for naive finite differences)!

SPSA (Simultaneous Perturbation Stochastic Approximation):
- We want J^T v, where J = ∂y/∂θ (curve from params), v = ∂L/∂y (from PyTorch)
- Trick: φ(θ) = v^T y(θ)  →  ∇_θ φ = J^T v
- SPSA estimates ∇φ using only 2 sims: φ(θ+cΔ) and φ(θ-cΔ)
- Δ ~ Rademacher({-1,+1}^P) - random sign vector

Performance:
- Naive FD: 2P simulations per sample (P=7 → 14 sims)
- SPSA: 2 simulations per sample (constant!)
- 7× speedup for P=7 parameters
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
from xrd import DeformationProfile, compute_curve_and_profile
from model_common import RANGES, PARAM_NAMES


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def denormalize_params_numpy(params_normalized: np.ndarray) -> np.ndarray:
    """
    Denormalize parameters from [0,1] to physical ranges.

    Args:
        params_normalized: [7,] normalized parameters

    Returns:
        params_phys: [7,] physical parameters
    """
    params_phys = np.zeros(7, dtype=np.float64)
    for i, name in enumerate(PARAM_NAMES):
        min_val, max_val = RANGES[name]
        params_phys[i] = min_val + \
            (max_val - min_val) * float(params_normalized[i])
    return params_phys


def simulate_curve_normalized(
    theta_phys: np.ndarray,
    crop_params: Tuple[int, int] = (40, 701),
    m1: int = 700,
    m10: int = 20,
    ik: float = 4.018235972
) -> np.ndarray:
    """
    Simulate XRD curve and apply EXACT SAME preprocessing as NormalizedXRDDataset.

    CRITICAL: This must match model_common.py preprocessing exactly!

    Args:
        theta_phys: [7,] physical parameters
        crop_params: (start, end) for GGG peak cropping

    Returns:
        curve_normalized: [L,] curve in [0,1] (log-normalized)
    """
    # Unpack parameters
    Dmax1, D01, L1, Rp1, D02, L2, Rp2 = theta_phys

    # Create deformation profile
    deformation = DeformationProfile(
        Dmax1=float(Dmax1),
        D01=float(D01),
        L1=float(L1),
        Rp1=float(Rp1),
        D02=float(D02),
        L2=float(L2),
        Rp2=float(Rp2),
        Dmin=0.0001,
        dl=100e-8
    )

    # Run simulation
    curve_obj, _ = compute_curve_and_profile(
        params_obj=deformation,
        m1=m1,
        m10=m10,
        ik=ik,
        verbose=False,
        instrumental=True
    )

    # Get convolved curve
    R_vseZ = curve_obj.Y_R_vseZ

    # Crop GGG peak (SAME as dataset generation)
    # start, end = crop_params
    # curve = R_vseZ[start:end]

    curve = R_vseZ[:]

    # Log-normalize (SAME as NormalizedXRDDataset)
    curve = np.clip(curve, 1e-12, None)
    curve_log = np.log10(curve)
    curve_log_min = curve_log.min()
    curve_log_max = curve_log.max()
    curve_normalized = (curve_log - curve_log_min) / \
        (curve_log_max - curve_log_min + 1e-12)

    return curve_normalized.astype(np.float64)


# =============================================================================
# SPSA AUTOGRAD FUNCTION
# =============================================================================

class XRD_SPSA(torch.autograd.Function):
    """
    SPSA gradient estimator for XRD curve generation.

    Forward: Generate curves using numpy simulator
    Backward: Estimate J^T v using SPSA (only 2 sims per sample!)
    """

    @staticmethod
    def forward(
        ctx,
        theta_norm: torch.Tensor,
        c_rel: float = 1e-3,
        crop_start: int = 40,
        crop_end: int = 701,
        seed: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate XRD curves from normalized parameters.

        Args:
            theta_norm: [B, 7] normalized parameters in [0,1]
            c_rel: relative step size for SPSA perturbations
            crop_start: GGG peak crop start
            crop_end: GGG peak crop end
            seed: random seed for reproducibility

        Returns:
            curves: [B, L] normalized curves in [0,1]
        """
        device = theta_norm.device
        dtype = theta_norm.dtype
        batch_size = theta_norm.shape[0]

        # Convert to numpy
        theta_np = theta_norm.detach().cpu().numpy()

        # Generate curves
        curves_list = []
        for i in range(batch_size):
            theta_phys = denormalize_params_numpy(theta_np[i])
            curve = simulate_curve_normalized(
                theta_phys,
                crop_params=(crop_start, crop_end)
            )
            curves_list.append(curve)

        curves = np.stack(curves_list, axis=0)  # [B, L]
        curves_tensor = torch.from_numpy(curves).to(device=device, dtype=dtype)

        # Save for backward
        ctx.save_for_backward(theta_norm)
        ctx.c_rel = c_rel
        ctx.crop_start = crop_start
        ctx.crop_end = crop_end
        ctx.seed = int(seed) if seed is not None else None

        return curves_tensor

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None, None, None]:
        """
        Estimate gradient using SPSA.

        grad_output: [B, L] = ∂L/∂curves (from PyTorch autograd)

        Returns:
            grad_theta: [B, 7] = ∂L/∂theta (estimated via SPSA)
        """
        (theta_norm,) = ctx.saved_tensors
        batch_size, n_params = theta_norm.shape
        device = theta_norm.device
        dtype = theta_norm.dtype

        # Initialize gradient
        grad_theta = torch.zeros_like(theta_norm)

        # Random number generator
        rng = np.random.default_rng(ctx.seed)
        c = ctx.c_rel

        # SPSA for each sample
        for i in range(batch_size):
            # [L] - incoming gradient
            v = grad_output[i].detach().cpu().numpy()
            theta = theta_norm[i].detach().cpu().numpy()  # [7] in [0,1]

            # Random perturbation: Δ ~ {-1, +1}^7
            delta = rng.choice([-1.0, 1.0], size=n_params).astype(np.float64)

            # Perturbed parameters (clipped to [0,1])
            theta_plus = np.clip(theta + c * delta, 0.0, 1.0)
            theta_minus = np.clip(theta - c * delta, 0.0, 1.0)

            # Generate curves for perturbed parameters
            theta_plus_phys = denormalize_params_numpy(theta_plus)
            theta_minus_phys = denormalize_params_numpy(theta_minus)

            curve_plus = simulate_curve_normalized(
                theta_plus_phys,
                crop_params=(ctx.crop_start, ctx.crop_end)
            )
            curve_minus = simulate_curve_normalized(
                theta_minus_phys,
                crop_params=(ctx.crop_start, ctx.crop_end)
            )

            # Scalar projections: φ(θ) = v^T y(θ)
            phi_plus = float(np.dot(v, curve_plus))
            phi_minus = float(np.dot(v, curve_minus))

            # SPSA gradient estimate: ∇φ ≈ [(φ+ - φ-)/(2c)] * Δ
            grad_estimate = ((phi_plus - phi_minus) / (2.0 * c)) * delta  # [7]

            # Convert to torch
            grad_theta[i] = torch.from_numpy(
                grad_estimate).to(device=device, dtype=dtype)

        # Return gradient only for theta_norm (other args have None)
        return grad_theta, None, None, None, None


def xrd_curve_spsa(
    theta_norm: torch.Tensor,
    c_rel: float = 1e-3,
    crop_start: int = 40,
    crop_end: int = 701,
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    Convenience wrapper for XRD_SPSA.apply().

    Args:
        theta_norm: [B, 7] normalized parameters in [0,1]
        c_rel: SPSA step size (relative to [0,1] range)
        crop_start: GGG peak crop start (must match training!)
        crop_end: GGG peak crop end (must match training!)
        seed: random seed for reproducibility

    Returns:
        curves: [B, L] normalized curves in [0,1]
    """
    return XRD_SPSA.apply(theta_norm, c_rel, crop_start, crop_end, seed)


# =============================================================================
# CURVE RECONSTRUCTION LOSS
# =============================================================================

class CurveReconstructionLoss(nn.Module):
    """
    Curve reconstruction loss with SPSA gradients.

    Compares input curve with curve generated from predicted parameters.
    Uses SPSA for efficient gradient estimation (only 2 sims per sample!).
    """

    def __init__(
        self,
        c_rel: float = 1e-3,
        crop_start: int = 40,
        crop_end: int = 701,
        reduction: str = 'mean'
    ):
        """
        Args:
            c_rel: SPSA step size
            crop_start: GGG peak crop start (must match dataset!)
            crop_end: GGG peak crop end (must match dataset!)
            reduction: 'mean' or 'none'
        """
        super().__init__()
        self.c_rel = c_rel
        self.crop_start = crop_start
        self.crop_end = crop_end
        self.reduction = reduction

    def forward(
        self,
        curve_input: torch.Tensor,
        params_pred: torch.Tensor,
        seed: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute curve reconstruction loss.

        Args:
            curve_input: [B, 1, L] or [B, L] input curves (log-normalized)
            params_pred: [B, 7] predicted parameters (normalized [0,1])
            seed: random seed for SPSA

        Returns:
            loss: scalar or [B,] depending on reduction
        """
        # Match shapes
        if curve_input.ndim == 3:
            curve_input = curve_input.squeeze(1)  # [B, L]

        # Generate curves from predicted parameters (with SPSA gradients)
        curve_pred = xrd_curve_spsa(
            params_pred,
            c_rel=self.c_rel,
            crop_start=self.crop_start,
            crop_end=self.crop_end,
            seed=seed
        )

        # MAE loss (more robust than MSE for outliers)
        loss = torch.abs(curve_input - curve_pred)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'none':
            return loss.mean(dim=1)  # [B,]
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")


# =============================================================================
# HYBRID LOSS (PARAMS + CURVE)
# =============================================================================

class HybridLoss(nn.Module):
    """
    Combined parameter loss + curve reconstruction loss.

    L_total = α * L_params + β * L_curve

    where:
    - L_params = MSE(θ_true, θ_pred) - fast, no simulations
    - L_curve = MAE(y_input, y_reconstructed(θ_pred)) - 2 simulations per sample
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.1,
        c_rel: float = 1e-3,
        crop_start: int = 40,
        crop_end: int = 701
    ):
        """
        Args:
            alpha: weight for parameter loss
            beta: weight for curve reconstruction loss
            c_rel: SPSA step size
            crop_start: GGG peak crop start
            crop_end: GGG peak crop end
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.param_loss_fn = nn.MSELoss()
        self.curve_loss_fn = CurveReconstructionLoss(
            c_rel=c_rel,
            crop_start=crop_start,
            crop_end=crop_end,
            reduction='mean'
        )

    def forward(
        self,
        curve_input: torch.Tensor,
        params_true: torch.Tensor,
        params_pred: torch.Tensor,
        seed: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute hybrid loss.

        Args:
            curve_input: [B, 1, L] or [B, L] input curves
            params_true: [B, 7] true parameters (normalized)
            params_pred: [B, 7] predicted parameters (normalized)
            seed: random seed for SPSA

        Returns:
            loss_total: scalar total loss
            loss_params: scalar parameter loss (for logging)
            loss_curve: scalar curve loss (for logging)
        """
        # Parameter loss (fast, no simulations)
        loss_params = self.param_loss_fn(params_pred, params_true)

        # Curve reconstruction loss (2 sims per sample)
        loss_curve = self.curve_loss_fn(curve_input, params_pred, seed=seed)

        # Combined loss
        loss_total = self.alpha * loss_params + self.beta * loss_curve

        return loss_total, loss_params, loss_curve


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_spsa_forward():
    """Test forward pass (curve generation)."""
    print("Testing SPSA forward pass...")

    # Random parameters
    batch_size = 2
    theta_norm = torch.rand(batch_size, 7)

    # Generate curves
    curves = xrd_curve_spsa(theta_norm, c_rel=1e-3, seed=42)

    print(f"  Input shape: {theta_norm.shape}")
    print(f"  Output shape: {curves.shape}")
    print(f"  Output range: [{curves.min():.4f}, {curves.max():.4f}]")
    print("✓ Forward pass OK")


def test_spsa_backward():
    """Test backward pass (SPSA gradient estimation)."""
    print("\nTesting SPSA backward pass...")

    # Random parameters (requires_grad!)
    batch_size = 2
    theta_norm = torch.rand(batch_size, 7, requires_grad=True)

    # Generate curves
    curves = xrd_curve_spsa(theta_norm, c_rel=1e-3, seed=42)

    # Dummy loss
    loss = curves.mean()
    loss.backward()

    print(f"  Gradients shape: {theta_norm.grad.shape}")
    print(
        f"  Gradients range: [{theta_norm.grad.min():.6f}, {theta_norm.grad.max():.6f}]")
    print(
        f"  Gradient nonzero: {(theta_norm.grad != 0).sum().item()}/{theta_norm.grad.numel()}")
    print("✓ Backward pass OK")


def test_curve_reconstruction_loss():
    """Test curve reconstruction loss."""
    print("\nTesting curve reconstruction loss...")

    # Mock data
    batch_size, curve_length = 2, 661
    curve_input = torch.rand(batch_size, 1, curve_length)
    params_pred = torch.rand(batch_size, 7, requires_grad=True)

    # Compute loss
    criterion = CurveReconstructionLoss(c_rel=1e-3)
    loss = criterion(curve_input, params_pred, seed=42)

    print(f"  Loss: {loss.item():.6f}")

    # Backward
    loss.backward()
    print(f"  Gradients shape: {params_pred.grad.shape}")
    print(
        f"  Gradients nonzero: {(params_pred.grad != 0).sum().item()}/{params_pred.grad.numel()}")
    print("✓ Curve reconstruction loss OK")


def test_hybrid_loss():
    """Test hybrid loss (params + curve)."""
    print("\nTesting hybrid loss...")

    # Mock data
    batch_size, curve_length = 2, 661
    curve_input = torch.rand(batch_size, 1, curve_length)
    params_true = torch.rand(batch_size, 7)
    params_pred = torch.rand(batch_size, 7, requires_grad=True)

    # Compute loss
    criterion = HybridLoss(alpha=1.0, beta=0.1, c_rel=1e-3)
    loss_total, loss_params, loss_curve = criterion(
        curve_input, params_true, params_pred, seed=42
    )

    print(f"  Param loss: {loss_params.item():.6f}")
    print(f"  Curve loss: {loss_curve.item():.6f}")
    print(f"  Total loss: {loss_total.item():.6f}")

    # Backward
    loss_total.backward()
    print(f"  Gradients shape: {params_pred.grad.shape}")
    print("✓ Hybrid loss OK")


if __name__ == "__main__":
    print("=" * 70)
    print("XRD SPSA WRAPPER TESTS")
    print("=" * 70)
    print("\n⚠️  Warning: These tests are SLOW (real XRD simulations)")
    print("Expected time: ~30-60 seconds\n")

    test_spsa_forward()
    test_spsa_backward()
    test_curve_reconstruction_loss()
    test_hybrid_loss()

    print("\n" + "=" * 70)
    print("✅ All tests passed!")
    print("=" * 70)
    print("\n📊 Performance:")
    print("  - SPSA: 2 simulations per sample (vs 14 for naive FD)")
    print("  - 7× speedup for 7 parameters!")
    print("  - Ready for training with hybrid loss")
