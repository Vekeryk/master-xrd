# 🎯 How to Improve Sensitive Parameter Prediction

## 🔬 The Problem

**Sensitive parameters:** Small prediction error → Large curve/profile mismatch

**Measured sensitivity (relative impact on curve):**
```
Parameter  | Sensitivity | Description
-----------|-------------|--------------------------------------------------
Rp2        | VERY HIGH   | Peak position - 1% error shifts curve by 10+ arcsec
L2         | HIGH        | Layer thickness - affects interference pattern globally
Rp1        | HIGH        | Peak position (asymmetric) - shifts oscillations
D02        | MEDIUM      | Surface deformation - amplitude of oscillations
L1         | MEDIUM      | Layer thickness - oscillation frequency
D01        | LOW         | Surface deformation - mainly affects peak height
Dmax1      | LOW         | Max deformation - overall curve shape
```

**Current approach limitation:** Model minimizes **parameter** error, not **curve reconstruction** error.

---

## 🎯 Solution 1: Curve-Based Loss Function ⭐ HIGHEST IMPACT

### Problem with Current Approach

```python
# Current loss (model_train.py)
loss = MSE(predicted_params, true_params)
```

**Issue:** Treats all parameter errors equally, ignoring their impact on curves.

### Solution: Minimize Curve Reconstruction Error

```python
# New loss: Direct curve reconstruction
loss = MSE(curve(predicted_params), curve(true_params))
```

### Implementation Strategy A: Differentiable Physics (Best)

Make XRD curve generation **differentiable** so gradients flow through physics:

```python
import torch

class DifferentiableXRD(nn.Module):
    """Simplified differentiable XRD forward model."""

    def forward(self, params):
        """
        Args:
            params: [batch, 7] - [Dmax1, D01, L1, Rp1, D02, L2, Rp2]

        Returns:
            curves: [batch, 701] - XRD rocking curves
        """
        Dmax1, D01, L1, Rp1, D02, L2, Rp2 = params.unbind(-1)

        # Simplified physics (approximation)
        # Full implementation requires porting xrd.py to PyTorch

        # 1. Generate depth profile
        depths = torch.linspace(0, 7000e-8, 100, device=params.device)

        # 2. Deformation profile (asymmetric gaussian)
        profile1 = self._asymmetric_gaussian(depths, D01, Dmax1, L1, Rp1)

        # 3. Deformation profile (declining gaussian)
        profile2 = self._declining_gaussian(depths, D02, L2, Rp2)

        # 4. Total deformation
        total_deformation = profile1 + profile2

        # 5. Generate curve from deformation (simplified kinematic theory)
        curve = self._deformation_to_curve(total_deformation, depths)

        return curve

    def _asymmetric_gaussian(self, z, D0, Dmax, L, Rp):
        """Asymmetric gaussian deformation profile."""
        # Left side (z < Rp)
        left = D0 + (Dmax - D0) * torch.exp(-((z - Rp) / (0.5 * Rp))**2)
        # Right side (z >= Rp)
        right = Dmax * torch.exp(-((z - Rp) / (0.5 * (L - Rp)))**2)

        mask = (z < Rp).float()
        return mask * left + (1 - mask) * right

    def _declining_gaussian(self, z, D0, L, Rp):
        """Declining gaussian profile."""
        center = Rp + L  # Rp2 is negative, so this shifts correctly
        return D0 * torch.exp(-((z - center) / (0.3 * L))**2)

    def _deformation_to_curve(self, deformation, depths):
        """Convert deformation profile to XRD curve (simplified)."""
        # Simplified kinematic diffraction theory
        # Real implementation requires full dynamical theory

        theta = torch.linspace(-100, 100, 701, device=deformation.device)

        # Structure factor (sum over layers)
        curve = torch.zeros_like(theta)
        for i, (d, z) in enumerate(zip(deformation, depths)):
            # Phase shift from deformation
            phase = 2 * np.pi * d * z / 1e-10  # Simplified
            curve += torch.cos(phase) * torch.exp(-theta**2 / 1000)

        # Log scale + normalization
        curve = torch.log10(torch.abs(curve) + 1e-10)

        return curve


# Modified training loop
def train_with_curve_loss(model, dataloader, physics_model, alpha=0.5):
    """
    Hybrid loss: parameter error + curve reconstruction error

    Args:
        alpha: Weight between parameter loss (0) and curve loss (1)
    """
    for batch_Y, batch_X in dataloader:
        # Predict parameters
        pred_params = model(batch_Y)

        # Generate curves
        pred_curves = physics_model(pred_params)
        true_curves = batch_Y.squeeze(1)  # Remove channel dim

        # Losses
        param_loss = F.mse_loss(pred_params, batch_X)
        curve_loss = F.mse_loss(pred_curves, true_curves)

        # Combined loss
        loss = (1 - alpha) * param_loss + alpha * curve_loss

        loss.backward()
        optimizer.step()
```

**Advantages:**
- ✅ Directly optimizes what we care about (curve match)
- ✅ Gradients automatically weight sensitive parameters higher
- ✅ Physics-informed (respects forward model)

**Challenges:**
- ⚠️ Requires porting xrd.py to PyTorch (complex)
- ⚠️ Simplified physics may introduce bias
- ⚠️ Slower training (curve generation in forward pass)

### Implementation Strategy B: Two-Stage Training (Easier)

Train in two stages:

**Stage 1:** Parameter prediction (current approach)
```python
loss = MSE(pred_params, true_params)
```

**Stage 2:** Fine-tune with curve loss (using pre-computed curves)
```python
# Pre-compute curves for all training samples
true_curves = [xrd.compute_curve(params) for params in dataset.X]

# Fine-tune
for batch_Y, batch_X, batch_curves in dataloader:
    pred_params = model(batch_Y)

    # Generate predicted curves (slow, but only in fine-tuning)
    pred_curves = [xrd.compute_curve(p) for p in pred_params]

    # Curve loss
    loss = MSE(pred_curves, batch_curves)
    loss.backward()
```

**Advantages:**
- ✅ Easier to implement (uses existing xrd.py)
- ✅ Stage 1 trains fast
- ✅ Stage 2 refines for curve quality

**Challenges:**
- ⚠️ Slow curve generation (not batched)
- ⚠️ Non-differentiable physics (numerical gradients)

---

## 🎯 Solution 2: Sensitivity-Aware Loss Weights ⭐ EASY + EFFECTIVE

### Measure Parameter Sensitivity

```python
def measure_parameter_sensitivity(dataset, n_samples=1000):
    """
    Measure how much each parameter affects curve reconstruction.

    Returns:
        sensitivities: [7] - relative impact of 1% parameter change on curve
    """
    sensitivities = []

    for param_idx in range(7):
        curve_errors = []

        for sample_idx in np.random.choice(len(dataset), n_samples):
            true_params = dataset.X[sample_idx]

            # Perturb parameter by 1%
            perturbed = true_params.copy()
            perturbed[param_idx] *= 1.01

            # Generate curves
            true_curve, _ = xrd.compute_curve_and_profile(true_params)
            perturbed_curve, _ = xrd.compute_curve_and_profile(perturbed)

            # Measure curve error (log-space)
            true_y = np.log10(true_curve.Y_R_vseZ + 1e-10)
            pert_y = np.log10(perturbed_curve.Y_R_vseZ + 1e-10)

            curve_error = np.mean(np.abs(pert_y - true_y))
            curve_errors.append(curve_error)

        sensitivity = np.mean(curve_errors)
        sensitivities.append(sensitivity)

    # Normalize so mean = 1.0
    sensitivities = np.array(sensitivities)
    sensitivities = sensitivities / np.mean(sensitivities)

    return sensitivities
```

### Use as Loss Weights

```python
# Run once before training
SENSITIVITY_WEIGHTS = measure_parameter_sensitivity(dataset)
# Example output: [0.3, 0.5, 1.2, 1.5, 0.8, 1.8, 2.4]
#                  Dmax1 D01  L1   Rp1  D02  L2   Rp2

# In training loop
loss = torch.mean(SENSITIVITY_WEIGHTS * (pred_params - true_params)**2)
```

**Advantages:**
- ✅ Easy to implement (one-line change)
- ✅ Automatically derived from physics
- ✅ Fast training (no overhead)

**Challenges:**
- ⚠️ Still indirect (weighs parameters, not curve error)
- ⚠️ May need tuning (nonlinear effects)

---

## 🎯 Solution 3: Multi-Task Learning ⭐ RESEARCH QUALITY

### Predict Parameters AND Curves Simultaneously

```python
class MultiTaskXRDRegressor(nn.Module):
    """Predict both parameters and curve residuals."""

    def __init__(self):
        super().__init__()

        # Shared encoder (existing CNN backbone)
        self.encoder = XRDRegressor()  # Reuse existing architecture

        # Remove final layer
        self.encoder.head = nn.Identity()

        # Task-specific heads
        self.param_head = nn.Linear(128, 7)  # Parameters
        self.curve_head = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 701)  # Curve residual correction
        )

    def forward(self, x):
        # Shared features
        features = self.encoder(x)

        # Predict parameters
        params = self.param_head(features)

        # Predict curve correction
        curve_residual = self.curve_head(features)

        return params, curve_residual


def train_multi_task(model, dataloader, beta=0.5):
    """
    Loss = (1-β) * param_loss + β * curve_loss

    Args:
        beta: Weight for curve reconstruction loss
    """
    for batch_Y, batch_X in dataloader:
        pred_params, pred_residual = model(batch_Y)

        # Parameter loss
        param_loss = F.mse_loss(pred_params, batch_X)

        # Curve loss
        # pred_residual corrects the initial curve
        corrected_curve = batch_Y.squeeze(1) + pred_residual
        curve_loss = F.mse_loss(corrected_curve, batch_Y.squeeze(1))

        # Combined loss
        loss = (1 - beta) * param_loss + beta * curve_loss

        loss.backward()
```

**Advantages:**
- ✅ Learns to correct curve errors
- ✅ Shares features (efficient)
- ✅ More robust predictions

**Challenges:**
- ⚠️ More complex training
- ⚠️ Requires tuning β

---

## 🎯 Solution 4: Output Space Constraints ⭐ POST-PROCESSING

### Constrained Optimization After Prediction

Instead of directly using predicted parameters, **refine** them to minimize curve error:

```python
def refine_prediction_with_curve_matching(initial_params, target_curve, max_iter=50):
    """
    Post-process: Optimize parameters to match curve.

    Args:
        initial_params: [7] - Model's initial prediction
        target_curve: [701] - Experimental curve
        max_iter: Optimization iterations

    Returns:
        refined_params: [7] - Refined to match curve better
    """
    from scipy.optimize import minimize

    def objective(params):
        """Curve reconstruction error."""
        pred_curve, _ = xrd.compute_curve_and_profile(params.tolist())
        pred_y = np.log10(pred_curve.Y_R_vseZ + 1e-10)
        true_y = np.log10(target_curve + 1e-10)

        return np.mean((pred_y - true_y)**2)

    # Bounds (physical constraints)
    bounds = [
        (0.001, 0.030),   # Dmax1
        (0.002, None),    # D01 (will add constraint)
        (1000e-8, 7000e-8),  # L1
        (0, None),        # Rp1 (will add constraint)
        (0.002, 0.030),   # D02
        (1000e-8, 7000e-8),  # L2
        (-6000e-8, 0),    # Rp2
    ]

    # Constraints
    constraints = [
        {'type': 'ineq', 'fun': lambda p: p[0] - p[1]},  # D01 <= Dmax1
        {'type': 'ineq', 'fun': lambda p: 0.03 - p[1] - p[4]},  # D01 + D02 <= 0.03
        {'type': 'ineq', 'fun': lambda p: p[2] - p[3]},  # Rp1 <= L1
        {'type': 'ineq', 'fun': lambda p: p[2] - p[5]},  # L2 <= L1
    ]

    # Optimize starting from model prediction
    result = minimize(
        objective,
        x0=initial_params,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': max_iter}
    )

    return result.x


# Usage
pred_params = model(curve)
refined_params = refine_prediction_with_curve_matching(pred_params, curve)
```

**Advantages:**
- ✅ Guarantees curve match (directly optimizes it!)
- ✅ No retraining needed (post-processing)
- ✅ Uses model as "warm start" (faster than random init)

**Challenges:**
- ⚠️ Slow (optimization per sample)
- ⚠️ May get stuck in local minima
- ⚠️ Needs good initialization (model prediction)

**When to use:**
- Critical experimental samples
- Publication-quality results
- When speed is not critical

---

## 🎯 Solution 5: Hierarchical/Coarse-to-Fine Prediction

### Predict in Multiple Stages

```python
class CoarseToFineRegressor(nn.Module):
    """Two-stage predictor: coarse → fine refinement."""

    def __init__(self):
        super().__init__()

        # Stage 1: Coarse prediction (all parameters)
        self.coarse_net = XRDRegressor(n_out=7)

        # Stage 2: Fine-tune sensitive parameters only
        self.refinement_net = nn.Sequential(
            nn.Linear(7 + 128, 128),  # Input: coarse params + features
            nn.ReLU(),
            nn.Linear(128, 3)  # Output: corrections for [Rp1, L2, Rp2]
        )

        self.encoder = XRDRegressor()
        self.encoder.head = nn.Identity()

    def forward(self, x):
        # Extract features
        features = self.encoder(x)

        # Stage 1: Coarse prediction
        coarse_params = self.coarse_net(x)

        # Stage 2: Refine sensitive parameters
        combined = torch.cat([coarse_params, features], dim=-1)
        corrections = self.refinement_net(combined)

        # Apply corrections to sensitive params [Rp1, L2, Rp2] = indices [3, 5, 6]
        refined_params = coarse_params.clone()
        refined_params[:, [3, 5, 6]] += corrections

        return refined_params


def train_hierarchical(model, dataloader):
    """Train coarse network first, then refinement."""
    # Phase 1: Train coarse network (10 epochs)
    for epoch in range(10):
        for batch_Y, batch_X in dataloader:
            model.coarse_net.train()
            model.refinement_net.eval()

            coarse_pred = model.coarse_net(batch_Y)
            loss = F.mse_loss(coarse_pred, batch_X)
            loss.backward()

    # Phase 2: Train refinement (5 epochs, frozen coarse)
    for epoch in range(5):
        for batch_Y, batch_X in dataloader:
            model.coarse_net.eval()
            model.refinement_net.train()

            refined_pred = model(batch_Y)

            # Focus on sensitive parameters
            sensitive_indices = [3, 5, 6]  # Rp1, L2, Rp2
            loss = F.mse_loss(
                refined_pred[:, sensitive_indices],
                batch_X[:, sensitive_indices]
            )
            loss.backward()
```

**Advantages:**
- ✅ Specializes refinement for hard parameters
- ✅ Coarse prediction stabilizes training
- ✅ Can use stronger supervision for refinement

**Challenges:**
- ⚠️ More complex training protocol
- ⚠️ Requires identifying sensitive parameters beforehand

---

## 🎯 Solution 6: Higher Resolution Sampling for Sensitive Parameters

### Augment Dataset with Dense Sampling

Current sampling: Uniform grid

**Problem:** Equal spacing in parameter space ≠ equal spacing in curve space

**Solution:** Sample more densely for sensitive parameters

```python
def generate_sensitivity_aware_dataset(n_samples=100000):
    """Generate dataset with higher resolution for sensitive params."""

    # Sensitivity factors (measured empirically)
    sensitivity = {
        'Dmax1': 1.0,
        'D01': 1.0,
        'L1': 1.5,
        'Rp1': 2.0,  # More samples
        'D02': 1.2,
        'L2': 2.5,   # More samples
        'Rp2': 3.0,  # MOST samples
    }

    # Grid steps inversely proportional to sensitivity
    steps = {}
    for param, sens in sensitivity.items():
        range_size = RANGES[param][1] - RANGES[param][0]
        # More sensitive → smaller steps → more samples
        steps[param] = int(20 * sens)  # Adjust multiplier for desired total

    # Generate samples
    samples = []
    for Dmax1 in np.linspace(RANGES['Dmax1'][0], RANGES['Dmax1'][1], steps['Dmax1']):
        for D01 in np.linspace(RANGES['D01'][0], min(Dmax1, RANGES['D01'][1]), steps['D01']):
            if D01 + D02 > 0.03:
                continue
            for L1 in np.linspace(RANGES['L1'][0], RANGES['L1'][1], steps['L1']):
                for Rp1 in np.linspace(RANGES['Rp1'][0], min(L1, RANGES['Rp1'][1]), steps['Rp1']):
                    for D02 in np.linspace(RANGES['D02'][0], 0.03 - D01, steps['D02']):
                        for L2 in np.linspace(RANGES['L2'][0], min(L1, RANGES['L2'][1]), steps['L2']):
                            # Extra dense for Rp2!
                            for Rp2 in np.linspace(RANGES['Rp2'][0], RANGES['Rp2'][1], steps['Rp2']):
                                params = [Dmax1, D01, L1, Rp1, D02, L2, Rp2]
                                samples.append(params)

    return samples
```

**Advantages:**
- ✅ Better coverage where it matters
- ✅ Model sees more variation in sensitive params
- ✅ Reduces interpolation error

**Challenges:**
- ⚠️ Dataset size explodes (may need >500k samples)
- ⚠️ Computation time for curve generation

---

## 🎯 Solution 7: Attention Mechanisms for Position Parameters

### Architecture Modification for Rp1, Rp2

Position parameters (Rp1, Rp2) require knowing WHERE on the curve the peak/features are.

Current model: Uses global pooling (AttentionPool1d), but may not capture fine position.

**Enhancement:** Multi-scale attention

```python
class PositionAwareRegressor(nn.Module):
    """Enhanced model with position-sensitive heads."""

    def __init__(self):
        super().__init__()

        # Shared encoder (existing)
        self.encoder = XRDRegressor()

        # Position-sensitive branch for Rp1, Rp2
        self.position_attention = nn.MultiheadAttention(
            embed_dim=128,
            num_heads=8,
            dropout=0.1
        )

        # Separate heads
        self.param_head = nn.Linear(128, 5)  # [Dmax1, D01, L1, D02, L2]
        self.position_head = nn.Linear(128, 2)  # [Rp1, Rp2]

    def forward(self, x):
        # Encode
        features = self.encoder(x)  # [batch, 128, L]

        # Global features for amplitude/thickness params
        global_features = torch.mean(features, dim=-1)  # [batch, 128]
        params_amp = self.param_head(global_features)

        # Position-aware features for Rp1, Rp2
        # Self-attention over spatial dimension
        features_pos = features.permute(2, 0, 1)  # [L, batch, 128]
        attended, _ = self.position_attention(features_pos, features_pos, features_pos)
        attended = attended.permute(1, 0, 2)  # [batch, L, 128]
        attended = torch.mean(attended, dim=1)  # [batch, 128]

        params_pos = self.position_head(attended)

        # Combine
        all_params = torch.cat([
            params_amp[:, :2],   # Dmax1, D01
            params_amp[:, 2:3],  # L1
            params_pos[:, 0:1],  # Rp1
            params_amp[:, 3:4],  # D02
            params_amp[:, 4:5],  # L2
            params_pos[:, 1:2],  # Rp2
        ], dim=-1)

        return all_params
```

**Advantages:**
- ✅ Specialized architecture for different parameter types
- ✅ Better position encoding
- ✅ More expressive

**Challenges:**
- ⚠️ More complex model
- ⚠️ Longer training time

---

## 📊 Recommended Implementation Priority

### **For your thesis - Practical Order:**

#### 1. **Sensitivity-Aware Loss Weights** (Week 1) ⭐ START HERE
   - **Effort:** 1 day
   - **Code:** 30 lines
   - **Impact:** Medium (10-20% improvement expected)
   - **File:** `model_train_sensitivity.py`

#### 2. **Post-Processing Refinement** (Week 1-2) ⭐ CRITICAL SAMPLES
   - **Effort:** 2-3 days
   - **Code:** 100 lines
   - **Impact:** High for specific cases (your experiment!)
   - **File:** `refine_prediction.py`

#### 3. **Higher Resolution Sampling** (Week 2-3)
   - **Effort:** 1 week (dataset generation)
   - **Code:** Modify `dataset_stratified_7d.py`
   - **Impact:** High (better coverage)
   - **Dataset:** `dataset_100000_adaptive_7d.pkl`

#### 4. **Multi-Task Learning** (Optional - Research Extension)
   - **Effort:** 2-3 weeks
   - **Code:** New model architecture
   - **Impact:** High (publication quality)
   - **For:** Master's thesis extension / future paper

#### 5. **Differentiable Physics** (Future Work)
   - **Effort:** 1-2 months
   - **Code:** Port xrd.py to PyTorch
   - **Impact:** Very high (but risky)
   - **For:** PhD thesis / next project

---

## 🚀 Quick Start: Implement Solution #1 NOW

I'll create the implementation files for sensitivity-aware training...
