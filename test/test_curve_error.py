#!/usr/bin/env python3
"""Debug curve error calculation."""

import torch
import numpy as np
from model_common import XRDRegressor, NormalizedXRDDataset, load_dataset, denorm_params
import xrd

# Load dataset
X, Y = load_dataset('datasets/dataset_1000_dl100_7d.pkl', use_full_curve=False)

# Load model
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = XRDRegressor(n_out=7).to(device)
checkpoint = torch.load('checkpoints/approach_baseline.pt', map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model'])
model.eval()

# Create dataset
ds = NormalizedXRDDataset(X, Y, log_space=True, train=False)

# Get one sample
idx = 100
y_input, x_true = ds[idx]

print("="*80)
print("DEBUG: Testing curve error calculation")
print("="*80)

# Predict
with torch.no_grad():
    x_pred = model(y_input.unsqueeze(0).to(device))

# Denormalize
pred_params = denorm_params(x_pred.cpu())[0].numpy()
true_params = X[idx].numpy()

print(f"\nTrue params:  {true_params}")
print(f"Pred params:  {pred_params}")
print(f"Param error:  {np.abs(pred_params - true_params)}")

# Generate curves
print(f"\nGenerating curves...")
pred_curve, _ = xrd.compute_curve_and_profile(pred_params.tolist(), dl=100e-8)
true_curve, _ = xrd.compute_curve_and_profile(true_params.tolist(), dl=100e-8)

print(f"pred_curve.Y_R_vseZ shape: {pred_curve.Y_R_vseZ.shape}")
print(f"true_curve.Y_R_vseZ shape: {true_curve.Y_R_vseZ.shape}")

# Apply cropping
pred_y = pred_curve.Y_R_vseZ[50:701]
true_y = true_curve.Y_R_vseZ[50:701]

print(f"\nAfter cropping [50:701]:")
print(f"pred_y shape: {pred_y.shape}")
print(f"true_y shape: {true_y.shape}")
print(f"pred_y range: [{pred_y.min():.2e}, {pred_y.max():.2e}]")
print(f"true_y range: [{true_y.min():.2e}, {true_y.max():.2e}]")

# Log space
pred_y_log = np.log10(pred_y + 1e-10)
true_y_log = np.log10(true_y + 1e-10)

print(f"\nLog space:")
print(f"pred_y_log range: [{pred_y_log.min():.2f}, {pred_y_log.max():.2f}]")
print(f"true_y_log range: [{true_y_log.min():.2f}, {true_y_log.max():.2f}]")

# Difference
diff = pred_y_log - true_y_log
print(f"\nDifference (pred - true):")
print(f"  min: {diff.min():.6f}")
print(f"  max: {diff.max():.6f}")
print(f"  mean: {diff.mean():.6f}")
print(f"  std: {diff.std():.6f}")

# MSE
mse = np.mean((pred_y_log - true_y_log) ** 2)
print(f"\nMSE (log space): {mse:.12f}")

# Check if arrays are identical
are_identical = np.array_equal(pred_y, true_y)
print(f"\nAre arrays identical? {are_identical}")

if are_identical:
    print("❌ PROBLEM: Predicted and true curves are IDENTICAL!")
    print("   This suggests xrd.compute_curve_and_profile() is returning same curve for different params")
else:
    print(f"✓ Arrays are different (as expected)")

print("\n" + "="*80)
