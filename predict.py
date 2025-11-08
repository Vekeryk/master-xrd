"""
XRD Curve Predictor
===================
Predicts deformation parameters from XRD curve.

Usage:
    predict.exe input_curve.txt output_params.txt

Input: Full XRD curve (701 points)
Output: 7 deformation parameters

Exit: 0 = success, 1 = error
"""

import sys
import torch
import numpy as np
from pathlib import Path
from model_common import XRDRegressor, RANGES, PARAM_NAMES, apply_noise_tail

# --- Check arguments ---
if len(sys.argv) != 3:
    try:
        with open('predict_error.log', 'w') as f:
            f.write("Error: Missing arguments\n")
            f.write("Usage: predict.exe input_curve.txt output_params.txt\n")
            f.write(f"Got {len(sys.argv)-1} arguments\n")
    except:
        pass
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]
model_path = 'checkpoints/dataset_1000_dl100_7d_curve_val_best_curve.pt'

# --- Check files exist ---
if not Path(input_file).exists():
    try:
        with open('predict_error.log', 'w') as f:
            f.write(f"Error: Input file not found: {input_file}\n")
    except:
        pass
    sys.exit(1)

if not Path(model_path).exists():
    try:
        with open('predict_error.log', 'w') as f:
            f.write(f"Error: Model checkpoint not found: {model_path}\n")
    except:
        pass
    sys.exit(1)

try:
    # --- Load input curve ---
    with open(input_file, 'r') as f:
        curve_raw = np.array([float(line.strip())
                             for line in f if line.strip()], dtype=np.float32)

    # --- Apply noise tail (crop by peak + noise processing) ---
    curve_cropped = apply_noise_tail(curve_raw)

    # --- Preprocessing (must match NormalizedXRDDataset) ---
    # Clip to avoid log(0)
    curve_clipped = np.clip(curve_cropped, 1e-12, None)

    # Log10 transform
    curve_log = np.log10(curve_clipped)

    # Normalize to [0,1]
    curve_min = curve_log.min()
    curve_max = curve_log.max()
    curve_norm = (curve_log - curve_min) / (curve_max - curve_min + 1e-12)

    # --- Load model ---
    device = torch.device('cpu')
    model = XRDRegressor().to(device)

    checkpoint = torch.load(
        model_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model']

    # Remove hann_window if present
    if 'hann_window' in state_dict:
        state_dict.pop('hann_window')

    model.load_state_dict(state_dict, strict=False)
    model.eval()  # CRITICAL!

    # --- Predict ---
    with torch.no_grad():
        # Convert to tensor [1, 1, 661]
        curve_tensor = torch.from_numpy(
            curve_norm).unsqueeze(0).unsqueeze(0).to(device)

        # Get normalized predictions [1, 7]
        params_norm = model(curve_tensor)[0].cpu().numpy()

    # --- Denormalize to physical units ---
    params_phys = np.zeros(7, dtype=np.float64)
    for i, name in enumerate(PARAM_NAMES):
        min_val, max_val = RANGES[name]
        params_phys[i] = min_val + (max_val - min_val) * params_norm[i]

    # --- Save output ---
    with open(output_file, 'w') as f:
        f.write("# XRD Deformation Parameters\n")
        for i, name in enumerate(PARAM_NAMES):
            value = params_phys[i]
            # Format: lengths in scientific, deformations in fixed
            if name in ['L1', 'Rp1', 'L2', 'Rp2']:
                f.write(f"{name:<8} {value:.6e}\n")
            else:
                f.write(f"{name:<8} {value:.6f}\n")

    # Success
    sys.exit(0)

except Exception as e:
    # Error - log and exit
    try:
        with open('predict_error.log', 'w') as f:
            f.write(f"Error: {str(e)}\n")
    except:
        pass
    sys.exit(1)
