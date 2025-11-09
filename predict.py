"""
XRD Curve Predictor
===================
Predicts deformation parameters from XRD curve.

Usage:
    predict.exe model_path input_curve.txt output_params.txt

Arguments:
    model_path: Path to trained model checkpoint (.pt file)
    input_curve.txt: Full XRD curve (701 points or variable)
    output_params.txt: Output file (7 parameters, values only, no names)

Exit: 0 = success, 1 = error
"""

import sys
import torch
import numpy as np
from pathlib import Path
from model_common import XRDRegressor, RANGES, PARAM_NAMES, NormalizedXRDDataset, preprocess_curve

# --- Check arguments ---
if len(sys.argv) != 4:
    try:
        with open('predict_error.log', 'w') as f:
            f.write("Error: Missing arguments\n")
            f.write(
                "Usage: predict.exe model_path input_curve.txt output_params.txt\n")
            f.write(f"Got {len(sys.argv)-1} arguments\n")
    except:
        pass
    sys.exit(1)

model_path = sys.argv[1]
input_file = sys.argv[2]
output_file = sys.argv[3]

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
            f.write(f"Expected: {Path(model_path).absolute()}\n")
    except:
        pass
    sys.exit(1)

try:
    device = torch.device('cpu')
    checkpoint = torch.load(
        model_path, map_location=device, weights_only=False)

    # Get expected curve length from checkpoint metadata
    # Default 651 if not in metadata
    expected_length = checkpoint.get('L', 651)

    model = XRDRegressor().to(device)
    state_dict = checkpoint['model']

    # Remove hann_window if present
    if 'hann_window' in state_dict:
        state_dict.pop('hann_window')

    model.load_state_dict(state_dict, strict=False)
    model.eval()  # CRITICAL!

    # --- Load input curve ---
    with open(input_file, 'r') as f:
        curve_raw = np.array([float(line.strip())
                             for line in f if line.strip()], dtype=np.float32)

    curve_cropped = preprocess_curve(
        curve_raw, crop_by_peak=True, target_length=expected_length)

    X_dummy = torch.zeros((1, 7))
    Y_input = torch.from_numpy(curve_cropped).unsqueeze(0).float()

    dataset = NormalizedXRDDataset(
        X_dummy, Y_input, log_space=True, train=False)

    curve_preprocessed, _ = dataset[0]

    # --- Predict ---
    with torch.no_grad():
        # Add batch dimension [1, 1, L] (channel already added by dataset)
        curve_tensor = curve_preprocessed.unsqueeze(0).to(device)

        # Get normalized predictions [1, 7]
        params_norm = model(curve_tensor)[0].cpu().numpy()

    # --- Denormalize to physical units ---
    params_phys = np.zeros(7, dtype=np.float64)
    for i, name in enumerate(PARAM_NAMES):
        min_val, max_val = RANGES[name]
        params_phys[i] = min_val + (max_val - min_val) * params_norm[i]

    with open(output_file, 'w') as f:
        for value in params_phys:
            # Always use scientific notation for consistency
            f.write(f"{value:.6e}\n")

    # Success
    sys.exit(0)

except Exception as e:
    # Error - log and exit
    try:
        with open('predict_error.log', 'w') as f:
            f.write(f"Error: {str(e)}\n")
            import traceback
            f.write(traceback.format_exc())
    except:
        pass
    sys.exit(1)
