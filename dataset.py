import numpy as np
import torch
import xrd
from helpers import get_device
from importlib import reload
reload(xrd)


# Функція для генерації датасету
def generate_train_dataset(n_samples):
    X = []
    Y = []
    i = 0
    while i < n_samples:
        rng = np.random.default_rng()
        # Use normal distribution for _Dmax1 and _D01
        # _Dmax1 = float(rng.normal(loc=0.016, scale=0.005))
        # _Dmax1 = np.clip(_Dmax1, 0.002, 0.030)

        # _D01 = float(rng.normal(loc=0.004, scale=0.002))
        # _D01 = np.clip(_D01, 0.002, _Dmax1)

        _Dmax1 = float(rng.uniform(0.002, 0.030))

        # (_D01 + _D02) < 0.030
        _D01 = float(rng.uniform(0.002, _Dmax1))

        if _Dmax1 - _D01 < 0.002:
            print("Invalid D01 value:", _D01, _Dmax1 - _D01, i)
            continue

        _D02 = float(rng.uniform(0.002, _Dmax1 - _D01))

        # Structural parameters - discrete with 10 Å step
        # _Rp1 <= 0.75 * _L1
        # _L2

        _L1 = np.random.choice(np.arange(1000, 7001, 10))
        _Rp1 = np.random.choice(np.arange(20, int(0.75 * _L1), 10))

        if int(0.75 * _L1) < 500:
            print("Invalid L1 value:", _L1, int(0.75 * _L1))
            continue

        _L2 = np.random.choice(np.arange(500, int(0.75 * _L1), 10))
        _Rp2 = np.random.choice(np.arange(-6000, -20, 10))

        _L1 *= 1e-8
        _Rp1 *= 1e-8
        _L2 *= 1e-8
        _Rp2 *= 1e-8

        # Dmax1 = 0.01305
        # D01 = 0.0017
        # L1 = 5800e-8
        # Rp1 = 3500e-8
        # D02 = 0.004845
        # L2 = 4000e-8
        # Rp2 = -500e-8
        # L1, Rp1, L2, Rp2

        params_obj = xrd.DeformationProfile(
            Dmax1=_Dmax1,
            D01=_D01,
            L1=_L1,
            Rp1=_Rp1,
            D02=_D02,
            L2=_L2,
            Rp2=_Rp2,
            Dmin=0.0001,
            dl=100e-8
        )

        curve, profile = xrd.compute_curve_and_profile(params_obj=params_obj)

        X.append([_Dmax1, _D01, _L1, _Rp1, _D02, _L2, _Rp2])
        Y.append(curve.ML_Y)  # Keep as raw data until final conversion

        i += 1

    device = get_device()
    # Convert both X and Y to tensors consistently at the end
    X = torch.tensor(X, dtype=torch.float32, device=device)
    Y = torch.tensor(Y, dtype=torch.float32, device=device)

    return X, Y
