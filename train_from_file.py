#!/usr/bin/env python3
"""
Train a 1D‑CNN regressor directly on your pickled dataset
========================================================

Assumes a pickle file with a dict like: {"X": tensor/ndarray [N,3], "Y": tensor/ndarray [N,170]}
Where:
  • X = [Dmax1, D01, D02] (targets we want to predict)
  • Y = rocking curve samples (inputs)

Run:
  pip install torch numpy
  python train_from_pickle.py --data dataset_10_000.pkl --epochs 80

It will save a model to checkpoints/pickle_model.pt and print quick MAE metrics.
"""

from __future__ import annotations
import argparse
import pickle
from pathlib import Path
import numpy as np
import torch
import time
import torch.nn as nn
import torch.nn.functional as F

MODEL_PATH = "checkpoints/pickle_model_100k_00490.pt"

# -------------------
# Parameter ranges (for target normalization)
# Simple numeric bounds only (no references). Order of keys defines
# the parameter order and must match the columns of X in your dataset.
# To add new parameters, just append here and ensure X has the same order.
# -------------------
# RANGES = {
#     "Dmax1": (0.0125, 0.0135),
#     # If you prefer: "D01": (0.0010, 0.0135)
#     "D01": (0.0010, 0.0100),
#     "D02": (0.0010, 0.0100),
# }

# Example of extended numeric-only spec (uncomment when X has 7 cols in same order):

RANGES = {
    "Dmax1": (0.002, 0.030),
    "D01": (0.0010, 0.030),
    "L1": (1000e-8, 7000e-8),
    "Rp1": (0.0, 7000e-8),
    "D02": (0.0020, 0.0300),
    "L2": (1000e-8, 7000e-8),
    "Rp2": (-6000e-8, 0.0),
}

RANGES = {
    "Dmax1": (0.002, 0.030),
    "D01": (0.0010, 0.030),
    # "L1": (1000.0, 7000.0),
    # "Rp1": (0.0, 7000.0),
    "L1": (1000e-8, 7000e-8),
    "Rp1": (0.0, 7000e-8),
    "D02": (0.0020, 0.0300),
    # "L2": (1000.0, 7000.0),
    # "Rp2": (-6000.0, 0.0),
    "L2": (1000e-8, 7000e-8),
    "Rp2": (-6000e-8, 0.0),
}

# RANGES = {
#     "Dmax1": (0.0010, 0.0300),
#     # simple numeric bounds; dataset already respects D01 ≤ Dmax1
#     "D01": (0.0010, 0.0135),
#     "L1": (1000e-8, 7000e-8),
#     "Rp1": (0.0, 7000e-8),
#     "D02": (0.0020, 0.0300),
#     "L2": (1000e-8, 7000e-8),
#     "Rp2": (-6000e-8, 0.0),
# }

# Centralized param names (order matters & drives model output dim)
PARAM_NAMES = list(RANGES.keys())


def set_seed(seed: int = 1234):
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device():
    if torch.backends.mps.is_available():
        print("MPS available:", torch.backends.mps.is_available())
        return torch.device("mps")

    if torch.cuda.is_available():
        print("CUDA available:", torch.cuda.is_available())
        return torch.device("cuda")

    print("No GPU available, using CPU")
    return torch.device("cpu")


def load_dataset(path: Path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    X = torch.as_tensor(data["X"]).float()  # [N,P]
    Y = torch.as_tensor(data["Y"]).float()  # [N,L]
    assert X.ndim == 2, f"X must be 2D [N,P], got {tuple(X.shape)}"
    assert Y.ndim == 2, f"Y must be 2D [N,L], got {tuple(Y.shape)}"
    assert X.size(1) == len(PARAM_NAMES), (
        f"X has {X.size(1)} columns but RANGES defines {len(PARAM_NAMES)} params.\n"
        f"PARAM_NAMES={PARAM_NAMES}\n"
        "Update RANGES or your dataset so they match in size & order.")

    print(f"Loaded dataset from {path}")
    print(f"X: {tuple(X.shape)}  Y: {tuple(Y.shape)}")
    return X, Y


class PickleXRDDataset(torch.utils.data.Dataset):
    def __init__(self, X: torch.Tensor, Y: torch.Tensor, log_space: bool = True, train: bool = True):
        self.X = X.clone()
        self.Y = Y.clone()
        self.train = train
        self.log_space = log_space
        self.param_names = PARAM_NAMES

        # Normalize inputs (curves) per‑sample to [0,1] (optionally after log10)
        if self.log_space:
            # Yp = torch.log10(
            #     self.Y / (self.Y.amax(dim=1, keepdim=True) + 1e-12) + 1e-8)

            Y_safe = self.Y + 1e-10  # запобігти log(0)
            Yp = torch.log10(Y_safe)
            # Потім нормалізувати
            Yp = (Yp - Yp.amin(dim=1, keepdim=True)) / \
                (Yp.amax(dim=1, keepdim=True) -
                 Yp.amin(dim=1, keepdim=True) + 1e-12)
        else:
            Yp = self.Y / (self.Y.amax(dim=1, keepdim=True) + 1e-12)
        Yp = (Yp - Yp.amin(dim=1, keepdim=True)) / (Yp.amax(dim=1,
                                                            keepdim=True) - Yp.amin(dim=1, keepdim=True) + 1e-12)
        self.Yn = Yp.unsqueeze(1)  # [N,1,L]

        # Normalize targets to [0,1] using simple numeric RANGES
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
        return self.Yn[idx], self.Tn[idx]


class ResidualBlock(nn.Module):
    def __init__(self, c: int, dilation: int = 1):
        super().__init__()
        # pad = dilation * 3

        kernel_size = 7
        # Правильный padding для сохранения размерности
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
    def __init__(self, n_out: int | None = None):
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
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)


@torch.no_grad()
def denorm_params(p_norm: torch.Tensor) -> torch.Tensor:
    """Denormalize predictions p_norm [B,P] → physical params [B,P] using numeric RANGES."""
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


def train(args):
    set_seed(args.seed)
    device = get_device()
    X, Y = load_dataset(Path(args.data))
    n = X.size(0)

    # Train/val split
    # ~20% val, but never >= n, and capped at 1000
    idx = torch.randperm(n)
    n_val = int(0.2 * n)
    n_val = max(1, min(n_val, n - 1, 1000))
    tr_idx, va_idx = idx[n_val:], idx[:n_val]
    print(f"Split: train={len(tr_idx)} | val={len(va_idx)}")

    ds_tr = PickleXRDDataset(X[tr_idx], Y[tr_idx],
                             log_space=not args.no_log, train=True)
    ds_va = PickleXRDDataset(X[va_idx], Y[va_idx],
                             log_space=not args.no_log, train=False)
    dl_tr = torch.utils.data.DataLoader(
        ds_tr, batch_size=args.bs, shuffle=True)
    dl_va = torch.utils.data.DataLoader(
        ds_va, batch_size=args.bs, shuffle=False)

    model = XRDRegressor().to(device)

    # opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    opt = torch.optim.AdamW(model.parameters(), lr=0.0015, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )

    best = float("inf")
    Path("checkpoints").mkdir(exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # Set model to training mode
        model.train()
        run = 0.0
        for y, t in dl_tr:
            y, t = y.to(device), t.to(device)
            # 1. Forward pass
            p = model(y)
            # 2. Compute loss
            loss = F.smooth_l1_loss(p, t)
            # 3. Optimizer zero grad
            opt.zero_grad()
            # 4. Perform backpropagation on the loss with respect to the model parameters
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # w = torch.tensor([1.0, 1.0, 1.15, 1.0, 1.0, 1.15, 1.6], device=y.device)[:p.size(1)]
            # loss = (w * F.smooth_l1_loss(p, t, reduction='none')).mean()

            # 5. Optimizer step (perform gradient descent)
            opt.step()
            run += loss.item() * y.size(0)
        tr_loss = run / len(ds_tr)

        # Set model to evaluation mode (turn off gradient tracking)
        model.eval()
        with torch.no_grad():
            run = 0.0
            for y, t in dl_va:
                y, t = y.to(device), t.to(device)
                p = model(y)
                run += F.smooth_l1_loss(p, t).item() * y.size(0)
            va_loss = run / len(ds_va)

        sched.step(va_loss)

        curr_lr = opt.param_groups[0]['lr']
        print(
            f"Epoch {epoch:03d} | train {tr_loss:.5f} | val {va_loss:.5f} | lr {curr_lr:.2e}")

        if va_loss < best:
            best = va_loss
            torch.save({"model": model.state_dict(), "L": Y.size(1)},
                       MODEL_PATH)


@torch.no_grad()
def quick_eval(args):
    device = get_device()
    X, Y = load_dataset(Path(args.data))

    ckpt = torch.load(MODEL_PATH, map_location=device)
    model = XRDRegressor().to(device)
    model.load_state_dict(ckpt["model"])  # best checkpoint from training
    model.eval()

    ds = PickleXRDDataset(X, Y, log_space=not args.no_log, train=False)
    dl = torch.utils.data.DataLoader(ds, batch_size=256)

    preds = []
    with torch.no_grad():
        for y, _ in dl:
            p = model(y.to(device))
            preds.append(p.to(device))
    P = torch.cat(preds, dim=0)
    Theta_hat = denorm_params(P)

    # Aggregate metrics
    abs_err = torch.abs(Theta_hat - X)  # [N,P]
    mae = abs_err.mean(dim=0)
    rng = torch.tensor([float(RANGES[k][1] - RANGES[k][0])
                       for k in PARAM_NAMES], device=device)
    mae_pct_range = (mae / rng) * 100.0
    mean_true = X.mean(dim=0)
    mae_pct_true = (mae / (mean_true + 1e-12)) * 100.0

    print("MAE by parameter (abs / % of range / % of mean):")
    for j, k in enumerate(PARAM_NAMES):
        print(
            f"  {k}: {mae[j].item():.6f}  |  {mae_pct_range[j].item():.2f}% rng  |  {mae_pct_true[j].item():.2f}% mean")

    # Optional: show a few random examples
    output_to_copy = ''
    if getattr(args, 'show', 0) > 0:
        print("Examples (true vs predicted):")
        idx = torch.randint(0, X.size(0), (args.show,))
        for i in idx.tolist():
            t = X[i]
            p = Theta_hat[i]
            e = (p - t)
            t_str = ",".join(f"{v:.6f}" if i in [
                             0, 1, 5] else f"{v:.3e}" for i, v in enumerate(t.tolist()))
            p_str = ",".join(f"{v:.6f}" if i in [
                             0, 1, 5] else f"{v:.3e}" for i, v in enumerate(p.tolist()))
            e_str = ",".join(f"{v:+.3e}" for v in e.tolist())
            print(
                f"  #{i:05d} | err=[{e_str}]")
            output_to_copy += f"  ([{t_str}], [{p_str}]),\n"
        print("\nCopyable output (for e.g. error analysis):\n")
        print(output_to_copy)


def build_parser():
    p = argparse.ArgumentParser(
        description="Train CNN on pickled HRXRD dataset")
    p.add_argument("--data", type=str, required=True,
                   help="Path to dataset_*.pkl with keys X,Y")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--no-log", action="store_true",
                   help="Disable log10 normalization of curves")
    p.add_argument("--eval", action="store_true",
                   help="Only run quick evaluation of saved model")
    p.add_argument("--show", type=int, default=0,
                   help="Print N random examples (true vs pred)")
    return p


def main():
    args = build_parser().parse_args()
    if args.eval:
        quick_eval(args)
    else:
        start_time = time.time()
        train(args)
        elapsed = time.time() - start_time
        print(f"Total training time: {elapsed/60:.2f} minutes")
        quick_eval(args)


if __name__ == "__main__":
    main()
