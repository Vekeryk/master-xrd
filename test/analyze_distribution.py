"""Analyze distribution of parameters in 7D dataset"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from model_common import PARAM_NAMES

# Load dataset
DATASET_PATH = "datasets/dataset_10000_dl100_7d_20251030_124511.pkl"

with open(DATASET_PATH, "rb") as f:
    data = pickle.load(f)

X = data['X']

# Analyze distribution for each parameter
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

for i, param in enumerate(PARAM_NAMES):
    ax = axes[i]

    # Convert to Angstroms for L, Rp
    if param in ['L1', 'Rp1', 'L2', 'Rp2']:
        values = X[:, i] * 1e8
        unit = ' (Å)'
    else:
        values = X[:, i]
        unit = ''

    # Count occurrences of each unique value
    unique_vals, counts = np.unique(values, return_counts=True)

    # Bar plot
    ax.bar(range(len(unique_vals)), counts, alpha=0.7, edgecolor='black')
    ax.set_xlabel(f'{param}{unit}')
    ax.set_ylabel('Count')
    ax.set_title(f'{param} Distribution ({len(unique_vals)} unique values)')
    ax.set_xticks(range(len(unique_vals)))
    ax.set_xticklabels([f'{v:.0f}' if param in ['L1', 'Rp1', 'L2', 'Rp2'] else f'{v:.4f}' for v in unique_vals],
                       rotation=45, ha='right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # Print distribution
    print(f"\n{param}:")
    print(f"  Unique values: {len(unique_vals)}")
    print(f"  Min count: {counts.min():,}")
    print(f"  Max count: {counts.max():,}")
    print(f"  Mean count: {counts.mean():.1f}")
    print(f"  Std count: {counts.std():.1f}")
    print(f"  Ratio max/min: {counts.max() / counts.min():.2f}x")

# Remove extra subplots
for i in range(len(PARAM_NAMES), len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.savefig('analysis_report/7d_distributions_bar.png',
            dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 80)
print("✅ Saved to analysis_report/7d_distributions_bar.png")
print("=" * 80)
