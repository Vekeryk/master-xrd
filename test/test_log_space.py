"""
Test Log-Space Transform Impact
================================
Demonstrates the difference between log-space and linear-space normalization
for XRD curves.
"""

import numpy as np
import matplotlib.pyplot as plt

# Simulate XRD rocking curve with peak and oscillations
x = np.linspace(-100, 3200, 700)
peak = 0.02 * np.exp(-((x - 50) ** 2) / (2 * 100 ** 2))
oscillations = 0.001 * np.sin(x / 100) * np.exp(-x / 1000)
curve = peak + oscillations + 1e-5  # Add small baseline

# Linear-space normalization
curve_linear = curve / curve.max()

# Log-space normalization
curve_log_raw = np.log10(curve + 1e-10)
curve_log = (curve_log_raw - curve_log_raw.min()) / (curve_log_raw.max() - curve_log_raw.min())

# Plot comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Original curve
axes[0].plot(x, curve)
axes[0].set_title("Original XRD Curve")
axes[0].set_xlabel("Δθ (arcsec)")
axes[0].set_ylabel("Intensity")
axes[0].set_ylim(0, 0.022)
axes[0].grid(True, alpha=0.3)

# Linear normalization
axes[1].plot(x, curve_linear)
axes[1].set_title("Linear Normalization")
axes[1].set_xlabel("Δθ (arcsec)")
axes[1].set_ylabel("Normalized Intensity")
axes[1].set_ylim(-0.1, 1.1)
axes[1].grid(True, alpha=0.3)

# Log-space normalization
axes[2].plot(x, curve_log)
axes[2].set_title("Log-Space Normalization")
axes[2].set_xlabel("Δθ (arcsec)")
axes[2].set_ylabel("Normalized Intensity")
axes[2].set_ylim(-0.1, 1.1)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("log_space_comparison.png", dpi=150)
print("✅ Saved comparison to log_space_comparison.png")

# Show statistics
print("\n📊 Comparison Statistics:")
print(f"Linear norm - min: {curve_linear.min():.6f}, max: {curve_linear.max():.6f}")
print(f"Log norm    - min: {curve_log.min():.6f}, max: {curve_log.max():.6f}")

# Dynamic range in oscillation region (x > 500)
osc_region = x > 500
linear_range = curve_linear[osc_region].max() - curve_linear[osc_region].min()
log_range = curve_log[osc_region].max() - curve_log[osc_region].min()

print(f"\nDynamic range in oscillation region (Δθ > 500):")
print(f"Linear: {linear_range:.6f}")
print(f"Log:    {log_range:.6f}")
print(f"Improvement: {log_range / linear_range:.2f}x")

print("\n💡 Log-space enhances low-intensity features (oscillations, tails)")
print("   → Better for XRD curve analysis with interference patterns")
