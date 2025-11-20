#!/usr/bin/env python3
"""
Приклад генерації датасету з окремими відсотками для кожного параметра
========================================================================
"""

from generate_targeted_dataset import generate_targeted_dataset, EXPERIMENT_PARAMS

# =============================================================================
# CUSTOM RANGES (відсотки від центрального значення)
# =============================================================================

range_pct_dict = {
    'Dmax1': 50,  # ±50% від 0.008094 = [0.004047, 0.012141]
    'D01': 60,    # ±60% від 0.000943 = [0.000377, 0.001509]
    'L1': 10,     # ±10% від 5200Å = [4680Å, 5720Å]
    'Rp1': 20,    # ±20% від 3500Å = [2800Å, 4200Å]
    'D02': 40,    # ±40% від 0.00255 = [0.00153, 0.00357]
    'L2': 15,     # ±15% від 3000Å = [2550Å, 3450Å]
    'Rp2': 200,   # ±200% від -50Å = [-150Å, +100Å]
}

# =============================================================================
# CUSTOM GRID STEPS (відсотки від ширини range)
# =============================================================================

step_pct_dict = {
    'Dmax1': 1,   # 1% від range width - дрібна сітка
    'D01': 2,     # 2% від range width
    'L1': 5,      # 5% від range width
    'Rp1': 5,     # 5% від range width
    'D02': 2,     # 2% від range width
    'L2': 5,      # 5% від range width
    'Rp2': 10,    # 10% від range width - груба сітка
}

# =============================================================================
# ГЕНЕРАЦІЯ
# =============================================================================

if __name__ == "__main__":
    dataset = generate_targeted_dataset(
        experiment_params=EXPERIMENT_PARAMS,
        n_samples=5000,
        dl=100e-8,
        range_pct_dict=range_pct_dict,  # Custom ranges
        step_pct_dict=step_pct_dict,    # Custom steps
        output_dir="datasets"
    )

    print(f"\n✅ Dataset generated with custom percentages per parameter!")
    print(f"   File: {dataset['generation_params'].get('timestamp')}")
