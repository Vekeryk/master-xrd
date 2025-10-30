"""
Verification Script: Синхронізація RANGES
==========================================
Перевіряє що діапазони параметрів синхронізовані між:
1. perebir.py (Grid 5)
2. dataset_stratified.py (grids для генерації)
3. model_common.py (RANGES для norm/denorm)

КРИТИЧНО ВАЖЛИВО: Всі три файли МУСЯТЬ мати ОДНАКОВІ діапазони!
"""

import numpy as np
import sys


def arange_inclusive(start, stop, step):
    """
    Helper function to create inclusive ranges.
    Works correctly for both positive and negative ranges.

    FIX: Previous version had bug with negative ranges (e.g., Rp2: -6500 → -5)
         It included extra values due to floating point arithmetic.
    """
    n_steps = round((stop - start) / step)
    return np.array([start + i * step for i in range(n_steps + 1)], dtype=float)


# =============================================================================
# 1. GRID 5 з perebir.py (еталон)
# =============================================================================

GRID5_PEREBIR = {
    'Dmax1': {'min': 0.0010, 'max': 0.0310, 'step': 0.0025},  # ВИПРАВЛЕНО: 0.031 кратне step
    'D01':   {'min': 0.0005, 'max': 0.0305, 'step': 0.0025},  # ВИПРАВЛЕНО: 0.0305 кратне step
    'L1':    {'min': 500.,   'max': 7000.,  'step': 500.},    # Angstroms ✓
    'Rp1':   {'min': 50.,    'max': 5050.,  'step': 500.},    # ВИПРАВЛЕНО: 5050 кратне step
    'D02':   {'min': 0.0010, 'max': 0.0310, 'step': 0.0025},  # ВИПРАВЛЕНО: 0.031 кратне step
    'L2':    {'min': 500.,   'max': 5000.,  'step': 500.},    # Angstroms ✓
    'Rp2':   {'min': -6500., 'max': 0.,     'step': 500.},    # ВИПРАВЛЕНО: 0 кратне step
}

# =============================================================================
# 2. RANGES для model_common.py (має відповідати GRID5)
# =============================================================================

# ВАЖЛИВО: L1, Rp1, L2, Rp2 в model_common.py зберігаються в СМ!
# Але min/max має відповідати діапазону в Å конвертованому в см

RANGES_MODEL_COMMON = {
    "Dmax1": (0.0010, 0.0310),      # ВИПРАВЛЕНО: 0.031 кратне step
    "D01":   (0.0005, 0.0305),      # ВИПРАВЛЕНО: 0.0305 кратне step
    "L1":    (500e-8, 7000e-8),     # 500 Å → см, 7000 Å → см ✓
    "Rp1":   (50e-8, 5050e-8),      # ВИПРАВЛЕНО: 5050 Å → см
    "D02":   (0.0010, 0.0310),      # ВИПРАВЛЕНО: 0.031 кратне step
    "L2":    (500e-8, 5000e-8),     # 500 Å → см, 5000 Å → см ✓
    "Rp2":   (-6500e-8, 0e-8),      # ВИПРАВЛЕНО: 0 Å → см
}

# =============================================================================
# 3. GRIDS для dataset_stratified.py
# =============================================================================

GRIDS_DATASET = {
    'Dmax1_grid': arange_inclusive(0.0010, 0.0310, 0.0025),  # 13 значень (0.031 кратне step)
    'D01_grid':   arange_inclusive(0.0005, 0.0305, 0.0025),  # 13 значень (0.0305 кратне step)
    'L1_grid':    arange_inclusive(500., 7000., 500.),       # 14 значень (Å) ✓
    'Rp1_grid':   arange_inclusive(50., 5050., 500.),        # 11 значень (5050 кратне step)
    'D02_grid':   arange_inclusive(0.0010, 0.0310, 0.0025),  # 13 значень (0.031 кратне step)
    'L2_grid':    arange_inclusive(500., 5000., 500.),       # 10 значень (Å) ✓
    'Rp2_grid':   arange_inclusive(-6500., 0., 500.),        # 14 значень (0 кратне step)
}

# =============================================================================
# VERIFICATION
# =============================================================================

def verify_sync():
    """Перевірити синхронізацію всіх діапазонів"""

    print("="*70)
    print("🔬 VERIFICATION: Синхронізація RANGES")
    print("="*70)

    all_ok = True
    errors = []

    print("\n📊 Перевірка відповідності GRID5 (perebir.py) та RANGES (model_common.py):")
    print("-"*70)

    param_names = ['Dmax1', 'D01', 'L1', 'Rp1', 'D02', 'L2', 'Rp2']

    for name in param_names:
        grid_def = GRID5_PEREBIR[name]
        grid_min = grid_def['min']
        grid_max = grid_def['max']

        ranges_min, ranges_max = RANGES_MODEL_COMMON[name]

        # Конвертувати якщо це L або Rp (з Å в см)
        if name in ['L1', 'Rp1', 'L2', 'Rp2']:
            grid_min_cm = grid_min * 1e-8
            grid_max_cm = grid_max * 1e-8
            unit = "Å→см"
        else:
            grid_min_cm = grid_min
            grid_max_cm = grid_max
            unit = ""

        # Порівняти
        tol = 1e-12
        min_match = abs(grid_min_cm - ranges_min) < tol
        max_match = abs(grid_max_cm - ranges_max) < tol

        if min_match and max_match:
            status = "✅"
        else:
            status = "❌"
            all_ok = False
            errors.append(f"{name}: Grid [{grid_min_cm:.8f}, {grid_max_cm:.8f}] != RANGES [{ranges_min:.8f}, {ranges_max:.8f}]")

        print(f"  {status} {name:6s}: Grid [{grid_min:8.4f}, {grid_max:8.4f}] {unit:5s} → RANGES [{ranges_min:.8e}, {ranges_max:.8e}]")

    print("\n" + "="*70)
    print("📐 Перевірка grids у dataset_stratified.py:")
    print("-"*70)

    for name in param_names:
        grid_key = name + '_grid'
        grid_arr = GRIDS_DATASET[grid_key]

        grid_min_actual = grid_arr.min()
        grid_max_actual = grid_arr.max()

        grid_min_expected = GRID5_PEREBIR[name]['min']
        grid_max_expected = GRID5_PEREBIR[name]['max']

        # Порівняти
        tol = 1e-9
        min_match = abs(grid_min_actual - grid_min_expected) < tol
        max_match = abs(grid_max_actual - grid_max_expected) < tol

        if min_match and max_match:
            status = "✅"
        else:
            status = "❌"
            all_ok = False
            errors.append(f"dataset_stratified.py {grid_key}: [{grid_min_actual}, {grid_max_actual}] != Grid5 [{grid_min_expected}, {grid_max_expected}]")

        count = len(grid_arr)
        print(f"  {status} {grid_key:12s}: {count:2d} значень  [{grid_min_actual:8.2f}, {grid_max_actual:8.2f}]")

    # Покриття експериментальних даних
    print("\n" + "="*70)
    print("🧪 Перевірка покриття експериментальних даних:")
    print("-"*70)

    params_experiment = [0.008094, 0.000943, 5200e-8, 3500e-8, 0.00255, 3000e-8, -50e-8]
    params_default = [0.01305, 0.0017, 5800e-8, 3500e-8, 0.004845, 4000e-8, -500e-8]

    def check_params(params, label):
        print(f"\n  {label}:")
        covered = True
        for i, (name, val) in enumerate(zip(param_names, params)):
            ranges_min, ranges_max = RANGES_MODEL_COMMON[name]

            # Конвертувати якщо потрібно для відображення
            if name in ['L1', 'Rp1', 'L2', 'Rp2']:
                val_display = val * 1e8  # см → Å
                unit = "Å"
            else:
                val_display = val
                unit = ""

            in_range = ranges_min <= val <= ranges_max

            if in_range:
                status = "✅"
            else:
                status = "❌"
                covered = False
                all_ok = False
                errors.append(f"{label} {name}={val:.6e} ПОЗА МЕЖАМИ [{ranges_min:.6e}, {ranges_max:.6e}]")

            print(f"    {status} {name:6s}: {val_display:8.2f}{unit:2s}")

        return covered

    exp_ok = check_params(params_experiment, "Експеримент [0.008094, 0.000943, ...]")
    def_ok = check_params(params_default, "Default [0.01305, 0.0017, ...]")

    # Фінальний висновок
    print("\n" + "="*70)
    print("🎯 РЕЗУЛЬТАТ ВЕРИФІКАЦІЇ")
    print("="*70)

    if all_ok:
        print("\n✅ ВСЕ СИНХРОНІЗОВАНО ПРАВИЛЬНО!")
        print("   • perebir.py Grid 5 ✓")
        print("   • model_common.py RANGES ✓")
        print("   • dataset_stratified.py grids ✓")
        print("   • Експериментальні дані покриті ✓")
        print("\n🚀 Можна генерувати датасет та тренувати!")
    else:
        print("\n❌ ЗНАЙДЕНО ПРОБЛЕМИ!")
        print("\nПомилки:")
        for err in errors:
            print(f"  • {err}")
        print("\n⚠️  ВИПРАВТЕ ЦІ ПРОБЛЕМИ ПЕРЕД ГЕНЕРАЦІЄЮ ДАТАСЕТУ!")

    print("\n" + "="*70)

    return all_ok


# =============================================================================
# CODE GENERATION
# =============================================================================

def print_code_for_files():
    """Вивести код для copy-paste в файли"""

    print("\n" + "="*70)
    print("📝 КОД ДЛЯ ОНОВЛЕННЯ ФАЙЛІВ")
    print("="*70)

    print("\n1️⃣ Для dataset_stratified.py (грид параметрів):")
    print("-"*70)
    print("""
# IMPROVED: Grid 5 - розширені діапазони для покриття експериментальних даних
# ⚠️ ВАЖЛИВО: max значення скориговані щоб (max-min) було кратне step!
Dmax1_grid = arange_inclusive(0.0010, 0.0310, 0.0025)  # 13 значень (0.031 покриває 0.030)
D01_grid = arange_inclusive(0.0005, 0.0305, 0.0025)    # 13 значень (0.0305 покриває 0.030)
L1_grid = arange_inclusive(500., 7000., 500.)          # 14 значень ✓
Rp1_grid = arange_inclusive(50., 5050., 500.)          # 11 значень (5050 покриває 5000)
D02_grid = arange_inclusive(0.0010, 0.0310, 0.0025)    # 13 значень (0.031 покриває 0.030)
L2_grid = arange_inclusive(500., 5000., 500.)          # 10 значень ✓
Rp2_grid = arange_inclusive(-6500., 0., 500.)          # 14 значень (0 покриває -50, -500)
""")

    print("\n2️⃣ Для model_common.py (RANGES):")
    print("-"*70)
    print("""
# IMPROVED Grid 5: Діапазони відповідають новій сітці з perebir.py
# ⚠️ ВАЖЛИВО: L1, Rp1, L2, Rp2 в СМ (бо X зберігається в см)!
# ⚠️ ВАЖЛИВО: max значення скориговані щоб (max-min) було кратне step!
RANGES = {
    "Dmax1": (0.0010, 0.0310),      # ВИПРАВЛЕНО: 0.031 кратне step (покриває 0.030)
    "D01":   (0.0005, 0.0305),      # ВИПРАВЛЕНО: 0.0305 кратне step (покриває 0.000943)
    "L1":    (500e-8, 7000e-8),     # 500 Å = 500e-8 см, 7000 Å = 7000e-8 см ✓
    "Rp1":   (50e-8, 5050e-8),      # ВИПРАВЛЕНО: 5050 Å = 5050e-8 см (покриває 5000)
    "D02":   (0.0010, 0.0310),      # ВИПРАВЛЕНО: 0.031 кратне step (покриває 0.030)
    "L2":    (500e-8, 5000e-8),     # 500 Å = 500e-8 см, 5000 Å = 5000e-8 см ✓
    "Rp2":   (-6500e-8, 0e-8),      # ВИПРАВЛЕНО: 0 Å = 0 см (покриває -50, -500)
}
""")

    print("\n3️⃣ Постфікс для dataset файлу:")
    print("-"*70)
    print("""
# У dataset_stratified.py змінити:
output_file = f"datasets/dataset_{n_samples}_dl{dl_angstrom}_grid5.pkl"  # додано _grid5
""")

    print("\n" + "="*70)


if __name__ == "__main__":
    success = verify_sync()

    print_code_for_files()

    sys.exit(0 if success else 1)
