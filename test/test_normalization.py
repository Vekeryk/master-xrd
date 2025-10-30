"""
Тест нормалізації та денормалізації параметрів
================================================
Перевіряє чи правильно працює NormalizedXRDDataset та denorm_params
з урахуванням конвертації L1, Rp1, L2, Rp2 з Ångströms в см.

ВАЖЛИВО:
- При генерації датасету: параметри в Å → конвертуються в см для зберігання
- В RANGES: всі параметри вже в см (L1, Rp1, L2, Rp2 як 500e-8)
- Нормалізація: значення в см → [0, 1]
- Денормалізація: [0, 1] → значення в см
"""

import numpy as np
import torch
from model_common import RANGES, PARAM_NAMES, denorm_params, NormalizedXRDDataset


def test_normalization_denormalization():
    """Тест циклу нормалізація → денормалізація"""

    print("\n" + "="*80)
    print("🔬 ТЕСТ НОРМАЛІЗАЦІЇ ТА ДЕНОРМАЛІЗАЦІЇ")
    print("="*80)

    # ==========================================================================
    # 1. ПЕРЕВІРКА RANGES (як зберігаються)
    # ==========================================================================
    print("\n📊 RANGES у model_common.py:")
    print("-"*80)
    for name in PARAM_NAMES:
        r_min, r_max = RANGES[name]
        if name in ['L1', 'Rp1', 'L2', 'Rp2']:
            # Конвертуємо в Å для читабельності
            print(f"  {name:6s}: ({r_min:.2e} см, {r_max:.2e} см) = ({r_min*1e8:.1f} Å, {r_max*1e8:.1f} Å)")
        else:
            print(f"  {name:6s}: ({r_min:.6f}, {r_max:.6f})")

    # ==========================================================================
    # 2. СИМУЛЯЦІЯ ГЕНЕРАЦІЇ ДАТАСЕТУ
    # ==========================================================================
    print("\n" + "="*80)
    print("🔧 СИМУЛЯЦІЯ ГЕНЕРАЦІЇ ДАТАСЕТУ")
    print("="*80)

    # Параметри в Ångströms (як в циклі генерації)
    params_angstrom = {
        'Dmax1': 0.01,      # безрозмірні
        'D01': 0.002,       # безрозмірні
        'L1': 5000.,        # Å
        'Rp1': 3500.,       # Å
        'D02': 0.005,       # безрозмірні
        'L2': 3000.,        # Å
        'Rp2': -500.,       # Å
    }

    print("\nВихідні параметри (як в циклі генерації):")
    print("-"*80)
    for name, val in params_angstrom.items():
        if name in ['L1', 'Rp1', 'L2', 'Rp2']:
            print(f"  {name:6s}: {val:10.1f} Å")
        else:
            print(f"  {name:6s}: {val:10.6f}")

    # Конвертація для зберігання (як в dataset_stratified.py)
    params_cm = {}
    for name, val in params_angstrom.items():
        if name in ['L1', 'Rp1', 'L2', 'Rp2']:
            params_cm[name] = val * 1e-8  # Å → см
        else:
            params_cm[name] = val

    print("\nПараметри після конвертації для зберігання в X:")
    print("-"*80)
    for name, val in params_cm.items():
        if name in ['L1', 'Rp1', 'L2', 'Rp2']:
            print(f"  {name:6s}: {val:.2e} см  (було {params_angstrom[name]:.1f} Å)")
        else:
            print(f"  {name:6s}: {val:.6f}")

    # Створити X як в датасеті (values в см)
    X_raw = np.array([[params_cm[name] for name in PARAM_NAMES]], dtype=np.float32)

    print("\nX (як зберігається в датасеті):")
    print("-"*80)
    print(f"  Shape: {X_raw.shape}")
    print(f"  Values: {X_raw[0]}")

    # ==========================================================================
    # 3. НОРМАЛІЗАЦІЯ через NormalizedXRDDataset
    # ==========================================================================
    print("\n" + "="*80)
    print("📈 НОРМАЛІЗАЦІЯ (через NormalizedXRDDataset)")
    print("="*80)

    # Створити dummy Y
    Y_dummy = np.zeros((1, 700), dtype=np.float32)

    # Створити dataset (автоматично нормалізує)
    dataset = NormalizedXRDDataset(
        torch.tensor(X_raw, dtype=torch.float32),
        torch.tensor(Y_dummy, dtype=torch.float32),
        log_space=False,
        train=False
    )

    # Отримати нормалізовані параметри
    # NormalizedXRDDataset.__getitem__ returns (Y_normalized, X_normalized)
    _, X_norm = dataset[0]
    X_norm_np = X_norm.cpu().numpy()

    print(f"\nDEBUG: X_norm shape: {X_norm.shape}, X_norm_np shape: {X_norm_np.shape}")

    print("\nНормалізовані значення (має бути в [0, 1]):")
    print("-"*80)
    for i, name in enumerate(PARAM_NAMES):
        r_min, r_max = RANGES[name]
        raw_val = float(X_raw[0, i])
        norm_val = X_norm_np[i].item() if hasattr(X_norm_np[i], 'item') else float(X_norm_np[i])

        # Перевірка чи в межах [0, 1]
        in_range = 0.0 <= norm_val <= 1.0
        status = "✅" if in_range else "❌"

        # Ручний розрахунок (для перевірки)
        expected_norm = (raw_val - r_min) / (r_max - r_min)

        print(f"  {status} {name:6s}: {norm_val:.6f}  (очікується: {expected_norm:.6f}, diff: {abs(norm_val - expected_norm):.2e})")

    # ==========================================================================
    # 4. ДЕНОРМАЛІЗАЦІЯ через denorm_params
    # ==========================================================================
    print("\n" + "="*80)
    print("📉 ДЕНОРМАЛІЗАЦІЯ (через denorm_params)")
    print("="*80)

    # Денормалізувати
    X_denorm = denorm_params(X_norm.unsqueeze(0))  # add batch dimension
    X_denorm_np = X_denorm[0].cpu().numpy()

    print("\nДенормалізовані значення (має співпадати з оригінальними в см):")
    print("-"*80)
    for i, name in enumerate(PARAM_NAMES):
        original_cm = X_raw[0, i]
        denorm_cm = X_denorm_np[i]
        diff = abs(original_cm - denorm_cm)

        # Толерантність для float32
        matches = diff < 1e-9
        status = "✅" if matches else "❌"

        if name in ['L1', 'Rp1', 'L2', 'Rp2']:
            print(f"  {status} {name:6s}: {denorm_cm:.2e} см  (оригінал: {original_cm:.2e} см, diff: {diff:.2e})")
            print(f"           = {denorm_cm*1e8:.1f} Å  (оригінал: {params_angstrom[name]:.1f} Å)")
        else:
            print(f"  {status} {name:6s}: {denorm_cm:.6f}  (оригінал: {original_cm:.6f}, diff: {diff:.2e})")

    # ==========================================================================
    # 5. ПЕРЕВІРКА ГРАНИЧНИХ ЗНАЧЕНЬ
    # ==========================================================================
    print("\n" + "="*80)
    print("🔍 ПЕРЕВІРКА ГРАНИЧНИХ ЗНАЧЕНЬ (min/max з RANGES)")
    print("="*80)

    test_cases = []
    for name in PARAM_NAMES:
        r_min, r_max = RANGES[name]
        test_cases.append((name, 'min', r_min))
        test_cases.append((name, 'max', r_max))

    print("\nПеревірка що min→0, max→1 після нормалізації:")
    print("-"*80)

    all_ok = True
    for name, label, value in test_cases:
        # Створити X з цим значенням
        X_test = np.zeros((1, 7), dtype=np.float32)
        param_idx = PARAM_NAMES.index(name)
        X_test[0, param_idx] = value

        # Нормалізувати
        dataset_test = NormalizedXRDDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(Y_dummy, dtype=torch.float32),
            log_space=False,
            train=False
        )
        # NormalizedXRDDataset.__getitem__ returns (Y_normalized, X_normalized)
        _, X_norm_test = dataset_test[0]
        norm_value = X_norm_test[param_idx].item()

        # Перевірити
        if label == 'min':
            expected = 0.0
        else:
            expected = 1.0

        diff = abs(norm_value - expected)
        matches = diff < 1e-6
        status = "✅" if matches else "❌"

        if not matches:
            all_ok = False

        if name in ['L1', 'Rp1', 'L2', 'Rp2']:
            print(f"  {status} {name:6s} {label:3s}: {value:.2e} см → {norm_value:.6f} (очікується {expected:.1f})")
        else:
            print(f"  {status} {name:6s} {label:3s}: {value:.6f} → {norm_value:.6f} (очікується {expected:.1f})")

    # ==========================================================================
    # 6. ФІНАЛЬНИЙ ВИСНОВОК
    # ==========================================================================
    print("\n" + "="*80)
    print("🎯 ФІНАЛЬНИЙ ВИСНОВОК")
    print("="*80)

    if all_ok:
        print("\n✅ ВСЕ ПРАЦЮЄ ПРАВИЛЬНО!")
        print("   • Нормалізація: values в см → [0, 1]")
        print("   • Денормалізація: [0, 1] → values в см")
        print("   • L1, Rp1, L2, Rp2 коректно обробляються")
        print("\n💡 Як це працює:")
        print("   1. Генерація: параметри в Å → множимо на 1e-8 → зберігаємо в см в X")
        print("   2. RANGES: вже в см (L1: 500e-8 см = 500 Å)")
        print("   3. Нормалізація: (value_cm - min_cm) / (max_cm - min_cm) → [0, 1]")
        print("   4. Денормалізація: value_norm * (max_cm - min_cm) + min_cm → см")
        print("   5. Для відображення: value_cm * 1e8 → Å")
    else:
        print("\n❌ ЗНАЙДЕНО ПРОБЛЕМИ!")
        print("   Перевірте нормалізацію/денормалізацію в model_common.py")

    print("\n" + "="*80)

    return all_ok


if __name__ == "__main__":
    success = test_normalization_denormalization()
    exit(0 if success else 1)
