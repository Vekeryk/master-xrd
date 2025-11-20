"""
Створення додаткових діаграм моделі XRDRegressor
використовуючи torchview, torchviz та ONNX
"""

import torch
from model_common import XRDRegressor, PARAM_NAMES, RANGES
from pathlib import Path

print("="*80)
print("Створення додаткових діаграм архітектури XRDRegressor")
print("="*80 + "\n")

# Створюємо модель
model = XRDRegressor(n_out=7, kernel_size=15)
model.eval()

# Створюємо приклад вхідних даних
batch_size = 1
curve_length = 700
dummy_input = torch.randn(batch_size, 1, curve_length)

print(f"Модель: {model.__class__.__name__}")
print(f"Вхідні дані: {dummy_input.shape}")
print(f"Вихідні дані: {model(dummy_input).shape}")
print()

# ============================================================================
# 1. TORCHVIEW - детальна архітектурна діаграма
# ============================================================================
print("📊 1. Створення діаграми з torchview...")
try:
    from torchview import draw_graph

    # Варіант 1: Горизонтальна діаграма з деталями
    model_graph_horizontal = draw_graph(
        model,
        input_data=dummy_input,
        expand_nested=True,
        graph_name='XRDRegressor',
        depth=3,
        device='cpu',
        graph_dir='LR',  # Left to Right
        hide_module_functions=False,
        hide_inner_tensors=False,
        roll=False,
        show_shapes=True,
        save_graph=False,
    )

    model_graph_horizontal.visual_graph.render(
        filename='figures/model_torchview_horizontal',
        format='png',
        cleanup=True
    )
    model_graph_horizontal.visual_graph.render(
        filename='figures/model_torchview_horizontal',
        format='pdf',
        cleanup=True
    )
    print("   ✓ Збережено: figures/model_torchview_horizontal.png")
    print("   ✓ Збережено: figures/model_torchview_horizontal.pdf")

    # Варіант 2: Вертикальна компактна діаграма
    model_graph_vertical = draw_graph(
        model,
        input_data=dummy_input,
        expand_nested=False,
        graph_name='XRDRegressor',
        depth=2,
        device='cpu',
        graph_dir='TB',  # Top to Bottom
        hide_module_functions=True,
        hide_inner_tensors=True,
        roll=True,
        show_shapes=True,
        save_graph=False,
    )

    model_graph_vertical.visual_graph.render(
        filename='figures/model_torchview_vertical',
        format='png',
        cleanup=True
    )
    model_graph_vertical.visual_graph.render(
        filename='figures/model_torchview_vertical',
        format='pdf',
        cleanup=True
    )
    print("   ✓ Збережено: figures/model_torchview_vertical.png")
    print("   ✓ Збережено: figures/model_torchview_vertical.pdf")

except Exception as e:
    print(f"   ❌ Помилка torchview: {e}")

print()

# ============================================================================
# 2. TORCHVIZ - computational graph
# ============================================================================
print("📊 2. Створення computational graph з torchviz...")
try:
    from torchviz import make_dot

    # Прогоняємо дані через модель
    output = model(dummy_input)

    # Створюємо граф обчислень
    dot = make_dot(
        output.mean(),  # Треба скалярне значення
        params=dict(model.named_parameters()),
        show_attrs=False,
        show_saved=False
    )

    dot.render(
        filename='model_computational_graph',
        directory='figures',
        format='png',
        cleanup=True
    )
    dot.render(
        filename='model_computational_graph',
        directory='figures',
        format='pdf',
        cleanup=True
    )
    print("   ✓ Збережено: figures/model_computational_graph.png")
    print("   ✓ Збережено: figures/model_computational_graph.pdf")

except Exception as e:
    print(f"   ❌ Помилка torchviz: {e}")

print()

# ============================================================================
# 3. ONNX для Netron
# ============================================================================
print("📊 3. Експорт моделі у ONNX для Netron...")
try:
    onnx_path = Path("figures/model_xrdregressor.onnx")

    # Експортуємо модель
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=18,  # Використовуємо новішу версію
        do_constant_folding=True,
        input_names=['xrd_curve'],
        output_names=['deformation_parameters'],
        dynamic_axes={
            'xrd_curve': {0: 'batch_size'},
            'deformation_parameters': {0: 'batch_size'}
        },
        verbose=False
    )

    print(f"   ✓ Збережено: {onnx_path}")
    print(f"   ℹ️  Відкрийте у Netron: https://netron.app/")
    print(f"      або командою: netron {onnx_path}")

except Exception as e:
    print(f"   ❌ Помилка ONNX: {e}")

print()

# ============================================================================
# 4. TORCHINFO - детальна статистика моделі
# ============================================================================
print("📊 4. Створення детального summary...")
try:
    from torchinfo import summary

    model_stats = summary(
        model,
        input_size=(batch_size, 1, curve_length),
        col_names=[
            "input_size",
            "output_size",
            "num_params",
            "params_percent",
            "kernel_size",
            "mult_adds"
        ],
        depth=6,
        verbose=0,
        row_settings=["var_names"]
    )

    # Зберігаємо у файл
    summary_path = Path("figures/model_detailed_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(str(model_stats))
        f.write("\n\n")
        f.write("="*80 + "\n")
        f.write("СТРУКТУРА МОДЕЛІ\n")
        f.write("="*80 + "\n\n")
        f.write(f"Вхід: XRD крива дифракційного відбивання [B, 1, {curve_length}]\n")
        f.write(f"Вихід: 7 параметрів профілю деформації [B, 7]\n")
        f.write(f"  {', '.join(PARAM_NAMES)}\n\n")

        f.write("Гілки моделі:\n")
        f.write("  1. CNN гілка: згорткова мережа з residual блоками\n")
        f.write("     - Stem: Conv1d(2→32) + BN + SiLU\n")
        f.write("     - 6 Residual блоків з прогресивним розширенням каналів:\n")
        f.write("       32 → 48 → 64 → 96 → 128 → 128\n")
        f.write("     - Dilations: 1, 2, 4, 8, 16, 32 (для великого receptive field)\n")
        f.write("     - Attention pooling (замість GAP)\n\n")

        f.write("  2. FFT гілка: спектральний аналіз\n")
        f.write("     - Hann window для зменшення spectral leakage\n")
        f.write("     - FFT → 50 частотних bins → MLP(50→64→32)\n")
        f.write("     - Критично для визначення L1, L2 (період осциляцій)\n\n")

        f.write("  3. Fusion Head: MLP\n")
        f.write("     - Вхід: 128 (CNN) + 32 (FFT) = 160 features\n")
        f.write("     - 160 → 256 → 128 → 7 з Dropout(0.2)\n")
        f.write("     - Sigmoid активація → [0, 1] (нормалізовані параметри)\n\n")

        f.write("Діапазони параметрів (denormalization):\n")
        for name in PARAM_NAMES:
            lo, hi = RANGES[name]
            f.write(f"  {name:8s}: [{lo:.6f}, {hi:.6f}]\n")

    print(f"   ✓ Збережено: {summary_path}")

    # Виводимо короткий summary у консоль
    print("\n" + "="*80)
    print("КОРОТКИЙ SUMMARY")
    print("="*80)
    print(f"Загальна кількість параметрів: {model_stats.total_params:,}")
    print(f"Trainable параметрів: {model_stats.trainable_params:,}")
    print(f"Розмір моделі: {model_stats.total_mult_adds / 1e9:.2f} GMult-Adds")
    print(f"Estimated memory: {model_stats.total_input + model_stats.total_output_bytes/1e6 + model_stats.total_param_bytes/1e6:.2f} MB")

except Exception as e:
    print(f"   ❌ Помилка torchinfo: {e}")

print()
print("="*80)
print("✅ ВІЗУАЛІЗАЦІЯ ЗАВЕРШЕНА!")
print("="*80)
print("\nСтворені файли:")
print("  📄 figures/model_torchview_horizontal.{png,pdf} - детальна горизонтальна діаграма")
print("  📄 figures/model_torchview_vertical.{png,pdf} - компактна вертикальна діаграма")
print("  📄 figures/model_computational_graph.{png,pdf} - граф обчислень")
print("  📄 figures/model_xrdregressor.onnx - для інтерактивного перегляду у Netron")
print("  📄 figures/model_detailed_summary.txt - детальна статистика моделі")
print("\nДля магістерської роботи рекомендую використовувати:")
print("  ✨ torchview_vertical.pdf - для загальної архітектури")
print("  ✨ xrd_model_architecture.pdf - для детальної схеми з поясненнями")
print("  ✨ model_detailed_summary.txt - для таблиць у тексті роботи")
