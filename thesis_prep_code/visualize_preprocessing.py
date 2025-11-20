"""
Візуалізація процесу попередньої обробки експериментальних КДВ
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add parent directory to path to import model_common
sys.path.insert(0, str(Path(__file__).parent.parent))
from model_common import preprocess_curve


def load_experiment_data(filepath):
    """Завантажити експериментальні дані з текстового файлу"""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # Пропустити порожні рядки
                try:
                    value = float(line)
                    data.append(value)
                except ValueError:
                    print(f"Не вдалося розпарсити: {line}")
    return np.array(data)


def main():
    # Завантажити експериментальну криву
    exp_file = Path(__file__).parent.parent / 'experiments' / 'experiment.txt'
    original_curve = load_experiment_data(exp_file)

    print(f"Завантажено точок: {len(original_curve)}")

    # Застосувати preprocessing
    processed_curve = preprocess_curve(
        original_curve.copy(),  # Створити копію щоб не змінювати оригінал
        crop_by_peak=True,
        peak_offset=30,
        target_length=700
    )

    print(f"Після обробки: {len(processed_curve)} точок")

    # Створити рисунок з двома графіками
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Ліворуч - оригінальна крива
    ax1.semilogy(range(len(original_curve)), original_curve, 'b-', linewidth=1.5)
    ax1.set_xlabel('Номер точки', fontsize=12)
    ax1.set_ylabel('Інтенсивність (відн. од.)', fontsize=12)
    ax1.set_title('а) Оригінальна експериментальна КДВ', fontsize=13, weight='bold')
    ax1.grid(True, alpha=0.3)

    # Позначити пік
    peak_idx = np.argmax(original_curve)
    ax1.axvline(x=peak_idx, color='r', linestyle='--', alpha=0.5, label=f'Пік підкладки (точка {peak_idx})')
    ax1.axvline(x=peak_idx + 30, color='g', linestyle='--', alpha=0.5, label='Початок аналізу (пік + 30)')
    ax1.legend(fontsize=10)

    # Праворуч - оброблена крива
    ax2.semilogy(range(len(processed_curve)), processed_curve, 'b-', linewidth=1.5)
    ax2.set_xlabel('Номер точки', fontsize=12)
    ax2.set_ylabel('Інтенсивність (відн. од.)', fontsize=12)
    ax2.set_title('б) КДВ після попередньої обробки', fontsize=13, weight='bold')
    ax2.grid(True, alpha=0.3)

    # Позначити зони
    ax2.axhline(y=0.00025, color='orange', linestyle=':', alpha=0.7, label='Поріг шуму (2.5×10⁻⁴)')
    ax2.legend(fontsize=10)

    plt.tight_layout()

    # Зберегти рисунок
    output_file = Path(__file__).parent.parent / 'content' / 'preprocessing_comparison.png'
    output_file.parent.mkdir(exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Рисунок збережено: {output_file}")

    # Вивести статистику
    print(f"\nСтатистика обробки:")
    print(f"  Оригінальна довжина: {len(original_curve)} точок")
    print(f"  Оброблена довжина: {len(processed_curve)} точок")
    print(f"  Позиція піка: {peak_idx}")
    print(f"  Початок аналізу: {peak_idx + 30}")
    print(f"  Максимальна інтенсивність: {original_curve[peak_idx]:.4e}")
    print(f"  Точок після обробки: {len(processed_curve)}")

    plt.show()


if __name__ == '__main__':
    main()
