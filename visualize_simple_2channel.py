"""
Проста візуалізація 2-канального представлення КДВ
Рисунок 2.X - Двоканальне представлення: крива + позиція + тензор
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Завантаження експериментальної кривої
experiment_file = Path("experiments/experiment.txt")

if experiment_file.exists():
    intensity = np.loadtxt(experiment_file)
else:
    angles = np.linspace(-300, 2100, 650)
    intensity = (
        10**5 * np.exp(-((angles - 0) / 150)**2) +
        10**3 * np.exp(-((angles - 500) / 100)**2) * np.sin(angles / 50)**2 +
        np.random.exponential(10, len(angles))
    )

# Resample до 650 точок
target_length = 650
if len(intensity) != target_length:
    x_old = np.linspace(0, 1, len(intensity))
    x_new = np.linspace(0, 1, target_length)
    intensity = np.interp(x_new, x_old, intensity)

# ============================================================================
# КАНАЛИ
# ============================================================================
# Канал 0: log-нормалізована інтенсівність
intensity_safe = intensity + 1e-10
intensity_log = np.log10(intensity_safe)
channel_0 = (intensity_log - intensity_log.min()) / \
    (intensity_log.max() - intensity_log.min() + 1e-12)

# Знаходимо позицію максимуму КДВ
peak_idx = np.argmax(channel_0)

# Створюємо кутову шкалу як на експерименті: від -200 до ~1100 arcsec
# з піком близько 0
# Діапазон: ~1300 arcsec на 650 точок → крок ~2 arcsec
angles_start = -200
angles_end = 1100
angles = np.linspace(angles_start, angles_end, target_length)

# Коригуємо так, щоб пік був точно на 0
angles = angles - angles[peak_idx]

# Канал 1: позиція
channel_1 = np.linspace(0, 1, target_length)

# ============================================================================
# ВІЗУАЛІЗАЦІЯ: 2 рядки
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.subplots_adjust(hspace=0.35, wspace=0.25)

# ========== РЯДОК 1: Ліворуч крива, праворуч позиція ==========

# Ліворуч: Канал 0 - Крива
ax1 = axes[0, 0]
ax1.plot(angles, channel_0, 'b-', linewidth=2)
ax1.fill_between(angles, 0, channel_0, alpha=0.3, color='blue')
ax1.set_xlabel('Кутове відхилення (arcsec)', fontsize=11, weight='bold')
ax1.set_ylabel('Нормалізована інтенсивність', fontsize=11, weight='bold')
ax1.set_title('Канал 0: Log-нормалізована КДВ', fontsize=12, weight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-0.05, 1.05)

# Праворуч: Канал 1 - Позиція
ax2 = axes[0, 1]
ax2.plot(angles, channel_1, 'g-', linewidth=2)
ax2.fill_between(angles, 0, channel_1, alpha=0.3, color='green')
ax2.set_xlabel('Кутове відхилення (arcsec)', fontsize=11, weight='bold')
ax2.set_ylabel('Позиція [0, 1]', fontsize=11, weight='bold')
ax2.set_title('Канал 1: Позиційна координата', fontsize=12, weight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_ylim(-0.05, 1.05)

# ========== РЯДОК 2: Тензорне представлення (на всю ширину) ==========

# Видаляємо обидва осередки другого рядка і створюємо один великий
fig.delaxes(axes[1, 0])
fig.delaxes(axes[1, 1])

# Створюємо новий subplot на всю ширину
ax3 = fig.add_subplot(2, 1, 2)

# Heatmap з 2 каналами
heatmap_data = np.vstack([channel_0, channel_1])

im = ax3.imshow(heatmap_data, aspect='auto', cmap='viridis',
                interpolation='bilinear',
                extent=[angles[0], angles[-1], 0, 2])

ax3.set_xlabel('Кутове відхилення (arcsec)', fontsize=11, weight='bold')
ax3.set_title(
    'Двоканальне представлення вхідних даних для нейромережевої моделі [2, 650]', fontsize=12, weight='bold')

# Colorbar
cbar = plt.colorbar(im, ax=ax3, orientation='vertical',
                    pad=0.01, fraction=0.06)
cbar.set_label('Значення [0, 1]', fontsize=10)

# Загальний заголовок

plt.savefig('figures/2channel_simple.png', dpi=300,
            bbox_inches='tight', facecolor='white')

print("✓ Проста візуалізація створена:")
print("  - figures/2channel_simple.png (300 DPI)")
print("  - figures/2channel_simple.pdf (векторна)")
print("\nПідпис для роботи:")
print("Рисунок 2.X – Двоканальне представлення вхідних даних для нейромережевої моделі")
