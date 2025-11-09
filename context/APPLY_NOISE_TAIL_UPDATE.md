# prepocess_curve (ранішня назва: apply_noise_tail) - Оновлення з auto-padding + exponential noise

**Дата:** 2025-11-09
**Статус:** ✅ ГОТОВО

---

## 🎯 Зміни

### 1. Оновлено prepocess_curve() (раніше apply_noise_tail)

**Додано параметр `target_length`** для автоматичного padding/truncate + exponential noise:

```python
def prepocess_curve(curve, crop_by_peak=True, peak_offset=30, target_length=None):
    """
    Apply noise tail and optionally pad/truncate to target length.

    Args:
        curve: Input curve (numpy array or torch tensor)
        crop_by_peak: If True, crop from peak position
        peak_offset: Offset after peak (default 30)
        target_length: If specified, pad or truncate to this length  # ← НОВЕ!

    Returns:
        numpy array with noise tail applied and adjusted to target_length
    """
```

**Логіка padding/truncate:**
```python
if target_length is not None:
    current_length = len(curve_np)

    if current_length < target_length:
        # Pad with constant 0.00025 then apply exponential noise (like line 117-129)
        pad_len = target_length - current_length
        pad_values = np.full(pad_len, 0.00025)
        curve_np = np.concatenate([curve_np, pad_values])

        # Apply exponential noise to padded section (±2% like line 129)
        curve_np[current_length:] *= np.exp(np.random.normal(0, 0.02, pad_len))

    elif current_length > target_length:
        # Truncate
        curve_np = curve_np[:target_length]
```

---

## 📝 Використання в predict.py

### БУЛО (ручний padding):
```python
# Apply noise tail
curve_cropped = apply_noise_tail(curve_raw)  # Стара назва

# Manually pad or truncate
if len(curve_cropped) < expected_length:
    pad_len = expected_length - len(curve_cropped)
    pad_noise = np.random.normal(2.2e-4, 0.15e-4, pad_len)
    curve_cropped = np.concatenate([curve_cropped, pad_noise])
elif len(curve_cropped) > expected_length:
    curve_cropped = curve_cropped[:expected_length]
```

### СТАЛО (автоматичний padding + exponential noise):
```python
# Apply noise tail with auto padding/truncate + exponential noise
curve_cropped = prepocess_curve(  # Нова назва
    curve_raw,
    crop_by_peak=True,
    peak_offset=30,
    target_length=expected_length  # ← З metadata checkpoint!
)
```

**Переваги:**
- ✅ Менше коду
- ✅ Одна функція робить все (crop + noise + padding + exponential noise)
- ✅ Consistency (однаковий padding + exponential noise як у load_dataset і noise tail)

---

## 🔧 Metadata у checkpoint

### Де зберігається:

**train_with_curve_validation.py (рядок 248):**
```python
checkpoint = {
    "model": model.state_dict(),
    "L": Y.size(1),  # ← Curve length metadata
    "epoch": epoch,
    "val_loss_params": val_loss_params,
    "val_loss_curve": val_loss_curve,
}
torch.save(checkpoint, save_path)
```

**model_train.py (рядок 273):**
```python
checkpoint = {
    "model": model.state_dict(),
    "L": Y.size(1),  # ← Curve length metadata
    "epoch": epoch,
    "val_loss": val_loss,
}
torch.save(checkpoint, save_path)
```

### Як читається:

**predict.py (рядок 62):**
```python
checkpoint = torch.load(model_path, weights_only=False)
expected_length = checkpoint.get('L', 651)  # Default 651
```

**Приклад:**
```python
>>> checkpoint = torch.load('checkpoints/10000_target_log_best_curve.pt')
>>> checkpoint['L']
651
```

---

## ✅ Переваги нового підходу

### 1. Автоматичне визначення розміру
- ❌ БУЛО: Треба знати розмір моделі вручну
- ✅ ТЕПЕР: Читається з checkpoint metadata

### 2. Consistency
- ❌ БУЛО: Padding в predict.py міг відрізнятися від load_dataset
- ✅ ТЕПЕР: Той самий код для padding + exponential noise у prepocess_curve

### 3. Менше коду
- ❌ БУЛО: ~15 рядків для padding у predict.py
- ✅ ТЕПЕР: 1 рядок з target_length параметром

### 4. Гнучкість
- ✅ Можна викликати БЕЗ target_length (як раніше)
- ✅ Можна викликати З target_length (автоматичний padding + exponential noise)

---

## 📊 Приклади використання

### Варіант 1: Без padding (як раніше)
```python
curve = prepocess_curve(raw_curve)
# Повертає змінну довжину
```

### Варіант 2: З padding до фіксованого розміру
```python
curve = prepocess_curve(raw_curve, target_length=651)
# Повертає точно 651 точку (з exponential noise на padding)
```

### Варіант 3: З metadata з checkpoint (predict.py)
```python
checkpoint = torch.load(model_path, weights_only=False)
expected_length = checkpoint.get('L', 651)
curve = prepocess_curve(raw_curve, target_length=expected_length)
# Автоматично підганяється під модель з exponential noise!
```

---

## 🧪 Тестування

### Тест 1: Padding (curve коротша за target)
```python
>>> curve = np.random.rand(500)
>>> result = prepocess_curve(curve, crop_by_peak=False, target_length=651)
>>> len(result)
651
>>> # Перші 500 точок = оригінал + noise tail processing
>>> # Останні 151 точок = 0.00025 + exponential noise
```

### Тест 2: Truncate (curve довша за target)
```python
>>> curve = np.random.rand(800)
>>> result = prepocess_curve(curve, crop_by_peak=False, target_length=651)
>>> len(result)
651
>>> # Обрізано до 651
```

### Тест 3: Без target_length (як раніше)
```python
>>> curve = np.random.rand(700)
>>> result = prepocess_curve(curve, crop_by_peak=False)
>>> len(result)
700  # Залишається оригінальна довжина
```

---

## 📋 Checklist

- [x] ✅ Додано параметр `target_length` в prepocess_curve() (раніше apply_noise_tail)
- [x] ✅ Реалізовано padding з константою 0.00025 + exponential noise (консистентно з line 117-129)
- [x] ✅ Реалізовано truncate
- [x] ✅ Оновлено predict.py для використання target_length з prepocess_curve
- [x] ✅ Metadata 'L' вже зберігається у checkpoint (train_with_curve_validation.py, model_train.py)
- [x] ✅ Backward compatibility (target_length=None працює як раніше)
- [x] ✅ Функцію перейменовано на prepocess_curve для кращої читабельності
- [x] ✅ Документація

---

## 🚀 Готово до використання!

**predict.py тепер:**
1. ✅ Завантажує checkpoint
2. ✅ Читає metadata `L` (curve length)
3. ✅ Викликає `prepocess_curve(..., target_length=L)`
4. ✅ Отримує криву точно потрібного розміру (0.00025 + exponential noise для padding)
5. ✅ Передає в модель

**Все автоматично! Padding консистентний з noise tail! Не треба вручну padding!** 🎉
