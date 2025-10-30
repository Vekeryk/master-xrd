# XRD CNN Training Guide

## Unified Training Script

**File:** `model_train.py`

Єдиний скрипт для всіх режимів тренування з прапорцями конфігурації.

## Configuration Flags

### 1. `WEIGHTED_TRAINING` (True/False)

Контролює, чи використовувати різні ваги для параметрів у loss function.

```python
WEIGHTED_TRAINING = True   # Weighted: [1.0, 1.2, 1.0, 1.0, 1.5, 2.0, 2.5]
WEIGHTED_TRAINING = False  # Unweighted: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
```

**Weighted:** Більші ваги для складних параметрів (L2, Rp2)
**Unweighted:** Всі параметри рівноцінні (baseline)

### 2. `FULL_CURVE_TRAINING` (True/False)

Контролює, чи використовувати повну криву або тільки crop [50:701].

```python
FULL_CURVE_TRAINING = False  # Crop: Y[:, 50:701] (651 points)
FULL_CURVE_TRAINING = True   # Full: Y[:, 0:701] (701 points)
```

**Cropped:** Фокус на peak region, швидше
**Full:** Більше контексту, повільніше

## Model Naming Convention

Автоматично генерується на основі прапорців:

| WEIGHTED | FULL_CURVE | Model Name |
|----------|------------|------------|
| True | False | `dataset_100000_dl100_7d_v3.pt` |
| True | True | `dataset_100000_dl100_7d_v3_full.pt` |
| False | False | `dataset_100000_dl100_7d_v3_unweighted.pt` |
| False | True | `dataset_100000_dl100_7d_v3_unweighted_full.pt` |

## Usage Examples

### Example 1: Weighted + Cropped (Default)
```python
# model_train.py lines 258-263
WEIGHTED_TRAINING = True
FULL_CURVE_TRAINING = False
```
```bash
python model_train.py
```
→ Model: `checkpoints/dataset_100000_dl100_7d_v3.pt`

### Example 2: Weighted + Full Curve
```python
WEIGHTED_TRAINING = True
FULL_CURVE_TRAINING = True  # ← Enable full curve
```
```bash
python model_train.py
```
→ Model: `checkpoints/dataset_100000_dl100_7d_v3_full.pt`

### Example 3: Unweighted + Cropped (Baseline)
```python
WEIGHTED_TRAINING = False  # ← Disable weighted loss
FULL_CURVE_TRAINING = False
```
```bash
python model_train.py
```
→ Model: `checkpoints/dataset_100000_dl100_7d_v3_unweighted.pt`

### Example 4: Unweighted + Full Curve
```python
WEIGHTED_TRAINING = False
FULL_CURVE_TRAINING = True
```
```bash
python model_train.py
```
→ Model: `checkpoints/dataset_100000_dl100_7d_v3_unweighted_full.pt`

## Loss Weights Configuration

Ваги визначені в `__main__` блоці (lines 270-275):

```python
if WEIGHTED_TRAINING:
    # Higher weights for challenging parameters (L2, Rp2)
    LOSS_WEIGHTS = torch.tensor([1.0, 1.2, 1.0, 1.0, 1.5, 2.0, 2.5])
else:
    # Unweighted: all parameters equal importance
    LOSS_WEIGHTS = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
```

**Order:** `[Dmax1, D01, L1, Rp1, D02, L2, Rp2]`

## Training Output

Скрипт виводить summary перед початком:

```
======================================================================
TRAINING CONFIGURATION SUMMARY
======================================================================
Dataset: datasets/dataset_100000_dl100_7d.pkl
Model: checkpoints/dataset_100000_dl100_7d_v3_unweighted.pt
Weighted loss: False
Full curve: False
Loss weights: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
======================================================================
```

І під час тренування:

```
======================================================================
XRD CNN TRAINING
======================================================================
✓ Using MPS (Apple Silicon)
✓ Applying crop_params: Y[:, 50:701]  # Якщо FULL_CURVE_TRAINING=False

⚖️  Loss Configuration:
   UNWEIGHTED loss: [1. 1. 1. 1. 1. 1. 1.]
   All parameters have equal importance
   Physics constraints: D01≤Dmax1, D01+D02≤0.03, Rp1≤L1, L2≤L1
```

## Experiment Plan

Рекомендований порядок експериментів:

1. **Baseline:** Unweighted + Cropped → встановити baseline
2. **Weighted:** Weighted + Cropped → порівняти з baseline
3. **Full curve:** Weighted + Full → перевірити чи допомагає більше даних
4. **Full baseline:** Unweighted + Full → порівняти з (3)

## Files Modified

- ✅ **model_train.py** - unified training script з прапорцями
- ❌ **model_train_unweighted.py** - видалено (merged into model_train.py)

---

**Date:** 2025-10-30
**Status:** ✅ Ready for training
