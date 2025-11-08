# C++ Integration - Quick Summary

## 🎯 Яку версію вибрати?

### ⭐ **РЕКОМЕНДОВАНО: predict_integration_spawnl.cpp**

Використовує `_spawnl()` замість `system()`:
- ✅ Безпечний (аргументи як масив)
- ✅ Працює з шляхами з пробілами
- ✅ VCL інтеграція
- ✅ Надійні шляхи
- ⚠️ Блокує UI на 1-2 сек (це OK для ML)

### 🚀 **ADVANCED: predict_integration_timeout.cpp**

Використовує `CreateProcessA()`:
- ✅ Все що в _spawnl +
- ✅✅ **Timeout захист** (вбиває процес якщо завис)
- ✅✅ Приховує console window
- ✅✅ Async виконання через TThread
- ⚠️ Трохи складніший код

---

## 📋 Швидкий старт (3 кроки)

### 1. Додай файли до проекту

```
Difuz/
├── predict_integration.h
├── predict_integration_spawnl.cpp  ⭐
└── predict.exe (Windows build)
```

### 2. Include в Difuz.cpp

```cpp
#include "predict_integration.h"
```

### 3. Виклик з кнопки

```cpp
void __fastcall TForm1::PredictButtonClick(TObject *Sender)
{
    // 1. Підготувати дані (661 точка)
    double curve[661];
    for (int i = 0; i < 661; i++) {
        curve[i] = R_vseZ[i + 40];
    }

    // 2. Показати progress
    Cursor = crHourGlass;
    StatusBar->SimpleText = "ML prediction...";
    Application->ProcessMessages();

    // 3. PREDICT!
    DeformParams predicted;
    int success = PredictFromCurve(curve, 661, &predicted, "predict.exe");

    Cursor = crDefault;

    // 4. Застосувати результат
    if (success) {
        Edit1->Text = FloatToStrF(predicted.Dmax1, ffFixed, 8, 6);
        Edit2->Text = FloatToStrF(predicted.D01, ffFixed, 8, 6);
        Edit3->Text = FloatToStrF(predicted.L1, ffExponent, 8, 2);
        Edit4->Text = FloatToStrF(predicted.Rp1, ffExponent, 8, 2);
        Edit5->Text = FloatToStrF(predicted.D02, ffFixed, 8, 6);
        Edit6->Text = FloatToStrF(predicted.L2, ffExponent, 8, 2);
        Edit7->Text = FloatToStrF(predicted.Rp2, ffExponent, 8, 2);
        StatusBar->SimpleText = "Success!";
    } else {
        ShowMessage("Prediction failed!");
    }
}
```

**ГОТОВО!** 🎉

---

## 🔧 Зібрати predict.exe для Windows

На Windows машині:

```bash
pip install pyinstaller torch numpy scipy matplotlib tqdm
python build_predictor.py
```

Отримаєш: `dist/predict.exe` (~200-250 MB)

---

## 📊 Порівняння версій

| Версія | Метод | Безпека | Timeout | Рекомендація |
|--------|-------|---------|---------|--------------|
| predict_integration.cpp | system() | ❌ | ❌ | ❌ Не використовуй |
| predict_integration_improved.cpp | system() | ⚠️ | ❌ | ⚠️ Застарілий |
| predict_integration_vcl.cpp | system() | ⚠️ | ❌ | ⚠️ Застарілий |
| **predict_integration_spawnl.cpp** | **_spawnl** | **✅** | ❌ | **✅ ВИКОРИСТОВУЙ** |
| **predict_integration_timeout.cpp** | **CreateProcess** | **✅✅** | **✅** | **✅ ADVANCED** |

---

## ⚙️ Різниця між методами

### system() ❌
```cpp
system("\"predict.exe\" \"file.txt\"");
```
- Вразливий до command injection
- Проблеми з кавичками і пробілами
- Важко дебажити

### _spawnl() ✅
```cpp
_spawnl(_P_WAIT, "predict.exe", "predict.exe", "file.txt", NULL);
```
- Безпечний (аргументи як масив)
- Автоматичне екранування
- Просто і надійно

### CreateProcess() ✅✅
```cpp
CreateProcessA(...);
WaitForSingleObject(hProcess, 30000);  // 30 sec timeout
```
- Повний контроль
- Timeout захист
- Async можливість

---

## 🐛 Troubleshooting

### "Prediction failed"

**Перевір:**
1. `predict.exe` існує в папці з Difuz.exe
2. Curve має рівно 661 точку
3. Є права на запис (не в Program Files)

**Debug:**
```cpp
String appDir = ExtractFilePath(Application->ExeName);
String predictorPath = appDir + "predict.exe";
if (!FileExists(predictorPath)) {
    ShowMessage("predict.exe not found:\n" + predictorPath);
}
```

### UI зависає на 1-2 секунди

**Це нормально** - ML inference займає час.

**Рішення:**
- Додай `Cursor = crHourGlass` (вже в прикладі)
- Або використай timeout версію з TThread для async

---

## 📖 Детальна документація

Див. [CPP_INTEGRATION_GUIDE.md](CPP_INTEGRATION_GUIDE.md) для:
- Повного опису всіх версій
- Приклади async виконання
- Детальне troubleshooting
- Формат файлів обміну

---

## ✅ Checklist

- [ ] Скопіював `predict_integration.h` і `predict_integration_spawnl.cpp` в проект
- [ ] Зібрав `predict.exe` на Windows
- [ ] Помістив `predict.exe` поруч з Difuz.exe
- [ ] Додав `#include "predict_integration.h"` в Difuz.cpp
- [ ] Викликав `PredictFromCurve()` з кнопки
- [ ] Додав індикатор завантаження (Cursor + StatusBar)
- [ ] Протестував на реальних даних

**Готово до production!** 🚀
