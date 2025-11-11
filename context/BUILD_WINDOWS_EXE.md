# Компіляція Windows .exe через Wine

## ШВИДКА ІНСТРУКЦІЯ

### predict.py (основний predictor з PyTorch):

**Варіант 1 - `--onefile` (один файл):**

```bash
wine "C:\\Program Files\\Python310\\Scripts\\pyinstaller.exe" \
  --onefile \
  --name=predict \
  --console \
  --clean \
  --collect-all torch \
  predict.py
```

**Варіант 2 - `--onedir` (РЕКОМЕНДОВАНО - краще з DLL):**

```bash
wine "C:\\Program Files\\Python310\\Scripts\\pyinstaller.exe" \
  --onedir \
  --name=predict \
  --console \
  --clean \
  --collect-all torch \
  predict.py
```

```bash
wine "C:\\Program Files\\Python310\\Scripts\\pyinstaller.exe" \
  --noconfirm --clean --onedir \
  --name=predict --noconsole \
  --collect-binaries torch \
  --collect-submodules torch \
  predict.py
```

**⚠️ `--onedir` створює папку `dist/predict/` з .exe + DLL. Рідше проблеми з ініціалізацією!**

**Результат:** `dist/predict.exe` (~180-200 MB з PyTorch)

**⚠️ ВАЖЛИВО:**

- Model checkpoint НЕ вбудований в .exe (передається як аргумент)
- Usage: `predict.exe model.pt input_curve.txt output_params.txt`
- Copy `predict.exe` + `checkpoints/` folder разом

### test_predictor.py (тестовий без ML):

```bash
wine "C:\\Program Files\\Python310\\Scripts\\pyinstaller.exe" --onefile --name=test_predictor_win --console --clean test_predictor.py
```

### Новий скрипт:

```bash
wine "C:\\Program Files\\Python310\\Scripts\\pyinstaller.exe" --onefile --name=my_script_win --console --clean my_script.py
```

---

# ДЕТАЛЬНА ІНСТРУКЦІЯ

## Інструкція: Компіляція Windows .exe на macOS через Wine

## Передумови

### 1. Встановити Wine CrossOver

```bash
# Додати tap для Wine
brew tap gcenx/wine

# Встановити Wine CrossOver (краще для macOS)
brew install --cask --no-quarantine wine-crossover
```

### 2. Встановити Rosetta 2 (для Apple Silicon)

```bash
softwareupdate --install-rosetta --agree-to-license
```

### 3. Перевірити Wine

```bash
wine --version
# Очікуваний результат: wine-8.0.1 (CrossOverFOSS 23.7.1)
```

## Початкове налаштування (один раз)

### 1. Ініціалізувати Wine prefix

```bash
wineboot --init
# Wine створить ~/.wine з Windows середовищем
```

### 2. Завантажити Windows Python

```bash
cd /tmp
curl -L -o python-installer.exe https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe
```

### 3. Встановити Python в Wine

```bash
wine /tmp/python-installer.exe /quiet InstallAllUsers=1 PrependPath=1 Include_test=0
# Чекати ~1-2 хвилини
```

### 4. Перевірити встановлення Python

```bash
wine "C:\\Program Files\\Python310\\python.exe" --version
# Очікуваний результат: Python 3.10.11
```

### 5. Встановити залежності (PyTorch + PyInstaller + NumPy)

```bash
# PyTorch CPU version (для predict.py)
wine "C:\\Program Files\\Python310\\python.exe" -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# NumPy
wine "C:\\Program Files\\Python310\\python.exe" -m pip install numpy

# PyInstaller
wine "C:\\Program Files\\Python310\\python.exe" -m pip install pyinstaller
```

**⚠️ Увага:** Встановлення PyTorch займає ~5-10 хвилин через Wine

## Компіляція Windows .exe

### Команда для predict.py (з PyTorch):

**РЕКОМЕНДОВАНО - `--onedir` (краще з DLL):**

```bash
wine "C:\\Program Files\\Python310\\Scripts\\pyinstaller.exe" \
  --onedir \
  --name=predict \
  --console \
  --clean \
  --collect-all torch \
  predict.py
```

**Або `--onefile` (один файл, але можуть бути проблеми з DLL):**

```bash
wine "C:\\Program Files\\Python310\\Scripts\\pyinstaller.exe" \
  --onefile \
  --name=predict \
  --console \
  --clean \
  --collect-all torch \
  predict.py
```

**Параметри:**

- `--onefile` / `--onedir` - формат output
- `--name=predict` - ім'я .exe файлу
- `--console` - З консольним вікном (для debug)
- `--clean` - очистити кеш перед білдом
- `--collect-all torch` - збирає ВСЕ з PyTorch (binaries + submodules + data)

**⚠️ Компіляція займає ~5-10 хвилин через Wine + PyTorch**

### Результат:

**`--onefile`:**

```
dist/predict.exe - один файл (~180-200 MB з PyTorch)
```

**`--onedir`:**

```
dist/predict/
├── predict.exe      - виконуваний файл
├── torch/           - PyTorch DLL
└── _internal/       - інші залежності
```

### Команда для простих скриптів (БЕЗ ML):

```bash
wine "C:\\Program Files\\Python310\\Scripts\\pyinstaller.exe" \
  --onefile \
  --name=test_predictor_win \
  --console \
  --clean \
  test_predictor.py
```

**Результат:** `dist/test_predictor_win.exe` (~5-10 MB)

## Тестування на macOS

### Запустити predict.exe через Wine:

```bash
# Create test curve
python -c "import numpy as np; curve = np.random.rand(701) * 1e-3 + 1e-5; np.savetxt('test_curve.txt', curve, fmt='%.6e')"

# Run predict.exe
wine dist/predict.exe checkpoints/model.pt test_curve.txt test_output.txt

# Check output
cat test_output.txt
```

### Перевірити exit code:

```bash
wine dist/predict.exe checkpoints/model.pt test_curve.txt test_output.txt
echo $?
# 0 = успіх, 1 = помилка
```

### Перевірити тип файлу:

```bash
file dist/predict.exe
# Очікуваний результат: PE32+ executable (console) x86-64, for MS Windows
```

## Швидкий rebuild

Якщо змінили `predict.py` і треба перекомпілювати:

```bash
# Очистити попередній білд
rm -rf build/ dist/ *.spec

# Перекомпілювати predict.exe (РЕКОМЕНДОВАНО - onedir)
wine "C:\\Program Files\\Python310\\Scripts\\pyinstaller.exe" \
  --onedir --name=predict --console --clean \
  --collect-all torch \
  predict.py

# АБО onefile (один файл)
wine "C:\\Program Files\\Python310\\Scripts\\pyinstaller.exe" \
  --onefile --name=predict --console --clean \
  --collect-all torch \
  predict.py
```

**Час компіляції:** ~5-10 хвилин з PyTorch

## Troubleshooting

### Якщо Wine не запускається:

```bash
# Перевірити Rosetta 2
pgrep oahd || echo "Rosetta 2 не працює!"

# Перезапустити Wine
wineboot -k  # kill wineserver
wineboot --init  # restart
```

### Якщо PyInstaller не знайдено:

```bash
# Перевстановити PyInstaller
wine "C:\\Program Files\\Python310\\python.exe" -m pip uninstall pyinstaller -y
wine "C:\\Program Files\\Python310\\python.exe" -m pip install pyinstaller
```

### Очистити Wine повністю:

```bash
# УВАГА: це видалить весь Windows Python!
rm -rf ~/.wine
wineboot --init  # потім заново встановити Python
```

## Примітки

### predict.exe (з PyTorch):

1. **Час компіляції:** ~5-10 хвилин (Wine + PyTorch)
2. **Розмір .exe:** ~180-200 MB (з PyTorch CPU)
3. **Model checkpoint:** НЕ вбудований, передається як аргумент
4. **Usage:** `predict.exe model.pt input.txt output.txt`
5. **Працює тільки на Windows** - не запускається нативно на macOS!
6. **Тестування на macOS** можливе лише через Wine
7. **Фінальне тестування** робити на реальній Windows машині

### Прості скрипти (БЕЗ ML):

1. **Час компіляції:** ~30-60 секунд
2. **Розмір .exe:** ~5-10 MB

## Структура проекту

```
master-project-light/
├── predict.py                 # ML predictor скрипт
├── model_common.py            # Model architecture
├── checkpoints/               # Model checkpoints (НЕ в .exe)
│   └── model.pt
└── dist/
    └── predict.exe            # Windows executable (~180-200 MB)
```

**Для розгортання на Windows:**

```
your-project/
├── predict.exe               # Скопіювати з dist/
├── checkpoints/              # Скопіювати всю папку
│   └── model.pt
└── data/                     # Ваші дані
    ├── input_curve.txt
    └── output_params.txt
```

## Автоматизація (опціонально)

### Скрипт `build_windows.sh`:

```bash
#!/bin/bash
set -e

echo "Building Windows predict.exe with PyTorch..."

# Clean
rm -rf build/ dist/ *.spec

# Build (onedir - краще для Windows DLL)
wine "C:\\Program Files\\Python310\\Scripts\\pyinstaller.exe" \
  --onedir --name=predict --console --clean \
  --collect-all torch \
  predict.py

# Verify
if [ -d "dist/predict" ] && [ -f "dist/predict/predict.exe" ]; then
    echo "✓ Build successful!"
    ls -lh dist/predict/predict.exe
    du -sh dist/predict/
    echo ""
    echo "📦 Package for Windows:"
    echo "   - Copy entire dist/predict/ folder"
    echo "   - Copy checkpoints/ folder"
    echo ""
    echo "Usage: predict.exe model.pt input.txt output.txt"
elif [ -f "dist/predict.exe" ]; then
    echo "✓ Build successful (onefile)!"
    ls -lh dist/predict.exe
    file dist/predict.exe
else
    echo "✗ Build failed!"
    exit 1
fi
```

Використання:

```bash
chmod +x build_windows.sh
./build_windows.sh
```

**⚠️ Очікуваний час:** ~5-10 хвилин
