# Компіляція Windows .exe через Wine

## ШВИДКА ІНСТРУКЦІЯ

### test_predictor.py (тестовий):
```bash
wine "C:\\Program Files\\Python310\\Scripts\\pyinstaller.exe" --onefile --name=test_predictor_win --noconsole --clean test_predictor.py
```

### predict.py (основний predictor):
```bash
wine "C:\\Program Files\\Python310\\Scripts\\pyinstaller.exe" --onefile --name=predict --noconsole --clean --add-data="checkpoints/dataset_1000_dl100_7d_curve_val_best_curve.pt;checkpoints" predict.py
```

### Новий скрипт:
```bash
wine "C:\\Program Files\\Python310\\Scripts\\pyinstaller.exe" --onefile --name=my_script_win --noconsole --clean my_script.py
```

**Результат:** `dist/назва_файлу.exe`

**Важливо:** Для predict.py потрібно додати model checkpoint через `--add-data`

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

### 5. Встановити PyInstaller
```bash
wine "C:\\Program Files\\Python310\\python.exe" -m pip install pyinstaller
```

## Компіляція Windows .exe

### Команда для білду:
```bash
wine "C:\\Program Files\\Python310\\Scripts\\pyinstaller.exe" \
  --onefile \
  --name=test_predictor_win \
  --noconsole \
  --clean \
  test_predictor.py
```

**Параметри:**
- `--onefile` - один виконуваний файл (не папка)
- `--name=test_predictor_win` - ім'я .exe файлу
- `--noconsole` - БЕЗ чорного вікна консолі (важливо!)
- `--clean` - очистити кеш перед білдом

### Результат:
```
dist/test_predictor_win.exe - Windows виконуваний файл (5.3 MB)
```

## Тестування на macOS

### Запустити через Wine:
```bash
wine dist/test_predictor_win.exe input.txt output.txt
```

### Перевірити exit code:
```bash
wine dist/test_predictor_win.exe input.txt output.txt
echo $?
# 0 = успіх, 1 = помилка
```

### Перевірити тип файлу:
```bash
file dist/test_predictor_win.exe
# Очікуваний результат: PE32+ executable (GUI) x86-64, for MS Windows
```

## Швидкий rebuild

Якщо змінили `test_predictor.py` і треба перекомпілювати:

```bash
# Очистити попередній білд
rm -rf build/ dist/ *.spec

# Перекомпілювати
wine "C:\\Program Files\\Python310\\Scripts\\pyinstaller.exe" \
  --onefile --name=test_predictor_win --noconsole --clean \
  test_predictor.py
```

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

1. **Час компіляції:** ~30-60 секунд
2. **Розмір .exe:** ~5-6 MB
3. **Працює тільки на Windows** - не запускається нативно на macOS!
4. **Тестування на macOS** можливе лише через Wine
5. **Фінальне тестування** робити на реальній Windows машині

## Структура проекту

```
master-project-light/
├── test_predictor.py          # Python скрипт
└── dist/
    └── test_predictor_win.exe # Windows executable
```

## Автоматизація (опціонально)

Можна створити bash скрипт `build_windows.sh`:

```bash
#!/bin/bash
set -e

echo "Building Windows executable..."

# Clean
rm -rf build/ dist/ *.spec

# Build
wine "C:\\Program Files\\Python310\\Scripts\\pyinstaller.exe" \
  --onefile --name=test_predictor_win --noconsole --clean \
  test_predictor.py

# Verify
if [ -f "dist/test_predictor_win.exe" ]; then
    echo "✓ Build successful!"
    ls -lh dist/test_predictor_win.exe
    file dist/test_predictor_win.exe
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
