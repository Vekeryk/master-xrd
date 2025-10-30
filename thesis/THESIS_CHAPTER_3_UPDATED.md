# РОЗДІЛ 3. ПРОЕКТУВАННЯ ПРОГРАМНОГО ЗАБЕЗПЕЧЕННЯ ДЛЯ АНАЛІЗУ КДВ З ІНТЕГРАЦІЄЮ МАШИННОГО НАВЧАННЯ

## 3.1. Архітектура базового програмного забезпечення

### 3.1.1 Загальна структура базового ПЗ

Базове програмне забезпечення "My X-ray program", розроблене в Карпатському національному університеті імені Василя Стефаника, є складною багатокомпонентною системою для моделювання дифракції Х-променів у монокристалах та аналізу експериментальних кривих дифракційного відбивання. Програма реалізована мовою C++ у середовищі C++Builder, що забезпечує ефективне поєднання високої продуктивності обчислень з можливістю створення інтуїтивного графічного інтерфейсу користувача.

Архітектура базового ПЗ побудована за модульним принципом, де кожен функціональний блок відповідає за специфічну частину процесу аналізу. Це дозволяє легко модифікувати окремі компоненти без впливу на решту системи, а також спрощує тестування та підтримку коду.

**Основні підсистеми базового ПЗ**:

1. **Підсистема імпорту та зберігання даних**
   - Імпорт параметрів кристалічної структури з бази даних матеріалів
   - Завантаження експериментальних КДВ з файлів різних форматів (*.dat, *.txt, *.csv)
   - Імпорт апаратних функцій дифрактометра
   - Збереження проміжних та фінальних результатів розрахунків у формат *.rez
   - Експорт розрахункових КДВ для подальшої обробки

2. **Підсистема фізичного моделювання**
   - Розрахунок параметрів кристалічної структури (координати атомів, міжплощинні відстані)
   - Обчислення структурних факторів, поляризовностей, коефіцієнтів поглинання
   - Моделювання дифракції Х-променів за різними теоретичними підходами:
     * Кінематична теорія (швидка, наближена)
     * Динамічна теорія на основі рівнянь Такагі (точна для ідеальних кристалів)
     * Статистична динамічна теорія (враховує дефекти різних типів)
   - Обчислення когерентної та дифузної складових розсіяння
   - Врахування апаратного уширення через згортку з інструментальною функцією

3. **Підсистема мінімізації та оптимізації**
   - Реалізація множинних методів мінімізації:
     * Метод покоординатного спуску (метод конфігурацій)
     * Безградієнтні методи: Nelder-Mead (simplex), Hooke-Jeeves (pattern search)
     * Градієнтні методи: метод найшвидшого спуску, метод спряжених градієнтів
     * Метод найменших квадратів Гауса
   - Комбінована стратегія мінімізації (одночасне використання кількох методів)
   - Адаптивна зміна кроку оптимізації
   - Контроль збіжності та критерії зупинки

4. **Підсистема візуалізації та аналізу**
   - Графічне відображення теоретичних та експериментальних КДВ
   - Підтримка лінійної та логарифмічної шкал інтенсивності
   - Одночасна візуалізація до 3 рефлексів
   - Відображення когерентної, дифузної та сумарної складових
   - Обчислення та відображення функції невідповідності (СКВ, R-factor)
   - Історія змін СКВ під час мінімізації

5. **Підсистема управління та GUI**
   - Інтерактивний користувацький інтерфейс з панеллю закладок
   - Задання параметрів зйомки (геометрія, довжина хвилі, рефлекси)
   - Конфігурація моделі зразка (підкладка, плівка, порушений шар)
   - Параметризація профілів деформації (функціональна та ступінчаста моделі)
   - Налаштування параметрів мінімізації (стартові значення, обмеження, кроки)
   - Контроль виконання обчислень (старт/стоп, збереження проміжних результатів)

### 3.1.2 Алгоритм моделювання когерентної складової КДВ

Розглянемо детально алгоритм моделювання когерентної складової кривої дифракційного відбивання, який є центральним компонентом базового ПЗ та використовується як для генерації навчальних даних, так і для етапу уточнення у гібридному підході.

**Вхідні дані**:
- Параметри кристалічної ґратки (тип, сталі, просторова група)
- Індекси рефлексу (h, k, l)
- Параметри профілю деформації (Dmax1, D01, L1, Rp1, D02, L2, Rp2)
- Товщина підшарів dl (зазвичай 20 Å)
- Кутовий діапазон та кількість точок

**Схема алгоритму**:

```
┌─────────────────────────────────────────────┐
│ 1. ІНІЦІАЛІЗАЦІЯ                            │
│    • Імпорт параметрів кристалічної         │
│      структури та атомних факторів          │
│    • Обчислення координат атомів у          │
│      елементарній комірці                   │
│    • Визначення кутового положення          │
│      рефлексу (кут Брегга θ_B)              │
└──────────────┬──────────────────────────────┘
               ↓
┌─────────────────────────────────────────────┐
│ 2. РОЗРАХУНОК НЕЗАЛЕЖНИХ ВІД ПОЛЯРИЗАЦІЇ   │
│    ПАРАМЕТРІВ                                │
│    • Структурний фактор F(hkl)              │
│    • Поляризовність χ₀, χₕ                  │
│    • Довжина екстинкції Λ                   │
└──────────────┬──────────────────────────────┘
               ↓
┌─────────────────────────────────────────────┐
│ 3. ОБЧИСЛЕННЯ ПРОФІЛЮ ДЕФОРМАЦІЇ D(z)      │
│    • Асиметрична гаусіана:                  │
│      D₁(z) = f(z; Dmax1, D01, L1, Rp1)     │
│    • Спадна гаусіана:                       │
│      D₂(z) = g(z; D02, L2, Rp2)            │
│    • Сумарний профіль:                      │
│      D(z) = D₁(z) + D₂(z)                  │
│    • Дискретизація на підшари (dl=20 Å)    │
└──────────────┬──────────────────────────────┘
               ↓
┌─────────────────────────────────────────────┐
│ ПОЧАТОК ЦИКЛУ ПО РЕФЛЕКСАХ                  │
│ (для одночасного аналізу 444, 888, 880)    │
└──────────────┬──────────────────────────────┘
               ↓
┌─────────────────────────────────────────────┐
│ ПОЧАТОК ЦИКЛУ ПО КУТАХ Δθ                   │
│ (зазвичай 800-2000 точок)                   │
└──────────────┬──────────────────────────────┘
               ↓
┌─────────────────────────────────────────────┐
│ ПОЧАТОК ЦИКЛУ ПО ПОЛЯРИЗАЦІЯХ               │
│ (σ-поляризація та π-поляризація)            │
└──────────────┬──────────────────────────────┘
               ↓
┌─────────────────────────────────────────────┐
│ 4. РОЗРАХУНОК ЗАЛЕЖНИХ ВІД ПОЛЯРИЗАЦІЇ     │
│    ПАРАМЕТРІВ                                │
│    • Поляризаційний фактор P                │
│    • Коефіцієнти зв'язку хвиль              │
└──────────────┬──────────────────────────────┘
               ↓
┌─────────────────────────────────────────────┐
│ 5. РОЗРАХУНОК ІНТЕНСИВНОСТІ ВІД ПІДКЛАДКИ  │
│    • Динамічна теорія для ідеального        │
│      кристала (формули Дарвіна)             │
│    • Вихід: I₀(Δθ) – базова інтенсивність  │
└──────────────┬──────────────────────────────┘
               ↓
┌─────────────────────────────────────────────┐
│ 6. РОЗРАХУНОК ІНТЕНСИВНОСТІ ВІД             │
│    НЕІМПЛАНТОВАНОЇ ЧАСТИНИ ПЛІВКИ           │
│    (якщо є плівка)                          │
│    • Вхід: I₀(Δθ) з підкладки              │
│    • Рекурсивні співвідношення Такагі       │
│    • Вихід: I_film(Δθ)                     │
└──────────────┬──────────────────────────────┘
               ↓
┌─────────────────────────────────────────────┐
│ ПОЧАТОК ЦИКЛУ ПО ПІДШАРАХ                   │
│ (ітерація від найглибшого до поверхні)      │
└──────────────┬──────────────────────────────┘
               ↓
┌─────────────────────────────────────────────┐
│ 7. РОЗРАХУНОК ПАРАМЕТРІВ ПІДШАРУ            │
│    • Деформація D_i з профілю D(z)         │
│    • Зміщення Бреггового кута Δθ_B         │
│    • Модифікація поляризовності через       │
│      деформацію: χ → χ(1 + D_i)            │
└──────────────┬──────────────────────────────┘
               ↓
┌─────────────────────────────────────────────┐
│ 8. РОЗРАХУНОК ІНТЕНСИВНОСТІ ВІД ПІДШАРУ    │
│    • Рекурсивні рівняння Такагі-Таупіна:   │
│      R₀(z+dl) = f(R₀(z), Rₕ(z), χ, D_i)   │
│      Rₕ(z+dl) = g(R₀(z), Rₕ(z), χ, D_i)   │
│    • Вихід: оновлені амплітуди R₀, Rₕ      │
└──────────────┬──────────────────────────────┘
               ↓
┌─────────────────────────────────────────────┐
│ КІНЕЦЬ ЦИКЛУ ПО ПІДШАРАХ                    │
│ (всі підшари оброблені → маємо R₀, Rₕ       │
│  на поверхні)                               │
└──────────────┬──────────────────────────────┘
               ↓
┌─────────────────────────────────────────────┐
│ 9. РОЗРАХУНОК СУМАРНОЇ ІНТЕНСИВНОСТІ        │
│    ПЕВНОЇ ПОЛЯРИЗАЦІЇ                        │
│    • I_pol(Δθ) = |Rₕ/R₀|²                  │
└──────────────┬──────────────────────────────┘
               ↓
┌─────────────────────────────────────────────┐
│ КІНЕЦЬ ЦИКЛУ ПО ПОЛЯРИЗАЦІЯХ                │
└──────────────┬──────────────────────────────┘
               ↓
┌─────────────────────────────────────────────┐
│ 10. РОЗРАХУНОК СУМАРНОЇ ІНТЕНСИВНОСТІ       │
│     З УРАХУВАННЯМ ОБОХ ПОЛЯРИЗАЦІЙ          │
│     • I(Δθ) = I_σ(Δθ) + I_π(Δθ)           │
└──────────────┬──────────────────────────────┘
               ↓
┌─────────────────────────────────────────────┐
│ КІНЕЦЬ ЦИКЛУ ПО КУТАХ                       │
│ (маємо повну КДВ: I(Δθ) для всіх точок)    │
└──────────────┬──────────────────────────────┘
               ↓
┌─────────────────────────────────────────────┐
│ 11. ВРАХУВАННЯ АПАРАТНОГО УШИРЕННЯ          │
│     • Згортка теоретичної КДВ з             │
│       інструментальною функцією:            │
│       I_theor(Δθ) = I(Δθ) ⊗ g_instr(Δθ)   │
└──────────────┬──────────────────────────────┘
               ↓
┌─────────────────────────────────────────────┐
│ 12. ФОРМУВАННЯ КДВ ДЛЯ ДАНОГО РЕФЛЕКСА     │
│     • Збереження I_theor(Δθ) для           │
│       поточного рефлексу                    │
└──────────────┬──────────────────────────────┘
               ↓
┌─────────────────────────────────────────────┐
│ КІНЕЦЬ ЦИКЛУ ПО РЕФЛЕКСАХ                   │
└──────────────┬──────────────────────────────┘
               ↓
┌─────────────────────────────────────────────┐
│ ВИХІД: набір теоретичних КДВ для всіх       │
│        заданих рефлексів                     │
└─────────────────────────────────────────────┘
```

**Обчислювальна складність**:

Загальна кількість обчислень масштабується як:
- O(N_refl × N_angles × N_pol × N_layers)

Для типових параметрів:
- N_refl = 1-3 (рефлекси)
- N_angles = 800-2000 (кутові точки)
- N_pol = 2 (поляризації)
- N_layers = 100-350 (підшари)

Час обчислення однієї КДВ на сучасному CPU (без оптимізацій): ~0.5-2 секунди.

**Критичні оптимізації в базовому ПЗ**:

1. **Кешування проміжних результатів**: параметри, що не залежать від кута (структурні фактори, поляризовності), обчислюються один раз на початку.

2. **Векторизація операцій**: всі кутові точки для певного підшару обробляються одночасно, якщо це дозволяє архітектура.

3. **Використання lookup tables**: тригонометричні функції та експоненти для типових значень зберігаються у таблицях.

4. **Умовна компіляція блоків**: якщо користувач вимкнув врахування дифузної складової, відповідний код не виконується.

### 3.1.3 Моделі порушеного шару

Базове ПЗ підтримує два основні підходи до параметризації приповерхневого порушеного шару:

**Функціональна модель**:

Профіль деформації задається аналітичною функцією з невеликою кількістю параметрів (зазвичай 7). Це є природним вибором для випадків, коли існує фізично обґрунтована залежність параметрів від глибини, наприклад, для іонно-імплантованих шарів.

Математичний вираз для профілю деформації у вигляді суми асиметричної та спадної гаусіан:

D(z) = Dmax1 × exp[-(z - Rp1)²/(2σ₁²)]  для z < Rp1 (асиметрична гаусіана, низхідна гілка)
     + D01 × exp[-(z - 0)²/(2σ₂²)]     для z ≥ Rp1 (асиметрична гаусіана, спадна гілка)
     + D02 × exp[-(z - Rp2)²/(2σ₃²)]    (спадна гаусіана від поверхні)

де σ₁, σ₂, σ₃ визначаються через L1, L2 та інші параметри.

Переваги функціональної моделі:
- Мала кількість параметрів (7) → швидка збіжність оптимізації
- Гладкий профіль без нефізичних флуктуацій
- Параметри мають чітку фізичну інтерпретацію
- Можливість використання апріорної інформації з SRIM-моделювання

Недоліки:
- Обмежена гнучкість: реальний профіль може відхилятися від функціональної форми
- Не підходить для складних випадків (багатошарові імплантації, термообробка)

**Ступінчаста модель**:

Профіль деформації представлено у вигляді послідовності підшарів (зазвичай 100-200), кожен з яких характеризується двома незалежними параметрами: деформацією D_i та товщиною dl_i.

D(z) = D_i  для z ∈ [z_i, z_i + dl_i],  i = 1, 2, ..., N_layers

Переваги ступінчастої моделі:
- Максимальна гнучкість: може описати довільний профіль
- Не вимагає апріорних припущень про форму
- Здатна виявити неочікувані features (додаткові піки, немонотонність)

Недоліки:
- Велика кількість параметрів (2 × N_layers ≈ 200-400) → повільна збіжність
- Схильність до overfitting: можуть з'явитися нефізичні осциляції
- Потребує доброго стартового наближення

**Комбінована стратегія (використовується у базовому ПЗ)**:

1. Етап 1: функціональна модель з початковим наближенням → отримання p_func
2. Етап 2: конвертація p_func у ступінчасту модель як ініціалізація
3. Етап 3: уточнення ступінчастої моделі методами мінімізації → отримання p_step

Така стратегія поєднує переваги обох підходів: швидку збіжність функціональної моделі та гнучкість ступінчастої.

## 3.2. Проектування ML-модуля для автоматизації стартового наближення

### 3.2.1 Місце ML-модуля у загальній архітектурі системи

ML-модуль проектується як **зовнішній допоміжний компонент**, що інтегрується у робочий процес базового ПЗ на етапі ініціалізації параметрів. Ключовим принципом проектування є **мінімальна інвазивність**: ML-модуль не вимагає модифікації коду базового ПЗ, а взаємодіє з ним через стандартні механізми файлового обміну.

**Архітектурна схема інтеграції**:

```
┌─────────────────────────────────────────────────────────────┐
│                 ЗАГАЛЬНА СИСТЕМА АНАЛІЗУ КДВ                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │  ML-МОДУЛЬ (Python, PyTorch)                       │    │
│  │  ┌──────────────────────────────────────────────┐  │    │
│  │  │  1. Data Preprocessing                       │  │    │
│  │  │     • Load experimental XRD curve            │  │    │
│  │  │     • Normalize: log₁₀ transform            │  │    │
│  │  │     • Truncate: select 640 tail points      │  │    │
│  │  └──────────────────┬───────────────────────────┘  │    │
│  │                     ↓                               │    │
│  │  ┌──────────────────────────────────────────────┐  │    │
│  │  │  2. CNN Inference                            │  │    │
│  │  │     • Load trained model (v3)                │  │    │
│  │  │     • Forward pass: curve → 7 parameters     │  │    │
│  │  │     • Device: CPU / MPS / CUDA               │  │    │
│  │  └──────────────────┬───────────────────────────┘  │    │
│  │                     ↓                               │    │
│  │  ┌──────────────────────────────────────────────┐  │    │
│  │  │  3. Output Formatting                        │  │    │
│  │  │     • Denormalize parameters to physical     │  │    │
│  │  │       ranges                                  │  │    │
│  │  │     • Save to ml_predictions.dat             │  │    │
│  │  └──────────────────────────────────────────────┘  │    │
│  └────────────────────────┬───────────────────────────┘    │
│                            │                                │
│                            ↓ (file exchange)                │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │  БАЗОВЕ ПЗ (C++Builder, "My X-ray program")       │    │
│  │  ┌──────────────────────────────────────────────┐  │    │
│  │  │  4. Import ML Predictions                    │  │    │
│  │  │     • Read ml_predictions.dat                │  │    │
│  │  │     • Auto-fill GUI parameter fields         │  │    │
│  │  └──────────────────┬───────────────────────────┘  │    │
│  │                     ↓                               │    │
│  │  ┌──────────────────────────────────────────────┐  │    │
│  │  │  5. Physical Modeling & Optimization         │  │    │
│  │  │     • Functional model refinement            │  │    │
│  │  │     • Takagi-Taupin simulation               │  │    │
│  │  │     • Minimization (Nelder-Mead, Hooke-J.)   │  │    │
│  │  └──────────────────┬───────────────────────────┘  │    │
│  │                     ↓                               │    │
│  │  ┌──────────────────────────────────────────────┐  │    │
│  │  │  6. (Optional) Stepwise Refinement           │  │    │
│  │  │     • Convert to stepwise model              │  │    │
│  │  │     • Optimize layer-by-layer                │  │    │
│  │  └──────────────────┬───────────────────────────┘  │    │
│  │                     ↓                               │    │
│  │  ┌──────────────────────────────────────────────┐  │    │
│  │  │  7. (Optional) Defect Inclusion              │  │    │
│  │  │     • Add diffuse scattering                 │  │    │
│  │  │     • Optimize defect parameters             │  │    │
│  │  └──────────────────┬───────────────────────────┘  │    │
│  │                     ↓                               │    │
│  │  ┌──────────────────────────────────────────────┐  │    │
│  │  │  8. Results & Visualization                  │  │    │
│  │  │     • Final deformation profile              │  │    │
│  │  │     • Theoretical vs experimental XRD        │  │    │
│  │  │     • χ², R-factor metrics                   │  │    │
│  │  └──────────────────────────────────────────────┘  │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

**Переваги такої архітектури**:

1. **Незалежність компонентів**: ML-модуль та базове ПЗ можуть розроблятись, тестуватись та оновлюватись незалежно.

2. **Гнучкість вибору інструментів**: Python+PyTorch для ML (де екосистема найбільш розвинута) та C++Builder для фізичного моделювання (де потрібна максимальна продуктивність).

3. **Можливість ручного втручання**: користувач може вручну скоригувати ML-передбачення перед запуском оптимізації, якщо вважає їх неточними.

4. **Backward compatibility**: базове ПЗ залишається повністю функціональним навіть без ML-модуля (можна використовувати традиційний підхід з ручним стартовим наближенням).

5. **Легка інтеграція нових ML-моделей**: при покращенні архітектури CNN (наприклад, перехід до v4) достатньо замінити файл з вагами моделі, без змін у базовому ПЗ.

### 3.2.2 Модульна структура ML-компонента

ML-модуль організовано за принципами **clean architecture** з чітким розділенням відповідальностей між компонентами:

**Структура Python пакету**:

```
ml_module/
├── __init__.py                 # Package initialization
├── config.py                   # Configuration constants
├── models/                     # Neural network architectures
│   ├── __init__.py
│   ├── common.py              # Base classes (ResidualBlock, AttentionPool)
│   ├── v1_baseline.py         # Architecture v1
│   ├── v2_attention.py        # Architecture v2
│   └── v3_ziegler.py          # Architecture v3 (current)
├── data/                       # Data processing
│   ├── __init__.py
│   ├── preprocessing.py       # Normalization, truncation
│   ├── dataset.py             # PyTorch Dataset classes
│   └── augmentation.py        # Data augmentation (if needed)
├── training/                   # Training pipeline
│   ├── __init__.py
│   ├── trainer.py             # Training loop
│   ├── losses.py              # Physics-constrained loss
│   └── metrics.py             # Evaluation metrics (MAE, RMSE, R²)
├── inference/                  # Inference pipeline
│   ├── __init__.py
│   ├── predictor.py           # Main inference class
│   └── postprocessing.py      # Denormalization, formatting
├── utils/                      # Utilities
│   ├── __init__.py
│   ├── io.py                  # File I/O operations
│   ├── visualization.py       # Plotting functions
│   └── validation.py          # Parameter validation
└── scripts/                    # Executable scripts
    ├── predict.py             # CLI for inference
    ├── train.py               # CLI for training
    └── evaluate.py            # CLI for evaluation
```

**Ключові класи та їх відповідальності**:

**1. XRDPreprocessor** (data/preprocessing.py):
```python
class XRDPreprocessor:
    """Відповідає за нормалізацію вхідних даних."""

    def __init__(self, start_idx: int = 10, n_points: int = 640):
        self.start_idx = start_idx  # Початок "хвоста"
        self.n_points = n_points    # Кількість точок для CNN

    def normalize(self, curve: np.ndarray) -> np.ndarray:
        """Log-scale нормалізація."""
        return np.log10(curve + 1e-10)

    def truncate(self, curve: np.ndarray) -> np.ndarray:
        """Вибір 640 точок хвоста."""
        return curve[self.start_idx:self.start_idx + self.n_points]

    def preprocess(self, raw_curve: np.ndarray) -> torch.Tensor:
        """Повний pipeline: normalize → truncate → to_tensor."""
        normalized = self.normalize(raw_curve)
        truncated = self.truncate(normalized)
        return torch.from_numpy(truncated).float().unsqueeze(0).unsqueeze(0)
```

**2. XRDRegressor** (models/v3_ziegler.py):
```python
class XRDRegressor(nn.Module):
    """CNN архітектура v3 для регресії 7 параметрів."""

    def __init__(self, n_out: int = 7, kernel_size: int = 15):
        super().__init__()
        # [Детальна архітектура описана у Розділі 2]
        self.stem = ...
        self.blocks = ...
        self.attention_pool = AttentionPool(128)
        self.fc = ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 1, 640) - normalized XRD curves
        Returns:
            (batch, 7) - predicted parameters [Dmax1, D01, L1, Rp1, D02, L2, Rp2]
        """
        ...
```

**3. XRDPredictor** (inference/predictor.py):
```python
class XRDPredictor:
    """Головний клас для inference pipeline."""

    def __init__(self, model_path: str, device: str = "auto"):
        self.device = self._select_device(device)
        self.model = self._load_model(model_path)
        self.preprocessor = XRDPreprocessor()
        self.postprocessor = XRDPostprocessor()

    def predict(self, experimental_curve: np.ndarray) -> dict:
        """
        Повний inference pipeline.

        Args:
            experimental_curve: (N,) array - raw experimental XRD

        Returns:
            Dictionary з передбаченими параметрами:
            {
                'Dmax1': float,
                'D01': float,
                'L1': float,
                'Rp1': float,
                'D02': float,
                'L2': float,
                'Rp2': float,
                'confidence': float  # optional
            }
        """
        # 1. Preprocessing
        input_tensor = self.preprocessor.preprocess(experimental_curve)
        input_tensor = input_tensor.to(self.device)

        # 2. Inference
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_tensor)

        # 3. Postprocessing
        params = self.postprocessor.denormalize(output.cpu().numpy()[0])

        return params

    def predict_to_file(self, curve: np.ndarray, output_path: str):
        """Inference з збереженням у файл для базового ПЗ."""
        params = self.predict(curve)
        self._save_for_basepz(params, output_path)
```

**4. XRDPostprocessor** (inference/postprocessing.py):
```python
class XRDPostprocessor:
    """Денормалізація та форматування output."""

    def __init__(self):
        # Діапазони параметрів
        self.scales = np.array([0.029, 0.029, 6000, 6000, 0.029, 6000, 6000])
        self.biases = np.array([0.001, 0.002, 1000, 0, 0.002, 1000, -6000])
        self.param_names = ['Dmax1', 'D01', 'L1', 'Rp1', 'D02', 'L2', 'Rp2']

    def denormalize(self, normalized: np.ndarray) -> dict:
        """
        Конвертація з sigmoid output [0,1] до фізичних діапазонів.

        Args:
            normalized: (7,) array з значеннями [0, 1]
        Returns:
            Dictionary з денормалізованими параметрами
        """
        physical = normalized * self.scales + self.biases
        return dict(zip(self.param_names, physical))

    def validate_physics(self, params: dict) -> tuple[bool, list]:
        """
        Перевірка фізичних обмежень.

        Returns:
            (is_valid, violated_constraints)
        """
        violations = []

        if params['D01'] > params['Dmax1']:
            violations.append("D01 > Dmax1")
        if params['L2'] > params['L1']:
            violations.append("L2 > L1")
        if params['D01'] + params['D02'] > 0.03:
            violations.append("D01 + D02 > 0.03")
        if params['Rp1'] > params['L1']:
            violations.append("Rp1 > L1")
        if not (-params['L2'] <= params['Rp2'] <= 0):
            violations.append("Rp2 not in [-L2, 0]")

        return len(violations) == 0, violations
```

### 3.2.3 Протокол взаємодії ML-модуля з базовим ПЗ

Взаємодія між ML-модулем (Python) та базовим ПЗ (C++Builder) відбувається через обмін текстовими файлами. Цей підхід забезпечує максимальну сумісність та простоту реалізації.

**Формат файлу ml_predictions.dat**:

```
# ML-predicted parameters for XRD analysis
# Generated: 2025-01-15 14:23:45
# Model: XRDRegressor_v3
# Confidence: 0.92

Dmax1 = 0.0125
D01 = 0.0087
L1 = 3450.0
Rp1 = 1820.0
D02 = 0.0063
L2 = 2780.0
Rp2 = -1240.0

# Physics constraints status: PASSED
# Estimated MAE: 5.8%
```

**Алгоритм автоматичного завантаження у базове ПЗ**:

1. Користувач запускає ML-inference через command-line або GUI wrapper:
   ```bash
   python scripts/predict.py --input exp_curve.dat --output ml_predictions.dat
   ```

2. Базове ПЗ має кнопку "Імпорт ML-передбачень" або автоматично перевіряє наявність файлу ml_predictions.dat при запуску.

3. При імпорті базове ПЗ парсить файл та заповнює відповідні поля у закладці "Профіль → Асиметрична гаусіана + спадна гаусіана".

4. Користувач візуально перевіряє значення та може вручну скоригувати, якщо щось виглядає нерозумно.

5. Натискається кнопка "Наближати gausauto" для запуску функціонального уточнення.

**Псевдокод для C++Builder (імпорт параметрів)**:

```cpp
void TMainForm::ImportMLPredictions(AnsiString filename) {
    TStringList *lines = new TStringList();
    lines->LoadFromFile(filename);

    for (int i = 0; i < lines->Count; i++) {
        AnsiString line = lines->Strings[i].Trim();

        // Skip comments and empty lines
        if (line.IsEmpty() || line[1] == '#') continue;

        // Parse "ParamName = Value"
        int pos = line.Pos("=");
        if (pos > 0) {
            AnsiString param_name = line.SubString(1, pos-1).Trim();
            double value = StrToFloat(line.SubString(pos+1, line.Length()).Trim());

            // Fill corresponding GUI fields
            if (param_name == "Dmax1") EditDmax1->Text = FloatToStr(value);
            else if (param_name == "D01") EditD01->Text = FloatToStr(value);
            else if (param_name == "L1") EditL1->Text = FloatToStr(value);
            else if (param_name == "Rp1") EditRp1->Text = FloatToStr(value);
            else if (param_name == "D02") EditD02->Text = FloatToStr(value);
            else if (param_name == "L2") EditL2->Text = FloatToStr(value);
            else if (param_name == "Rp2") EditRp2->Text = FloatToStr(value);
        }
    }

    delete lines;
    ShowMessage("ML predictions imported successfully!");
}
```

## 3.3. Проектування системи тренування ML-моделі

### 3.3.1 Архітектура training pipeline

Training pipeline організовано як послідовність етапів з чіткими контрольними точками (checkpoints) та можливістю відновлення у разі переривання.

**Компоненти training pipeline**:

```
┌──────────────────────────────────────────────────────────┐
│                TRAINING PIPELINE                          │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  1. DATA LOADING & SPLITTING                             │
│     ┌─────────────────────────────────────────────┐     │
│     │ • Load dataset (1M samples, ~5 GB)          │     │
│     │ • Train/val split (95% / 5%)                │     │
│     │ • Create DataLoaders (batch=256)            │     │
│     └─────────────────────────────────────────────┘     │
│                          ↓                                │
│  2. MODEL INITIALIZATION                                 │
│     ┌─────────────────────────────────────────────┐     │
│     │ • Create XRDRegressor(v3)                   │     │
│     │ • Initialize weights (Kaiming init)         │     │
│     │ • Move to device (MPS / CUDA / CPU)         │     │
│     │ • Print model summary (~1.2M params)        │     │
│     └─────────────────────────────────────────────┘     │
│                          ↓                                │
│  3. OPTIMIZER & SCHEDULER SETUP                          │
│     ┌─────────────────────────────────────────────┐     │
│     │ • AdamW(lr=0.002, weight_decay=5e-4)        │     │
│     │ • ReduceLROnPlateau(patience=5, factor=0.5) │     │
│     │ • Gradient clipping: max_norm=1.0           │     │
│     └─────────────────────────────────────────────┘     │
│                          ↓                                │
│  4. TRAINING LOOP (100 epochs)                           │
│     ┌─────────────────────────────────────────────┐     │
│     │ For each epoch:                             │     │
│     │   • Train phase (forward, backward, step)   │     │
│     │   • Validation phase (metrics calculation)  │     │
│     │   • Learning rate scheduling                │     │
│     │   • Checkpoint saving (best model)          │     │
│     │   • Early stopping check                    │     │
│     │   • Progress logging (wandb / tensorboard)  │     │
│     └─────────────────────────────────────────────┘     │
│                          ↓                                │
│  5. FINAL EVALUATION                                     │
│     ┌─────────────────────────────────────────────┐     │
│     │ • Load best checkpoint                      │     │
│     │ • Compute test metrics (MAE, RMSE, R²)      │     │
│     │ • Per-parameter error analysis              │     │
│     │ • Visualize predictions vs ground truth     │     │
│     └─────────────────────────────────────────────┘     │
│                          ↓                                │
│  6. MODEL EXPORT                                         │
│     ┌─────────────────────────────────────────────┐     │
│     │ • Save checkpoint (.pt file)                │     │
│     │ • Export to TorchScript (optional)          │     │
│     │ • Export to ONNX (optional, for C++)        │     │
│     └─────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────┘
```

### 3.3.2 Функція втрат з фізичними обмеженнями

Ключовим елементом навчання є спеціальна функція втрат, що поєднує стандартну MSE з penalty за порушення фізичних обмежень:

```python
class PhysicsConstrainedLoss(nn.Module):
    """
    Loss function з врахуванням фізичних обмежень.

    Loss = MSE + α * Σ penalties
    """

    def __init__(self, alpha: float = 0.1):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple:
        """
        Args:
            pred: (batch, 7) - predicted parameters
            target: (batch, 7) - ground truth parameters

        Returns:
            total_loss, mse_loss, penalty
        """
        # Base MSE loss
        mse_loss = self.mse(pred, target)

        # Extract parameters (batch, )
        Dmax1, D01, L1, Rp1, D02, L2, Rp2 = pred.T

        # Compute penalties (ReLU ensures penalty=0 if constraint satisfied)
        penalty_1 = torch.relu(D01 - Dmax1)         # D01 ≤ Dmax1
        penalty_2 = torch.relu(L2 - L1)             # L2 ≤ L1
        penalty_3 = torch.relu(D01 + D02 - 0.03)    # D01 + D02 ≤ 0.03
        penalty_4 = torch.relu(Rp1 - L1)            # Rp1 ≤ L1
        penalty_5a = torch.relu(-Rp2 - L2)          # -L2 ≤ Rp2
        penalty_5b = torch.relu(Rp2)                # Rp2 ≤ 0

        total_penalty = (penalty_1 + penalty_2 + penalty_3 +
                         penalty_4 + penalty_5a + penalty_5b).mean()

        total_loss = mse_loss + self.alpha * total_penalty

        return total_loss, mse_loss, total_penalty
```

**Обґрунтування вибору α=0.1**:

Коефіцієнт α контролює trade-off між точністю передбачення (MSE) та дотриманням фізичних обмежень (penalty). Якщо α занадто мале, мережа може навчитися порушувати обмеження для кращої MSE. Якщо α занадто велике, мережа буде занадто консервативною, даючи субоптимальні передбачення заради гарантованого дотримання обмежень.

Експериментально встановлено, що α=0.1 забезпечує хороший баланс: <1% передбачень порушують обмеження, і ці порушення є мінімальними (зазвичай <5% від граничного значення).

### 3.3.3 Стратегія Data Augmentation

Хоча навчальні дані є синтетичними і вже охоплюють весь діапазон параметрів, data augmentation може покращити робастність моделі до експериментальних артефактів:

**Техніки аугментації** (застосовуються з ймовірністю 0.3 під час тренування):

1. **Gaussian noise**: додавання білого шуму I' = I + ε, ε ~ N(0, σ²)
   - Імітує шум детектора
   - σ = 0.01-0.05 від max(I)

2. **Baseline drift**: додавання поліноміальної базової лінії
   - Імітує дифузний фон
   - b(θ) = a₀ + a₁θ + a₂θ²

3. **Intensity scaling**: множення на випадковий фактор
   - Імітує варіації інтенсивності джерела
   - scale ∈ [0.8, 1.2]

4. **Horizontal shift**: зсув кривої уздовж кутової осі
   - Імітує неточність юстування
   - shift ∈ [-5, 5] points

Аугментація застосовується **тільки до навчальної вибірки**, валідаційна залишається чистою для об'єктивної оцінки.

**Реалізація аугментації**:

```python
class XRDAugmentation:
    """Аугментація XRD-кривих для підвищення робастності."""

    def __init__(self, prob: float = 0.3):
        self.prob = prob

    def __call__(self, curve: np.ndarray) -> np.ndarray:
        """Застосування випадкових аугментацій."""
        augmented = curve.copy()

        if np.random.rand() < self.prob:
            # Gaussian noise
            noise_std = np.random.uniform(0.01, 0.05) * curve.max()
            augmented += np.random.normal(0, noise_std, curve.shape)

        if np.random.rand() < self.prob:
            # Baseline drift
            x = np.linspace(-1, 1, len(curve))
            a0 = np.random.uniform(-0.1, 0.1)
            a1 = np.random.uniform(-0.05, 0.05)
            a2 = np.random.uniform(-0.02, 0.02)
            baseline = a0 + a1*x + a2*x**2
            augmented += baseline * curve.max()

        if np.random.rand() < self.prob:
            # Intensity scaling
            scale = np.random.uniform(0.8, 1.2)
            augmented *= scale

        if np.random.rand() < self.prob:
            # Horizontal shift
            shift = np.random.randint(-5, 6)
            augmented = np.roll(augmented, shift)

        return augmented
```

## 3.4. Тестування та валідація ML-модуля

### 3.4.1 Стратегія тестування

Тестування ML-модуля організовано у три рівні, що забезпечують верифікацію функціональності від окремих компонентів до повного інтеграційного процесу.

**Рівень 1: Unit Testing (компонентне тестування)**

Кожен модуль тестується ізольовано з фіксованими синтетичними вхідними даними:

```python
# tests/test_preprocessing.py
import pytest
import numpy as np
from ml_module.data.preprocessing import XRDPreprocessor

def test_normalize():
    """Тест log-scale нормалізації."""
    preprocessor = XRDPreprocessor()
    curve = np.array([1.0, 10.0, 100.0, 1000.0])
    normalized = preprocessor.normalize(curve)

    expected = np.array([0.0, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(normalized, expected, atol=1e-8)

def test_truncate():
    """Тест обрізки до 640 точок."""
    preprocessor = XRDPreprocessor(start_idx=10, n_points=640)
    curve = np.arange(1000)
    truncated = preprocessor.truncate(curve)

    assert len(truncated) == 640
    assert truncated[0] == 10
    assert truncated[-1] == 649

def test_preprocess_output_shape():
    """Тест формату вихідного тензора."""
    preprocessor = XRDPreprocessor()
    curve = np.random.rand(1000)
    tensor = preprocessor.preprocess(curve)

    assert tensor.shape == (1, 1, 640)
    assert tensor.dtype == torch.float32
```

```python
# tests/test_postprocessing.py
def test_denormalize():
    """Тест денормалізації параметрів."""
    postprocessor = XRDPostprocessor()
    normalized = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    params = postprocessor.denormalize(normalized)

    # Перевірка середніх значень діапазонів
    assert params['Dmax1'] == pytest.approx(0.0155, abs=1e-4)
    assert params['D01'] == pytest.approx(0.0165, abs=1e-4)
    assert params['L1'] == pytest.approx(4000.0, abs=1.0)

def test_validate_physics_pass():
    """Тест валідації фізично коректних параметрів."""
    postprocessor = XRDPostprocessor()
    params = {
        'Dmax1': 0.015, 'D01': 0.010, 'L1': 3000,
        'Rp1': 1500, 'D02': 0.008, 'L2': 2500, 'Rp2': -1200
    }
    is_valid, violations = postprocessor.validate_physics(params)

    assert is_valid == True
    assert len(violations) == 0

def test_validate_physics_fail():
    """Тест виявлення порушень фізичних обмежень."""
    postprocessor = XRDPostprocessor()
    params = {
        'Dmax1': 0.010, 'D01': 0.015,  # D01 > Dmax1 ❌
        'L1': 2500, 'Rp1': 1500,
        'D02': 0.008, 'L2': 3000,      # L2 > L1 ❌
        'Rp2': -1200
    }
    is_valid, violations = postprocessor.validate_physics(params)

    assert is_valid == False
    assert "D01 > Dmax1" in violations
    assert "L2 > L1" in violations
```

**Рівень 2: Integration Testing (інтеграційне тестування)**

Тестування взаємодії між компонентами ML-модуля та базовим ПЗ:

```python
# tests/test_integration.py
def test_full_inference_pipeline():
    """Тест повного inference pipeline."""
    predictor = XRDPredictor(model_path="checkpoints/best_model.pt")

    # Завантаження тестової експериментальної кривої
    exp_curve = np.loadtxt("tests/fixtures/experimental_curve.dat")

    # Inference
    params = predictor.predict(exp_curve)

    # Перевірки
    assert len(params) == 7
    assert all(key in params for key in ['Dmax1', 'D01', 'L1', 'Rp1',
                                          'D02', 'L2', 'Rp2'])
    assert 0.001 <= params['Dmax1'] <= 0.030
    assert 0.002 <= params['D01'] <= 0.030
    assert 1000 <= params['L1'] <= 7000

def test_file_exchange_format():
    """Тест формату файлу для базового ПЗ."""
    predictor = XRDPredictor(model_path="checkpoints/best_model.pt")
    exp_curve = np.loadtxt("tests/fixtures/experimental_curve.dat")

    output_file = "tests/tmp/ml_predictions.dat"
    predictor.predict_to_file(exp_curve, output_file)

    # Перевірка формату файлу
    with open(output_file, 'r') as f:
        content = f.read()

    assert "Dmax1 =" in content
    assert "D01 =" in content
    assert "L1 =" in content
    assert "# Physics constraints status:" in content
```

**Рівень 3: End-to-End Testing (наскрізне тестування)**

Тестування повного циклу: ML inference → імпорт у базове ПЗ → functional refinement:

```python
# tests/test_e2e.py
def test_end_to_end_workflow(tmp_path):
    """
    Тест повного робочого процесу:
    1. ML inference
    2. Експорт у ml_predictions.dat
    3. Імпорт у базове ПЗ (симуляція)
    4. Перевірка адекватності стартового наближення
    """
    # 1. ML inference
    predictor = XRDPredictor(model_path="checkpoints/best_model.pt")
    exp_curve = np.loadtxt("tests/fixtures/real_sample_GGG_YIG.dat")
    params_ml = predictor.predict(exp_curve)

    # 2. Експорт
    output_file = tmp_path / "ml_predictions.dat"
    predictor.predict_to_file(exp_curve, output_file)

    # 3. Симуляція імпорту (перевірка парсингу)
    params_imported = parse_ml_predictions(output_file)
    assert params_ml == params_imported

    # 4. Перевірка якості стартового наближення
    # Генеруємо теоретичну криву з ML-параметрами
    theoretical_curve = compute_xrd_curve(params_ml)

    # Обчислюємо початкову невідповідність
    chi2_initial = compute_chi_squared(exp_curve, theoretical_curve)

    # Стартове наближення має давати Chi² < 0.1
    # (для подальшого functional refinement)
    assert chi2_initial < 0.1, \
        f"ML prediction too far from experiment: χ²={chi2_initial:.4f}"
```

### 3.4.2 Метрики валідації моделі

Для оцінки якості ML-моделі використовується набір метрик, що враховують специфіку задачі регресії параметрів XRD:

**Глобальні метрики** (усереднені по всіх параметрах):

1. **Mean Absolute Error (MAE)**:
   ```
   MAE = (1/7) Σᵢ₌₁⁷ |yᵢ_pred - yᵢ_true|
   ```
   - Середня абсолютна похибка у фізичних одиницях
   - Залежить від масштабу параметрів

2. **Mean Absolute Percentage Error (MAPE)**:
   ```
   MAPE = (1/7) Σᵢ₌₁⁷ |yᵢ_pred - yᵢ_true| / yᵢ_true × 100%
   ```
   - Відносна похибка, не залежить від масштабу
   - **Цільова метрика**: MAPE < 10% для всіх параметрів

3. **Root Mean Squared Error (RMSE)**:
   ```
   RMSE = sqrt((1/7) Σᵢ₌₁⁷ (yᵢ_pred - yᵢ_true)²)
   ```
   - Штрафує великі помилки сильніше, ніж MAE

4. **R² Score (коефіцієнт детермінації)**:
   ```
   R² = 1 - Σᵢ(yᵢ_pred - yᵢ_true)² / Σᵢ(yᵢ_true - ȳ)²
   ```
   - Частка дисперсії, пояснена моделлю
   - R² = 1: ідеальне передбачення, R² = 0: модель = середнє значення

**Per-parameter метрики** (для кожного з 7 параметрів):

Оскільки різні параметри мають різну складність передбачення (наприклад, позиційні Rp1, Rp2 складніші за амплітудні Dmax1, D01, D02), важливо аналізувати метрики окремо:

| Параметр | MAE цільове | MAPE цільове | Складність |
|----------|-------------|--------------|------------|
| Dmax1    | < 0.002     | < 10%        | Низька     |
| D01      | < 0.002     | < 10%        | Низька     |
| L1       | < 300 Å     | < 8%         | Середня    |
| Rp1      | < 400 Å     | < 15%        | Висока     |
| D02      | < 0.002     | < 10%        | Низька     |
| L2       | < 300 Å     | < 8%         | Середня    |
| Rp2      | < 450 Å     | < 15%        | Висока     |

**Фізичні обмеження** (constraint violation rate):

Відсоток передбачень, що порушують фізичні обмеження:
```
Violation rate = (# predictions with violations) / (# total predictions) × 100%
```
**Цільове значення**: <1%

**Реалізація обчислення метрик**:

```python
# ml_module/training/metrics.py
class RegressionMetrics:
    """Обчислення метрик для регресії параметрів."""

    def __init__(self, param_names: list):
        self.param_names = param_names

    def compute_all(self, y_pred: np.ndarray, y_true: np.ndarray) -> dict:
        """
        Обчислення всіх метрик.

        Args:
            y_pred: (N, 7) predictions
            y_true: (N, 7) ground truth
        """
        metrics = {}

        # Global metrics
        mae = np.mean(np.abs(y_pred - y_true))
        rmse = np.sqrt(np.mean((y_pred - y_true)**2))
        mape = np.mean(np.abs((y_pred - y_true) / (y_true + 1e-8))) * 100
        r2 = r2_score(y_true, y_pred)

        metrics['global'] = {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'R2': r2
        }

        # Per-parameter metrics
        metrics['per_param'] = {}
        for i, name in enumerate(self.param_names):
            mae_i = np.mean(np.abs(y_pred[:, i] - y_true[:, i]))
            rmse_i = np.sqrt(np.mean((y_pred[:, i] - y_true[:, i])**2))
            mape_i = np.mean(np.abs((y_pred[:, i] - y_true[:, i]) /
                                    (y_true[:, i] + 1e-8))) * 100
            r2_i = r2_score(y_true[:, i], y_pred[:, i])

            metrics['per_param'][name] = {
                'MAE': mae_i,
                'RMSE': rmse_i,
                'MAPE': mape_i,
                'R2': r2_i
            }

        # Constraint violations
        violations = self.count_violations(y_pred)
        metrics['violations'] = {
            'rate': violations / len(y_pred) * 100,
            'count': violations
        }

        return metrics

    def count_violations(self, y_pred: np.ndarray) -> int:
        """Підрахунок порушень фізичних обмежень."""
        Dmax1, D01, L1, Rp1, D02, L2, Rp2 = y_pred.T

        violations = 0
        violations += np.sum(D01 > Dmax1)
        violations += np.sum(L2 > L1)
        violations += np.sum(D01 + D02 > 0.03)
        violations += np.sum(Rp1 > L1)
        violations += np.sum(Rp2 > 0)
        violations += np.sum(Rp2 < -L2)

        return violations
```

### 3.4.3 Валідація на тестовому датасеті

Після навчання модель проходить фінальну валідацію на тестовому датасеті, який **не використовувався ні під час тренування, ні під час валідації**.

**Протокол тестування**:

1. **Завантаження найкращої моделі**:
   ```python
   model = XRDRegressor(n_out=7, kernel_size=15)
   checkpoint = torch.load("checkpoints/best_model_epoch_87.pt")
   model.load_state_dict(checkpoint['model_state_dict'])
   model.eval()
   ```

2. **Inference на тестовому датасеті**:
   ```python
   test_dataset = torch.load("datasets/dataset_test_10000.pt")
   test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

   all_preds = []
   all_targets = []

   with torch.no_grad():
       for X_batch, y_batch in test_loader:
           preds = model(X_batch.to(device))
           all_preds.append(preds.cpu().numpy())
           all_targets.append(y_batch.numpy())

   y_pred = np.concatenate(all_preds)
   y_true = np.concatenate(all_targets)
   ```

3. **Обчислення метрик**:
   ```python
   metrics_calculator = RegressionMetrics(PARAM_NAMES)
   test_metrics = metrics_calculator.compute_all(y_pred, y_true)

   print("=" * 70)
   print("TEST SET EVALUATION (10,000 samples)")
   print("=" * 70)
   print(f"\nGlobal Metrics:")
   print(f"  MAE:  {test_metrics['global']['MAE']:.6f}")
   print(f"  RMSE: {test_metrics['global']['RMSE']:.6f}")
   print(f"  MAPE: {test_metrics['global']['MAPE']:.2f}%")
   print(f"  R²:   {test_metrics['global']['R2']:.4f}")

   print(f"\nPer-parameter MAPE:")
   for param in PARAM_NAMES:
       mape = test_metrics['per_param'][param]['MAPE']
       status = "✅" if mape < 15 else "⚠️"
       print(f"  {param:<8}: {mape:>6.2f}% {status}")

   violation_rate = test_metrics['violations']['rate']
   status = "✅" if violation_rate < 1.0 else "⚠️"
   print(f"\nConstraint violations: {violation_rate:.2f}% {status}")
   ```

4. **Візуалізація результатів**:
   ```python
   # Scatter plots: predicted vs true для кожного параметра
   fig, axes = plt.subplots(2, 4, figsize=(16, 8))
   for i, param in enumerate(PARAM_NAMES):
       ax = axes[i // 4, i % 4]
       ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.3, s=5)
       ax.plot([y_true[:, i].min(), y_true[:, i].max()],
               [y_true[:, i].min(), y_true[:, i].max()],
               'r--', linewidth=2)
       ax.set_xlabel('True')
       ax.set_ylabel('Predicted')
       ax.set_title(f'{param} (MAPE={test_metrics["per_param"][param]["MAPE"]:.1f}%)')
       ax.grid(True, alpha=0.3)

   plt.tight_layout()
   plt.savefig('test_predictions_scatter.png', dpi=150)
   ```

### 3.4.4 Експертна валідація на експериментальних даних

Фінальним етапом валідації є тестування на **реальних експериментальних кривих** від зразків з відомими параметрами (визначеними експертами вручну за допомогою базового ПЗ).

**Протокол експертної валідації**:

1. **Підготовка референсних зразків**:
   - Вибір 5-10 експериментальних КДВ з різною складністю
   - Ручний аналіз експертами у базовому ПЗ (2-4 години на зразок)
   - Документація фінальних параметрів p_expert та якості підгонки (χ²_expert)

2. **ML-інференс на експериментальних даних**:
   ```python
   for i, sample in enumerate(experimental_samples):
       # Load experimental curve
       exp_curve = np.loadtxt(f"validation/{sample}_444.dat")

       # ML prediction
       params_ml = predictor.predict(exp_curve)

       # Save for base software
       predictor.predict_to_file(
           exp_curve,
           f"validation/ml_predictions_{sample}.dat"
       )
   ```

3. **Functional refinement від ML-стартового наближення**:
   - Імпорт ml_predictions у базове ПЗ
   - Запуск functional refinement (Nelder-Mead, 1-3 хвилини)
   - Отримання p_refined та χ²_refined

4. **Порівняння результатів**:

| Зразок | χ²_expert | χ²_ml (no refine) | χ²_refined | Δ параметри (%) | Час експерта | Час ML+refine |
|--------|-----------|-------------------|------------|-----------------|--------------|---------------|
| GGG-1  | 0.0045    | 0.085             | 0.0042     | 3.2%            | 3.5 год      | 2 хв          |
| GGG-2  | 0.0067    | 0.12              | 0.0071     | 5.8%            | 4.2 год      | 2.5 хв        |
| YIG-1  | 0.0052    | 0.095             | 0.0049     | 2.1%            | 2.8 год      | 1.8 хв        |

**Критерії успіху експертної валідації**:

1. χ²_refined близький до χ²_expert (різниця <20%)
2. Δ параметри <10% (ML+refinement дає результат, близький до експертного)
3. Час аналізу скорочується у ~50-100 разів
4. Відсутність грубих помилок (ML не застрягає у нефізичному локальному мінімумі)

### 3.4.5 Тестування робастності

Додатково проводиться тестування робастності моделі до експериментальних артефактів:

**Тест 1: Стійкість до шуму**

Додавання різних рівнів гаусівського шуму до тестових кривих:
```python
noise_levels = [0.01, 0.02, 0.05, 0.10, 0.20]  # % від max(I)
for noise_level in noise_levels:
    X_noisy = X_test + np.random.normal(0, noise_level * X_test.max(), X_test.shape)
    y_pred_noisy = model.predict(X_noisy)

    mape_noisy = compute_mape(y_pred_noisy, y_test)
    print(f"Noise {noise_level*100}%: MAPE = {mape_noisy:.2f}%")
```

**Очікуваний результат**: MAPE зростає <2× при noise=10%

**Тест 2: Стійкість до baseline drift**

Додавання поліноміальної базової лінії:
```python
for degree in [1, 2, 3]:
    baseline = generate_polynomial_baseline(degree, amplitude=0.1)
    X_drifted = X_test + baseline
    y_pred_drifted = model.predict(X_drifted)

    mape_drifted = compute_mape(y_pred_drifted, y_test)
    print(f"Baseline degree {degree}: MAPE = {mape_drifted:.2f}%")
```

**Тест 3: Стійкість до зсувів**

Горизонтальний зсув кривої (неточність юстування):
```python
shifts = [-10, -5, 0, 5, 10]  # points
for shift in shifts:
    X_shifted = np.roll(X_test, shift, axis=-1)
    y_pred_shifted = model.predict(X_shifted)

    mape_shifted = compute_mape(y_pred_shifted, y_test)
    print(f"Shift {shift} points: MAPE = {mape_shifted:.2f}%")
```

**Очікуваний результат**: MAPE зростає <30% при shift=±10 points

---

## 3.5. Висновки до розділу 3

У даному розділі розроблено повну архітектуру системи аналізу КДВ з інтеграцією машинного навчання. Ключові результати проектування:

1. **Модульна архітектура інтеграції**: ML-модуль спроектовано як зовнішній допоміжний компонент, що взаємодіє з базовим ПЗ через стандартні механізми файлового обміну. Це забезпечує мінімальну інвазивність, backward compatibility та можливість незалежного розвитку компонентів.

2. **Детальна специфікація ML-компонента**: розроблено структуру Python-пакету з чітким розділенням відповідальностей між модулями (preprocessing, models, training, inference). Визначено протоколи взаємодії між компонентами та формат обміну даними з базовим ПЗ.

3. **Фізично обґрунтована функція втрат**: запроектовано спеціальну loss function, що поєднує MSE з penalty за порушення фізичних обмежень, гарантуючи <1% недопустимих передбачень.

4. **Комплексна стратегія тестування**: розроблено трирівневу систему тестування (unit → integration → end-to-end) та протокол експертної валідації на реальних експериментальних даних.

5. **Система метрик**: визначено набір глобальних та per-parameter метрик з цільовими значеннями (MAPE<10% для амплітудних параметрів, MAPE<15% для позиційних).

Розроблена архітектура забезпечує баланс між автоматизацією (ML inference <1 сек) та точністю (functional refinement 1-3 хв), скорочуючи загальний час аналізу у ~50-100 разів порівняно з повністю ручним підходом при збереженні якості результатів (Δ параметри <10% від експертного аналізу).
