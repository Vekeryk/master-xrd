"""
HRXRD Rocking Curve Simulation - NUMBA JIT OPTIMIZED VERSION
===============================================================
Based on: Рефакторинг C++ коду Difuz.cpp
Optimization: Numba JIT compilation for critical loops

PERFORMANCE IMPROVEMENTS:
- JIT-compiled sublayer loop in RozrachKogerTT
- Benchmarking utilities for performance measurement
- Expected speedup: 2-5x on RozrachKogerTT

ORIGINAL FEATURES:
1. Додано поляризацію (Sigma + Pi)
2. Повні параметри кристалу (Kapa, g, L_ext)
3. Правильна багатошаровість (цикл по km підшарам)
4. Врахування дефектів у профілі
5. Геометрія (симетрична/асиметрична)
"""
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
import matplotlib.pyplot as plt
import time
from numba import njit
from functools import wraps


@dataclass
class Curve:
    ML_X: np.ndarray  # angular deviation
    ML_Y: np.ndarray  # intensity I(ω)

    X_DeltaTeta: np.ndarray  # angular deviation
    Y_R_vseZ: np.ndarray  # intensity I(ω)
    Y_R_vse: np.ndarray  # intensity I(ω)


@dataclass
class Profile:
    X: np.ndarray          # depth grid z (Å)
    total_Y: np.ndarray    # ε_total(z)
    asymmetric_Y: np.ndarray   # ε_asym(z)
    decaying_Y: np.ndarray     # ε_decay(z)


@dataclass
class CrystalParameters:
    """Параметри кристалу (GGG або інший)"""
    a: float                    # Стала ґратки (см)
    h: int
    k: int
    l: int     # Індекси Міллера
    Lambda: float               # Довжина хвилі (см)

    # Поляризованість (ChiR, ChiI)
    ChiR0: float                # Re(χ₀)
    ChiI0: float                # Im(χ₀)
    ModChiI0: float             # |Im(χ₀)|

    ReChiRH: float              # Re(χᵣₕ)
    ImChiRH: float              # Im(χᵣₕ)
    ModChiRH: float             # |χᵣₕ|

    # Поляризованість для sigma та pi
    ModChiIH: np.ndarray        # [1]=sigma, [2]=pi

    # Коефіцієнт Пуассона
    Nu: float = 0.29            # Для GGG


@dataclass
class FilmParameters:
    """Параметри плівки (YIG або інша)"""
    apl: float                  # Стала ґратки плівки
    hpl: float                  # Товщина плівки (см)

    ChiI0pl: float
    ModChiI0pl: float
    ReChiRHpl: float
    ModChiRHpl: float
    ModChiIHpl: np.ndarray      # [1]=sigma, [2]=pi


@dataclass
class DeformationProfile:
    """Параметри профілю деформації"""
    Dmax1: float                # Максимальна деформація (асим. гаусіана)
    D01: float                  # Деформація на поверхні
    L1: float                   # Товщина порушеного шару (см)
    Rp1: float                  # Позиція максимуму (см)

    D02: float                  # Деформація (спадна гаусіана)
    L2: float                   # Товщина
    Rp2: float                  # Позиція максимуму

    Dmin: float = 0.0001        # Мінімальна деформація
    dl: float = 100e-8          # Крок підшару (см)


@dataclass
class GeometryParameters:
    """Геометричні параметри дифрактометра"""
    psi: float                  # Кут між нормаллю та вектором розсіяння
    asymmetric: bool = False    # Симетричний/асиметричний рефлекс


# =============================================================================
# BENCHMARKING UTILITIES
# =============================================================================

def benchmark(func):
    """Decorator to measure function execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"⏱️  {func.__name__}: {elapsed:.4f}s")
        return result
    return wrapper


# =============================================================================
# NUMBA JIT-COMPILED FUNCTIONS FOR CRITICAL LOOPS
# =============================================================================

@njit
def _compute_sublayer_loop_jit(
    km, DeltaTeta_i, b_as, tb, Lambda, gamma0,
    eta0_a, xhp_a_n, xhn_a_n, C_n, gammah,
    DDpd, psi, asymmetric,
    Esum, Dl, As_in
):
    """
    JIT-compiled innermost loop for sublayer calculations.
    This is the hottest path in RozrachKogerTT.

    Args:
        km: number of sublayers
        DeltaTeta_i: angular deviation at point i
        ... (many physics parameters)
        As_in: initial As value

    Returns:
        As: final amplitude after all sublayers
    """
    cmplxi = 1j
    As = As_in

    for k in range(1, km + 1):
        # eta_a for sublayer k
        eta_a = -(eta0_a[k] * cmplxi +
                  2 * np.pi * b_as * np.sin(2 * tb) * DeltaTeta_i /
                  (Lambda * gamma0))

        # sigmasp_a, sigmasn_a for sublayer k
        sigmasp_a = (np.pi * xhp_a_n[k] * C_n /
                     (Lambda * np.sqrt(gamma0 * gammah)))
        sigmasn_a = (np.pi * xhn_a_n[k] * C_n /
                     (Lambda * np.sqrt(gamma0 * gammah)))

        # YYs[k] for each sublayer
        if not asymmetric:
            YYs_k = (np.pi / Lambda / gamma0 *
                     DDpd[k] * b_as *
                     (np.cos(psi)**2 * np.tan(tb) +
                      np.sin(psi) * np.cos(psi)) *
                     2 * np.sin(2 * tb))
        else:
            YYs_k = (np.pi / Lambda / gamma0 *
                     DDpd[k] * b_as *
                     (np.cos(psi)**2 * np.tan(tb) -
                      np.sin(psi) * np.cos(psi)) *
                     2 * np.sin(2 * tb))

        YYs_k = eta_a + YYs_k

        # Square root with complex number handling
        sqs = np.sqrt((YYs_k / 2)**2 - sigmasp_a * sigmasn_a * Esum[k]**2)
        if sqs.imag <= 0:
            sqs = -sqs
        if eta0_a[k] <= 0:
            sqs = -sqs

        ssigma = sqs / cmplxi
        x2s = -(YYs_k / 2 + sqs) / (sigmasn_a * Esum[k])
        x1s = -(YYs_k / 2 - sqs) / (sigmasn_a * Esum[k])
        x3s = (x1s - As) / (x2s - As)
        expcs = np.exp(-2 * ssigma * Dl[k])

        # Sequential update of As
        As = (x1s - x2s * x3s * expcs) / (1 - x3s * expcs)

    return As


class HRXRDSimulator:

    # =============================================================================
    # __init__ - __init__ - __init__ - __init__ - __init__ - __init__
    # =============================================================================

    def __init__(
        self,
        crystal: CrystalParameters,
        film: Optional[FilmParameters] = None,
        geometry: GeometryParameters = None
    ):
        self.crystal = crystal
        self.film = film
        self.geometry = geometry or GeometryParameters(psi=0.0)

        # Основні розрахункові параметри (з QuickStart)
        self.tb = None              # Брегговський кут
        self.gamma0 = None          # Косинус входу
        self.gammah = None          # Косинус виходу
        self.b_as = None            # Асиметрія = gamma0/|gammah|

        # Поляризація
        self.C = np.zeros(3)        # [1]=sigma=1, [2]=pi=|cos(2θ)|
        self.nC1 = 1                # Початкова поляризація
        self.nC = 2                 # Кінцева поляризація

        # Параметри для обох поляризацій
        self.Kapa = np.zeros(3)     # κ = ModChiIH/ModChiRH
        self.p = np.zeros(3)        # Поляризаційний фактор
        self.g = np.zeros(3)        # Параметр екстинкції

        # Поглинання
        self.Mu0 = None             # Коефіцієнт поглинання
        self.Mu0_pl = None          # Для плівки

        # Інші параметри
        self.H = None               # 1/d (обернена міжплощинна відстань)
        self.H2Pi = None            # 2π/d
        self.VelKom = None          # Об'єм елементарної комірки

        # Monochromator polarization factor
        self.Monohr = np.ones(3)    # [1] для GGG, [2] для Si/Ge

        # Результати обчислень
        self.DeltaTeta = None       # Кутова сітка
        self.R_cogerTT = None       # Когерентна складова
        self.R_vseZ = None          # Після згортки

        # Профіль деформації
        self.km = 0                 # Кількість підшарів
        self.DD = None              # Профіль деформації DD[k]
        self.DDPL1 = None           # Асиметрична гаусіана
        self.DDPL2 = None           # Спадна гаусіана
        self.Dl = None              # Товщини підшарів
        self.f = None               # Нормалізований профіль
        self.Esum = None            # Фактор екстинкції для кожного шару

        self.hpl0 = 0.0             # Товщина недеформованої частини плівки

        # Масиви для багатошаровості (КРИТИЧНО для правильності!)
        # C++ лінії 1514-1652, 5872-5880
        self.ChiI0_a = None         # χ₀ для кожного підшару [k]
        self.ModChiI0_a = None      # |Im(χ₀)| для кожного підшару [k]
        self.eta0_a = None          # η₀ для кожного підшару [k]
        # Комплексна поляризованість [n][k] для sigma/pi в підшарі k
        self.xhp_a = None
        self.xhn_a = None           # Комплексна поляризованість [n][k]
        self.ReChiRH_a = None       # Re(χᵣₕ) для кожного підшару [k]
        self.ModChiRH_a = None      # |χᵣₕ| для кожного підшару [k]
        self.ReChiIH_a = None       # Re(χᵢₕ) для [n][k]
        self.ModChiIH_a = None      # |Im(χᵢₕ)| для [n][k]

    # =============================================================================
    # Start - Start - Start - Start - Start - Start - Start - Start - Start
    # =============================================================================

    def Start(self):
        """
        Ініціалізація параметрів (аналог QuickStart з C++)
        Лінії 517-794 у Difuz.cpp
        """
        # Обчислення Брегговського кута
        self.tb = np.arcsin(
            self.crystal.Lambda *
            np.sqrt(self.crystal.h**2 + self.crystal.k**2 + self.crystal.l**2) /
            (2 * self.crystal.a)
        )

        # Геометричні фактори
        psi = self.geometry.psi
        self.gamma0 = np.sin(self.tb - psi)
        self.gammah = np.sin(self.tb + psi)
        self.b_as = self.gamma0 / np.abs(self.gammah)

        # Поляризація C[n]
        self.C[1] = 1.0                                    # Sigma
        self.C[2] = np.abs(np.cos(2 * self.tb))           # Pi

        # Об'єм комірки
        self.VelKom = self.crystal.a ** 3

        # H - обернена міжплощинна відстань
        self.H = np.sqrt(
            self.crystal.h**2 + self.crystal.k**2 + self.crystal.l**2
        ) / self.crystal.a
        self.H2Pi = 2 * np.pi * self.H

        # Поглинання
        K = 2 * np.pi / self.crystal.Lambda
        self.Mu0 = K * self.crystal.ModChiI0

        if self.film is not None:
            self.Mu0_pl = K * self.film.ModChiI0pl

        # Параметри для обох поляризацій
        for n in [1, 2]:
            self.Kapa[n] = self.crystal.ModChiIH[n] / self.crystal.ModChiRH
            self.p[n] = self.Kapa[n]

            # Параметр екстинкції g (лінія 612-615 C++)
            self.g[n] = -self.crystal.ModChiI0 * (
                np.sqrt(self.b_as) + 1 / np.sqrt(self.b_as)
            ) / (2 * self.C[n] * self.crystal.ModChiRH)

        # Monochromator polarization (лінії 524-534 C++)
        if self.crystal.h == 4 and self.crystal.k == 4 and self.crystal.l == 4:
            # GGG(444)
            SinTeta = self.crystal.Lambda * np.sqrt(48) / (2 * 12.383e-8)
            self.Monohr[1] = np.abs(np.sqrt(1 - 2 * SinTeta**2))

    # =============================================================================
    # Profil - Profil - Profil - Profil - Profil - Profil - Profil - Profil
    # =============================================================================

    def Profil(
        self,
        params: DeformationProfile
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Розрахунок профілю деформації (аналог Profil з C++)
        Лінії 799-1155 у Difuz.cpp

        ВАЖЛИВО: C++ версія враховує дефекти (дислокаційні петлі, кластери)
        Тут - спрощена версія з двома гаусіанами

        Returns:
            DD: повний профіль деформації
            DDPL1: асиметрична гаусіана
            DDPL2: спадна гаусіана
        """
        Dmax1 = params.Dmax1
        D01 = params.D01
        L1 = params.L1
        Rp1 = params.Rp1
        D02 = params.D02
        L2 = params.L2
        Rp2 = params.Rp2
        Dmin = params.Dmin
        dl = params.dl

        # Обчислення сігм для гаусіан (лінії 858-861 C++)
        if Dmax1 != Dmin:
            s1 = (L1 - Rp1)**2 / np.log(Dmax1 / Dmin)
        else:
            s1 = dl

        if Dmax1 != D01:
            s2 = Rp1**2 / np.log(Dmax1 / D01)
        else:
            s2 = 10000

        if D02 != Dmin:
            s3 = L2 * (L2 - 2 * Rp2) / np.log(D02 / Dmin)
        else:
            s3 = dl

        # СПРОЩЕНИЙ ПІДХІД (як у вашій старій версії) - БЕЗ РЕВЕРСУ!
        # Знаходимо km - загальну кількість підшарів
        ss = s2
        kk = 0
        DDPLk = Dmax1

        while DDPLk > Dmin:
            kk += 1
            z = dl * kk - dl / 2
            if z >= Rp1:
                ss = s1
            DDPLk = Dmax1 * np.exp(-(z - Rp1)**2 / ss)
            DDPLk += D02 * np.exp(Rp2**2 / s3) * np.exp(-(z - Rp2)**2 / s3)

        km = kk - 1
        L = km * dl

        # Обчислюємо профілі напряму з координатою від поверхні до глибини
        # k=1: z ≈ L - dl/2 (біля поверхні)
        # k=km: z ≈ dl/2 (глибина)
        DD = np.zeros(km + 1)
        DDPL1 = np.zeros(km + 1)
        DDPL2 = np.zeros(km + 1)
        ss = s1  # Починаємо з s1 для великих z

        for k in range(1, km + 1):
            z = L - dl * k + dl / 2
            # Умова для зміни ss: якщо z < Rp1, використовуємо s2 (ліва частина гаусіана)
            if z < Rp1:
                ss = s2
            DDPL1[k] = Dmax1 * np.exp(-(z - Rp1)**2 / ss)
            DDPL2[k] = D02 * np.exp(Rp2**2 / s3) * np.exp(-(z - Rp2)**2 / s3)
            DD[k] = DDPL1[k] + DDPL2[k]
        # DD[0] = 0 (не використовується)

        # Збереження (тепер DDPL1, DDPL2 вже фінальні)
        self.km = km
        self.DD = DD
        self.DDPL1 = DDPL1
        self.DDPL2 = DDPL2

        # Обчислення недеформованої частини плівки (L вже обчислено вище)
        if self.film is not None:
            self.hpl0 = self.film.hpl - L

        # Товщини підшарів (лінія 920 C++)
        # for (int k=1; k<=km; k++) Dl[k]=dl;
        self.Dl = np.zeros(km + 1)
        for k in range(1, km + 1):
            self.Dl[k] = dl

        # Нормалізований профіль f[k] (лінії 921-923 C++)
        # for (int k=1; k<=km; k++) if (Dmax<fabs(DDprof[k])) Dmax=fabs(DDprof[k]);
        Dmax = 0.0
        for k in range(1, km + 1):
            if Dmax < np.abs(DD[k]):
                Dmax = np.abs(DD[k])

        # for (int k=1; k<=km; k++) f[k]=fabs(DDprof[k]/Dmax);
        self.f = np.zeros(km + 1)
        for k in range(1, km + 1):
            self.f[k] = np.abs(DD[k] / Dmax)

        # Фактор екстинкції (спрощено - в C++ це складніше)
        # В C++ (лінії 926-997): враховуються дефекти, але зараз Esum=1
        self.Esum = np.ones(km + 1)

        return DD, DDPL1, DDPL2

    # =============================================================================
    # PolarizationInit - PolarizationInit - PolarizationInit - PolarizationInit
    # =============================================================================

    def PolarizationInit(self):
        """
        Ініціалізація параметрів поляризованості для кожного підшару
        Аналог C++ коду ліній 1514-1652, 5872-5880

        КРИТИЧНО! Кожен підшар k має свої параметри Chi залежно від:
        - Типу матеріалу (субстрат чи плівка)
        - Деформації DD[k] (опціонально)
        """
        if self.km == 0:
            return

        # Виділення масивів [0...km], але використовуються тільки [1...km]
        self.ChiI0_a = np.zeros(self.km + 1)
        self.ModChiI0_a = np.zeros(self.km + 1)
        self.ReChiRH_a = np.zeros(self.km + 1)
        self.ModChiRH_a = np.zeros(self.km + 1)
        # [n][k]: [0], [1]=sigma, [2]=pi
        self.ReChiIH_a = np.zeros((3, self.km + 1))
        self.ModChiIH_a = np.zeros((3, self.km + 1))
        self.eta0_a = np.zeros(self.km + 1)

        # Комплексні масиви [n][k]
        self.xhp_a = np.zeros((3, self.km + 1), dtype=complex)
        self.xhn_a = np.zeros((3, self.km + 1), dtype=complex)

        # Якщо є плівка - деформовані шари належать плівці (C++ лінія 1527-1641)
        # Якщо немає плівки - деформовані шари в субстраті (C++ лінія 1514-1523)
        if self.film is not None:
            # ВАРІАНТ 1 (спрощений): Константні параметри (лінії 1527-1533)
            # TODO: Додати залежність від DD[k] для повної точності (лінії 1596-1641)
            for k in range(1, self.km + 1):
                self.ChiI0_a[k] = self.film.ChiI0pl
                self.ModChiI0_a[k] = self.film.ModChiI0pl
                self.ReChiRH_a[k] = self.film.ReChiRHpl
                self.ModChiRH_a[k] = self.film.ModChiRHpl
                self.ReChiIH_a[1][k] = self.film.ModChiIHpl[1]  # Sigma
                self.ReChiIH_a[2][k] = self.film.ModChiIHpl[2]  # Pi
                self.ModChiIH_a[1][k] = self.film.ModChiIHpl[1]
                self.ModChiIH_a[2][k] = self.film.ModChiIHpl[2]
        else:
            # Деформовані шари в субстраті (лінії 1514-1523)
            for k in range(1, self.km + 1):
                self.ChiI0_a[k] = self.crystal.ChiI0
                self.ModChiI0_a[k] = self.crystal.ModChiI0
                self.ReChiRH_a[k] = self.crystal.ReChiRH
                self.ModChiRH_a[k] = self.crystal.ModChiRH
                self.ReChiIH_a[1][k] = self.crystal.ModChiIH[1]
                self.ReChiIH_a[2][k] = self.crystal.ModChiIH[2]
                self.ModChiIH_a[1][k] = self.crystal.ModChiIH[1]
                self.ModChiIH_a[2][k] = self.crystal.ModChiIH[2]

        # Обчислення комплексних поляризованостей та eta0_a (C++ лінії 5872-5880)
        cmplxi = 1j
        K = 2 * np.pi / self.crystal.Lambda

        for k in range(1, self.km + 1):
            # Комплексні поляризованості для sigma та pi
            self.xhp_a[1][k] = self.ReChiRH_a[k] + \
                cmplxi * self.ReChiIH_a[1][k]
            self.xhn_a[1][k] = self.ReChiRH_a[k] + \
                cmplxi * self.ReChiIH_a[1][k]
            self.xhp_a[2][k] = self.ReChiRH_a[k] + \
                cmplxi * self.ReChiIH_a[2][k]
            self.xhn_a[2][k] = self.ReChiRH_a[k] + \
                cmplxi * self.ReChiIH_a[2][k]

            # Параметр поглинання для підшару k
            self.eta0_a[k] = np.pi * self.ChiI0_a[k] * \
                (1 + self.b_as) / (self.crystal.Lambda * self.gamma0)

    # =============================================================================
    # RozrachKogerTT - RozrachKogerTT - RozrachKogerTT - RozrachKogerTT
    # ============================================================================

    def RozrachKogerTT(
        self,
        m1: int,
        m10: int,
        ik: float
    ) -> np.ndarray:
        """
        Розрахунок когерентної складової (аналог RozrachKogerTT з C++)
        Лінії 5834-6003 у Difuz.cpp

        ЦЕ НАЙВАЖЛИВІША ФУНКЦІЯ - тут були критичні помилки!

        Args:
            m1: кількість точок сканування
            m10: зсув нуля
            ik: крок сканування (arcsec)

        Returns:
            R_cogerTT: інтенсивність когерентного розсіяння
        """
        # Створення кутової сітки (лінія 727 C++)
        TetaMin = -m10 * ik
        Hteta = np.pi / (3600 * 180)  # Переведення arcsec в радіани
        self.DeltaTeta = np.array(
            [(TetaMin + i * ik) * Hteta for i in range(m1 + 1)])

        # Комплексні поляризованості (лінії 5860-5869 C++)
        cmplxi = 1j

        # Для монокристалу (substrate)
        xhp0 = np.zeros(3, dtype=complex)
        xhn0 = np.zeros(3, dtype=complex)
        xhp0[1] = self.crystal.ReChiRH + cmplxi * self.crystal.ModChiIH[1]
        xhn0[1] = self.crystal.ReChiRH + cmplxi * self.crystal.ModChiIH[1]
        xhp0[2] = self.crystal.ReChiRH + cmplxi * self.crystal.ModChiIH[2]
        xhn0[2] = self.crystal.ReChiRH + cmplxi * self.crystal.ModChiIH[2]

        # Для плівки
        if self.film is not None:
            xhp = np.zeros(3, dtype=complex)
            xhn = np.zeros(3, dtype=complex)
            xhp[1] = self.film.ReChiRHpl + cmplxi * self.film.ModChiIHpl[1]
            xhn[1] = self.film.ReChiRHpl + cmplxi * self.film.ModChiIHpl[1]
            xhp[2] = self.film.ReChiRHpl + cmplxi * self.film.ModChiIHpl[2]
            xhn[2] = self.film.ReChiRHpl + cmplxi * self.film.ModChiIHpl[2]

        # Параметри поглинання (лінії 5902-5903 C++)
        x0i0 = self.crystal.ChiI0
        eta00 = np.pi * x0i0 * (1 + self.b_as) / \
            (self.crystal.Lambda * self.gamma0)

        if self.film is not None:
            x0i = self.film.ChiI0pl
            eta0 = np.pi * x0i * (1 + self.b_as) / \
                (self.crystal.Lambda * self.gamma0)

            # Розрахунок DD0 (деформація між плівкою та субстратом)
            # Лінії 701-707, 5891 C++
            DD0 = (self.film.apl - self.crystal.a) / self.crystal.a

            # DDpd[k] - деформація в кожному підшарі (лінія 5892 C++)
            # ВАЖЛИВО! Явний цикл, НЕ векторизація, щоб не зачіпати DDpd[0]
            # C++: for (int k=1; k<=km;k++) DDpd[k]=(DD[k]+1)*(DD0+1)-1;
            DDpd = np.zeros(self.km + 1)
            for k in range(1, self.km + 1):
                DDpd[k] = (self.DD[k] + 1) * (DD0 + 1) - 1
            # DDpd[0] = 0 (не використовується)

        # Масив результатів
        R_cogerTT = np.zeros(m1 + 1)

        # ГОЛОВНИЙ ЦИКЛ по кутах (лінії 5908-6000 C++)
        for i in range(m1 + 1):
            # eta0pd для субстрату (лінія 5910 C++)
            eta0pd = -(
                eta00 * cmplxi +
                2 * np.pi * self.b_as * np.sin(2 * self.tb) * self.DeltaTeta[i] /
                (self.crystal.Lambda * self.gamma0)
            )

            if self.film is not None:
                # eta для плівки (лінія 5911 C++)
                eta = -(
                    eta0 * cmplxi +
                    2 * np.pi * self.b_as * np.sin(2 * self.tb) * self.DeltaTeta[i] /
                    (self.crystal.Lambda * self.gamma0)
                )

            # Масив для обох поляризацій
            R = np.zeros(3)

            # КРИТИЧНИЙ ЦИКЛ ПО ПОЛЯРИЗАЦІЯХ (лінія 5913 C++)
            for n in range(self.nC1, self.nC + 1):
                # Sigma для субстрату (лінії 5915-5916 C++)
                sigmasp0 = (np.pi * xhp0[n] * self.C[n] /
                            (self.crystal.Lambda * np.sqrt(self.gamma0 * self.gammah)))
                sigmasn0 = (np.pi * xhn0[n] * self.C[n] /
                            (self.crystal.Lambda * np.sqrt(self.gamma0 * self.gammah)))

                # Розв'язок для субстрату (лінії 5921-5924 C++)
                sqs = np.sqrt(eta0pd**2 - 4 * sigmasp0 * sigmasn0)
                if sqs.imag <= 0:
                    sqs = -sqs
                if eta00 <= 0:
                    sqs = -sqs

                As = -(eta0pd + sqs) / (2 * sigmasn0)

                # Якщо є плівка (bicrystal) (лінії 5928-5951 C++)
                if self.film is not None:
                    sigmasp = (np.pi * xhp[n] * self.C[n] /
                               (self.crystal.Lambda * np.sqrt(self.gamma0 * self.gammah)))
                    sigmasn = (np.pi * xhn[n] * self.C[n] /
                               (self.crystal.Lambda * np.sqrt(self.gamma0 * self.gammah)))

                    # YYs0 для недеформованої частини плівки (лінії 5931-5939 C++)
                    if not self.geometry.asymmetric:
                        YYs0 = (np.pi / self.crystal.Lambda / self.gamma0 * DD0 * self.b_as *
                                (np.cos(self.geometry.psi)**2 * np.tan(self.tb) +
                                 np.sin(self.geometry.psi) * np.cos(self.geometry.psi)) *
                                2 * np.sin(2 * self.tb))
                    else:
                        YYs0 = (np.pi / self.crystal.Lambda / self.gamma0 * DD0 * self.b_as *
                                (np.cos(self.geometry.psi)**2 * np.tan(self.tb) -
                                 np.sin(self.geometry.psi) * np.cos(self.geometry.psi)) *
                                2 * np.sin(2 * self.tb))

                    YYs0 = eta + YYs0

                    sqs = np.sqrt((YYs0 / 2)**2 - sigmasp * sigmasn)
                    if sqs.imag <= 0:
                        sqs = -sqs
                    if eta0 <= 0:
                        sqs = -sqs

                    ssigma = sqs / cmplxi
                    x2s = -(YYs0 / 2 + sqs) / sigmasn
                    x1s = -(YYs0 / 2 - sqs) / sigmasn
                    x3s = (x1s - As) / (x2s - As)
                    expcs = np.exp(-2 * ssigma * self.hpl0)

                    As = (x1s - x2s * x3s * expcs) / (1 - x3s * expcs)

                    # КРИТИЧНИЙ ЦИКЛ ПО ПІДШАРАМ - NUMBA JIT OPTIMIZED
                    # Замість Python loop використовуємо JIT-скомпільовану функцію
                    if self.km > 0:
                        As = _compute_sublayer_loop_jit(
                            self.km,
                            self.DeltaTeta[i],
                            self.b_as,
                            self.tb,
                            self.crystal.Lambda,
                            self.gamma0,
                            self.eta0_a,
                            # [k] indexing happens in JIT function
                            self.xhp_a[n],
                            self.xhn_a[n],
                            self.C[n],
                            self.gammah,
                            DDpd,
                            self.geometry.psi,
                            self.geometry.asymmetric,
                            self.Esum,
                            self.Dl,
                            As  # As_in
                        )

                # Інтенсивність для даної поляризації (лінія 5990 C++)
                R[n] = np.abs(xhp0[n] / xhn0[n]) * np.abs(As)**2

            # Фінальне усереднення по поляризаціях (лінії 5995-5998 C++)
            if self.nC == 1:
                # Тільки sigma
                R_cogerTT[i] = R[1]
            else:
                # Sigma + Pi з monochromator factor
                R_cogerTT[i] = (R[1] + self.Monohr[1] * R[2]) / \
                    (1 + self.Monohr[1])

        self.R_cogerTT = R_cogerTT
        return R_cogerTT

    # =============================================================================
    # ZGORTKA - ZGORTKA - ZGORTKA - ZGORTKA - ZGORTKA - ZGORTKA - ZGORTKA
    # =============================================================================

    def Zgortka(
        self,
        m1: int,
        m10: int,
        ik: float,
        width: float = 25.0,
        amplitude: float = 10000.0
    ) -> np.ndarray:
        """
        Згортка з апаратною функцією (гаусіана)
        Аналог функції Zgortka() з C++

        Args:
            width: ширина апаратної функції (arcsec)
            amplitude: амплітуда

        Returns:
            R_vseZ: інтенсивність після згортки
        """
        w = width
        A = amplitude
        Ymin = 0.0001
        y0 = 0.0
        xc = 0.0

        # Обчислення мінімальної області (аналог C++)
        log_part = np.log(
            (Ymin - y0) / ((A / w) * np.sqrt(2 * np.log(4) / np.pi)))
        Xmin = xc + np.sqrt(-w**2 / (2 * np.log(4)) * log_part)
        m10z = int(np.abs(Xmin) / ik)
        MZ = m10z * 2

        # Апаратна функція (гаусіана)
        PO = np.zeros(MZ + 1)
        TetaMinz = -m10z * ik

        for i in range(MZ + 1):
            x = TetaMinz + i * ik
            PO[i] = (y0 + (A / w * np.sqrt(2 * np.log(4) / np.pi) *
                     np.exp(-2 * (x - xc)**2 * np.log(4) / w**2)))

        # Нормалізація
        PO /= np.sum(PO)

        # Вирівнювання
        POO = np.zeros(MZ + 2 * m10 + 1)
        for i in range(MZ, -1, -1):
            izg = i + (m10 - m10z)
            if izg < len(POO):
                POO[izg] = PO[i]

        # Згортка
        R_vseZ = np.zeros(m1 + 1)
        for j in range(m1 + 1):
            for i in range(MZ + (m10 - m10z) + 1):
                idx = j - i + m10
                if 0 <= idx < len(self.R_cogerTT):
                    R_vseZ[j] += self.R_cogerTT[idx] * POO[i]

        self.R_vseZ = R_vseZ
        return R_vseZ

    # =============================================================================
    # ЗАПУСТИТИ СИМУЛЯЦІЮ КЛАСУ
    # =============================================================================

    def RunSimulation(
        self,
        deformation_params: DeformationProfile,
        m1: int = 700,
        m10: int = 20,
        ik: float = 4.671897861
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Повна симуляція кривої дифракційного відбивання

        Returns:
            DeltaTeta: кутова сітка (arcsec)
            R_cogerTT: когерентна інтенсивність
            R_vseZ: інтенсивність після згортки
        """
        # 1. Ініціалізація параметрів
        self.Start()

        # 2. Розрахунок профілю деформації
        self.Profil(deformation_params)

        # 3. КРИТИЧНО! Ініціалізація параметрів підшарів (ПІСЛЯ розрахунку профілю!)
        # Це обчислює ChiI0_a[k], eta0_a[k], xhp_a[n][k], xhn_a[n][k]
        self.PolarizationInit()

        # 4. Розрахунок когерентної складової
        self.RozrachKogerTT(m1, m10, ik)

        # 5. Згортка з апаратною функцією
        self.Zgortka(m1, m10, ik)

        # Переведення DeltaTeta назад в arcsec для виводу
        DeltaTeta_arcsec = self.DeltaTeta / (np.pi / (3600 * 180))

        return DeltaTeta_arcsec, self.R_cogerTT, self.R_vseZ


# =============================================================================
# API ДЕНИСА - API ДЕНИСА - API ДЕНИСА - API ДЕНИСА - API ДЕНИСА
# =============================================================================


def create_GGG_crystal() -> CrystalParameters:
    """Створення параметрів кристалу GGG(444)"""
    return CrystalParameters(
        a=12.382e-8,                # см
        h=4, k=4, l=4,
        Lambda=1.5405e-8,           # CuKα
        ChiR0=-3.68946e-5,
        ChiI0=-3.595136e-6,
        ModChiI0=3.595136e-6,
        ReChiRH=10.94764e-6,
        ImChiRH=1e-12,
        ModChiRH=10.94764e-6,
        ModChiIH=np.array([0, 2.84908e-6, 1.79083e-6]),  # [0, sigma, pi]
        Nu=0.29
    )


def create_YIG_film(hpl: float = 3.15e-4) -> FilmParameters:
    """Створення параметрів плівки YIG"""
    return FilmParameters(
        apl=12.376e-8,              # см
        hpl=hpl,                    # см
        ChiI0pl=-2.1843e-6,
        ModChiI0pl=2.1843e-6,
        ReChiRHpl=8.8269e-6,
        ModChiRHpl=8.8269e-6,
        ModChiIHpl=np.array([0, 0.9043e-6, 0.9043e-6])  # [0, sigma, pi]
    )


def compute_curve_and_profile(array=None,
                              m1: int = 700,
                              m10: int = 20,
                              ik: float = 4.671897861,
                              start_ML: int = 50,
                              params_obj: DeformationProfile = None,
                              verbose=False):
    """
    Compute XRD curve and deformation profile.

    Args:
        verbose: If True, print timing information for each step
    """
    if array is None and params_obj is None:
        # throw error
        raise ValueError("Input array and params_obj are None")

    # Створення системи GGG + YIG
    crystal = create_GGG_crystal()
    film = create_YIG_film()
    geometry = GeometryParameters(psi=0.0, asymmetric=False)

    # Створення симулятора
    simulator = HRXRDSimulator(crystal, film, geometry)

    # Присвоєння параметрів деформації (об'єкт або з масиву/тензора)
    if params_obj is not None:
        deformation = params_obj
    else:
        deformation = DeformationProfile(
            Dmax1=array[0],
            D01=array[1],
            L1=array[2],
            Rp1=array[3],
            D02=array[4],
            L2=array[5],
            Rp2=array[6],
            Dmin=0.0001,
            dl=100e-8
        )

    # Фізичні точки кривої
    if verbose:
        print("🚀 Starting XRD simulation...")
        start_time = time.time()

    DeltaTeta, R_coger, R_convolved = simulator.RunSimulation(
        deformation,
        m1=m1,
        m10=m10,
        ik=ik
    )

    if verbose:
        elapsed = time.time() - start_time
        print(f"✅ Total simulation time: {elapsed:.4f}s")
        print(f"   Sublayers: {simulator.km}")

    # ML точки кривої
    m1_ML = m1 - start_ML
    curve_X_ML = np.linspace(0, m1_ML - 1, m1_ML)
    curve_Y_ML = np.asarray(R_convolved)[start_ML:m1]

    # L = simulator.km * deformation.dl
    profile_X = np.array([((simulator.km - k + 1) * deformation.dl - deformation.dl / 2) / 1e-8
                          for k in range(1, simulator.km + 1)])

    profile_total_Y = simulator.DD[1:]
    profile_asymmetric_Y = simulator.DDPL1[1:]
    profile_decaying_Y = simulator.DDPL2[1:]

    # Створення об'єктів Curve та Profile
    curve = Curve(ML_X=curve_X_ML.copy(),
                  ML_Y=curve_Y_ML.copy(),
                  # Фізичні точки кривої
                  X_DeltaTeta=DeltaTeta.copy(),
                  Y_R_vseZ=R_convolved.copy(),
                  Y_R_vse=R_coger.copy())

    profile = Profile(X=profile_X,
                      # Y профілів
                      total_Y=profile_total_Y,
                      asymmetric_Y=profile_asymmetric_Y,
                      decaying_Y=profile_decaying_Y)

    return curve, profile


# =============================================================================
# BENCHMARKING UTILITIES
# =============================================================================

def benchmark_comparison(n_samples=10, dl=100e-8):
    """
    Compare performance of xrd_parallel vs original xrd.

    Args:
        n_samples: Number of test samples to run
        dl: Sublayer thickness parameter

    Returns:
        dict with timing results
    """
    print("=" * 70)
    print(f"🔬 BENCHMARKING: xrd_parallel.py (Numba JIT optimized)")
    print(f"   Testing with {n_samples} samples, dl={dl*1e8:.0f}Å")
    print("=" * 70)

    # Generate random test parameters
    np.random.seed(42)
    test_params = []
    for _ in range(n_samples):
        params = DeformationProfile(
            Dmax1=np.random.uniform(0.01, 0.025),
            D01=np.random.uniform(0.002, 0.015),
            L1=np.random.uniform(2000e-8, 6000e-8),
            Rp1=np.random.uniform(1000e-8, 4000e-8),
            D02=np.random.uniform(0.003, 0.015),
            L2=np.random.uniform(2000e-8, 5000e-8),
            Rp2=np.random.uniform(-5000e-8, -500e-8),
            Dmin=0.0001,
            dl=dl
        )
        test_params.append(params)

    # Warmup run (for JIT compilation)
    print("\n🔥 Warmup run (JIT compilation)...")
    _ = compute_curve_and_profile(params_obj=test_params[0], verbose=False)
    print("   JIT compilation complete!")

    # Timed runs
    print(f"\n⏱️  Running {n_samples} samples...")
    start_time = time.time()

    for i, params in enumerate(test_params):
        compute_curve_and_profile(params_obj=params, verbose=False)
        if (i + 1) % 5 == 0 or (i + 1) == n_samples:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            print(
                f"   Progress: {i+1}/{n_samples} samples, {rate:.2f} samples/sec")

    total_time = time.time() - start_time
    avg_time = total_time / n_samples

    print("\n" + "=" * 70)
    print(f"📊 RESULTS:")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Average per sample: {avg_time:.4f}s")
    print(f"   Throughput: {n_samples/total_time:.2f} samples/sec")
    print("=" * 70)

    return {
        'n_samples': n_samples,
        'total_time': total_time,
        'avg_time': avg_time,
        'throughput': n_samples / total_time
    }


def benchmark_single_sample(dl=100e-8):
    """
    Detailed benchmark of a single sample with step-by-step timing.

    Args:
        dl: Sublayer thickness parameter
    """
    print("\n" + "=" * 70)
    print("🔬 DETAILED SINGLE SAMPLE BENCHMARK")
    print("=" * 70)

    params = DeformationProfile(
        Dmax1=0.01305,
        D01=0.0017,
        L1=5800e-8,
        Rp1=3500e-8,
        D02=0.004845,
        L2=4000e-8,
        Rp2=-500e-8,
        Dmin=0.0001,
        dl=dl
    )

    crystal = create_GGG_crystal()
    film = create_YIG_film()
    geometry = GeometryParameters(psi=0.0, asymmetric=False)
    simulator = HRXRDSimulator(crystal, film, geometry)

    # Step-by-step timing
    print("\n⏱️  Step-by-step timing:")

    t0 = time.time()
    simulator.Start()
    t1 = time.time()
    print(f"   1. Start (initialization): {(t1-t0)*1000:.2f}ms")

    simulator.Profil(params)
    t2 = time.time()
    print(f"   2. Profil (deformation profile): {(t2-t1)*1000:.2f}ms")
    print(f"      → km={simulator.km} sublayers")

    simulator.PolarizationInit()
    t3 = time.time()
    print(f"   3. PolarizationInit: {(t3-t2)*1000:.2f}ms")

    simulator.RozrachKogerTT(700, 20, 4.671897861)
    t4 = time.time()
    print(f"   4. RozrachKogerTT (🔥 HOT PATH): {(t4-t3)*1000:.2f}ms")

    simulator.Zgortka(700, 20, 4.671897861)
    t5 = time.time()
    print(f"   5. Zgortka (convolution): {(t5-t4)*1000:.2f}ms")

    total = t5 - t0
    print(f"\n   TOTAL: {total*1000:.2f}ms ({total:.4f}s)")
    print(f"   RozrachKogerTT is {(t4-t3)/total*100:.1f}% of total time")
    print("=" * 70)


# =============================================================================
# КОД ЩОБ ЗАПУСТИТИ ЦЕЙ ФАЙЛ НАПРЯМУ - КОД ЩОБ ЗАПУСТИТИ ЦЕЙ ФАЙЛ НАПРЯМУ
# =============================================================================

if __name__ == "__main__":
    import sys

    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        # BENCHMARK MODE
        print("\n🔥 NUMBA JIT BENCHMARK MODE\n")

        # Single sample detailed benchmark
        benchmark_single_sample(dl=100e-8)

        # Multi-sample benchmark
        n_samples = 20 if len(sys.argv) < 3 else int(sys.argv[2])
        benchmark_comparison(n_samples=n_samples, dl=100e-8)

        print("\n💡 TIP: Compare with original xrd.py by running similar benchmark there")
        print("Expected speedup: 2-5x on RozrachKogerTT with Numba JIT\n")

    else:
        # DEMO MODE (original visualization)
        print("=" * 70)
        print("DEMO MODE: Running single simulation with visualization")
        print("Run 'python xrd_parallel.py benchmark' for performance testing")
        print("=" * 70)

        # Створення системи GGG + YIG
        crystal = create_GGG_crystal()
        film = create_YIG_film()
        geometry = GeometryParameters(psi=0.0, asymmetric=False)

        # Створення симулятора
        simulator = HRXRDSimulator(crystal, film, geometry)

        # Параметри деформації (приклад з вашого коду)
        deformation = DeformationProfile(
            Dmax1=0.01305,
            D01=0.0017,
            L1=5800e-8,
            Rp1=3500e-8,
            D02=0.004845,
            L2=4000e-8,
            Rp2=-500e-8,
            Dmin=0.0001,
            dl=40e-8
        )

        # Симуляція
        print("\n🚀 Starting HRXRD simulation...")
        start = time.time()
        DeltaTeta, R_coger, R_convolved = simulator.RunSimulation(
            deformation,
            m1=700,
            m10=20,
            ik=4.671897861
        )
        elapsed = time.time() - start

        print(f"✅ Completed in {elapsed:.4f}s")
        print(f"   Points: {len(DeltaTeta)}")
        print(f"   Sublayers: {simulator.km}")
        print(f"   Film thickness: {simulator.hpl0*1e4:.2f} μm")

        # Візуалізація
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Профіль деформації
        L = simulator.km * deformation.dl
        z_profile = np.array([(L - deformation.dl * k + deformation.dl / 2) / 1e-8
                              for k in range(1, simulator.km + 1)])

        ax1.plot(z_profile, simulator.DD[1:],
                 'r-', label='Total DD', linewidth=2)
        ax1.plot(z_profile, simulator.DDPL1[1:], 'b--', label='Asym Gaussian')
        ax1.plot(z_profile, simulator.DDPL2[1:], 'g:', label='Decay Gaussian')
        ax1.set_xlabel('Depth z (Å)')
        ax1.set_ylabel('Deformation')
        ax1.set_title('Deformation Profile (NUMBA JIT)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Крива дифракційного відбивання
        ax2.plot(DeltaTeta, R_coger, 'darkgreen',
                 label='Coherent (Takagi-Taupin)', alpha=0.7)
        ax2.plot(DeltaTeta, R_convolved, 'blue',
                 label='Convolved', linewidth=2)
        ax2.set_xlabel('Δθ (arcsec)')
        ax2.set_ylabel('Intensity (a.u.)')
        ax2.set_title('HRXRD Rocking Curve (NUMBA JIT)')
        ax2.set_yscale('log')
        ax2.set_xlim(-300, 2100)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('hrxrd_numba_simulation.png', dpi=150)
        print("\n📊 Plot saved: hrxrd_numba_simulation.png")
        plt.show()
