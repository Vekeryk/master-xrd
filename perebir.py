import numpy as np
# from itertools import product
# from numba import njit, prange

'''  
# Дуже ідеальна густа сітка
Dmax1min =  .0020    #  Dmax1min  - нижня  межа максимальної деформацiї
Dmax1max =  .0300    #  Dmax1max  - верхня межа максимальної деформацiї
dDmax1   =  .0010
D01min   =  .0020    #  D01min  - нижня  межа максимальної деформацiї
D01max   =  .0300    #  D01max  - верхня межа максимальної деформацiї
dD01     =  .0010
L1min    =   500.    #  L1min     - нижня  межа товщини порушеного шару в ангстремах
L1max    =  7000.    #  L1max     - верхня межа товщини порушеного шару в ангстремах
dL1      =   100.    #
Rp1min   =    40.    #  Rp1min   - нижня  межа проекцiйного пробiгу =n*dRp-dl/2
Rp1max   =  6990.    #  Rp1max   - верхня межа проекцiйного пробiгу =n*dRp-dl/2
dRp1     =    50.    #  dRp      - крок змiни проекцiйного пробiгу
D02min   =  .0020    #  D02min - нижня  межа максимальної деформацiї спа
D02max   =  .0300    #  D02max - верхня межа максимальної деформацiї
dD02     =  .0010
L2min    =   500.    #  Lmin     - нижня  межа товщини порушеного шару в ангстремах
L2max    =  7000.    #  Lmax     - верхня межа товщини порушеного шару в ангстремах
dL2      =   100.    #
Rp2min   = -6050.    #  Rpmin2   - нижня  межа проекцiйного пробiгу =n*dRp-dl/2
Rp2max   =   -50.    #  Rpmax2   - верхня межа проекцiйного пробiгу =n*dRp-dl/2
dRp2     =   100.    #  dRp      - крок змiни проекцiйного пробiгу
#Розміри осей: {'Dmax1': 29, 'D01': 29, 'L1': 66, 'Rp1': 140, 'D02': 29, 'L2': 66, 'Rp2': 61}
#Теоретично комбінацій: 907,276,653,360
'''
'''
# Відносно густа сітка
Dmax1min =  .0020    #  Dmax1min  - нижня  межа максимальної деформацiї
Dmax1max =  .0300    #  Dmax1max  - верхня межа максимальної деформацiї
dDmax1   =  .0020
D01min   =  .0020    #  D01min  - нижня  межа максимальної деформацiї
D01max   =  .0300    #  D01max  - верхня межа максимальної деформацiї
dD01     =  .0020
L1min    =   500.    #  L1min     - нижня  межа товщини порушеного шару в ангстремах
L1max    =  7000.    #  L1max     - верхня межа товщини порушеного шару в ангстремах
dL1      =   200.    #
Rp1min   =    40.    #  Rp1min   - нижня  межа проекцiйного пробiгу =n*dRp-dl/2
Rp1max   =  6990.    #  Rp1max   - верхня межа проекцiйного пробiгу =n*dRp-dl/2
dRp1     =   200.    #  dRp      - крок змiни проекцiйного пробiгу
D02min   =  .0020    #  D02min - нижня  межа максимальної деформацiї спа
D02max   =  .0300    #  D02max - верхня межа максимальної деформацiї
dD02     =  .0020
L2min    =   500.    #  Lmin     - нижня  межа товщини порушеного шару в ангстремах
L2max    =  7000.    #  Lmax     - верхня межа товщини порушеного шару в ангстремах
dL2      =   200.    #
Rp2min   = -6050.    #  Rpmin2   - нижня  межа проекцiйного пробiгу =n*dRp-dl/2
Rp2max   =   -50.    #  Rpmax2   - верхня межа проекцiйного пробiгу =n*dRp-dl/2
dRp2     =   400.    #  dRp      - крок змiни проекцiйного пробiгу
#Розміри осей: {'Dmax1': 15, 'D01': 15, 'L1': 33, 'Rp1': 36, 'D02': 15, 'L2': 33, 'Rp2': 16}
#Теоретично комбінацій:                2,117,016,000
#Практично комбінацій(розумний перебір): 229,773,632
'''
'''
# Перевірочна сітка
Dmax1min =  .020    #  Dmax1min  - нижня  межа максимальної деформацiї
Dmax1max =  .030    #  Dmax1max  - верхня межа максимальної деформацiї
dDmax1   =  .010
D01min   =  .020    #  D01min  - нижня  межа максимальної деформацiї
D01max   =  .030    #  D01max  - верхня межа максимальної деформацiї
dD01     =  .010
L1min    =  5000.    #  L1min     - нижня  межа товщини порушеного шару в ангстремах
L1max    =  7000.    #  L1max     - верхня межа товщини порушеного шару в ангстремах
dL1      =  1000.    #
Rp1min   =  4010.    #  Rp1min   - нижня  межа проекцiйного пробiгу =n*dRp-dl/2
Rp1max   =  6990.    #  Rp1max   - верхня межа проекцiйного пробiгу =n*dRp-dl/2
dRp1     =  1000.    #  dRp      - крок змiни проекцiйного пробiгу
D02min   =  .010    #  D02min - нижня  межа максимальної деформацiї спа
D02max   =  .030    #  D02max - верхня межа максимальної деформацiї
dD02     =  .010
L2min    =  4000.    #  Lmin     - нижня  межа товщини порушеного шару в ангстремах
L2max    =  7000.    #  Lmax     - верхня межа товщини порушеного шару в ангстремах
dL2      =  1000.    #
Rp2min   = -6010.    #  Rpmin2   - нижня  межа проекцiйного пробiгу =n*dRp-dl/2
Rp2max   =   -5010.    #  Rpmax2   - верхня межа проекцiйного пробiгу =n*dRp-dl/2
dRp2     =   1000.    #  dRp      - крок змiни проекцiйного пробiгу
'''

# Покращена оптимальна сітка Grid 5 (IMPROVED - покриває експериментальні дані)
# ⚠️ ВАЖЛИВО: max значення скориговані щоб (max-min) було кратне step!
Dmax1min = .0010  # Dmax1min  - нижня  межа максимальної деформацiї
Dmax1max = .0310  # Dmax1max  - ВИПРАВЛЕНО: 0.031 кратне step (покриває 0.030)
dDmax1 = .0025
D01min = .0005  # D01min  - нижня  межа (РОЗШИРЕНО для покриття 0.000943)
D01max = .0305  # D01max  - ВИПРАВЛЕНО: 0.0305 кратне step (покриває 0.030)
dD01 = .0025
L1min = 500.  # L1min     - нижня  межа товщини порушеного шару в ангстремах
L1max = 7000.  # L1max     - верхня межа (УЖЕ кратне step) ✓
dL1 = 500.    #
Rp1min = 50.  # Rp1min   - нижня  межа проекцiйного пробiгу =n*dRp-dl/2
Rp1max = 5050.  # Rp1max   - ВИПРАВЛЕНО: 5050 кратне step (покриває 5000)
dRp1 = 500.  # dRp      - крок змiни проекцiйного пробiгу
D02min = .0010  # D02min - нижня  межа максимальної деформацiї спа
D02max = .0310  # D02max - ВИПРАВЛЕНО: 0.031 кратне step (покриває 0.030)
dD02 = .0025
L2min = 500.  # Lmin     - нижня  межа товщини порушеного шару в ангстремах
L2max = 5000.  # Lmax     - верхня межа (УЖЕ кратне step) ✓
dL2 = 500.    #
Rp2min = -6500.  # Rpmin2   - нижня  межа (РОЗШИРЕНО для покриття -50, -500)
Rp2max = 0.  # Rpmax2   - ВИПРАВЛЕНО: 0 кратне step (покриває -50, -500)
dRp2 = 500.  # dRp      - крок змiни проекцiйного пробiгу


def arange_inclusive(start, stop, step):
    """
    Create inclusive range. Works correctly for both positive and negative ranges.

    FIX: Previous version had bug with negative ranges (e.g., Rp2: -6500 → -5)
         It included extra values due to floating point arithmetic.
    """
    # Calculate exact number of steps
    n_steps = round((stop - start) / step)
    # Generate array with exact number of steps (more reliable than np.arange)
    return np.array([start + i * step for i in range(n_steps + 1)], dtype=float)


Dmax1 = arange_inclusive(Dmax1min, Dmax1max, dDmax1)
D01 = arange_inclusive(D01min, D01max, dD01)
L1 = arange_inclusive(L1min, L1max, dL1)
Rp1 = arange_inclusive(Rp1min, Rp1max, dRp1)
D02 = arange_inclusive(D02min, D02max, dD02)
L2 = arange_inclusive(L2min, L2max, dL2)
Rp2 = arange_inclusive(Rp2min, Rp2max, dRp2)

# Оцінка розміру повної сітки (теоретична) --------------------------------------
sizes = dict(Dmax1=len(Dmax1), D01=len(D01), L1=len(L1), Rp1=len(Rp1),
             D02=len(D02), L2=len(L2), Rp2=len(Rp2))
# total = np.prod(list(sizes.values()))
total = np.prod(list(sizes.values()), dtype=np.int64)

print("Розміри осей:", sizes)
print(f"Теоретично комбінацій: {total:,}")

"""
   Рахує кількість комбінацій для умов:
      d01 <= d1
      d01 + d02 <= limit
      r1 <= l1
      l2 <= l1
    ітерацій по Rp2 не обрізаємо (усі підходять).
"""

limit = 0.03
# Оцінка розміру повної сітки з умовами (розумний перебір)--------------------------------------
n = 0
for d1 in Dmax1:
    for d01 in D01:
        if d01 > d1:
            break
        for d02 in D02:
            if d01 + d02 > 0.03:
                break
            for l1 in L1:
                for r1 in Rp1:
                    if r1 > l1:
                        break
                    for l2 in L2:
                        if l2 > l1:
                            break
                        for r2 in Rp2:
                            n += 1
                #      print (n,d1,d01,l1,r1,d02,l2,r2)
print(f"Практично комбінацій(розумний перебір): {n}")


'''
# Оцінка розміру повної сітки з умовами (простий перебір)--------------------------------------
i=0
n=0
for d1 in Dmax1:
  for d01 in D01:
    for l1 in L1:
      for r1 in Rp1:
        for d02 in D02:
          for l2 in L2:
            for r2 in Rp2:
              i+=1
       #       print (i,d1,d01,l1,r1,d02,l2,r2,end='\t')
              if d01 > d1 or d01+d02>limit or r1>l1 or l2>l1: 
                  pass
       #           print("NO!",end='\n')
              else: 
                  n+=1
       #           print (end='\n')
print(f"Практично комбінацій(простий перебір): {n}")
'''
