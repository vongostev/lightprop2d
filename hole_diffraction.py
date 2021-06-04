"""
Тестирование алгоритма распространения излучения
Дифракция на круглом и квадратном отверстиях
Ахманов С.А., Никитин С.Ю. Физическая оптика. Лекция 14
"""
import numpy as np
from scipy.special import fresnel
import matplotlib.pyplot as plt

from beam2d import Beam2D, round_hole, square_hole

# XY grid dimensions
Nx = Ny = 1000
# All input data are in cm
# XY grid widening
Lx = Ly = 5e-1
# Wavelength in cm
wl0 = 532e-7

"""
Дифракция на круглом отверстии
"""
# Round hole radius
R = 0.01


def rh_init(x, y): return round_hole(x, y, R)


beam = Beam2D(init_field_gen=rh_init, wl=wl0, Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny)

# Z grid for a propagation
dz = 0.02
z_grid = np.arange(14, 200) * dz
intensities = []

beam.propagate(z_grid[0])
for z in z_grid:
    if z > z_grid[0]:
        beam.propagate(dz)
    intensities.append(beam.central_intensity())

z_normalized = z_grid * beam.wl * 2 / R ** 2
plt.plot(z_normalized, np.array(intensities) / (3e10 / 8 / np.pi), label='Calc')
plt.plot(z_normalized, 4 * np.sin(np.pi / z_normalized) ** 2,
         '--', label='Theory')
plt.axhline(4, linestyle=':')
plt.xlabel(r'$2\lambda z/R^2$')
plt.ylabel(r'$\frac{I_0(z)}{I_0(0)}$' + '\t   ', rotation=0, fontsize=14)
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig('round_hole_diffraction.png', dpi=300)
plt.show()

"""
Дифракция на квадратном отверстии
"""
# XY grid dimensions
Nx = Ny = 2000
# All input data are in cm
# XY grid widening
Lx = Ly = 4e-1
# Wavelength in cm
wl0 = 532e-7
# Square hole width
d = 0.02


def sh_init(x, y): return square_hole(x, y, d)


def lfunc_sqr(a, b):
    # Formula 14.9
    c1, s1 = fresnel(a)
    c2, s2 = fresnel(b)
    return (c1 - c2) ** 2 + (s1 - s2) ** 2


beam = Beam2D(init_field_gen=sh_init, wl=wl0, Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny)

# Z grid for a propagation
dz = 0.01
z_grid = np.arange(40, 100) * dz
intensities = []

beam.propagate(z_grid[0])
for z in z_grid:
    if z > z_grid[0]:
        beam.propagate(dz)
    intensities.append(beam.central_intensity())

z_normalized = z_grid * beam.wl * 2 / d ** 2
plt.plot(z_normalized, np.array(intensities) / (3e10 / 8 / np.pi), label='Calc')
plt.plot(z_normalized,
         0.25 * lfunc_sqr(- np.sqrt(1 / z_normalized),
                          np.sqrt(1 / z_normalized)) ** 2,
         '--', label='Theory')
plt.xlabel(r'$2\lambda z/d^2$')
plt.ylabel(r'$\frac{I_0(z)}{I_0(0)}$' + '\t   ', rotation=0, fontsize=14)
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig('square_hole_diffraction.png', dpi=300)
plt.show()
