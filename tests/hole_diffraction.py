"""
Тестирование алгоритма распространения излучения
Дифракция на круглом и квадратном отверстиях
Ахманов С.А., Никитин С.Ю. Физическая оптика. Лекция 14
"""
import __init__
import numpy as np
from scipy.special import fresnel
import matplotlib.pyplot as plt

from lightprop2d import round_hole, square_hole
from lightprop2d import Beam2D, um, mm

# XY grid dimensions
npoints = 512
# All input data are in cm
# XY grid widening
area_size = 3 * mm
# Wavelength in cm
wl0 = 532e-7
dz = 0.02
z_grid = np.arange(40, 200) * dz

"""
Дифракция на круглом отверстии
"""
# Round hole radius
R = 150 * um

beam = Beam2D(area_size, npoints, wl0, init_field_gen=round_hole,
              init_gen_args=(R,), use_gpu=1)

# Z grid for a propagation
intensities = []
iprofiles = [beam.iprofile]

beam.propagate(z_grid[0])
for z in z_grid:
    intensities.append(beam.centroid_intensity)
    if int(z / dz) % 10 == 0:
        iprofiles.append(beam.iprofile)
    beam.propagate(dz)

z_normalized = z_grid * beam.wl * 2 / R ** 2
plt.plot(z_normalized, intensities, label='Calc')
plt.plot(z_normalized, 4 * np.sin(np.pi / z_normalized) ** 2,
         '--', label='Theory')
plt.axhline(4, linestyle=':')
plt.xlabel(r'$z/z_d=2\lambda z/R^2$')
plt.ylabel(r'$\frac{I_0(z)}{I_0(0)}$' + '\t   ', rotation=0, fontsize=14)
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig('img/round_hole_diffraction.png', dpi=300)
plt.show()

bounds = [-area_size / 4 / mm, area_size / 4 / mm] * 2
fig, axes = plt.subplots(1, 3, figsize=(9, 5))
axes[0].imshow(iprofiles[0][128:-128, 128:-128], extent=bounds)
axes[1].imshow(iprofiles[len(iprofiles) // 2]
               [128:-128, 128:-128], extent=bounds)
axes[2].imshow(iprofiles[-1][128:-128, 128:-128], extent=bounds)
axes[0].set_xlabel('$z/z_d=0$')
axes[1].set_xlabel(f'$z/z_d={z_normalized[len(iprofiles) * 5 - 20]:.2f}$')
axes[2].set_xlabel(f'$z/z_d={z_normalized[len(iprofiles) * 10 - 20]:.2f}$')
plt.tight_layout()
# plt.savefig('img/round_hole_iprofiles.png', dpi=300)
plt.show()

# """
# Дифракция на квадратном отверстии
# """

d = area_size / 8


def lfunc_sqr(a, b):
    # Formula 14.9
    c1, s1 = fresnel(a)
    c2, s2 = fresnel(b)
    return (c1 - c2) ** 2 + (s1 - s2) ** 2


beam = Beam2D(area_size, npoints, wl0, init_field_gen=square_hole,
              init_gen_args=(d,), use_gpu=True)

# Z grid for a propagation
intensities = []

beam.propagate(z_grid[0])
for z in z_grid:
    intensities.append(beam.centroid_intensity)
    # iprofiles.append(beam.iprofile)
    beam.propagate(dz)

z_normalized = z_grid * beam.wl * 2 / d ** 2
plt.plot(z_normalized, intensities, label='Calc')
plt.plot(z_normalized,
         0.25 * lfunc_sqr(- np.sqrt(1 / z_normalized),
                          np.sqrt(1 / z_normalized)) ** 2,
         '--', label='Theory')
plt.xlabel(r'$2\lambda z/d^2$')
plt.ylabel(r'$\frac{I_0(z)}{I_0(0)}$' + '\t   ', rotation=0, fontsize=14)
plt.legend(frameon=False)
plt.tight_layout()
# plt.savefig('img/square_hole_diffraction.png', dpi=300)
plt.show()
