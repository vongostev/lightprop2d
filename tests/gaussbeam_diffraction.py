# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 17:33:23 2021

@author: Pavel Gostev

Тестирование алгоритма распространения излучения
Дифракция гауссова пучка
Ахманов С.А., Никитин С.Ю. Физическая оптика. Лекция 14
"""
import __init__
import numpy as np
import matplotlib.pyplot as plt

from lightprop2d import gaussian_beam, plane_wave
from lightprop2d import Beam2D, cm, mm, um

# XY grid dimensions
npoints = 512
# All input data are in cm
# XY grid widening
area_size = 2 * mm
# Wavelength in cm
wl0 = 0.532 * um

# Round hole radius
R = 0.1 * mm


beam = Beam2D(area_size, npoints, wl0, init_field_gen=gaussian_beam,
              init_gen_args=(1, R), use_gpu=True)
# beam = Beam2D(area_size, npoints, wl0, init_field_gen=plane_wave, use_gpu=1)
plt.imshow(beam.iprofile)
plt.show()
# Z grid for a propagation
dz = 2 * mm
z_grid = np.arange(0, 40, dz)

intensities = []
widths = []
phases = []

beam.propagate(z_grid[0])
for z in z_grid:
    if z > z_grid[0]:
        beam.propagate(dz)
    intensities.append(beam.centroid_intensity)
    widths.append(beam.D4sigma[0] / 2)
    phases.append(beam.phiprofile[beam.centroid[2:]])

z_normalized = 2 * z_grid / (2 * np.pi * beam.k0 * R ** 2)  # Formula 14.42

intensities_calc = np.array(intensities)
intensities_theory = 1 / (1 + z_normalized ** 2)
plt.plot(z_normalized, intensities_calc, label='Calc')
plt.plot(z_normalized, intensities_theory, '--', label='Theory')
plt.xlabel(r'$z / z_d$')
plt.ylabel(r'$\frac{I_0(z)}{I_0(0)}$' + '\t   ', rotation=0, fontsize=14)
plt.legend(frameon=False)
plt.tight_layout()
# plt.savefig('img/gaussbeam_diffraction.png', dpi=300)
plt.show()

widths_calc = np.array(widths) / R
widths_theory = np.sqrt(1 + (wl0 * z_grid / np.pi / R ** 2) ** 2)
plt.plot(z_normalized, widths_calc, label='Calc')
plt.plot(z_normalized, widths_theory,
         '--', label='Theory')
plt.xlabel(r'$z / z_d$')
plt.ylabel(r'$\frac{w(z)}{w(0)}$', rotation=0, fontsize=14)
plt.legend(frameon=False)
plt.tight_layout()
# plt.savefig('img/gaussbeam_diffraction.png', dpi=300)
plt.show()

phases_theory = np.arctan(wl0 * z_grid / np.pi / R ** 2)
phases_calc = np.array(phases) - 2 * np.pi * \
    (np.trunc(beam.k0 * z_grid) - beam.k0 * z_grid)
phases_calc[phases_calc > 2 * np.pi] -= 2 * np.pi

plt.plot(z_normalized, phases_calc, label='Calc')
plt.plot(z_normalized, phases_theory, '--', label='Theory')
plt.xlabel(r'$z / z_d$')
plt.ylabel(r'$\varphi_c - k_0z$', fontsize=14)
plt.legend(frameon=False)
plt.tight_layout()
# plt.savefig('img/gaussbeam_diffraction.png', dpi=300)
plt.show()
