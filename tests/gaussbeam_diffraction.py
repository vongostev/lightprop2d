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

from lightprop2d import gaussian_beam
from lightprop2d import Beam2D

# XY grid dimensions
npoints = 1024
# All input data are in cm
# XY grid widening
area_size = 2e-1
# Wavelength in cm
wl0 = 532e-7

# Round hole radius
R = 0.01


beam = Beam2D(area_size, npoints, wl0, init_field_gen=gaussian_beam,
              init_gen_args=(1, R), use_gpu=1)
plt.imshow(beam.iprofile.get())
plt.show()
# Z grid for a propagation
dz = 0.2
z_grid = np.arange(14, 200) * dz
intensities = []

beam.propagate(z_grid[0])
# beam.lens(2)
for z in z_grid:
    if z > z_grid[0]:
        beam.propagate(dz)
    intensities.append(beam.central_intensity.get())

z_normalized = z_grid / 2 / np.pi / beam.k0 / R ** 2
plt.plot(z_normalized, np.array(intensities), label='Calc')
plt.plot(z_normalized, 1 / (1 + z_normalized ** 2),
         '--', label='Theory')
plt.xlabel(r'$z / z_d$')
plt.ylabel(r'$\frac{I_0(z)}{I_0(0)}$' + '\t   ', rotation=0, fontsize=14)
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig('img/gaussbeam_diffraction.png', dpi=300)
plt.show()
