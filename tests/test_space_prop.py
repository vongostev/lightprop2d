# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 16:24:40 2022

@author: von.gostev
"""

import __init__
import numpy as np
import matplotlib.pyplot as plt

from lightprop2d import gaussian_beam, plane_wave
from lightprop2d import Beam2D, cm, mm, um


def get_quanta_energy(wavelen):
    return 3e10 * 2 * np.pi * 1.054e-34 / wavelen


def rho(z, R, wl):
    return R * np.sqrt(1 + (z * wl / 2 / np.pi / R ** 2) ** 2)
    


def phnumber(z, P0, r0, wl):
    return 2 * P0 * r0 ** 2 / (1.054e-34 * 3e10 * wl * z ** 2)


# XY grid dimensions
npoints = 512
# All input data are in cm
# XY grid widening
area_size = 2 * mm
# Wavelength in cm
wl0 = 0.405 * um

R = 1 * mm

beam = Beam2D(area_size, npoints, wl0, init_field_gen=gaussian_beam,
              init_gen_args=(1, R), use_gpu=False)

p_grid = np.arange(0, 1, 0.01)
Z = 3000e5

for R in [0.1, 1, 10]:
    
    rho1 = rho(Z, R, wl0)
    
    intensities_theory = p_grid / np.pi / rho1 ** 2 / get_quanta_energy(wl0) / 0.17e9 * np.pi * 5 ** 2
    plt.semilogy(p_grid[1:], intensities_theory[1:], label=r'$\rho_0=' + f'{R}' + r'cm$')
    plt.semilogy(p_grid[1:], phnumber(Z, p_grid, R, wl0)[1:] / 0.17e9 * np.pi * 5 ** 2, '--', label=r'$\rho_0=' + f'{R}' + r'cm$')

plt.xlabel(r'$P_0,W$')
plt.ylabel(r'$F, \frac{1}{cm^2s}$' + '\t', rotation=0, fontsize=14)
plt.legend(frameon=False)
plt.tight_layout()
# plt.savefig('img/gaussbeam_diffraction.png', dpi=300)
plt.show()