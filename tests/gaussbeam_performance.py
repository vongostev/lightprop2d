# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 17:33:23 2021

@author: Pavel Gostev

Тестирование алгоритма распространения излучения на производительность
"""
import __init__
import time

from lightprop2d import gaussian_beam
from lightprop2d import Beam2DGPU, Beam2DCPU

# XY grid dimensions
npoints = [1024, 2048, 4096]
# All input data are in cm
# XY grid widening
area_size = 2e-1
# Wavelength in cm
wl0 = 532e-7

# Round hole radius
R = 0.01

for Beam2D in [Beam2DCPU, Beam2DGPU]:
    for np in npoints:
        t = time.time()
        beam = Beam2D(area_size, np, wl0, init_field_gen=gaussian_beam,
                      init_gen_args=(1, R))
        print(Beam2D.__name__, np, 'points', 'construction time', f'{time.time() - t:.3g} s')
        t = time.time()
        beam.propagate(10)
        print(Beam2D.__name__, np, 'points', 'propagation time', f'{(time.time() - t) * 1e3:.3g} ms')

