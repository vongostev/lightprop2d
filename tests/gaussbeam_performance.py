# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 17:33:23 2021

@author: Pavel Gostev

Тестирование алгоритма распространения излучения на производительность
"""
import __init__
import time

from lightprop2d import gaussian_beam, rectangle_hole
from lightprop2d import Beam2D

# XY grid dimensions
npoints = [2048, 4096, 8192]
# All input data are in cm
# XY grid widening
area_size = 4e-1
# Wavelength in cm
wl0 = 532e-7

# Round hole radius
R = 0.01

for np in npoints:
    try:
        t = time.time()
        beam = Beam2D(area_size, np, wl0, init_field_gen=rectangle_hole,
                      init_gen_args=(2e-1, 1e-1), use_gpu=1)
        print(Beam2D.__name__, np, 'points', '[Info]',
              'construction time', f'{time.time() - t:.3g} s')
    except Exception as E:
        print(Beam2D.__name__, np, 'points', '[Error]', E)
    try:
        t = time.time()
        beam.propagate(10)
        print(Beam2D.__name__, np, 'points', '[Info]', 'propagation time',
              f'{(time.time() - t) * 1e3:.3g} ms')
        del beam
    except Exception as E:
        print(Beam2D.__name__, np, 'points', '[Error]', E)
