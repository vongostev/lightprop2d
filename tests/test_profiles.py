# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 18:47:39 2021

@author: vonGostev
"""
import __init__
import numpy as np
import sys
from lightprop2d.profiles import *
from logging import Logger, StreamHandler, Formatter


log = Logger('test.profiles')

handler = StreamHandler(sys.stdout)
handler.setLevel(10)
formatter = Formatter(
    "%(asctime)s - %(name)-10s [%(levelname)-7.7s]  %(message)s")
handler.setFormatter(formatter)
log.addHandler(handler)

def _tprofiles(xp, npoints=256, dL=1e-4):
    x = xp.arange(-npoints // 2, npoints // 2, 1) * dL
    y = x.reshape((-1, 1))

    R = d = rho0 = dL * npoints / 4
    A0 = 1.
    dx = dy = dL * npoints / 4

    assert gaussian_beam(x, y, A0, rho0)[npoints // 2, npoints // 2] == A0
    assert xp.sum(plane_wave(x, y)) == npoints ** 2
    random_wave(x, y)
    round_hole(x, y, R)
    random_round_hole(x, y, R)
    assert xp.allclose(square_hole(x, y, d), rectangle_hole(x, y, dx, dy))
    assert xp.allclose(square_slits(x, y, d, d),
                       square_hole(x, y, d, x0=-d/2) + square_hole(x, y, d, x0=d/2))


if __name__ == "__main__":
    try:
        import cupy as cp
        _tprofiles(cp)
        log.info('Cupy tested')
    except Exception as E:
        log.error(f'Cupy test error: {E}')
    try:
        import numpy as np
        _tprofiles(np)
        log.info('Numpy tested')
    except Exception as E:
        log.error(f'Numpy test error: {E}')
