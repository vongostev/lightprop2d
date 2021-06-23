# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 18:47:39 2021

@author: vonGostev
"""

import numpy as np
import cupy as cp

from profiles import *


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
    square_slits(x, y, d, d)


def test_numpy():
    return _tprofiles(np)


def test_cupy():
    return _tprofiles(cp)
