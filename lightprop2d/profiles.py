# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 15:21:29 2021

@author: vonGostev
"""
import numpy as np
try:
    import cupy as cp
except ImportError:
    print("ImportError: CuPy didn't find, 'use_gpu' key is meaningless.")

__all__ = ('plane_wave', 'random_wave', 'gaussian_beam',
           'round_hole', 'random_round_hole',
           'rectangle_hole', 'square_hole',
           'square_slits')


def _get_array_module(x):
    try:
        return cp.get_array_module(x)
    except Exception:
        return np


def plane_wave(x, y):
    xp = _get_array_module(x)
    return xp.ones((len(y), len(x)))


def random_wave(x, y):
    xp = _get_array_module(x)
    return xp.random.random(size=(len(y), len(x)))


def gaussian_beam(x, y, A0, rho0):
    xp = _get_array_module(x)
    return A0 * xp.exp(- (x ** 2 + y ** 2) / 2 / rho0 ** 2)


def round_hole(x, y, R, x0=0, y0=0):
    xp = _get_array_module(x)
    d = gaussian_beam(x - x0, y - y0, 1, R)
    field = d >= 1 / xp.exp(0.5)
    return xp.array(field, dtype=xp.int8)


def random_round_hole(x, y, R, x0=0, y0=0):
    field = random_wave(x, y)
    field[round_hole(x, y, R, x0, y0) == 0] = 0
    return field


def rectangle_hole(x, y, dx, dy, x0=0, y0=0):
    xp = _get_array_module(x)
    return (xp.abs(x - x0) < (dx / 2)) & (xp.abs(y - y0) < (dy / 2))


def square_hole(x, y, d, x0=0., y0=0.):
    return rectangle_hole(x, y, d, d, x0, y0)


def square_slits(x, y, d, slits_distance, x0=0, y0=0):
    dl = slits_distance / 2
    return square_hole(x, y, d, x0 - dl, y0) | square_hole(x, y, d, x0 + dl, y0)
