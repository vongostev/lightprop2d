# -*- coding: utf-8 -*-
"""
The algorithm based on the paper
Delen, N., & Hooker, B. (1998).
Free-space beam propagation between arbitrarily oriented planes
based on full diffraction theory: a fast Fourier transform approach.
JOSA A, 15(4), 857-867.
"""
import numpy as np


__all__ = ('plane_wave', 'random_wave', 'gaussian_beam',
           'round_hole', 'random_round_hole',
           'rectangle_hole', 'square_hole',
           'square_slits')


@np.vectorize
def plane_wave(x, y):
    return 1


@np.vectorize
def random_wave(x, y):
    return np.random.random()


@np.vectorize
def gaussian_beam(x, y, A0, rho0):
    return A0 * np.exp(- (x ** 2 + y ** 2) / 2 / rho0 ** 2)


def round_hole(x, y, R, x0=0, y0=0):
    d = gaussian_beam(x - x0, y - y0, 1, R)
    field = d >= 1 / np.exp(0.5)
    return np.array(field, dtype=int)


def random_round_hole(x, y, R, x0=0, y0=0):
    field = random_wave(x, y)
    field[round_hole(x, y, R, x0, y0) == 0] = 0
    return field


@np.vectorize
def rectangle_hole(x, y, dx, dy, x0=0, y0=0):
    return (abs(x - x0) < (dx / 2)) & (abs(y - y0) < (dy / 2))


def square_hole(x, y, d, x0=0, y0=0):
    return rectangle_hole(x, y, d, d, x0, y0)


def square_slits(x, y, d, slits_distance, x0=0, y0=0):
    l = slits_distance / 2
    return square_hole(x, y, d, x0 - l, y0) | square_hole(x, y, d, x0 + l, y0)
