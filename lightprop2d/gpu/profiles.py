# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 15:21:29 2021

@author: vonGostev
"""
import cupy as cp


__all__ = ('plane_wave', 'random_wave', 'gaussian_beam',
           'round_hole', 'random_round_hole',
           'rectangle_hole', 'square_hole',
           'square_slits')


@cp.fuse
def plane_wave(x, y):
    return 1


def random_wave(x, y):
    return cp.random.random(size=(len(x), len(y)))


@cp.fuse
def gaussian_beam(x, y, A0, rho0):
    return A0 * cp.exp(- (x ** 2 + y ** 2) / 2 / rho0 ** 2)


def round_hole(x, y, R, x0=0, y0=0):
    d = gaussian_beam(x - x0, y - y0, 1, R)
    field = d >= 1 / cp.exp(0.5)
    return cp.array(field, dtype=cp.int8)


def random_round_hole(x, y, R, x0=0, y0=0):
    field = random_wave(x, y)
    field[round_hole(x, y, R, x0, y0) == 0] = 0
    return field


@cp.fuse
def rectangle_hole(x, y, dx, dy, x0=0, y0=0):
    return (cp.abs(x - x0) < (dx / 2)) & (cp.abs(y - y0) < (dy / 2))


def square_hole(x, y, d, x0=0., y0=0.):
    return rectangle_hole(x, y, d, d, x0, y0)


def square_slits(x, y, d, slits_distance, x0=0, y0=0):
    dl = slits_distance / 2
    return square_hole(x, y, d, x0 - dl, y0) | square_hole(x, y, d, x0 + dl, y0)
