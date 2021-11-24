# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 18:15:54 2021

@author: vonGostev
"""

import __init__
import sys
import numpy as np
from scipy.special import fresnel
import matplotlib.pyplot as plt

from lightprop2d import gaussian_beam, round_hole, square_hole
from lightprop2d import Beam2D
from logging import Logger, StreamHandler, Formatter


log = Logger('test.analytics')

handler = StreamHandler(sys.stdout)
handler.setLevel(10)
formatter = Formatter(
    "%(asctime)s - %(name)-10s [%(levelname)-7.7s]  %(message)s")
handler.setFormatter(formatter)
log.addHandler(handler)

# XY grid dimensions
npoints = 512
# All input data are in cm
# XY grid widening
area_size = 3e-1
# Wavelength in cm
wl0 = 532e-7
dz = 0.02


def test_gaussian(R, tol):
    # Round hole radius
    beam = Beam2D(area_size, npoints, wl0, init_field_gen=gaussian_beam,
                  init_gen_args=(1, R), use_gpu=1)
    # Z grid for a propagation
    z_grid = np.arange(14, 100) * dz
    intensities = []

    beam.propagate(z_grid[0])
    # beam.lens(2)
    for z in z_grid:
        if z > z_grid[0]:
            beam.propagate(dz)
        intensities.append(beam.centroid_intensity)

    z_normalized = z_grid / 2 / np.pi / beam.k0 / R ** 2
    assert np.allclose(np.array(intensities), 1 /
                       (1 + z_normalized ** 2), 1e-3)


def test_round(R, tol):
    beam = Beam2D(area_size, npoints, wl0, init_field_gen=round_hole,
                  init_gen_args=(R,), use_gpu=1)

    # Z grid for a propagation
    # dz = 0.02
    z_grid = np.arange(14, 100) * dz
    intensities = []

    beam.propagate(z_grid[0])
    for z in z_grid:
        if z > z_grid[0]:
            beam.propagate(dz)
        intensities.append(beam.centroid_intensity)

    z_normalized = z_grid * beam.wl * 2 / R ** 2
    assert np.allclose(np.array(intensities), 4 *
                       np.sin(np.pi / z_normalized) ** 2, tol)


def test_square(d, tol):
    def lfunc_sqr(a, b):
        # Formula 14.9
        c1, s1 = fresnel(a)
        c2, s2 = fresnel(b)
        return (c1 - c2) ** 2 + (s1 - s2) ** 2

    beam = Beam2D(area_size, npoints, wl0, init_field_gen=square_hole,
                  init_gen_args=(d,))

    # Z grid for a propagation
    # dz = 0.02
    z_grid = np.arange(14, 100) * dz
    intensities = []

    beam.propagate(z_grid[0])
    for z in z_grid:
        if z > z_grid[0]:
            beam.propagate(dz)
        intensities.append(beam.iprofile[npoints // 2, npoints // 2])

    z_normalized = z_grid * beam.wl * 2 / d ** 2
    assert np.allclose(
        intensities,
        0.25 * lfunc_sqr(- np.sqrt(1 / z_normalized),
                         np.sqrt(1 / z_normalized)) ** 2,
        tol)


tests = {
    'Gaussian beam': test_gaussian,
    'Round hole': test_round,
    'Square hole': test_square}
tols = [0.05, 1e-2, 1e-3, 1e-4, 1e-5]

if __name__ == "__main__":
    for tol in tols:
        log.info(f'Tolerance set to {tol}')
        for test in tests:
            log.info(f'Test set to `{test}`')
            passed = 0
            for i in range(10):
                R = np.random.random() * area_size / 10
                try:
                    tests[test](R, tol)
                    passed += 1
                    log.info(
                        f'{test} with relative size {R / area_size:.4f}: PASSED')
                except Exception as E:
                    log.error(
                        f'{test} with relative size {R / area_size:.4f}: FAILED')
            log.info(f'Passed {passed} from 10 tests')
