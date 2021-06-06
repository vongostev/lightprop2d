# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 16:47:15 2021

@author: vonGostev
"""
import __init__
import pyMMF
import numpy as np
import matplotlib.pyplot as plt

from lightprop2d import Beam2D, gaussian_beam, round_hole, random_wave

# Parameters
NA = 0.27
radius = 25  # in microns
n1 = 1.45
wl = 0.6328  # wavelength in microns

# calculate the field on an area larger than the diameter of the fiber
areaSize = 3.5*radius
npoints = 2**7  # resolution of the window


def plot_i(ibeam):
    plt.imshow(ibeam.iprofile(),
               extent=(-ibeam.L / 2e-4, ibeam.L / 2e-4, -ibeam.L / 2e-4, ibeam.L / 2e-4))
    plt.xlabel(r'x, $\mu m$')
    plt.ylabel(r'y, $\mu m$')
    plt.colorbar()
    plt.tight_layout()
    plt.show()


def plot_modes(modes_coeffs):
    plt.plot(np.real(modes_coeffs))
    plt.plot(np.imag(modes_coeffs))
    plt.xlabel('Mode number')
    plt.ylabel('Coefficient')
    plt.title('Modes series before the fiber')
    plt.tight_layout()
    plt.show()


def g_init(x, y, random=0):
    if random:
        field = random_wave(x, y)
        field[round_hole(x, y, (radius - 1) * 1e-4) == 0] = 0
    else:
        field = round_hole(x, y, (radius - 2) * 1e-4)
        # * np.cos((x) * 1e4) * np.cos((y) * 1e4)
    return field


ibeam = Beam2D(2 * areaSize * 1e-4, 2 * npoints,
               wl * 1e-4, init_field_gen=g_init)
plot_i(ibeam)


# Create the fiber object
profile = pyMMF.IndexProfile(npoints=npoints, areaSize=areaSize)
# Initialize the index profile
profile.initStepIndex(n1=n1, a=radius, NA=NA)
# Instantiate the solver
solver = pyMMF.propagationModeSolver()
# Set the profile to the solver
solver.setIndexProfile(profile)
# Set the wavelength
solver.setWL(wl)

# Estimate the number of modes for a graded index fiber
Nmodes_estim = pyMMF.estimateNumModesSI(wl, radius, NA, pola=1)

# modes_semianalytical = solver.solve(mode='SI', curvature=None)
modes_eig = solver.solve(nmodesMax=500, boundary='close',
                         mode='eig', curvature=None, propag_only=True)
modes_list = modes_eig

modes_coeffs = ibeam.deconstruct_by_modes(modes_list.profiles)
plot_modes(modes_coeffs)

fiber_length = 50e4  # um
fiber_matrix = modes_eig.getPropagationMatrix(fiber_length)
tmodes_coeffs = fiber_matrix @ modes_coeffs

ibeam.construct_by_modes(modes_eig.profiles, tmodes_coeffs)
plot_i(ibeam)
plot_modes(tmodes_coeffs)

for i in range(5):
    ibeam.propagate(100e-4)
    plot_i(ibeam)
