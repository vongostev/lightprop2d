# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 16:47:15 2021

@author: vonGostev
"""
import __init__
import pyMMF
import numpy as np
import matplotlib.pyplot as plt

from lightprop2d import Beam2D, gaussian_beam, round_hole, random_wave, random_round_hole

# Parameters
NA = 0.27
radius = 25  # in microns
n1 = 1.45
wl = 0.6328  # wavelength in microns

# calculate the field on an area larger than the diameter of the fiber
area_size = 3.5*radius
npoints = 2**7  # resolution of the window

RAND = 1


def plot_i(ibeam):
    area_size = ibeam.area_size
    plt.imshow(ibeam.iprofile,
               extent=(-area_size / 2e-4, area_size / 2e-4, -area_size / 2e-4, area_size / 2e-4))
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
    # plt.title('Modes series after the fiber')
    plt.tight_layout()
    plt.show()


ibeam = Beam2D(2 * area_size * 1e-4, 2 * npoints,
               wl * 1e-4, init_field_gen=round_hole,
               init_gen_args=((radius - 1) * 1e-4,))
plot_i(ibeam)


# Create the fiber object
profile = pyMMF.IndexProfile(npoints=npoints, areaSize=area_size)
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
modes_eig = solver.solve(nmodesMax=50, boundary='close',
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
