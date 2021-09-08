# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 13:01:51 2021

@author: Pavel Gostev
"""
import __init__
import numpy as np
from os.path import join as pjoin
from lightprop2d import Beam2D, um, mm, cm, round_hole, gaussian_beam
import matplotlib.pyplot as plt
from tifffile import imread


def _n(p):
    return p / np.max(p)


def iprofile_crop(beam, ncrop):
    xc, yc, xnc, ync = beam.centroid

    xl = xnc - ncrop // 2
    xt = xnc + ncrop // 2
    yl = ync - ncrop // 2
    yt = ync + ncrop // 2
    return xl, xt, yl, yt, xnc, ync


def iprofile_imshow(beam, ax, ncrop, center=False, title=''):
    xl, xt, yl, yt, xnc, ync = iprofile_crop(beam, ncrop)
    img_extent = [beam.X[xl] / mm, beam.X[xt] / mm,
                  beam.X[yl] / mm, beam.X[yt] / mm]
    ax.imshow(_n(beam.iprofile)[yl:yt, xl:xt], extent=img_extent)

    if not center:
        xc, yc, _, _ = beam.centroid
    else:
        xc, yc = center
    ax.axvline(xc / mm, color='black', linestyle='--')
    ax.axhline(yc / mm, color='black', linestyle='--')
    ax.set_xlabel('x, mm')
    ax.set_ylabel('y, mm')
    ax.set_title(title)


def iprofile_plot(beam, ax, ncrop, axis=0, offset=(0, 0), calc_center=True, title=''):
    xl, xt, yl, yt, xnc, ync = iprofile_crop(beam, ncrop)
    x_offset, y_offset = offset
    if not calc_center:
        xnc, ync = beam.npoints // 2, beam.npoints // 2
    if axis == 0:
        ax.plot(beam.X[xl:xt] / mm,
                _n(beam.iprofile[ync - y_offset, x_offset:])[xl:xt])
    if axis == 1:
        ax.plot(beam.X[yl:yt] / mm,
                _n(beam.iprofile[y_offset:, xnc - x_offset])[yl:yt])
        # ax.set_xlabel('y, mm')
    ax.set_ylabel('$I / \max(I)$')
    ax.set_title(title)


# Z=0 Profile reading
# Path to experimental images
img_folder = '../../exp_beam/beam_imgs'
# Path to the image with z = 0
img_z0 = 'justbeam_z0.tif'
# Path to the image with z = 10
img_prop = 'beam_z9_9cm.tif'
beam_profile_z0 = imread(pjoin(img_folder, img_z0), key=0)[:, :, 0]
# Crop to square because of Beam2D requirements
beam_profile_z0_cropped = beam_profile_z0[:, 128:-128].astype(np.float64)
# Intensity aligning
intensity_transform_v = np.linspace(
    200, 1, beam_profile_z0_cropped.shape[1]).reshape((-1, 1)) ** 2
intensity_transform_matrix = np.repeat(
    intensity_transform_v, beam_profile_z0_cropped.shape[0], axis=1)
beam_profile_z0_cropped *= intensity_transform_matrix

# Parameters of Beam2D
wl = 0.632*um
dx = 6.45*um
npoints = beam_profile_z0_cropped.shape[0]
area_size = npoints * dx
hole_radius = 750*um
distance = 10.*cm

# Z=10 Profile reading
beam_profile_prop = imread(pjoin(img_folder, img_prop), key=0)[:, :, 0]
beam_profile_prop_cropped = beam_profile_prop[:, 128:-128].astype(np.float64)
# Image aligning
intensity_transform_v = np.linspace(
    100, 1, beam_profile_z0_cropped.shape[1]).reshape((-1, 1)) ** 1.5
intensity_transform_matrix = np.repeat(
    intensity_transform_v, beam_profile_z0_cropped.shape[0], axis=1)
beam_profile_prop_cropped *= intensity_transform_matrix

central_profile_prop = _n(beam_profile_prop_cropped[npoints // 2])
exp_beam_prop = Beam2D(area_size, npoints, wl,
                       init_field=beam_profile_prop_cropped)

x_offset = 7
y_offset = 13

# Calculated propagation from the experimental profile
beam_calc = Beam2D(area_size, npoints, wl, init_field=beam_profile_z0_cropped)
beam_calc.coordinate_filter(f_gen=round_hole, fargs=(
    hole_radius, -y_offset*dx, x_offset*dx))
beam_calc.propagate(distance)

# Calculated propagation from the gaussian profile
beam_gaussian = Beam2D(area_size, npoints, wl, init_field_gen=gaussian_beam,
                       init_gen_args=(1, 1100*um))
beam_gaussian.coordinate_filter(f_gen=round_hole, fargs=(
    hole_radius, -y_offset*dx, x_offset*dx))
beam_gaussian.propagate(distance)

beam_exp = Beam2D(area_size, npoints, wl, init_field=beam_profile_prop_cropped)


# Crop for depicting
ncrop = 300

gauss_title = f'Gaussian beam propagated to z={distance} cm'
exp_calc_title = f'Exp beam on z=0 propagated to z={distance} cm'
exp_title = f'Exp beam on z={distance} cm'

fig, ax = plt.subplots(1, 3, figsize=(10, 4))
iprofile_imshow(beam_gaussian, ax[0], ncrop, title=gauss_title)
iprofile_imshow(beam_calc, ax[1], ncrop, title=exp_calc_title)
iprofile_imshow(beam_exp, ax[2], ncrop, title=exp_title)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(3, 1, figsize=(5, 6))
iprofile_plot(beam_gaussian, ax[0], ncrop, title=gauss_title)
iprofile_plot(beam_calc, ax[1], ncrop, title=exp_calc_title)
iprofile_plot(beam_exp, ax[2], ncrop,  title=exp_title)
ax[2].set_xlabel('x, mm')
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(3, 1, figsize=(5, 6))
iprofile_plot(beam_gaussian, ax[0], ncrop, axis=1, title=gauss_title)
iprofile_plot(beam_calc, ax[1], ncrop, axis=1, title=exp_calc_title)
iprofile_plot(beam_exp, ax[2], ncrop, axis=1, title=exp_title)
ax[2].set_xlabel('y, mm')
plt.tight_layout()
plt.show()
