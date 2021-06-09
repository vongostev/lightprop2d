# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 15:21:37 2021

@author: vonGostev
"""

import __init__
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from lightprop2d import Beam2D, gaussian_beam, random_round_hole


# XY grid dimensions
npoints = 256
# All input data are in cm
# XY grid widening
area_size = 500e-4
# Wavelength in cm
wl0 = 532e-7

# Round hole radius
R = 50e-4


beam = Beam2D(area_size, npoints, wl0, init_field_gen=gaussian_beam,
              init_gen_args=(1, R,))
beam.lens(-0.1)

fps = 1

# First set up the figure, the axis, and the plot element we want to animate

fig, ax = plt.subplots(1, 1, figsize=(9, 8))
im = ax.imshow(beam.iprofile, interpolation='none',
               aspect='auto', vmin=0, vmax=1,
               extent=[-area_size / 2e-4, area_size / 2e-4] * 2)
ax.set_xlabel(r'x, $\mu m$')
ax.set_ylabel(r'y, $\mu m$')

plt.colorbar(im)

snapshots = []
for i in range(11):
    snapshots.append(beam.iprofile)
    beam.propagate(0.2e-1)


def animate_func(i):
    im.set_array(snapshots[i])
    ax.set_title(f'{0.2*(i):.1f} mm')
    return [im]


anim = animation.FuncAnimation(
    fig,
    animate_func,
    frames=len(snapshots),
    interval=1000 / fps,  # in ms
)

anim.save('gaussian.mp4', fps=fps, extra_args=['-vcodec', 'libx264'])
