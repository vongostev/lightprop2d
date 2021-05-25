import numpy as np
import matplotlib.pyplot as plt

from beam2d import Beam2D

# All input data are in cm
Nx = Ny = 50
Lx = Ly = 3e1
z = 1
wl0 = 532e-7

A0 = 1
P = 10
OpticalDensity = 0.5
QE = 0.8

rho0 = 0.5
omega0 = 1e9
quant_energy = 8e-19
S0 = 100 ** 2
Aperture = np.pi * 10 ** 2
FillFactor = 1

km = 1  # 00 * 1000


@np.vectorize
def beam_profile(x, y, A0, rho0):
    return A0 * np.exp(- (x ** 2 + y ** 2) / 2 / rho0 ** 2)


"""================= BEAM PROPAGATION ===================="""


def bp(x, y): return beam_profile(x, y, A0, rho0)


b = Beam2D(init_field_gen=bp, wl=wl0, Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny)
# b.propagate(z)
b.lens(1)
b.propagate(30)
print(b)


plt.imshow(np.abs(b.xyfprofile))
plt.show()
