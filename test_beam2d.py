import numpy as np
import matplotlib.pyplot as plt

from beam2d import Beam2D, square_slits, plane_wave

# All input data are in cm
Nx = Ny = 1000
Lx = Ly = 3e-1
z = 1
wl0 = 532e-7

A0 = 1
P = 10
OpticalDensity = 0.5
QE = 0.8

rho0 = 0.01
omega0 = 1e9
quant_energy = 8e-19
S0 = 100 ** 2
Aperture = np.pi * 10 ** 2
FillFactor = 1

km = 1  # 00 * 1000


@np.vectorize
def gaussian_beam(x, y, A0, rho0):
    return A0 * np.exp(- (x ** 2 + y ** 2) / 2 / rho0 ** 2)


def round_hole(x, y, R, x0=0, y0=0):
    d = gaussian_beam(x - x0, y - y0, 1, R)
    d = d >= 1 / np.exp(0.5)
    return np.array(d, dtype=int)


"""================= BEAM PROPAGATION ===================="""


def bp(x, y): return gaussian_beam(x, y, A0, rho0)


def ssf(x, y): return round_hole(x, y, 0.02)


intens = []
zs = []
b = Beam2D(init_field_gen=ssf, wl=wl0, Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny)
# b.coordinate_filter(ssf)
for i in range(1, 100):
    zs.append(0.05 * i)
    b.propagate(0.05 * i)
    intens.append(b.I0())

print(b)
zds = np.array(zs) * b.wl * 2 / 0.02 ** 2
plt.plot(zds, intens)
plt.plot(zds, 4 * 3e10 / 8 / np.pi *
         np.sin(b.k0 * 0.02 ** 2 / 4 / np.array(zs)) ** 2)
plt.show()

ip = np.abs(b.xyfprofile)
plt.imshow(ip)  # / np.sum(ip))
plt.show()
