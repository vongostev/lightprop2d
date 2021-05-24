import numpy as np
import matplotlib.pyplot as plt

from beam2d import Beam2D

# All input data are in cm
Nx = Ny = 1000
Lx = Ly = 10000e2
z = 1
wl0 = 1.550e-4

A0 = 1
P = 10
OpticalDensity = 0.5
QE = 0.8

rho0 = 1502e2
omega0 = 1e9
quant_energy = 8e-19
S0 = 100 ** 2
Aperture = np.pi * 10 ** 2
FillFactor = 1

km = 100 * 1000


@np.vectorize
def beam_profile(x, y, A0, rho0):
    return A0 * np.exp(- (x ** 2 + y ** 2) / 2 / rho0 ** 2)


"""================= BEAM PROPAGATION ===================="""


def bp(x, y): return beam_profile(x, y, A0, rho0)


b = Beam2D(init_field_gen=bp, wl=wl0, Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny)
b.propagate(z)
print(b)

photons = np.abs(b.xyprofile) / np.sum(b.xyprofile) * P / omega0 / \
    quant_energy / S0 * Aperture  # * OpticalDensity * QE / 4 * FillFactor


plt.plot(b.X[:, 0] / km, photons[Nx // 2])
plt.axvline(-rho0 / km, ls='--', color='black')
plt.axvline(rho0 / km, ls='--', color='black')

pmax = np.max(photons[Nx // 2])

plt.annotate(r'$\rho_0=%.2f$ km' % (rho0 / km) + '\n' + r'$\Delta\alpha=300\:\mu rad$', xy=(rho0 / km, pmax / 2),  xycoords='data',
             xytext=(0.95, 0.75), textcoords='axes fraction',
             arrowprops=dict(facecolor='black', shrink=0.05),
             horizontalalignment='right', verticalalignment='top',
             )

fivesigma = 62500 / km
plt.axvline(-fivesigma, ls='--', color='r')
plt.axvline(fivesigma, ls='--', color='r')

plt.annotate(r'$5\sigma = %.2f$ km' % (62500 / km) + '\n' + r'$\Delta\alpha=125\:\mu rad$', xy=(fivesigma, pmax / 7),  xycoords='data',
             xytext=(0.95, 0.5), textcoords='axes fraction',
             arrowprops=dict(facecolor='black', shrink=0.05),
             horizontalalignment='right', verticalalignment='top',
             )
plt.ylabel('Число фотоотсчетов на детектор')
plt.xlabel('Поперечная координата, км')
plt.show()
