# -*- coding: utf-8 -*-
import numpy as np
from scipy import fftpack
import pyfftw

fftpack = pyfftw.interfaces.scipy_fftpack
pyfftw.interfaces.cache.enable()


@np.vectorize
def beam_profile(x, y):
    return 1  # np.random.randint(GRAY)


@np.vectorize
def square_slits(y, x, wide_x, wide_y, slits_distance, x0=0, y0=0):
    l = slits_distance

    slit1 = (abs(x - x0 - l) < (wide_x / 2)) & (abs(y - y0) < (wide_y / 2))
    slit2 = (abs(x - x0 + l) < (wide_x / 2)) & (abs(y - y0) < (wide_y / 2))
    return slit1 | slit2


class Beam2D:

    def __init__(self, init_field, wl0, Lx, Ly, Nx, Ny, from_array=False):

        self.Lx = Lx
        self.Ly = Ly
        self.Nx = Nx
        self.Ny = Ny

        # create grid
        xxgrid, yygrid = np.mgrid[-Nx / 2:Nx / 2, -Ny / 2:Ny / 2]
        self.X = xxgrid * Lx / Nx
        self.Y = yygrid * Ly / Ny

        self.Kx = self._k_grid(Lx, Nx)[:, np.newaxis]
        self.Ky = self._k_grid(Ly, Ny)

        self.z = 0.
        self.wl0 = wl0
        self.k0 = 2. * np.pi / wl0

        self._check_input()

        if not from_array:
            self._construct_profile(init_field)
        else:
            self._adapt_profile(init_field)

    def _k_grid(self, l, n):
        return 2 * np.pi / l * np.concatenate(
            (np.arange(0, n // 2 + 1), np.arange(- n // 2 + 1, 0)))

    def _check_input(self):
        kx_cryt = np.trunc(self.Lx / self.wl0)
        if self.Nx / 2 > kx_cryt:
            raise ValueError(
                "Critical Kx (%d) must be bigger than Nx / 2 (%d)" % (kx_cryt, self.Nx / 2))

        ky_cryt = np.trunc(self.Ly / self.wl0)
        if self.Ny / 2 > ky_cryt:
            raise ValueError(
                "Critical Ky (%d) must be bigger than Ny / 2 (%d)" % (ky_cryt, self.Ny / 2))

    def _construct_profile(self, f):
        self.xyprofile = f(self.X, self.Y)
        self.kprofile = fftpack.fft(self.xyprofile)

        self.xyfprofile = self.xyprofile[:, :]
        self.kfprofile = self.kprofile[:, :]

    def _adapt_profile(self, init_field):
        self.xyprofile = init_field
        self.kprofile = fftpack.fft(init_field)

        self.xyfprofile = self.xyprofile[:, :]
        self.kfprofile = self.kprofile[:, :]

    def coordinate_filter(self, f):
        self.xyfprofile *= f(self.X, self.Y)
        self.kfprofile = fftpack.fft2(self.xyfprofile)

    def spectral_filter(self, f):
        self.kfprofile *= f(self.Kx, self.Ky)
        self.xyfprofile = fftpack.ifft2(self.kfprofile)

    def propagate(self, z):
        self.z = np.float64(z)

        kzz = np.real(np.emath.sqrt(self.k0**2 - self.Kx**2 - self.Ky**2) * z)
        delta = kzz - 2. * np.pi * np.trunc(kzz / 2. / np.pi)

        self.kfprofile *= np.exp(1.j * delta)
        self.xyfprofile = fftpack.ifft2(self.kfprofile)

    def lens(self, f):
        self.xyfprofile *= np.exp(1.j *
                                  (self.X ** 2 + self.Y ** 2) * self.k0 / 2 / f)

    def __repr__(self):
        return "Beam (%d, %.3f) (%d, %.3f) <wl0=%.3fmic, z=%.3fcm>" % \
            (self.Nx, self.Lx, self.Ny, self.Ly, self.wl0 * 10**4, self.z)
