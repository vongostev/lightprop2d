# -*- coding: utf-8 -*-
from dataclasses import dataclass
import numpy as np
from scipy import fftpack
from scipy.signal import peak_widths
import pyfftw

fftpack = pyfftw.interfaces.scipy_fftpack
pyfftw.interfaces.cache.enable()


@np.vectorize
def plane_wave(x, y):
    return 1  # np.random.randint(GRAY)


@np.vectorize
def square_slits(x, y, wide_x, wide_y, slits_distance, x0=0, y0=0):
    l = slits_distance

    slit1 = (abs(x - x0 - l) < (wide_x / 2)) & (abs(y - y0) < (wide_y / 2))
    slit2 = (abs(x - x0 + l) < (wide_x / 2)) & (abs(y - y0) < (wide_y / 2))
    return slit1 | slit2


@dataclass
class Beam2D:

    Lx: float
    Ly: float
    Nx: int
    Ny: int
    wl: float
    z: float = 0.
    init_field: np.ndarray = None
    init_field_gen: object = None

    def __post_init__(self):

        # create grid
        self.dX = self.Lx / self.Nx
        self.dY = self.Ly / self.Ny
        self.X = np.arange(-self.Nx / 2, self.Nx / 2, 1) * self.dX
        self.Y = np.arange(-self.Ny / 2, self.Ny / 2,
                           1)[:, np.newaxis] * self.dY

        self.Kx = self._k_grid(self.dX, self.Nx)
        self.Ky = self._k_grid(self.dY, self.Ny)[:, np.newaxis]

        self.k0 = 2. * np.pi / self.wl

        kx_cryt = np.trunc(self.Lx / self.wl)
        if self.Nx / 2 > kx_cryt:
            raise ValueError(
                f"Critical Kx {kx_cryt:d} must be bigger than {self.Nx // 2}")

        ky_cryt = np.trunc(self.Ly / self.wl)
        if self.Ny / 2 > ky_cryt:
            raise ValueError(
                f"Critical Ky {ky_cryt:d} must be bigger than {self.Ny // 2}")

        if self.init_field_gen is not None:
            self.xyprofile = np.complex128(self.init_field_gen(self.X, self.Y))
        elif self.init_field is not None:
            self.xyprofile = np.complex128(self.init_field)
        else:
            raise ValueError(
                "Init field data is None: " +
                "'init_field_gen' must be a function or 'init_field' must be an array.")
        self._construct_profile()

    def _k_grid(self, dL, N):
        return fftpack.fftfreq(N, d=dL)

    def _construct_profile(self):
        self.kprofile = fftpack.fft2(self.xyprofile)
        self.xyfprofile = self.xyprofile[:, :]
        self.kfprofile = self.kprofile[:, :]

    def coordinate_filter(self, f):
        self.xyfprofile *= f(self.X, self.Y)
        self.kfprofile = fftpack.fft2(self.xyfprofile)

    def spectral_filter(self, f):
        self.kfprofile *= f(self.Kx, self.Ky)
        self.xyfprofile = fftpack.ifft2(self.kfprofile)

    def propagate(self, z):
        self.z += z * 1.

        kz = np.real(np.emath.sqrt(self.k0**2 - self.Kx**2 - self.Ky**2))
        # delta = kz * z - 2. * np.pi * np.trunc(kz * z / 2. / np.pi)
        self.kfprofile *= np.exp(1.j * kz * z)
        self.xyfprofile = fftpack.ifft2(self.kfprofile)

    def lens(self, f):
        self.xyfprofile *= np.exp(1.j *
                                  (self.X ** 2 + self.Y ** 2) * self.k0 / 2 / f)
        self.kfprofile = fftpack.fft2(self.xyfprofile)

    def FWHM(self):
        xcentral_profile = np.abs(self.xyfprofile[self.Ny // 2, :])
        xwidths, _, _, _ = peak_widths(xcentral_profile, peaks=[self.Nx // 2])
        ycentral_profile = np.abs(self.xyfprofile[:, self.Nx // 2])
        ywidths, _, _, _ = peak_widths(ycentral_profile, peaks=[self.Ny // 2])
        return xwidths[0] * self.dX, ywidths[0] * self.dY

    def I0(self):
        return np.abs(self.xyfprofile[self.Ny // 2, self.Nx // 2]) ** 2 * 3e10 / 8 / np.pi

    def __repr__(self):
        return (f"Beam {self.Nx:d}x{self.Ny:d} points {self.Lx:.3g}x{self.Ly:.3g} cm " +
                f"<wl={self.wl * 1e7:.3g} nm, z={self.z:.3g} cm>")
