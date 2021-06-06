# -*- coding: utf-8 -*-
"""
The algorithm based on the paper
Delen, N., & Hooker, B. (1998).
Free-space beam propagation between arbitrarily oriented planes
based on full diffraction theory: a fast Fourier transform approach.
JOSA A, 15(4), 857-867.
"""
from dataclasses import dataclass
import numpy as np
from scipy import fftpack
from scipy.signal import peak_widths
from scipy.linalg import lstsq

try:
    import pyfftw
except ImportError:
    print("ImportError: PyFFTW didn't find, scipy.fftpack is used.")
else:
    print("PyFFTW is used to patch scipy.fftpack")
    fftpack = pyfftw.interfaces.scipy_fftpack
    pyfftw.interfaces.cache.enable()


@np.vectorize
def plane_wave(x, y):
    return 1


@np.vectorize
def random_wave(x, y):
    return np.random.random()


@np.vectorize
def gaussian_beam(x, y, A0, rho0):
    return A0 * np.exp(- (x ** 2 + y ** 2) / 2 / rho0 ** 2)


def round_hole(x, y, R, x0=0, y0=0):
    d = gaussian_beam(x - x0, y - y0, 1, R)
    d = d >= 1 / np.exp(0.5)
    return np.array(d, dtype=int)


@np.vectorize
def rectangle_hole(x, y, dx, dy, x0=0, y0=0):
    return (abs(x - x0) < (dx / 2)) & (abs(y - y0) < (dy / 2))


def square_hole(x, y, d, x0=0, y0=0):
    return rectangle_hole(x, y, d, d, x0, y0)


def square_slits(x, y, d, slits_distance, x0=0, y0=0):
    l = slits_distance / 2
    return square_hole(x, y, d, x0 - l, y0) | square_hole(x, y, d, x0 + l, y0)


@dataclass
class Beam2D:

    L: float
    N: int
    wl: float
    z: float = 0.
    init_field: np.ndarray = None
    init_field_gen: object = None

    def __post_init__(self):

        self.dL = self.L / self.N
        self.X = np.arange(-self.N // 2, self.N // 2, 1) * self.dL
        self.Y = self.X.reshape((-1, 1))

        self.k0 = 1 / self.wl
        self.Kx = self._k_grid(self.dL, self.N)
        self.Ky = self.Kx.reshape((-1, 1))
        self.Kz = 2 * np.pi * np.abs(
            np.emath.sqrt(self.k0**2 - self.Kx**2 - self.Ky**2))

        k_cryt = np.trunc(self.L / self.wl)
        if self.N / 2 > k_cryt:
            raise ValueError(
                f"Critical K⟂ {k_cryt:d} must be bigger than {self.N // 2}")

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

        _delta = self.Kz * z
        delta = _delta - 2. * np.pi * np.trunc(_delta / 2. / np.pi)

        self.kfprofile *= np.exp(- 1.j * delta)
        self.xyfprofile = fftpack.ifft2(self.kfprofile)

    def lens(self, f):
        self.xyfprofile *= \
            np.exp(1.j * (self.X ** 2 + self.Y ** 2) * self.k0 / 2 / f)
        self.kfprofile = fftpack.fft2(self.xyfprofile)

    def gaussian_fwhm(self):
        xcentral_profile = np.abs(self.xyfprofile[self.N // 2, :])
        xwidths, _, _, _ = peak_widths(xcentral_profile, peaks=[self.N // 2])
        ycentral_profile = np.abs(self.xyfprofile[:, self.N // 2])
        ywidths, _, _, _ = peak_widths(ycentral_profile, peaks=[self.N // 2])
        return xwidths[0] * self.dX, ywidths[0] * self.dY

    def _expand_basis(self, modes_list):
        Nb = np.sqrt(len(modes_list[0])).astype(int)
        if Nb != self.N:
            expanded_modes_list = []
            dN = (self.N - Nb) // 2
            for m in modes_list:
                me = np.zeros((self.N, self.N), dtype=np.complex128)
                me[dN:self.N - dN, dN:self.N - dN] = m.reshape((Nb, Nb))
                expanded_modes_list.append(me)
            return expanded_modes_list
        return modes_list

    def deconstruct_by_modes(self, modes_list):
        Nb = np.sqrt(len(modes_list[0])).astype(int)
        dN = (self.N - Nb) // 2
        modes_matrix = np.vstack(modes_list).T
        self.modes_coeffs = lstsq(modes_matrix,
                                  np.ravel(self.xyfprofile[dN:self.N - dN, dN:self.N - dN]))[0]
        return self.modes_coeffs

    def construct_by_modes(self, modes_list, modes_coeffs):
        modes_list_reshape = self._expand_basis(modes_list)
        self.xyfprofile = sum(modes_list_reshape[i] * modes_coeffs[i]
                              for i in range(len(modes_coeffs)))
        self.kfprofile = fftpack.fft2(self.xyfprofile)

    def iprofile(self):
        return np.abs(self.xyfprofile) ** 2

    def central_intensity(self):
        return self.iprofile()[self.N // 2, self.N // 2] * 3e10 / 8 / np.pi

    def __repr__(self):
        return (f"Beam {self.N:d}x{self.N:d} points {self.L:.3g}x{self.L:.3g} cm " +
                f"<wl={self.wl * 1e7:.3g} nm, z={self.z:.3g} cm>")
