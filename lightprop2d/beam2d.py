# -*- coding: utf-8 -*-
"""
The algorithm based on the paper
Delen, N., & Hooker, B. (1998).
Free-space beam propagation between arbitrarily oriented planes
based on full diffraction theory: a fast Fourier transform approach.
JOSA A, 15(4), 857-867.
"""
try:
    import cupy as cp
except ImportError:
    print("ImportError: CuPy didn't find, 'use_gpu' key is meaningless.")
else:
    print("CuPy found. GPU speed up can be used.")

import numpy as np
from dataclasses import dataclass
from scipy.signal import peak_widths


@dataclass
class Beam2D:

    area_size: float
    npoints: int
    wl: float
    z: float = 0.
    xp: object = np
    init_field: xp.ndarray = None
    init_field_gen: object = None
    init_gen_args: tuple = ()
    complex_bits: int = 64

    use_gpu: bool = False

    def __post_init__(self):

        if self.use_gpu:
            self.xp = cp
        else:
            self.xp = np

        if self.complex_bits == 64:
            self.complex = self.xp.complex64
        elif self.complex_bits == 128:
            self.complex = self.xp.complex128

        self.dL = self.area_size / self.npoints
        self.X = self.xp.arange(-self.npoints // 2,
                                self.npoints // 2, 1) * self.dL
        self.Y = self.X.reshape((-1, 1))

        self.k0 = 1 / self.wl
        self.Kx = self._k_grid(self.dL, self.npoints)
        self.Ky = self.Kx.reshape((-1, 1))
        self.Kz = 2 * self.xp.pi * self.xp.abs(
            self.xp.sqrt(self.k0**2 - self.Kx**2 - self.Ky**2))

        k_cryt = np.trunc(self.area_size / self.wl)
        if self.npoints / 2 > k_cryt:
            raise ValueError(
                f"Critical K⟂ {k_cryt:g} must be bigger than {self.npoints // 2}")

        if self.init_field_gen is not None:
            self.xyfprofile = self.xp.array(
                self.init_field_gen(
                    self.X, self.Y, *self.init_gen_args),
                dtype=self.complex)
        elif self.init_field is not None:
            self.xyfprofile = self.xp.array(
                self.init_field, dtype=self.complex)
        else:
            raise ValueError(
                "Init field data is None: " +
                "'init_field_gen' must be a function or 'init_field' must be an array.")

        self.kfprofile = self.xp.fft.fft2(self.xyfprofile)
        # self.xyfprofile = self.xyprofile.copy()
        # self.kfprofile = self.kprofile.copy()

    def _k_grid(self, dL, npoints):
        return self.xp.fft.fftfreq(npoints, d=dL)

    def coordinate_filter(self, f, fargs=()):
        self.xyfprofile *= f(self.X, self.Y, *fargs)
        self.kfprofile = self.xp.fft.fft2(self.xyfprofile)

    def spectral_filter(self, f, fargs=()):
        self.kfprofile *= f(self.Kx, self.Ky, *fargs)
        self.xyfprofile = self.xp.fft.ifft2(self.kfprofile)

    def expand(self, scale=2):
        old_npoints = self.npoints
        self.area_size *= scale
        self.npoints *= scale
        d_npoints = (self.npoints - old_npoints) // 2
        self.dL = self.area_size / self.npoints
        self.X = self.xp.arange(-self.npoints // 2,
                                self.npoints // 2, 1) * self.dL
        self.Y = self.X.reshape((-1, 1))

        self.k0 = 1 / self.wl
        self.Kx = self._k_grid(self.dL, self.npoints)
        self.Ky = self.Kx.reshape((-1, 1))
        self.Kz = 2 * self.xp.pi * self.xp.abs(
            self.xp.sqrt(self.k0**2 - self.Kx**2 - self.Ky**2))

        k_cryt = self.xp.trunc(self.area_size / self.wl)
        if self.npoints / 2 > k_cryt:
            raise ValueError(
                f"Critical K⟂ {k_cryt:g} must be bigger than {self.npoints // 2}")

        xyprofile = self.xp.zeros((self.npoints, self.npoints))
        xyprofile[d_npoints: -d_npoints,
                  d_npoints:-d_npoints] = self.xyfprofile
        self.xyfprofile = xyprofile
        del xyprofile
        self.kfprofile = self.xp.fft.fft2(self.xyfprofile)

    def propagate(self, z):
        self.z += z * 1.

        _delta = self.Kz * z
        delta = _delta - 2. * self.xp.pi * \
            self.xp.trunc(_delta / 2. / self.xp.pi)

        self.kfprofile *= self.xp.exp(- 1.j * delta)
        self.xyfprofile = self.xp.fft.ifft2(self.kfprofile)

    def lens(self, f):
        self.xyfprofile *= \
            self.xp.exp(1.j * (self.X ** 2 + self.Y ** 2) * self.k0 / 2 / f)
        self.kfprofile = self.xp.fft.fft2(self.xyfprofile)

    def _expand_basis(self, modes_list):
        Nb = np.sqrt(len(modes_list[0])).astype(int)
        if Nb != self.npoints:
            expanded_modes_list = []
            dN = (self.npoints - Nb) // 2
            Nt = self.npoints - dN
            Nl = dN
            for m in modes_list:
                me = np.zeros((self.npoints, self.npoints),
                              dtype=np.complex128)
                me[Nl:Nt, Nl:Nt] = m.reshape((Nb, Nb))
                expanded_modes_list.append(me)
            return np.array(expanded_modes_list)
        return np.array([m.reshape((Nb, Nb)) for m in modes_list])

    def deconstruct_by_modes(self, modes_list):
        """
        Return decomposed coefficients in given mode basis
        as least-square solution

        Parameters
        ----------
        modes_list : iterable
            List of flattened modes. Unified with pyMMF

        Returns
        -------
        ndarray
            Modes coefficients.

        """

        Nb = np.sqrt(len(modes_list[0])).astype(int)
        dN = (self.npoints - Nb) // 2
        modes_matrix = np.vstack(modes_list).T
        flatten_field = \
            self.xyfprofile[dN:self.npoints - dN,
                            dN:self.npoints - dN].flatten()
        self.modes_coeffs = self.xp.linalg.lstsq(
            self.xp.array(modes_matrix), flatten_field)[0]
        return self.modes_coeffs

    def fast_deconstruct_by_modes(self, modes_matrix_t,  modes_matrix_dot_t):
        """
        Return decomposed coefficients in given mode basis
        as least-square solution
        Fast version with pre-computations 

        Results can be a little different from `deconstruct_by_modes` ones

        Parameters
        ----------
        modes_matrix_t : ndarray
            If modes are flatten then modes_matrix_t is calculated as follows:
                >>> modes_matrix = self.xp.vstack(modes_list).T

        modes_matrix_dot_t : ndarray
            Linear system matrix. It is calculated so:
                >>> modes_matrix.T.dot(modes_matrix)

        Returns
        -------
        ndarray
            Modes coefficients.

        """
        modes_matrix_t = self.xp.array(modes_matrix_t)
        modes_matrix_dot_t = self.xp.array(modes_matrix_dot_t)
        Nb = np.sqrt(modes_matrix_t.shape[1]).astype(int)
        dN = (self.npoints - Nb) // 2
        flatten_field = \
            self.xyfprofile[dN:self.npoints - dN,
                            dN:self.npoints - dN].flatten()
        self.modes_coeffs = self.xp.linalg.solve(
            modes_matrix_dot_t, modes_matrix_t.dot(flatten_field))
        return self.modes_coeffs

    def construct_by_modes(self, modes_list, modes_coeffs):
        modes_list_reshape = self._expand_basis(modes_list)
        self.xyfprofile = self.xp.einsum(
            'ijk,i->jk', modes_list_reshape, modes_coeffs)
        self.kfprofile = self.xp.fft.fft2(self.xyfprofile)

    @property
    def gaussian_fwhm(self):
        xcentral_profile = self.xp.abs(
            self.xyfprofile[self.npoints // 2, :]).get()
        xwidths, _, _, _ = peak_widths(
            xcentral_profile, peaks=[self.npoints // 2])
        ycentral_profile = self.xp.abs(
            self.xyfprofile[:, self.npoints // 2]).get()
        ywidths, _, _, _ = peak_widths(
            ycentral_profile, peaks=[self.npoints // 2])
        return xwidths[0] * self.dL, ywidths[0] * self.dL

    @property
    def iprofile(self):
        return self.xp.abs(self.xyfprofile) ** 2

    @property
    def central_intensity(self):
        return self.iprofile[self.npoints // 2, self.npoints // 2] * 3e10 / 8 / self.xp.pi

    def __repr__(self):
        return (f"Beam {self.npoints:d}x{self.npoints:d} points {self.area_size:.3g}x{self.area_size:.3g} cm " +
                f"<wl={self.wl * 1e7:.3g} nm, z={self.z:.3g} cm>")
