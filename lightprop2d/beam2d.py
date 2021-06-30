# -*- coding: utf-8 -*-
"""
The algorithm based on the paper
Delen, N., & Hooker, B. (1998).
Free-space beam propagation between arbitrarily oriented planes
based on full diffraction theory: a fast Fourier transform approach.
JOSA A, 15(4), 857-867.
"""
import warnings
import numpy as np
from dataclasses import dataclass
from scipy.signal import peak_widths
from scipy.interpolate import interp2d

try:
    import cupy as cp
except ImportError as E:
    warnings.warn(
        f"ImportError: {E}, 'use_gpu' key must be False.")

try:
    import pyfftw
    from pyfftw.interfaces.numpy_fft import fft2, ifft2
    _fftargs = {'planner_effort': 'FFTW_ESTIMATE',
                'overwrite_input': True,
                'threads': -1}
    _using_pyfftw = True
except ImportError as E:
    warnings.warn(f"ImportError: {E}, numpy is using for fft.")
    _fftargs = {}
    _using_pyfftw = False


nm = 1e-7
um = 1e-4
mm = 1e-1
cm = 1
m = 100


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
    complex_bits: int = 128

    use_gpu: bool = False
    unsafe_fft: bool = False

    def __post_init__(self):

        if self.use_gpu:
            self.xp = cp
        else:
            self.xp = np

        if self.complex_bits == 64:
            self.complex = self.xp.complex64
        elif self.complex_bits == 128:
            self.complex = self.xp.complex128

        self._construct_grids()

        if self.init_field_gen is not None:
            self.field = self._xp(
                self.init_field_gen(
                    self.X, self.Y, *self.init_gen_args)).astype(self.complex)
        elif self.init_field is not None:
            self.field = self._xp(self.init_field).astype(self.complex)
        else:
            raise ValueError(
                "Init field data is None: " +
                "'init_field_gen' must be a function or 'init_field' must be an array.")

        self.spectrum = self._fft2(self.field)

    def _np(self, data):
        if self.xp == cp:
            return data.get()
        return data

    def _xp(self, data):
        if self.xp == cp:
            return self.xp.array(data)
        else:
            if type(data) == np.ndarray:
                return data
            if type(data) == cp.ndarray:
                return data.get()
        raise ValueError(
            "Unknown type of the given array, not numpy or cupy array")

    def _k_grid(self, dL, npoints):
        return self.xp.fft.fftfreq(npoints, d=dL)

    def _fft2(self, data):
        if _using_pyfftw and not self.use_gpu:
            return fft2(data, **_fftargs)
        return self.xp.fft.fft2(data)

    def _ifft2(self, data):
        if _using_pyfftw and not self.use_gpu:
            return ifft2(data, **_fftargs)
        return self.xp.fft.ifft2(data)

    def _construct_grids(self):
        self.dL = self.area_size / self.npoints
        self.X = self.xp.arange(-self.npoints // 2,
                                self.npoints // 2, 1) * self.dL
        self.Y = self.X.reshape((-1, 1))

        self.k0 = 1 / self.wl
        self.Kx = self._k_grid(self.dL, self.npoints)
        self.Ky = self.Kx.reshape((-1, 1))
        self.Kz = self.xp.sqrt(self.k0**2 - self.Kx**2 - self.Ky**2)

        k_cryt = self._np(self.xp.trunc(self.area_size / self.wl))
        if self.npoints / 2 > k_cryt and not self.unsafe_fft:
            raise ValueError(
                f"Critical K⟂ {k_cryt:g} must be bigger than {self.npoints // 2}")

    def coordinate_filter(self, f_init=None, f_gen=None, fargs=()):
        if f_gen is not None:
            self.field *= f_gen(self.X, self.Y, *fargs)
        if f_init is not None:
            self.field *= self._xp(f_init)
        self.spectrum = self._fft2(self.field)

    def spectral_filter(self, f_init=None, f_gen=None, fargs=()):
        if f_gen is not None:
            self.spectrum *= f_gen(self.Kx, self.Ky, *fargs)
        if f_init is not None:
            self.spectrum *= self._xp(f_init)
        self.field = self._ifft2(self.spectrum)

    def expand(self, area_size):
        old_npoints = self.npoints
        self.npoints *= int(area_size / self.area_size)
        self.area_size = area_size
        d_npoints = (self.npoints - old_npoints) // 2
        self._construct_grids()

        xyprofile = self.xp.zeros([self.npoints]*2)
        xyprofile[d_npoints:-d_npoints,
                  d_npoints:-d_npoints] = self.field
        self.field = xyprofile
        del xyprofile
        self.spectrum = self._fft2(self.field)

    def crop(self, area_size, npoints=0):
        old_X = self.X[self.xp.abs(self.X) < area_size / 2]
        Nc = self.npoints // 2
        n = len(old_X) // 2
        old_X = self._np(self.X)[Nc-n:Nc+n]
        if npoints:
            self.npoints = npoints
        self.area_size = area_size

        cropped_profile = self._np(self.field)[Nc-n:Nc+n, Nc-n:Nc+n]
        fieldgen_real = interp2d(old_X, old_X, np.real(cropped_profile))
        fieldgen_imag = interp2d(old_X, old_X, np.imag(cropped_profile))
        self._construct_grids()
        X = self._np(self.X)

        self.field = self._xp(fieldgen_real(X, X) + 1j * fieldgen_imag(X, X))
        self.spectrum = self._fft2(self.field)

    def propagate(self, z):
        self.z += z * 1.
        _deltak = self.Kz * z
        # clip to interval [-2pi, 0]
        phase = 2 * self.xp.pi * (self.xp.trunc(_deltak) - _deltak)
        # noticably faster than writing exp(1j*phase)
        self.spectrum *= self.xp.cos(phase) + 1j * self.xp.sin(phase)
        self.field = self._ifft2(self.spectrum)

    def lens(self, f):
        phase = (self.X ** 2 + self.Y ** 2) * self.k0 / (2 * f)
        # noticably faster than writing exp(1j*phase)
        self.field *= self.xp.cos(phase) + 1j * self.xp.sin(phase)
        self.spectrum = self._fft2(self.field)

    def lens_image(self, f, l1, l2):
        scale_factor = - l1 / l2
        X = self._np(self.X)
        fieldgen_real = interp2d(X, X, np.real(self._np(self.field)))
        fieldgen_imag = interp2d(X, X, np.imag(self._np(self.field)))
        phase = - (self.X ** 2 + self.Y ** 2) * self.k0 * (l1 + l2) / 2 / l2 ** 2
        X *= scale_factor

        self.field = scale_factor * \
            self._xp(fieldgen_real(X, X) + 1j * fieldgen_imag(X, X)) * \
            (self.xp.cos(phase) + 1j * self.xp.sin(phase))
        self.spectrum = self._fft2(self.field)

    def _expand_basis(self, modes_list):
        Nb = np.sqrt(len(modes_list[0])).astype(int)
        if Nb != self.npoints:
            expanded_modes_list = []
            dN = (self.npoints - Nb) // 2
            Nt = self.npoints - dN
            for m in modes_list:
                me = np.zeros([self.npoints]*2, dtype=np.complex128)
                me[dN:Nt, dN:Nt] = m.reshape((Nb, Nb))
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
            self.field[dN:self.npoints - dN,
                       dN:self.npoints - dN].flatten()
        self.modes_coeffs = self.xp.linalg.lstsq(
            self._xp(modes_matrix), flatten_field)[0]
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
        modes_matrix_t = self._xp(modes_matrix_t)
        modes_matrix_dot_t = self._xp(modes_matrix_dot_t)
        Nb = np.sqrt(modes_matrix_t.shape[1]).astype(int)
        dN = (self.npoints - Nb) // 2
        flatten_field = \
            self.field[dN:self.npoints - dN,
                       dN:self.npoints - dN].flatten()
        self.modes_coeffs = self.xp.linalg.solve(
            modes_matrix_dot_t, modes_matrix_t.dot(flatten_field))
        return self.modes_coeffs

    def construct_by_modes(self, modes_list, modes_coeffs):
        modes_list_reshape = self._expand_basis(modes_list)
        self.field = self.xp.einsum(
            'ijk,i->jk', modes_list_reshape, modes_coeffs)
        self.spectrum = self._fft2(self.field)

    @property
    def D4sigma(self):
        """
        Returns the width ( :math:`D4\sigma` ) of the intensity distribution.

        """
        n = self.npoints // 2
        Y, X = self.xp.mgrid[-n:n, -n:n] * self.dL
        I = self.iprofile
        I[I < self.xp.max(I) / self.xp.exp(2)] = 0
        Xc = self.xp.average(X, weights=I)
        Yc = self.xp.average(Y, weights=I)
        def sigma(x, xc): return self.xp.sqrt(
            self.xp.average((x-xc)*(x-xc), weights=I))
        return 4 * sigma(X, Xc), 4 * sigma(Y, Yc)

    @property
    def iprofile(self):
        return self.xp.abs(self.field) ** 2

    @property
    def central_intensity(self):
        n = self.npoints // 2
        return self.iprofile[n, n]

    def __repr__(self):
        return (f"Beam {self.npoints:d}x{self.npoints:d} points {self.area_size:.3g}x{self.area_size:.3g} cm " +
                f"<wl={self.wl * 1e7:.3g} nm, z={self.z:.3g} cm>")
