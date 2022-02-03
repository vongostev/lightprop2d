# -*- coding : utf-8 -*-

"""
The algorithm based on the paper
Delen, N., & Hooker, B. (1998).
Free-space beam propagation between arbitrarily oriented planes
based on full diffraction theory : a fast Fourier transform approach.
JOSA A, 15(4), 857-867.
"""

import warnings
import numpy as np
from dataclasses import dataclass
from scipy.interpolate import interp2d


try:
    import cupy as cp
    _using_cupy = True
except ImportError as E:
    _using_cupy = False
    warnings.warn(
        f"ImportError : {E}, 'use_gpu' key is meaningless.")

try:
    from pyfftw.interfaces.numpy_fft import fft2, ifft2
    _fftargs = {'planner_effort': 'FFTW_ESTIMATE',
                'overwrite_input': True,
                'threads': -1}
    _using_pyfftw = True
except ImportError as E:
    warnings.warn(f"ImportError : {E}, numpy is using for fft.")
    _fftargs = {}
    _using_pyfftw = False


nm = 1e-7
um = 1e-4
mm = 1e-1
cm = 1
m = 100


@dataclass
class Beam2D:
    """Electromagnetic field propagation using spectral method.

    Simple class to transform intitial field distribution using 2D fourier 
    transformation from x-y field profile to kx-ky spectrum.
    You can use both numpy and cupy backends with use_gpu key of the class.

    Parameters
    ----------
    area_size : float
        The beam calculation area size in centimetres.
    npoints : int
        Number of points by one axes.
    wl : float
        Beam central wavelength in centimetres.
    z : float = 0.
        Propagation distance in centimetres.
    xp : object = np
        Backend module. numpy (np) or cupy (cp). Controlled by 'use_gpu' key
    init_field : xp.ndarray = None
        Initial field distribution given as an array
    init_field_gen : object = None
        Initial field distribution given as a generating function
    init_gen_args : tuple = ()
        Additional arguments of 'init_field_gen' excluding
        the first two (X grid and Y grid)
    complex_bits : int = 128
        Precision of complex numbers. Can be 64 or 128
    use_gpu : bool = False
        Backend choice.
        If True, the class uses cupy backend with GPU support.
        If False, the class uses numpy backend
    """

    area_size: float
    npoints: int
    wl: float
    z: float = 0.
    xp: object = np
    init_field: xp.ndarray = None
    init_spectrum: xp.ndarray = None
    init_field_gen: object = None
    init_gen_args: tuple = ()
    complex_bits: int = 128

    use_gpu: bool = False
    numpy_output: bool = True

    def __post_init__(self):
        """Post-init parameters processing

        Raises
        ------
        ValueError

        Init field data is None:
            'init_field_gen' must be a function or 'init_field' must be an array.

        """

        # Choose module for arrays : numpy of cupy with gpu speed up
        if self.use_gpu and _using_cupy:
            self.xp = cp
        else:
            self.xp = np
        # Choose complex bits number
        if self.complex_bits == 64:
            self.complex = self.xp.complex64
        elif self.complex_bits == 128:
            self.complex = self.xp.complex128
        # Construct X, Y, Kx, Ky, Kz grids, k0 and dL
        self._construct_grids()
        # Initialize a field with the given function of array
        if self.init_field_gen is not None:
            self.field = self._asxp(
                self.init_field_gen(
                    self.X, self.Y, *self.init_gen_args))
        elif self.init_field is not None:
            self.field = self._asxp(self.init_field)
        else:
            raise ValueError(
                "Init field data is None : " +
                "'init_field_gen' must be a function or 'init_field' must be an array.")
        if self.field.dtype != self.complex:
            self.field = self.field.astype(self.complex)
        if self.init_spectrum is None:
            self.spectrum = self._fft2(self.field)
        else:
            self.spectrum = self.init_spectrum

    def _np(self, data):
        """Convert cupy or numpy arrays to numpy array.

        Parameters
        ----------
        data : Tuple[numpy.ndarray, cupy.ndarray]
            Input data.

        Returns
        -------
        data : numpy.ndarray
            Converted data.

        """
        # Return numpy array from numpy or cupy array
        if self.xp.__name__ == 'cupy':
            return data.get()
        return data

    def _asxp(self, data):
        """Convert cupy or numpy arrays to self.xp array.

        Parameters
        ----------
        data : Tuple[numpy.ndarray, cupy.ndarray]
            Input data.

        Raises
        ------
        ValueError
            Unknown type of the given array, not numpy or cupy array.

        Returns
        -------
        data : self.xp.ndarray
            Converted data.

        """
        # Return self.xp array from numpy or cupy array
        if not _using_cupy:
            return data
        if self.xp.__name__ == 'cupy':
            if type(data) == np.ndarray:
                return self.xp.array(data)
            if type(data) == cp.ndarray:
                return data
        else:
            if type(data) == np.ndarray:
                return data
            if type(data) == cp.ndarray:
                return data.get()
        raise ValueError(
            "Unknown type of the given array, not numpy or cupy array")

    def _k_grid(self, dL: float, npoints: int):
        """Return a grid for Kx or Ky values.

        Parameters
        ----------
        dL : float
            Spatial delta.
        npoints : int
            Number of points.

        Returns
        -------
        grid : self.xp.ndarray
            The grid for K values.

        """
        return self.xp.fft.fftfreq(npoints, d=dL)

    def _fft2(self, data):
        """2D FFT alias with a choice of the fft module.

        Parameters
        ----------
        data : self.xp.ndarray
            2d signal data with type of self.complex.
        """
        if _using_pyfftw and not self.use_gpu:
            return fft2(data, **_fftargs)
        return self.xp.fft.fft2(data)

    def _ifft2(self, data):
        """2D Inverse FFT alias with a choice of the fft module.

        Parameters
        ----------
        data : Tuple[numpy.ndarray, cupy.ndarray]
            2d signal data with type of self.complex.
        """
        if _using_pyfftw and not self.use_gpu:
            return ifft2(data, **_fftargs)
        return self.xp.fft.ifft2(data)

    def _update_obj(self, field, spectrum=None):
        """Fast updating of the beam field and spectrum.
        Very important for the sequential calculations.
        For example, with CPU:
        ```python
            >>> %timeit b = Beam2D(200, 1024, 0.632, init_field_gen=gaussian_beam, init_gen_args=(1, 50))
            81 ms ± 527 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

            >>> %timeit a = Beam2D(200, 1024, 0.632, init_field=b.field)
            55.8 ms ± 2.15 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

            >>> %timeit a._update_obj(b.field)
            36.4 ms ± 140 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

            >>> %timeit a = Beam2D(200, 1024, 0.632, init_field=b.field, init_spectrum=b.spectrum)
            17.2 ms ± 211 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

            >>> %timeit a._update_obj(b.field, spectrum=b.spectrum)
            1.12 µs ± 47.8 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
        ```

        And with GPU:
        ```python
            >>> %timeit b = Beam2D(200, 1024, 0.632, init_field_gen=gaussian_beam, init_gen_args=(1, 50), use_gpu=True)
            2.75 ms ± 16.3 µs per loop (mean ± std. dev. of 7 runs, 1 loop each)

            >>> %timeit a = Beam2D(200, 1024, 0.632, init_field=b.field, use_gpu=True)
            2.16 ms ± 63.2 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

            >>> %timeit -n10 a._update_obj(b.field)
            66.6 µs ± 23.1 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

            >>> a = Beam2D(200, 1024, 0.632, init_field=b.field, init_spectrum=b.spectrum, use_gpu=True)
            1.1 ms ± 44.2 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

            >>> %timeit -n10000 a._update_obj(b.field, spectrum=b.spectrum)
            1.39 µs ± 24 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)
        ```

        Parameters
        ----------
        field : Tuple[numpy.ndarray, cupy.ndarray]
            Field distribuion of complex type.
        spectrum : Tuple[numpy.ndarray, cupy.ndarray], optional
            Field spatial spectrum of complex type. The default is None.

        Returns
        -------
        None.

        """
        self.z = 0
        self.field = self._asxp(field)
        if spectrum is None:
            self.spectrum = self._fft2(self.field)
        else:
            self.spectrum = self._asxp(spectrum)

    def _construct_grids(self):
        """Construction of X, Y, Kx, Ky grids.

        Raises
        ------
        ValueError
            FFT may be unphysical.

        """
        self.dL = self.area_size / self.npoints
        self.X = self.xp.arange(-self.npoints // 2,
                                self.npoints // 2, 1) * self.dL
        self.Y = self.X.reshape((-1, 1))

        self.k0 = 1 / self.wl
        self.Kx = self._k_grid(self.dL, self.npoints)
        self.Ky = self.Kx.reshape((-1, 1))
        self.Kz = self.xp.nan_to_num(self.xp.sqrt(
            self.k0**2 - self.Kx**2 - self.Ky**2))

    def coordinate_filter(self, f_init=None, f_gen=None, fargs=()):
        """Apply a mask to the field profile.

        Parameters
        ----------
        f_init : Tuple[numpy.ndarray, cupy.ndarray], optional
            A mask as an array. The default is None.
        f_gen : function, optional
            A function to generate a mask. The default is None.
        fargs : tuple, optional
            Additional arguments of f_gen function. The default is ().

        Notes
        -----
        The mask function `f_gen` can be user defined and must be in form:
        >>> func(X, Y, *fargs)
        Where X, Y are 1D grids:
        >>> X = arange(-npoints // 2, npoints // 2, 1) * dL
        >>> Y = X.reshape((-1, 1))

        For example see **lightprop2d.gaussian_beam**
        """
        if f_gen is not None:
            assert isinstance(f_gen, object)
            assert isinstance(fargs, (tuple, list))
            self.field *= f_gen(self.X, self.Y, *fargs)
        if f_init is not None:
            assert isinstance(f_init, (np.ndarray, self.xp.ndarray))
            self.field *= self._asxp(f_init)
        self.spectrum = self._fft2(self.field)

    def spectral_filter(self, f_init=None, f_gen=None, fargs=()):
        """Apply a mask to the field spectrum.

        Parameters
        ----------
        f_init : Tuple[numpy.ndarray, cupy.ndarray], optional
            A mask as an array. The default is None.
        f_gen : function, optional
            A function to generate a mask. The default is None.
            The mask function can be user defined and must be in form
            >>> func(Kx, Ky, *fargs)

            Where Kx, Ky are 1D grids
            >>> Kx = fftfreq(npoints, d=dL)
            >>> Ky = Kx.reshape((-1, 1))
        fargs : tuple, optional
            Additional arguments of f_gen function. The default is ().

        """
        if f_gen is not None:
            self.spectrum *= f_gen(self.Kx, self.Ky, *fargs)
        if f_init is not None:
            self.spectrum *= self._asxp(f_init)
        self.field = self._ifft2(self.spectrum)

    def expand(self, area_size: float):
        r"""Expand the beam calculation area to the given area_size.
        with proportional `self.npoints` increasing.
        `self.dL` remains constant.

        Parameters
        ----------
        area_size : float
            Wanted area size in centimetres.
        """
        old_npoints = self.npoints
        self.npoints = int(self.npoints * area_size / self.area_size)
        self.npoints -= self.npoints % 2
        self.area_size = area_size
        d_npoints = (self.npoints - old_npoints) // 2
        self._construct_grids()
        # Remove step to zero on the bounds of image
        _f = self.xp.hstack((
            self.xp.repeat(self.field[:, 0].reshape(
                (-1, 1)), d_npoints, axis=1),
            self.field,
            self.xp.repeat(self.field[:, -1].reshape(
                (-1, 1)), d_npoints, axis=1)))
        _f = self.xp.vstack((
            self.xp.repeat(_f[0].reshape((1, -1)), d_npoints, axis=0),
            _f,
            self.xp.repeat(_f[-1].reshape((1, -1)), d_npoints, axis=0)))
        self.field = _f
        del _f
        self.spectrum = self._fft2(self.field)

    def crop(self, area_size: float, npoints: int = 0):
        r"""Crop the field to the new area_size smaller than actual.
        with proportional `self.npoints` decreasing.
        `self.dL` remains constant.

        Parameters
        ----------
        area_size : float
            A size of the calculation area in centimetres.
        npoints : int, optional
            A number of points in one dimention.
            The default is 0 -- number of points isn't changed.
        """
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

        self.field = self._asxp(fieldgen_real(X, X) + 1j * fieldgen_imag(X, X))
        self.spectrum = self._fft2(self.field)

    def coarse(self, mean_order: int = 1):
        r"""Decrease `self.npoints` with a divider `mean_order`.
        Block average applies to `self.field` with size of blocks as `mean_order*mean_order`.
        `self.spectrum` is calculated from the averaged `self.field`.
        It is necessary to decrease numerical complexity when propagation
        distance is huge and the beam radius grows dramatically.

        Recommended to use it after `self.expand`. For example
        ```python
            >>> beam.expand(self.area_size * 2)
            >>> beam.coarse(2)
        ```
        Parameters
        ----------
        mean_order : int, optional
            Mean block size. The default is 1.
        """
        self.npoints //= mean_order
        self._construct_grids()
        self.field = \
            self.field.reshape(
                (self.npoints, mean_order, self.npoints, mean_order)
            ).mean((1, 3))
        self.spectrum = self._fft2(self.field)

    def propagate(self, z: float):
        r"""A field propagation with Fourier transformation to the distance `z`.

        Parameters
        ----------
        z : float
            A propagation distance in centimetres.

        Notes
        -----
        With field as :math:`A` we can write in paraxial approximation

        .. math::A(z) = \int d^2k e^{-ikr - i k_z(r) z} \int A(0)e^{ikr}d^2r

        In discrete way we can describe it with FFT:
        >>> A(z) = iFFT(FFT(A(0)) * exp(- i*kz*z))

        :math:`k_z` must be greater than :math:`\max(k_x),\max(k_y)`
        """
        if z == 0:
            return

        self.z += z
        _deltak = self.Kz * z
        # clip to interval [-2pi, 0]
        phase = 2 * self.xp.pi * (self.xp.trunc(_deltak) - _deltak)
        # noticably faster than writing exp(1j*phase)
        self.spectrum *= self.xp.cos(phase) + 1j * self.xp.sin(phase)
        self.field = self._ifft2(self.spectrum)

    def lens(self, f: float):
        """Lens representated as a phase multiplicator.

        Parameters
        ----------
        f : float
            A focal length.

        Notes
        -----
        We can describe a field after the Lens :math:`A'(r)` as follows

        .. math::A'(r) = A(r) e^{ik_0 r^2/2f}

        Here :math:`A(r)` is a field before lens and phase multiplicator describes a lens.
        """
        phase = (self.X ** 2 + self.Y ** 2) * self.k0 / (2 * f)
        # noticably faster than writing exp(1j*phase)
        self.field *= self.xp.cos(phase) + 1j * self.xp.sin(phase)
        self.spectrum = self._fft2(self.field)

    def lens_image(self, f: float, l1: float, l2: float):
        """Image transmitting through the lens between optically conjugated planes.

        Parameters
        ----------
        f : float
            A focal length.
        l1 : float
            A distance before the lens in centimetres.
        l2 : float
            A distance after the lens in centimetres.
        """
        scale_factor = - l1 / l2
        X = self._np(self.X)
        fieldgen_real = interp2d(X, X, np.real(self._np(self.field)))
        fieldgen_imag = interp2d(X, X, np.imag(self._np(self.field)))
        phase = - (self.X ** 2 + self.Y ** 2) * \
            self.k0 * (l1 + l2) / 2 / l2 ** 2
        X *= scale_factor

        self.field = scale_factor * \
            self._asxp(fieldgen_real(X, X) + 1j * fieldgen_imag(X, X)) * \
            (self.xp.cos(phase) + 1j * self.xp.sin(phase))
        self.spectrum = self._fft2(self.field)

    def _expand_basis(self, modes_list):
        """Expand modes basis to the self.npoints.        

        Parameters
        ----------
        modes_list : Tuple[numpy.ndarray, cupy.ndarray, list]
            List of flattened modes. Unified with pyMMF

        Returns
        -------
        modes_list : self.xp.ndarray
            List of flattened modes. Unified with pyMMF.

        """
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
        return self.xp.array([m.reshape((Nb, Nb)) for m in modes_list])

    def deconstruct_by_modes(self, modes_list):
        r"""Return decomposed coefficients in given mode basis as a least-square solution.

        Here denoted :math:`\mathbf{M}(r)` is the given mode basis,
        :math:`\mathbf{C}` is modes coefficients

        Here the field :math:`A(r)` is described as

        .. math::A(r) = \sum_i C_i M_i (r)

        Where :math:`\mathbf{C}` is calculated as

        .. math::\mathbf{C} = LSTSQ(\mathbf{M}(r), A(r))

        Parameters
        ----------
        modes_list : Tuple[numpy.ndarray, cupy.ndarray, list]
            List of flattened modes. Unified with pyMMF

        Returns
        -------
        modes_list : self.xp.ndarray
            Modes coefficients.
        """
        modes_array = self._asxp(modes_list)
        Nb = np.sqrt(len(modes_list[0])).astype(int)
        dN = (self.npoints - Nb) // 2
        modes_matrix = self.xp.vstack(modes_array).T
        flatten_field = \
            self.field[dN:self.npoints - dN,
                       dN:self.npoints - dN].flatten()
        self.modes_coeffs = self.xp.linalg.lstsq(
            modes_matrix, flatten_field, rcond=-1)[0]
        return self.modes_coeffs

    def fast_deconstruct_by_modes(self, modes_matrix_t,  modes_matrix_dot_t):
        r"""Return decomposed coefficients in given mode basis as a least-square solution. Fast version.

        Fast version with pre-computations
        Results can be a little different from `deconstruct_by_modes` ones
        because of full set of singular values is used.

        Parameters
        ----------
        modes_matrix_t : Tuple[numpy.ndarray, cupy.ndarray]
            Modes matrix. See Notes.

        modes_matrix_dot_t : Tuple[numpy.ndarray, cupy.ndarray]
            Linear system matrix. See Notes.

        Returns
        -------
        modes_list : self.xp.ndarray
            Modes coefficients.

        Notes
        -----
        If modes are flatten then modes_matrix_t is calculated as follows
            >>> modes_matrix = np.vstack(modes_list).T

        Linear system matrix is calculated so
            >>> modes_matrix.T.dot(modes_matrix)

        The field :math:`A(r)` is described as

        .. math::A(r)=\sum_i C_i M_i(r)

        Where :math:`\mathbf{C}` is calculated as

        .. math::\mathbf{C}=SOLVE(\mathbf{M}(r)^T\mathbf{M},A(r)) \equiv (\mathbf{M}(r)^T\mathbf{M})^{-1}A(r)
        """
        modes_matrix_t = self._asxp(modes_matrix_t)
        modes_matrix_dot_t = self._asxp(modes_matrix_dot_t)
        Nb = np.sqrt(modes_matrix_t.shape[1]).astype(int)
        dN = (self.npoints - Nb) // 2
        flatten_field = \
            self.field[dN:self.npoints - dN,
                       dN:self.npoints - dN].flatten()
        self.modes_coeffs = self.xp.linalg.lstsq(
            modes_matrix_dot_t, modes_matrix_t.dot(flatten_field), rcond=-1)[0]
        return self.modes_coeffs

    def construct_by_modes(self, modes_list, modes_coeffs):
        """Construct self.field from the given modes and modes coefficients.

        Parameters
        ----------
        modes_list : Tuple[numpy.ndarray, cupy.ndarray, list]
            List of flattened modes. Unified with pyMMF.
        modes_coeffs : self.xp.ndarray
            Modes coefficients.

        """
        modes_list_reshape = self._expand_basis(modes_list)
        self.field = self.xp.einsum(
            'ijk,i->jk', modes_list_reshape, modes_coeffs)
        self.spectrum = self._fft2(self.field)

    @property
    def centroid(self):
        """Return the centroid of the intensity distribution.

        The centroid is the arithmetic mean of all points weighted by the intensity profile.

        Returns
        -------
        Centroid coordinates : Tuple[float, float, int, int]
            The coordinates and the closests array indices of the centroid (Xc, Yc, nxc, nyc).
        """
        n = self.npoints // 2
        Y, X = self.xp.mgrid[-n:n, -n:n] * self.dL
        I = self.iprofile
        Xc = self.xp.average(X, weights=I)
        Yc = self.xp.average(Y, weights=I)
        return Xc, Yc, int(Xc / self.dL) + n, int(Yc / self.dL) + n

    @property
    def D4sigma(self):
        r"""Return the width :math:`D=4\sigma` of the intensity distribution.

        Returns
        -------
        D4sigma : (float, float)
            Diameter of the beam by x and y axes.
        """
        n = self.npoints // 2
        I = self.iprofile
        Y, X = self.xp.mgrid[-n:n, -n:n] * self.dL
        Xc, Yc, _, _ = self.centroid

        def sigma(x, xc): return self.xp.sqrt(
            self.xp.average((x-xc) ** 2, weights=I)).tolist()

        return 4 * sigma(X, Xc), 4 * sigma(Y, Yc)

    @property
    def iprofile(self):
        """Return the intensity profile of the field A

        .. math::I(r)=|A(r)|^2

        Returns
        -------
        iprofile : self.xp.ndarray
            Intensity profile of the field A
        """
        profile = self.xp.abs(self.field) ** 2
        if self.numpy_output:
            return self._np(profile)
        else:
            return profile

    @property
    def phiprofile(self):
        """Return the phase profile of the field A

        .. math::\varphi(r)=\text{arg}(A(r))

        Returns
        -------
        iprofile : self.xp.ndarray
            Phase profile of the field A
        """
        profile = self.xp.angle(self.field)
        if self.numpy_output:
            return self._np(profile)
        else:
            return profile

    @property
    def centroid_intensity(self):
        """Return the intensity value in the centroid coordinates.

        .. math::I_c=|A(Xc,Yc)|^2

        Returns
        -------
        centroid_intensity : float
            The light intensity value in the centroid coordinates.
        """
        _, _, nxc, nyc = self.centroid
        return self.iprofile[nyc, nxc].tolist()

    def __repr__(self):
        return (f"Beam {self.npoints:d}x{self.npoints:d} points" +
                f"\n\t{self.area_size:.3g}x{self.area_size:.3g} cm " +
                f"\n\t<wl={self.wl / nm:.3g} nm, z={self.z:.3g} cm>")
