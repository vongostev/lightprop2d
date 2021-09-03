<a id="beam2d"></a>

# beam2d

The algorithm based on the paper
Delen, N., & Hooker, B. (1998).
Free-space beam propagation between arbitrarily oriented planes
based on full diffraction theory: a fast Fourier transform approach.
JOSA A, 15(4), 857-867.

<a id="beam2d.Beam2D"></a>

## Beam2D Objects

```python
@dataclass
class Beam2D()
```

<a id="beam2d.Beam2D.coordinate_filter"></a>

#### coordinate\_filter

```python
def coordinate_filter(f_init=None, f_gen=None, fargs=())
```

Apply a mask to the field profile.

Parameters
----------
f_init : numpy.ndarray, cupy.ndarray, optional
    A mask as an array. The default is None.
f_gen : function, optional
    A function to generate a mask. The default is None.
    The mask function can be user defined and must be in form
        >>> func(X, Y, *fargs)

    Where X, Y are 1D grids 
        >>> X = arange(-npoints // 2, npoints // 2, 1) * dL
        >>> Y = X.reshape((-1, 1))
    For example see **lightprop2d.gaussian_beam**
fargs : tuple, optional
    Additional arguments of f_gen function. The default is ().

<a id="beam2d.Beam2D.spectral_filter"></a>

#### spectral\_filter

```python
def spectral_filter(f_init=None, f_gen=None, fargs=())
```

Apply a mask to the field spectrum.

Parameters
----------
f_init : numpy.ndarray, cupy.ndarray, optional
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

<a id="beam2d.Beam2D.expand"></a>

#### expand

```python
def expand(area_size: float)
```

Expand the beam calculation area to the given area_size.

Parameters
----------
area_size : float
    Wanted area size in centimetres.

<a id="beam2d.Beam2D.crop"></a>

#### crop

```python
def crop(area_size: float, npoints: int = 0)
```

Crop the field to the new area_size smaller than actual.

Parameters
----------
area_size : float
    A size of the calculation area in centimetres.
npoints : int, optional
    A number of points in one dimention. 
    The default is 0 -- number of points isn't changed.

<a id="beam2d.Beam2D.propagate"></a>

#### propagate

```python
def propagate(z: float)
```

A field propagation with Fourier transformation.

With field as `A` we can write in paraxial approximation

.. math:: A(z) = \int d^2k e^{-ikr - i k_z(r) z} \int A(0)e^{ikr}d^2r

In discrete way we can describe it with FFT:

>>> A(z) = iFFT(FFT(A(0)) * exp(- i*kz*z))

:math:`k_z` must be greater than :math:`\max(k_x),\max(k_y)`

Parameters
----------
z : float
    A propagation distance in centimetres.

<a id="beam2d.Beam2D.lens"></a>

#### lens

```python
def lens(f: float)
```

Lens representated as a phase multiplicator.
We can describe a field after the Lens :math:`A'(r)` as follows

.. math:: A'(r) = A(r) e^{ik_0 r^2/2f}

Here :math:`A(r)` is a field before lens and phase multiplicator describes a lens.

Parameters
----------
f : float
    A focal length.

<a id="beam2d.Beam2D.lens_image"></a>

#### lens\_image

```python
def lens_image(f: float, l1: float, l2: float)
```

Image transmitting through the lens between optically conjugated planes.

Parameters
----------
f : float
    A focal length.
l1 : float
    A distance before the lens in centimetres.
l2 : float
    A distance after the lens in centimetres.

<a id="beam2d.Beam2D.deconstruct_by_modes"></a>

#### deconstruct\_by\_modes

```python
def deconstruct_by_modes(modes_list)
```

Return decomposed coefficients :math:`\mathbf{C}` in given mode basis :math:`\mathbf{M}(r)`
as a least-square solution

Here the field :math:`A(r)` is described as

.. math:: A(r) = \sum_i C_i M_i (r)

Where :math:`\mathbf{C}` is calculated as

.. math:: \mathbf{C} = LSTSQ(\mathbf{M}(r), A(r))

Parameters
----------
modes_list : iterable
    List of flattened modes. Unified with pyMMF

Returns
-------
ndarray
    Modes coefficients.

<a id="beam2d.Beam2D.fast_deconstruct_by_modes"></a>

#### fast\_deconstruct\_by\_modes

```python
def fast_deconstruct_by_modes(modes_matrix_t, modes_matrix_dot_t)
```

Return decomposed coefficients in given mode basis
as least-square solution.
Fast version with pre-computations 

Here the field :math:`A(r)` is described as

.. math:: A(r) = \sum_i C_i M_i (r)

Where :math:`\mathbf{C}` is calculated as

.. math:: \mathbf{C} = SOLVE(\mathbf{M}(r)^T\mathbf{M}, A(r)) \equiv
    (\mathbf{M}(r)^T\mathbf{M})^{-1}A(r)

Results can be a little different from `deconstruct_by_modes` ones 
because of full set of singular values is used.

Parameters
----------
modes_matrix_t : ndarray
    If modes are flatten then modes_matrix_t is calculated as follows:
        >>> modes_matrix = np.vstack(modes_list).T

modes_matrix_dot_t : ndarray
    Linear system matrix. It is calculated so:
        >>> modes_matrix.T.dot(modes_matrix)

Returns
-------
ndarray
    Modes coefficients.

<a id="beam2d.Beam2D.centroid"></a>

#### centroid

```python
@property
def centroid()
```

Returns the centroid of the intensity distribution.
The centroid is the arithmetic mean of all points weighted by the intensity profile.

Returns
-------
Tuple[float, float, int, int]
    The coordinates and the closests array indices of the centroid: Xc, Yc, nxc, nyc.

<a id="beam2d.Beam2D.D4sigma"></a>

#### D4sigma

```python
@property
def D4sigma()
```

Returns the width :math:`D=4\sigma`  of the intensity distribution.

<a id="beam2d.Beam2D.iprofile"></a>

#### iprofile

```python
@property
def iprofile()
```

Intensity profile of the field A

.. math:: I(r) = |A(r)|^2

<a id="beam2d.Beam2D.centroid_intensity"></a>

#### centroid\_intensity

```python
@property
def centroid_intensity()
```

Intensity value in the centroid coordinates.

.. math:: I_c = |A(Xc, Yc)|^2

<a id="profiles"></a>

# profiles

Created on Tue Jun 22 15:21:29 2021

@author: vonGostev

<a id="__init__"></a>

# \_\_init\_\_

Created on Tue Jun 22 16:11:32 2021

@author: von.gostev

