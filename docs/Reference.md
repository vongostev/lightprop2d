# 

The algorithm based on the paper
Delen, N., & Hooker, B. (1998).
Free-space beam propagation between arbitrarily oriented planes
based on full diffraction theory : a fast Fourier transform approach.
JOSA A, 15(4), 857-867.


# Beam2D 

```python
class Beam2D
```

Electromagnetic field propagation using spectral method.

Simple class to transform intitial field distribution using 2D fourier 
transformation from x-y field profile to kx-ky spectrum.
You can use both numpy and cupy backends with use_gpu key of the class.

| Parameters    | Type             | Doc             |
|:-------|:-----------------|:----------------|
|     area_size | float |         The beam calculation area size in centimetres. | 
|     npoints | int |         Number of points by one axes. | 
|     wl | float |         Beam central wavelength in centimetres. | 
|     z | float = 0. |         Propagation distance in centimetres. | 
|     xp | object = np |         Backend module. numpy (np) or cupy (cp). Controlled by 'use_gpu' key | 
|     init_field | xp.ndarray = None |         Initial field distribution given as an array | 
|     init_field_gen | object = None |         Initial field distribution given as a generating function | 
|     init_gen_args | tuple = () |         Additional arguments of 'init_field_gen' excluding        the first two (X grid and Y grid) | 
|     complex_bits | int = 128 |         Precision of complex numbers. Can be 64 or 128 | 
|     use_gpu | bool = False |         Backend choice.        If True, the class uses cupy backend with GPU support.        If False, the class uses numpy backend | 


--------- 

## Methods 

 
| method    | Doc             |
|:-------|:----------------|
| _np | Convert cupy or numpy arrays to numpy array. | 
| _asxp | Convert cupy or numpy arrays to self.xp array. | 
| _k_grid | Return a grid for Kx or Ky values. | 
| _fft2 | 2D FFT alias with a choice of the fft module. | 
| _ifft2 | 2D Inverse FFT alias with a choice of the fft module. | 
| _update_obj | Fast updating of the beam field and spectrum. | 
| _construct_grids | Construction of X, Y, Kx, Ky grids. | 
| coordinate_filter | Apply a mask to the field profile. | 
| spectral_filter | Apply a mask to the field spectrum. | 
| expand | Expand the beam calculation area to the given area_size. | 
| crop | Crop the field to the new area_size smaller than actual. | 
| coarse | Decrease `self.npoints` with a divider `mean_order`. | 
| propagate | A field propagation with Fourier transformation to the distance `z`. | 
| lens | Lens representated as a phase multiplicator. | 
| lens_image | Image transmitting through the lens between optically conjugated planes. | 
| _expand_basis | Expand modes basis to the self.npoints. | 
| deconstruct_by_modes | Return decomposed coefficients in given mode basis as a least-square solution. | 
| fast_deconstruct_by_modes | Return decomposed coefficients in given mode basis as a least-square solution. Fast version. | 
| construct_by_modes | Construct self.field from the given modes and modes coefficients. | 
| centroid | Return the centroid of the intensity distribution. | 
| D4sigma | Return the width :math:`D=4\sigma` of the intensity distribution. | 
| iprofile | Return the intensity profile of the field . | 
| phiprofile | Return the phase profile of the field . | 
| centroid_intensity | Return the intensity value in the centroid coordinates. | 
 
 

### _np

```python
   _np(data)
```


Convert cupy or numpy arrays to numpy array.

| Parameters    | Type             | Doc             |
|:-------|:-----------------|:----------------|
|         data | Tuple[numpy.ndarray, cupy.ndarray] |             Input data. | 


| Returns    | Type             | Doc             |
|:-------|:-----------------|:----------------|
|         data | numpy.ndarray |             Converted data. | 


### _asxp

```python
   _asxp(data)
```


Convert cupy or numpy arrays to self.xp array.

| Parameters    | Type             | Doc             |
|:-------|:-----------------|:----------------|
|         data | Tuple[numpy.ndarray, cupy.ndarray] |             Input data. | 


| Returns    | Type             | Doc             |
|:-------|:-----------------|:----------------|
|         data | self.xp.ndarray |             Converted data. | 


### _k_grid

```python
   _k_grid(dL: float, npoints: int)
```


Return a grid for Kx or Ky values.

| Parameters    | Type             | Doc             |
|:-------|:-----------------|:----------------|
|         dL | float |             Spatial delta. | 
|         npoints | int |             Number of points. | 


| Returns    | Type             | Doc             |
|:-------|:-----------------|:----------------|
|         grid | self.xp.ndarray |             The grid for K values. | 


### _fft2

```python
   _fft2(data)
```


2D FFT alias with a choice of the fft module.

| Parameters    | Type             | Doc             |
|:-------|:-----------------|:----------------|
|         data | self.xp.ndarray |             2d signal data with type of self.complex. | 


### _ifft2

```python
   _ifft2(data)
```


2D Inverse FFT alias with a choice of the fft module.

| Parameters    | Type             | Doc             |
|:-------|:-----------------|:----------------|
|         data | Tuple[numpy.ndarray, cupy.ndarray] |             2d signal data with type of self.complex. | 


### _update_obj

```python
   _update_obj(field, spectrum=None)
```


Fast updating of the beam field and spectrum.

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

| Parameters    | Type             | Doc             |
|:-------|:-----------------|:----------------|
|         field | Tuple[numpy.ndarray, cupy.ndarray] |             Field distribuion of complex type. | 
|         spectrum | Tuple[numpy.ndarray, cupy.ndarray], optional |             Field spatial spectrum of complex type. The default is None. | 


| Returns    | Type             | Doc             |
|:-------|:-----------------|:----------------|


### _construct_grids

```python
   _construct_grids()
```


Construction of X, Y, Kx, Ky grids.

### coordinate_filter

```python
   coordinate_filter(f_init=None, f_gen=None, fargs=())
```


Apply a mask to the field profile.

| Parameters    | Type             | Doc             |
|:-------|:-----------------|:----------------|
|         f_init | Tuple[numpy.ndarray, cupy.ndarray], optional |             A mask as an array. The default is None. | 
|         f_gen | function, optional |             A function to generate a mask. The default is None. | 
|         fargs | tuple, optional |             Additional arguments of f_gen function. The default is (). | 


------- 

#### Notes

The mask function `f_gen` can be user defined and must be in form:


```python
    >>> func(X, Y, *fargs)
```


Where X, Y are 1D grids:


```python
    >>> X = arange(-npoints // 2, npoints // 2, 1) * dL
    >>> Y = X.reshape((-1, 1))
```


For example see **lightprop2d.gaussian_beam**


### spectral_filter

```python
   spectral_filter(f_init=None, f_gen=None, fargs=())
```


Apply a mask to the field spectrum.

| Parameters    | Type             | Doc             |
|:-------|:-----------------|:----------------|
|         f_init | Tuple[numpy.ndarray, cupy.ndarray], optional |             A mask as an array. The default is None. | 
|         f_gen | function, optional |             A function to generate a mask. The default is None.            The mask function can be user defined and must be in form            >>> func(Kx, Ky, *fargs)            Where Kx, Ky are 1D grids            >>> Kx = fftfreq(npoints, d=dL)            >>> Ky = Kx.reshape((-1, 1)) | 
|         fargs | tuple, optional |             Additional arguments of f_gen function. The default is (). | 


### expand

```python
   expand(area_size: float)
```


Expand the beam calculation area to the given area_size.

with proportional `self.npoints` increasing.
`self.dL` remains constant.

| Parameters    | Type             | Doc             |
|:-------|:-----------------|:----------------|
|         area_size | float |             Wanted area size in centimetres. | 


### crop

```python
   crop(area_size: float, npoints: int = 0)
```


Crop the field to the new area_size smaller than actual.

with proportional `self.npoints` decreasing.
`self.dL` remains constant.

| Parameters    | Type             | Doc             |
|:-------|:-----------------|:----------------|
|         area_size | float |             A size of the calculation area in centimetres. | 
|         npoints | int, optional |             A number of points in one dimention.            The default is 0 -- number of points isn't changed. | 


### coarse

```python
   coarse(mean_order: int = 1)
```


Decrease `self.npoints` with a divider `mean_order`.

Block average applies to `self.field` with size of blocks as `mean_order*mean_order`.
`self.spectrum` is calculated from the averaged `self.field`.
It is necessary to decrease numerical complexity when propagation
distance is huge and the beam radius grows dramatically.
Recommended to use it after `self.expand`. For example
```python
>>> beam.expand(self.area_size * 2)
>>> beam.coarse(2)
```

| Parameters    | Type             | Doc             |
|:-------|:-----------------|:----------------|
|         mean_order | int, optional |             Mean block size. The default is 1. | 


### propagate

```python
   propagate(z: float)
```


A field propagation with Fourier transformation to the distance `z`.

| Parameters    | Type             | Doc             |
|:-------|:-----------------|:----------------|
|         z | float |             A propagation distance in centimetres. | 


------- 

#### Notes

With field as <img src="https://render.githubusercontent.com/render/math?math=A"> we can write in paraxial approximation

<img src="https://render.githubusercontent.com/render/math?math=A(z) = \int d^2k e^{-ikr - i k_z(r) z} \int A(0)e^{ikr}d^2r">

In discrete way we can describe it with FFT:


```python
    >>> A(z) = iFFT(FFT(A(0)) * exp(- i*kz*z))
```


<img src="https://render.githubusercontent.com/render/math?math=k_z"> must be greater than <img src="https://render.githubusercontent.com/render/math?math=\max(k_x),\max(k_y)">


### lens

```python
   lens(f: float)
```


Lens representated as a phase multiplicator.

| Parameters    | Type             | Doc             |
|:-------|:-----------------|:----------------|
|         f | float |             A focal length. | 


------- 

#### Notes

We can describe a field after the Lens <img src="https://render.githubusercontent.com/render/math?math=A'(r)"> as follows

<img src="https://render.githubusercontent.com/render/math?math=A'(r) = A(r) e^{ik_0 r^2/2f}">

Here <img src="https://render.githubusercontent.com/render/math?math=A(r)"> is a field before lens and phase multiplicator describes a lens.




### lens_image

```python
   lens_image(f: float, l1: float, l2: float)
```


Image transmitting through the lens between optically conjugated planes.

| Parameters    | Type             | Doc             |
|:-------|:-----------------|:----------------|
|         f | float |             A focal length. | 
|         l1 | float |             A distance before the lens in centimetres. | 
|         l2 | float |             A distance after the lens in centimetres. | 


### _expand_basis

```python
   _expand_basis(modes_list)
```


Expand modes basis to the self.npoints.

| Parameters    | Type             | Doc             |
|:-------|:-----------------|:----------------|
|         modes_list | Tuple[numpy.ndarray, cupy.ndarray, list] |             List of flattened modes. Unified with pyMMF | 


| Returns    | Type             | Doc             |
|:-------|:-----------------|:----------------|
|         modes_list | self.xp.ndarray |             List of flattened modes. Unified with pyMMF. | 


### deconstruct_by_modes

```python
   deconstruct_by_modes(modes_list)
```


Return decomposed coefficients in given mode basis as a least-square solution.

Here denoted <img src="https://render.githubusercontent.com/render/math?math=\mathbf{M}(r)"> is the given mode basis,
<img src="https://render.githubusercontent.com/render/math?math=\mathbf{C}"> is modes coefficients
Here the field <img src="https://render.githubusercontent.com/render/math?math=A(r)"> is described as

<img src="https://render.githubusercontent.com/render/math?math=A(r) = \sum_i C_i M_i (r)">

Where <img src="https://render.githubusercontent.com/render/math?math=\mathbf{C}"> is calculated as

<img src="https://render.githubusercontent.com/render/math?math=\mathbf{C} = LSTSQ(\mathbf{M}(r), A(r))">


| Parameters    | Type             | Doc             |
|:-------|:-----------------|:----------------|
|         modes_list | Tuple[numpy.ndarray, cupy.ndarray, list] |             List of flattened modes. Unified with pyMMF | 


| Returns    | Type             | Doc             |
|:-------|:-----------------|:----------------|
|         modes_list | self.xp.ndarray |             Modes coefficients. | 


### fast_deconstruct_by_modes

```python
   fast_deconstruct_by_modes(modes_matrix_t,  modes_matrix_dot_t)
```


Return decomposed coefficients in given mode basis as a least-square solution. Fast version.

Fast version with pre-computations
Results can be a little different from `deconstruct_by_modes` ones
because of full set of singular values is used.

| Parameters    | Type             | Doc             |
|:-------|:-----------------|:----------------|
|         modes_matrix_t | Tuple[numpy.ndarray, cupy.ndarray] |             Modes matrix. See Notes. | 
|         modes_matrix_dot_t | Tuple[numpy.ndarray, cupy.ndarray] |             Linear system matrix. See Notes. | 


| Returns    | Type             | Doc             |
|:-------|:-----------------|:----------------|
|         modes_list | self.xp.ndarray |             Modes coefficients. | 


------- 

#### Notes

If modes are flatten then modes_matrix_t is calculated as follows

```python
        >>> modes_matrix = np.vstack(modes_list).T
```


Linear system matrix is calculated so

```python
        >>> modes_matrix.T.dot(modes_matrix)
```


The field <img src="https://render.githubusercontent.com/render/math?math=A(r)"> is described as

<img src="https://render.githubusercontent.com/render/math?math=A(r)=\sum_i C_i M_i(r)">

Where <img src="https://render.githubusercontent.com/render/math?math=\mathbf{C}"> is calculated as

<img src="https://render.githubusercontent.com/render/math?math=\mathbf{C}=SOLVE(\mathbf{M}(r)^T\mathbf{M},A(r)) \equiv (\mathbf{M}(r)^T\mathbf{M})^{-1}A(r)">



### construct_by_modes

```python
   construct_by_modes(modes_list, modes_coeffs)
```


Construct self.field from the given modes and modes coefficients.

| Parameters    | Type             | Doc             |
|:-------|:-----------------|:----------------|
|         modes_list | Tuple[numpy.ndarray, cupy.ndarray, list] |             List of flattened modes. Unified with pyMMF. | 
|         modes_coeffs | self.xp.ndarray |             Modes coefficients. | 


### centroid

```python
   centroid()
```


Return the centroid of the intensity distribution.

The centroid is the arithmetic mean of all points weighted by the intensity profile.

| Returns    | Type             | Doc             |
|:-------|:-----------------|:----------------|
|         Centroid coordinates | Tuple[float, float, int, int] |             The coordinates and the closests array indices of the centroid (Xc, Yc, nxc, nyc). | 


### D4sigma

```python
   D4sigma()
```


Return the width <img src="https://render.githubusercontent.com/render/math?math=D=4\sigma"> of the intensity distribution.

| Returns    | Type             | Doc             |
|:-------|:-----------------|:----------------|
|         D4sigma | (float, float) |             Diameter of the beam by x and y axes. | 


### iprofile

```python
   iprofile()
```


Return the intensity profile of the field .


<img src="https://render.githubusercontent.com/render/math?math=I(r)=|A(r)|^2">


| Returns    | Type             | Doc             |
|:-------|:-----------------|:----------------|
|         iprofile | self.xp.ndarray |             Intensity profile of the field A | 


### phiprofile

```python
   phiprofile()
```


Return the phase profile of the field .


<img src="https://render.githubusercontent.com/render/math?math=\varphi(r)=\text{arg}(A(r))">


| Returns    | Type             | Doc             |
|:-------|:-----------------|:----------------|
|         iprofile | self.xp.ndarray |             Phase profile of the field A | 


### centroid_intensity

```python
   centroid_intensity()
```


Return the intensity value in the centroid coordinates.


<img src="https://render.githubusercontent.com/render/math?math=I_c=|A(Xc,Yc)|^2">


| Returns    | Type             | Doc             |
|:-------|:-----------------|:----------------|
|         centroid_intensity | float |             The light intensity value in the centroid coordinates. | 
