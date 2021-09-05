# 

The algorithm based on the paper
Delen, N., & Hooker, B. (1998).
Free-space beam propagation between arbitrarily oriented planes
based on full diffraction theory : a fast Fourier transform approach.
JOSA A, 15(4), 857-867.


# Beam2D 

``` python 
 class Beam2D 
```

2D field propagation.

Simple class to transform intitial field distribution using fourier transformation
from x-y field profile to kx-ky spectrum.
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
|     init_gen_args | tuple = () |         Additional arguments of 'init_field_gen' excluding first two (X grid and Y grid) | 
|     complex_bits | int = 128 |         Precision of complex numbers. Can be 64 or 128 | 
|     use_gpu | bool = False |         Backend choice.        If True, the class uses cupy backend with GPU support.        If False, the class uses numpy backend | 
|     unsafe_fft | bool = False |         Check physical correctness of spectrum calculations            'Critical K⟂  must be bigger than self.npoints // 2'        If True, this check is disabled. | 


--------- 

## Methods 

 
| method    | Doc             |
|:-------|:----------------|
| _np | Convert cupy or numpy arrays to numpy array. | 
| _xp | Convert cupy or numpy arrays to self.xp array. | 
| _k_grid | Make a grid for Kx or Ky value. | 
| _fft2 | 2D FFT alias with a choise of the fft module. | 
| _ifft2 | 2D Inverse FFT alias with a choise of the fft module. | 
| _construct_grids | Construction of X, Y, Kx, Ky grids. | 
| coordinate_filter | Apply a mask to the field profile. | 
| spectral_filter | Apply a mask to the field spectrum. | 
| expand | Expand the beam calculation area to the given area_size. | 
| crop | Crop the field to the new area_size smaller than actual. | 
| propagate | A field propagation with Fourier transformation. | 
| lens | Lens representated as a phase multiplicator. | 
| lens_image | Image transmitting through the lens between optically conjugated planes. | 
| _expand_basis | Expand modes basis to the self.npoints. | 
| deconstruct_by_modes | Return decomposed coefficients :math:`\mathbf{C}` in given mode basis :math:`\mathbf{M}(r). | 
| fast_deconstruct_by_modes | Return decomposed coefficients in given mode basis as least-square solution. | 
| construct_by_modes | self.field construction from the givn modes and coefficients. | 
| centroid | Returns the centroid of the intensity distribution. | 
| D4sigma | The width :math:`D=4\sigma`of the intensity distribution. | 
| iprofile | Intensity profile of the field . | 
| centroid_intensity | Intensity value in the centroid coordinates. | 
 
 

### _np

``` python 
    _np(data) 
```


Convert cupy or numpy arrays to numpy array.

| Parameters    | Type             | Doc             |
|:-------|:-----------------|:----------------|
|         data | Tuple[numpy.ndarray, cupy.ndarray] |             Input data. | 


| Returns    | Type             | Doc             |
|:-------|:-----------------|:----------------|
|         data | numpy.ndarray |             Converted data. | 


### _xp

``` python 
    _xp(data) 
```


Convert cupy or numpy arrays to self.xp array.

| Parameters    | Type             | Doc             |
|:-------|:-----------------|:----------------|
|         data | Tuple[numpy.ndarray, cupy.ndarray] |             Input data. | 


| Returns    | Type             | Doc             |
|:-------|:-----------------|:----------------|
|         data | self.xp.ndarray |             Converted data. | 


### _k_grid

``` python 
    _k_grid(dL: float, npoints: int) 
```


Make a grid for Kx or Ky value.

| Parameters    | Type             | Doc             |
|:-------|:-----------------|:----------------|
|         dL | float |             Spatial delta. | 
|         npoints | int |             Number of points. | 


| Returns    | Type             | Doc             |
|:-------|:-----------------|:----------------|
|         grid | self.xp.ndarray |             The grid for K values. | 


### _fft2

``` python 
    _fft2(data) 
```


2D FFT alias with a choise of the fft module.

| Parameters    | Type             | Doc             |
|:-------|:-----------------|:----------------|
|         data | self.xp.ndarray |             2d signal data with type of self.complex. | 


### _ifft2

``` python 
    _ifft2(data) 
```


2D Inverse FFT alias with a choise of the fft module.

| Parameters    | Type             | Doc             |
|:-------|:-----------------|:----------------|
|         data | self.xp.ndarray |             2d signal data with type of self.complex. | 


### _construct_grids

``` python 
    _construct_grids() 
```


Construction of X, Y, Kx, Ky grids.

### coordinate_filter

``` python 
    coordinate_filter(f_init=None, f_gen=None, fargs=()) 
```


Apply a mask to the field profile.

| Parameters    | Type             | Doc             |
|:-------|:-----------------|:----------------|
|         f_init | numpy.ndarray, cupy.ndarray, optional |             A mask as an array. The default is None. | 
|         f_gen | function, optional |             A function to generate a mask. The default is None. | 
|         fargs | tuple, optional |             Additional arguments of f_gen function. The default is (). | 


------- 

#### Notes

The mask function `f_gen` can be user defined and must be in form

``` python 
             >>> func(X, Y, *fargs)
 
```


Where X, Y are 1D grids

``` python 
             >>> X = arange(-npoints // 2, npoints // 2, 1) * dL
            >>> Y = X.reshape((-1, 1))
 
```


For example see **lightprop2d.gaussian_beam**


### spectral_filter

``` python 
    spectral_filter(f_init=None, f_gen=None, fargs=()) 
```


Apply a mask to the field spectrum.

| Parameters    | Type             | Doc             |
|:-------|:-----------------|:----------------|
|         f_init | numpy.ndarray, cupy.ndarray, optional |             A mask as an array. The default is None. | 
|         f_gen | function, optional |             A function to generate a mask. The default is None.            The mask function can be user defined and must be in form                >>> func(Kx, Ky, *fargs)            Where Kx, Ky are 1D grids                >>> Kx = fftfreq(npoints, d=dL)                >>> Ky = Kx.reshape((-1, 1)) | 
|         fargs | tuple, optional |             Additional arguments of f_gen function. The default is (). | 


### expand

``` python 
    expand(area_size: float) 
```


Expand the beam calculation area to the given area_size.

| Parameters    | Type             | Doc             |
|:-------|:-----------------|:----------------|
|         area_size | float |             Wanted area size in centimetres. | 


### crop

``` python 
    crop(area_size: float, npoints: int = 0) 
```


Crop the field to the new area_size smaller than actual.

| Parameters    | Type             | Doc             |
|:-------|:-----------------|:----------------|
|         area_size | float |             A size of the calculation area in centimetres. | 
|         npoints | int, optional |             A number of points in one dimention.            The default is 0 -- number of points isn't changed. | 


### propagate

``` python 
    propagate(z: float) 
```


A field propagation with Fourier transformation.

| Parameters    | Type             | Doc             |
|:-------|:-----------------|:----------------|
|         z | float |             A propagation distance in centimetres. | 


------- 

#### Notes

With field as `A` we can write in paraxial approximation
.. math::A(z) = \int d^2k e^{-ikr - i k_z(r) z} \int A(0)e^{ikr}d^2r
In discrete way we can describe it with FFT:


``` python 
         >>> A(z) = iFFT(FFT(A(0)) * exp(- i*kz*z))
 
```


:math:`k_z` must be greater than :math:`max(k_x),max(k_y)`


### lens

``` python 
    lens(f: float) 
```


Lens representated as a phase multiplicator.

| Parameters    | Type             | Doc             |
|:-------|:-----------------|:----------------|
|         f | float |             A focal length. | 


------- 

#### Notes

We can describe a field after the Lens :math:`A'(r)` as follows
.. math::A'(r) = A(r) e^{ik_0 r^2/2f}
Here :math:`A(r)` is a field before lens and phase multiplicator describes a lens.




### lens_image

``` python 
    lens_image(f: float, l1: float, l2: float) 
```


Image transmitting through the lens between optically conjugated planes.

| Parameters    | Type             | Doc             |
|:-------|:-----------------|:----------------|
|         f | float |             A focal length. | 
|         l1 | float |             A distance before the lens in centimetres. | 
|         l2 | float |             A distance after the lens in centimetres. | 


### _expand_basis

``` python 
    _expand_basis(modes_list) 
```


Expand modes basis to the self.npoints.

| Parameters    | Type             | Doc             |
|:-------|:-----------------|:----------------|
|         modes_list | Tuple[numpy.ndarray, list] |             List of flattened modes. Unified with pyMMF | 


| Returns    | Type             | Doc             |
|:-------|:-----------------|:----------------|
|         modes_list | numpy.ndarray |             List of flattened modes. Unified with pyMMF. | 


### deconstruct_by_modes

``` python 
    deconstruct_by_modes(modes_list) 
```


Return decomposed coefficients :math:`\mathbf{C}` in given mode basis :math:`\mathbf{M}(r).

as a least-square solution
Here the field :math:`A(r)` is described as
.. math::A(r) = \sum_i C_i M_i (r)
Where :math:`\mathbf{C}` is calculated as
.. math::\mathbf{C} = LSTSQ(\mathbf{M}(r), A(r))

| Parameters    | Type             | Doc             |
|:-------|:-----------------|:----------------|
|         modes_list | Tuple[numpy.ndarray, list] |             List of flattened modes. Unified with pyMMF | 


| Returns    | Type             | Doc             |
|:-------|:-----------------|:----------------|
|         modes_list | self.xp.ndarray |             Modes coefficients. | 


### fast_deconstruct_by_modes

``` python 
    fast_deconstruct_by_modes(modes_matrix_t,  modes_matrix_dot_t) 
```


Return decomposed coefficients in given mode basis as least-square solution.

Fast version with pre-computations 
Results can be a little different from `deconstruct_by_modes` ones 
because of full set of singular values is used.

| Parameters    | Type             | Doc             |
|:-------|:-----------------|:----------------|
|         modes_matrix_t | numpy.ndarray |             Modes matrix. See Notes. | 
|         modes_matrix_dot_t | numpy.ndarray |             Linear system matrix. See Notes. | 


| Returns    | Type             | Doc             |
|:-------|:-----------------|:----------------|
|         modes_list | self.xp.ndarray |             Modes coefficients.             | 


------- 

#### Notes

If modes are flatten then modes_matrix_t is calculated as follows

``` python 
         >>> modes_matrix = np.vstack(modes_list).T
 
```


Linear system matrix is calculated so

``` python 
         >>> modes_matrix.T.dot(modes_matrix)
 
```



The field :math:`A(r)` is described as
.. math::A(r)=sum_i C_i M_i(r)
Where :math:`mathbf{C}` is calculated as
.. math: : mathbf{C} = SOLVE(mathbf{M}(r)^Tmathbf{M}, A(r)) equiv
(mathbf{M}(r)^Tmathbf{M})^{-1}A(r)


### construct_by_modes

``` python 
    construct_by_modes(modes_list, modes_coeffs) 
```


self.field construction from the givn modes and coefficients.

| Parameters    | Type             | Doc             |
|:-------|:-----------------|:----------------|
|         modes_list | Tuple[numpy.ndarray, list] |             List of flattened modes. Unified with pyMMF. | 
|         modes_coeffs | self.xp.ndarray |             Modes coefficients. | 


### centroid

``` python 
    centroid() 
```


Returns the centroid of the intensity distribution.

The centroid is the arithmetic mean of all points weighted by the intensity profile.

| Returns    | Type             | Doc             |
|:-------|:-----------------|:----------------|
|         Centroid coordinates | Tuple[float, float, int, int] |             The coordinates and the closests array indices of the centroid (Xc, Yc, nxc, nyc). | 


### D4sigma

``` python 
    D4sigma() 
```


The width :math:`D=4\sigma`of the intensity distribution.

| Returns    | Type             | Doc             |
|:-------|:-----------------|:----------------|
|         D4sigma | (float, float) |             Diameter of the beam by x and y axes. | 


### iprofile

``` python 
    iprofile() 
```


Intensity profile of the field .

.. math::I(r)=|A(r)|^2

| Returns    | Type             | Doc             |
|:-------|:-----------------|:----------------|
|         iprofile | self.xp.ndarray |             Intensity profile of the field A | 


### centroid_intensity

``` python 
    centroid_intensity() 
```


Intensity value in the centroid coordinates.

.. math::I_c=|A(Xc,Yc)|^2

| Returns    | Type             | Doc             |
|:-------|:-----------------|:----------------|
|         centroid_intensity | float |             Intensity value in the centroid coordinates. | 
