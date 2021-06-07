[![lightprop2d](https://github.com/vongostev/lightprop2d/actions/workflows/python-package.yml/badge.svg)](https://github.com/vongostev/lightprop2d/actions/workflows/python-package.yml)
[![lightprop2d](https://github.com/vongostev/lightprop2d/actions/workflows/python-publish.yml/badge.svg)](https://github.com/vongostev/lightprop2d/actions/workflows/python-publish.yml)

### Light propagation
Lightprop2d includes class 'Beam2D' to transform intitial field distribution
using fourier transform from x-y field profile to kx-ky spectrum.

### Example 1: Random beam propagation
```python
import matplotlib.pyplot as plt
from lightprop2d import Beam2D, random_round_hole

# All input data are in cm
# XY grid dimensions
npoints = 256
# XY grid widening
beam_radius = 25e-4 # 25 um
area_size = 200e-4 # 200 um
# Wavelength in cm
wl0 = 632e-7

beam = Beam2D(area_size, npoints, wl0, init_field_gen=random_round_hole, 
              init_gen_args=(beam_radius,))
              
plt.imshow(beam.iprofile)
plt.show()

beam.propagate(100e-4)

plt.imshow(beam.iprofile)
plt.show()
```
