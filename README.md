[![pypi](https://github.com/vongostev/lightprop2d/actions/workflows/python-publish.yml/badge.svg)](https://github.com/vongostev/lightprop2d/actions/workflows/python-publish.yml)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/vongostev/lightprop2d.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/vongostev/lightprop2d/context:python)

### Light propagation
Lightprop2d includes class 'Beam2D' to transform intitial field distribution
using fourier transform from x-y field profile to kx-ky spectrum.
You can use both numpy and cupy backends with `use_gpu` key of Beam2D class.

You can install it as follows
```
pip install lightprop2d==1.0rc3
```

Check out the [Code Reference](docs/Reference.md) for details on the 'Beam2D' methods.


### Example 1: Random beam propagation
```python
import matplotlib.pyplot as plt
from lightprop2d import Beam2D, random_round_hole, um, nm

# All input data are in cm
# XY grid dimensions
npoints = 256
# XY grid widening
beam_radius = 25*um # 25 um
area_size = 200*um # 200 um
# Wavelength in cm
wl0 = 632*nm

beam = Beam2D(area_size, npoints, wl0, init_field_gen=random_round_hole, 
              init_gen_args=(beam_radius,))
              
plt.imshow(beam.iprofile)
plt.show()

beam.propagate(100*um)

plt.imshow(beam.iprofile)
plt.show()
```
