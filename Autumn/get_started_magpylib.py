# https://magpylib.readthedocs.io/en/latest/_pages/reso_get_started.html 
import numpy as np
import magpylib as magpy
from scipy.spatial.transform import Rotation as R

# Create a Cuboid magnet with magnetic polarization
# of 1000 mT pointing in x-direction and sides of
# 1,2 and 3 mm respectively.


cube = magpy.magnet.Cuboid(magnetization=(1000,0,0), dimension=(1,2,3))


# Create a Sensor for measuring the field

sensor = magpy.Sensor()

# By default, the position of a Magpylib object is
# (0,0,0) and its orientation is the unit rotation,
# given by a scipy rotation object.

print(cube.position)                   # -> [0. 0. 0.]
print(cube.orientation.as_rotvec())    # -> [0. 0. 0.]

# Manipulate object position and orientation through
# the respective attributes:

cube.position = (1,0,0)
cube.orientation = R.from_rotvec((0,0,45), degrees=True)

print(cube.position)                            # -> [1. 0.  0.]
print(cube.orientation.as_rotvec(degrees=True)) # -> [0. 0. 45.]

# Apply relative motion with the powerful `move`
# and `rotate` methods.
sensor.move((-1,0,0))
sensor.rotate_from_angax(angle=-45, axis='z')

print(sensor.position)                            # -> [-1.  0.  0.]
print(sensor.orientation.as_rotvec(degrees=True)) # -> [ 0.  0. -45.]

magpy.show(cube, sensor, backend='matplotlib')

# Compute the B-field in units of mT for some points.

points = [(0,0,-1), (0,0,0), (0,0,1)]
B = magpy.getB(cube, points)

print(B.round()) # -> [[263.  68.  81.]
                 #     [276.  52.   0.]
                 #     [263.  68. -81.]]

# Compute the H-field in units of kA/m at the sensor.

H = magpy.getH(cube, sensor)

print(H.round()) # -> [220.  41.   0.]
