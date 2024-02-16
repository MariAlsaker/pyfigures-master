import my_classes.MRIcoil as MRIcoil
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.colors import to_hex
from stl import mesh
import magpylib as magpy
from scipy.optimize import curve_fit
from scipy.ndimage import rotate
import utils

center = (0,0)
radius = 10.3
x = np.linspace(center[0], center[0]+radius, 101, endpoint=True)
arc = np.array([np.sqrt(radius**2 - (x-center[0])**2)+center[1], x])
rot_angle = np.radians(90+45)
cos, sin = np.cos(rot_angle), np.sin(rot_angle)
rot_matrix = np.array([[cos, sin], [-np.sin(rot_angle), np.cos(rot_angle)]])
new_arc = np.dot(rot_matrix, arc)
num_points = 11
x_points = np.linspace(new_arc[0][0], new_arc[0][-1], num_points, endpoint=True)
arc_plot = np.zeros((2, num_points))
for i, x in enumerate(x_points):
    print(new_arc[0])
    print(x_points)
    index = np.where(new_arc[0]>=x)[0][0]
    keep_x, keep_y = new_arc[0][index], new_arc[1][index]
    arc_plot[0][i] = keep_x
    arc_plot[1][i] = keep_y
print(arc_plot)
plt.figure(figsize=[5,5])
plt.vlines([0], ymin=-radius, ymax=radius, colors=["k"])
plt.hlines([0], xmin=-radius, xmax=radius, colors=["k"])
plt.plot(arc[0], arc[1])
plt.plot(new_arc[0], new_arc[1])
plt.plot(arc_plot[0], arc_plot[1])
plt.show()