from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
import my_classes.MRIcoil as MRIcoil
from matplotlib.colors import to_hex
from stl import mesh
import magpylib as magpy
from scipy.optimize import curve_fit
from scipy.ndimage import rotate
import utils

cmap = mpl.colormaps.get_cmap("plasma")

d = 0.15#2*pdepth
pdepth= d/2#0.05 #m
s = np.sqrt(np.pi)*pdepth

t = np.linspace(0, 2*np.pi, 100)
x = d/2*np.cos(t)
y = d/2*np.sin(t)
fig, ax = plt.subplots(1, 1, figsize=(6,6))
ax.plot(x, y, color=cmap(0.1), linewidth=7.0, alpha=0.5)
s_array = np.linspace(-s/2, s/2, 4)
ax.hlines([-s/2, s/2], xmin=[-s/2, -s/2], xmax=[s/2, s/2], colors=cmap(0.5), linewidth=7.0)
ax.vlines([-s/2, s/2], ymin=[-s/2, -s/2], ymax=[s/2, s/2], colors=cmap(0.5), linewidth=7.0)
ax.scatter([-s/2, -s/2, s/2, s/2], [s/2, -s/2, s/2, -s/2], color=[cmap(0.5) for i in range(4)], s=25.0)
ax.axis("off")
#plt.savefig("circle_square_same_area.png", dpi=300)
plt.close("all")
#plt.show()

""" SQUARE SINGLE LOOP COIL """
s_cm = s*1000
# In the MRI machine we have the coil in the x-y-plane (but really the x-z plane)
slice = (5, -4)
vertices_coil = np.zeros((4, 3, 11))
for i, line in enumerate(vertices_coil):
    line[i%2] = np.linspace((-1)**(i)*s_cm/2, (-1)**(i+1)*s_cm/2, 11)
    line[1-i%2] = np.ones(shape=(1, 11))*s_cm/2
    if i<=1: 
        line[1-i%2] = line[1-i%2] *(-1)
        line[i%2] = np.flip(line[i%2])
curr_lines = []
for line in vertices_coil:
    curr_lines.append( magpy.current.Line(400, line.transpose()))
coil = magpy.Collection(curr_lines)
# Create figure
fig = plt.figure()
fig.set_figheight(10)
fig.set_figwidth(9)
spec = mpl.gridspec.GridSpec(ncols=2, nrows=3, wspace=0.5,
                         hspace=0.5, height_ratios=[2, 3, 3])
ax0 = fig.add_subplot(spec[0], projection="3d")
ax1 = fig.add_subplot(spec[1], projection="3d")
ax2 = fig.add_subplot(spec[2])
ax3 = fig.add_subplot(spec[3])
ax4 = fig.add_subplot(spec[4])
ax5 = fig.add_subplot(spec[5])
#fig.suptitle("Coils of same area")
magpy.show(coil, backend='matplotlib',return_fig=False, canvas=ax0)
ax0.set_xlabel("x (mm)")
ax0.set_ylabel("z (mm)")
ax0.set_zlabel("y (mm)")
ax0.get_legend().remove()

extent_xy = 80
X, Y = np.mgrid[-extent_xy:extent_xy:100j, -extent_xy:extent_xy:100j].transpose((0,2,1))
grid = np.stack([np.zeros((100,100)), X, Y], axis=2)
B_field = magpy.getB(coil, observers=grid)
utils.show_field_lines(grid, B_field, ax=ax2, fig=fig, slicein="x")
utils.show_field_magn(coil, grid, ax=ax4, fig=fig, vmax=8)

""" SQUARE SINGLE LOOP COIL """
d_mm = d*1000
coil1 = magpy.current.Loop(current=400, diameter=d_mm)
magpy.show(coil1, backend='matplotlib',return_fig=False, canvas=ax1)
ax1.set_xlabel("x (mm)")
ax1.set_ylabel("z (mm)")
ax1.set_zlabel("y (mm)")
ax1.get_legend().remove()
# surface_coil.show_field_magnitude("B", vmax=100)

B_field = magpy.getB(coil1, observers=grid)
utils.show_field_lines(grid, B_field, ax=ax3, fig=fig, slicein="x")
utils.show_field_magn(coil1, grid, ax=ax5, fig=fig, vmax=8)
plt.tight_layout(pad=0)
# plt.savefig("comparing_coil_shapes.png", dpi=300)
# plt.show()