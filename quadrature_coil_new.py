
import my_classes.MRIcoil as MRIcoil
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.colors import to_hex
from stl import mesh
import magpylib as magpy
from scipy.optimize import curve_fit
from scipy.ndimage import rotate
import utils

# Creating my arc in 2D
final_arc=utils.create_2D_arc(radius = 103)

arc_is = [(0, 11), (8, -1)]
vertices_2coils = np.zeros((2, 4, 3, 11))
for i, arc_i in enumerate(arc_is):
    # vert_negarc = vertices_2coils[i][0]
    # vert_negpos = vertices_2coils[i][1]
    # vert_posarc = vertices_2coils[i][2]
    # vert_posneg = vertices_2coils[i][3]
    vertices_2coils[i][0], vertices_2coils[i][2] = np.zeros((3,11)), np.zeros((3,11)) # fra minus til pluss, fra pluss til minus
    vertices_2coils[i][0][0], vertices_2coils[i][0][1], vertices_2coils[i][0][2] = np.ones((11))*-42.5, final_arc[0][arc_i[0]:arc_i[1]], final_arc[1][arc_i[0]:arc_i[1]]
    vertices_2coils[i][2][0], vertices_2coils[i][2][1], vertices_2coils[i][2][2] = np.ones((11))*42.5, np.flip(final_arc[0][arc_i[0]:arc_i[1]]), np.flip(final_arc[1][arc_i[0]:arc_i[1]])
    vertices_2coils[i][1], vertices_2coils[i][3] = np.zeros((3,11)), np.zeros((3,11))
    vertices_2coils[i][1][0], vertices_2coils[i][3][0] = np.linspace(-42.5, 42.5, 11, endpoint=True), np.linspace(42.5, -42.5, 11, endpoint=True)
    vertices_2coils[i][1][1], vertices_2coils[i][3][1] = np.ones((11))*vertices_2coils[i][0][1][-1], np.ones((11))*vertices_2coils[i][2][1][-1]
    vertices_2coils[i][1][2], vertices_2coils[i][3][2] = np.ones((11))*vertices_2coils[i][0][2][-1], np.ones((11))*vertices_2coils[i][2][2][-1]

curr_lines_2coils = [[], []]
currents = [400*np.cos(np.pi/4), 400*np.cos(np.pi/4 + np.pi/2)]
for i, vertices in enumerate(vertices_2coils):
    for j, line in enumerate(vertices):
        this_line =  magpy.current.Line(currents[i], line.transpose())
        this_line.current
        curr_lines_2coils[i].append(this_line)
coil1 = magpy.Collection(curr_lines_2coils[0])
coil2 = magpy.Collection(curr_lines_2coils[1])
both_coils = magpy.Collection((coil1, coil2), override_parent=True)
both_coils.move((0,0,60))

figured_coil=both_coils
# Create figure
fig1 = plt.figure(figsize=[5, 5])
extent_xy, z = 100, 0
ax1 = fig1.add_subplot(1,1,1, projection="3d")
magpy.show(figured_coil, backend='matplotlib', return_fig=False, canvas=ax1)
ax1.view_init(15, 20)
ax1.set_xlabel("z (mm)")
ax1.set_ylabel("x (mm)")
ax1.set_zlabel("y (mm)")
ax1.get_legend().remove()
# plt.savefig("quad_coil_mpl", dpi=300)
# plt.close("all")

fig2 = plt.figure(figsize=[6, 5])
ax2 = fig2.add_subplot(1,1,1)
X, Y = np.mgrid[-extent_xy:extent_xy:100j, -extent_xy:extent_xy:100j].transpose((0,2,1))
grid = np.stack([np.zeros((100,100)) + z, X, Y], axis=2)
B_field = magpy.getB(figured_coil, observers=grid)
utils.show_field_magn(grid_slice=grid, coil=figured_coil, ax=ax2, fig=fig2, vmax=8)
ax2.set_xlabel("x (mm)")
ax2.set_ylabel("y (mm)")
ticks=np.linspace(0, 100, 5)
tick_ls=np.linspace(-extent_xy, extent_xy, 5)
labels = [f"{lab:.0f}" for lab in tick_ls]
ax2.set_xticks(ticks, labels)
ax2.set_yticks(ticks, labels)
# plt.savefig(f"quad_fieldmagn", dpi=300)
# plt.close("all")

fig3 = plt.figure(figsize=[6, 5])
ax3 = fig3.add_subplot(1,1,1)
X, Y = np.mgrid[-extent_xy:extent_xy:100j, -extent_xy:extent_xy:100j].transpose((0,2,1))
grid = np.stack([ np.zeros((100,100)) + z, X, Y], axis=2)
B_field = magpy.getB(figured_coil, observers=grid)
utils.show_field_lines(grid_slice=grid, B_field=B_field, ax=ax3, fig=fig2, slicein="x")
# plt.savefig(f"quad_fieldlines", dpi=300)
# plt.close("all")

plt.show()