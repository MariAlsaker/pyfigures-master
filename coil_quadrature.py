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
center = (0,0)
radius = 103
x = np.linspace(center[0], center[0]+radius, 101, endpoint=True)
arc = np.array([np.sqrt(radius**2 - (x-center[0])**2)+center[1], x])
rot_angle = np.radians(90+45)
cos, sin = np.cos(rot_angle), np.sin(rot_angle)
rot_matrix = np.array([[cos, sin], [-np.sin(rot_angle), np.cos(rot_angle)]])
new_arc = np.dot(rot_matrix, arc)
num_points = 20
x_points = np.linspace(new_arc[0][0], new_arc[0][-1], num_points, endpoint=True)
final_arc = np.zeros((2, num_points))
for i, x in enumerate(x_points):
    index = np.where(new_arc[0]>=x)[0][0]
    keep_x, keep_y = new_arc[0][index], new_arc[1][index]
    final_arc[0][i] = keep_x
    final_arc[1][i] = keep_y
print(final_arc.shape) # x, y = 0, 1

# In the MRI machine we have the coil in the x-y-plane (but really the x-z plane)
slice_start_end = [(0, 11), (8, -1)]
vertices_2coils = np.zeros((2, 4, 3, 11))
for i, slice in enumerate(slice_start_end):
    # vert_negarc = vertices_2coils[i][0]
    # vert_negpos = vertices_2coils[i][1]
    # vert_posarc = vertices_2coils[i][2]
    # vert_posneg = vertices_2coils[i][3]
    vertices_2coils[i][0], vertices_2coils[i][2] = np.zeros((3,11)), np.zeros((3,11)) # fra minus til pluss, fra pluss til minus
    vertices_2coils[i][0][0], vertices_2coils[i][0][1], vertices_2coils[i][0][2] = np.ones((11))*-42.5, final_arc[0][slice[0]:slice[1]], final_arc[1][slice[0]:slice[1]]
    vertices_2coils[i][2][0], vertices_2coils[i][2][1], vertices_2coils[i][2][2] = np.ones((11))*42.5, np.flip(final_arc[0][slice[0]:slice[1]]), np.flip(final_arc[1][slice[0]:slice[1]])
    vertices_2coils[i][1], vertices_2coils[i][3] = np.zeros((3,11)), np.zeros((3,11))
    vertices_2coils[i][1][0], vertices_2coils[i][3][0] = np.linspace(-42.5, 42.5, 11, endpoint=True), np.linspace(42.5, -42.5, 11, endpoint=True)
    vertices_2coils[i][1][1], vertices_2coils[i][3][1] = np.ones((11))*vertices_2coils[i][0][1][-1], np.ones((11))*vertices_2coils[i][2][1][-1]
    vertices_2coils[i][1][2], vertices_2coils[i][3][2] = np.ones((11))*vertices_2coils[i][0][2][-1], np.ones((11))*vertices_2coils[i][2][2][-1]

curr_lines_2coils = [[], []]
for i, vertices in enumerate(vertices_2coils):
    for j, line in enumerate(vertices):
        curr_lines_2coils[i].append( magpy.current.Line(100, line.transpose()))
coil1 = magpy.Collection(curr_lines_2coils[0])
coil2 = magpy.Collection(curr_lines_2coils[1])
both_coils = magpy.Collection((coil1, coil2))
both_coils.move((0,0,100))
# Define coil
quad_loop = MRIcoil.MRIcoil(current=90, diameter=100, custom_coil=True, custom_coil_current_line=both_coils)
# Create figure
fig2 = plt.figure(figsize=[12, 8])
fig2.suptitle("Quadrature coil (s=8.5cm) with field lines in slice z = 0, y = 0 and x = 0")
ax1 = fig2.add_subplot(2,2,1, projection="3d")
ax2 = fig2.add_subplot(2,2,2)
ax3 = fig2.add_subplot(2,2,3)
ax4 = fig2.add_subplot(2,2,4)
quad_loop.show_coil(ax=ax1)
quad_loop.show_field_lines(slice="z50", ax=ax2, fig=fig2)
quad_loop.show_field_lines(slice="x50", ax=ax3, fig=fig2)
ax3.hlines(y=[40], xmin=[-50], xmax=[50], colors=['k'], linestyles='dashed')
quad_loop.show_field_lines(slice="y50", ax=ax4, fig=fig2)
ax4.hlines(y=[40], xmin=[-50], xmax=[50], colors=['k'], linestyles='dashed')
X, Y, Z = utils.plane_at("z=40")
ax1.plot_surface(X, Y, Z, alpha=.3, label="x = 0, slice 50")

ax1.set_xlabel("z (mm)")
ax1.set_ylabel("x (mm)")
ax1.set_zlabel("y (mm)")
ax2.set_xlabel("z (mm)")
ax2.set_ylabel("x (mm)")
ax3.set_xlabel("z (mm)")
ax3.set_ylabel("y (mm)")
ax4.set_xlabel("x (mm)")
ax4.set_ylabel("y (mm)")
#quad_loop.show_field_magnitude("B", gif_name="quad_Bmagn.mp4")
plt.show()