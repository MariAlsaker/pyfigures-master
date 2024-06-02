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
#print(final_arc) # x, y = 0, 1
print(final_arc)
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
currents = [100*np.cos(np.pi/4), 100*np.cos(np.pi/4 + np.pi/2)]
for i, vertices in enumerate(vertices_2coils):
    for j, line in enumerate(vertices):
        this_line =  magpy.current.Line(currents[i], line.transpose())
        curr_lines_2coils[i].append(this_line)
coil1 = magpy.Collection(curr_lines_2coils[0])
coil2 = magpy.Collection(curr_lines_2coils[1])
# Aiming for 8 different fields
angles = np.linspace(0, 2*np.pi, 8, endpoint=False)

# both_coils = magpy.Collection([coil1, coil2])
# both_coils.show()

extent_xy = 25
z = 0
X, Y = np.mgrid[-extent_xy:extent_xy:100j, -extent_xy:extent_xy:100j].transpose((0,2,1))
grid = np.stack([np.zeros((100,100)) + z, X, Y], axis=2)

fig = plt.figure(figsize=[8, 12])
start = 421
for ang in angles:
    for i, coil in enumerate([coil1, coil2]):
        for current_line in coil.sources_all:
            if i == 0:
                current_line.current = 100*np.cos(ang+np.pi/2)
            else:
                current_line.current = 100*np.cos(ang)
    both_coils = magpy.Collection((coil1, coil2), override_parent=True)
    B_field = magpy.getB(both_coils, observers=grid)
    ax = fig.add_subplot(start)
    start = start+1
    utils.show_field_lines(grid, B_field, ax=ax, fig=fig, slicein='x')
    ax.set_title(f"Current = ({100*np.cos(ang+np.pi/2):.0f}, {100*np.cos(ang):.0f}) A")
# plt.savefig("rotating_field_quad_dir_correct.png")

plt.show()
    