import my_classes.MRIcoil as MRIcoil
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.colors import to_hex
from stl import mesh
import magpylib as magpy
from scipy.optimize import curve_fit
from scipy.ndimage import rotate
import utils

""" SINGLE LOOP SURFACE COIL """
# single_loop_coil = MRIcoil.MRIcoil(15, 100)
# fig1 = plt.figure(figsize=[12, 6])
# fig1.suptitle("Single loop coil (D=150mm)")
# ax1 = fig1.add_subplot(1,2,1, projection="3d")
# ax2 = fig1.add_subplot(1,2,2) 
# single_loop_coil.show_coil(ax=ax1)
# single_loop_coil.show_field_lines(slice="x50", ax=ax2, fig=fig1)
# ax2.set_title("B-field - magnetic flux")
# plt.show()
# fig = plt.figure(figsize=[12, 4])
# ax1 = fig.add_subplot(1,3,1) 
# ax2 = fig.add_subplot(1,3,2) 
# ax3 = fig.add_subplot(1,3,3) 
# single_loop_coil.show_field_slice("B", slice=50, ax=ax1)
# single_loop_coil.show_field_slice("B", slice=97, ax=ax2)
# single_loop_coil.show_field_slice("B", slice=2, ax=ax3)
# plt.show()
# single_loop_coil.show_field_lines(slice="y99")
# single_loop_coil.show_field_magnitude("B", gif_name="norm_to_mid_val.mp4")

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

# In the MRI machine we have the coil in the x-y-plane (but really the x-z plane)
slice = (5, -4)
vertices_coil = np.zeros((4, 3, 11))
# negarc, negpos, posarc, posneg = vertices_coil[0], vertices_coil[1], vertices_coil[2], vertices_coil[3]
vertices_coil[0], vertices_coil[2] = np.zeros((3,11)), np.zeros((3,11)) # fra minus til pluss, fra pluss til minus
vertices_coil[0][0], vertices_coil[0][1], vertices_coil[0][2] = np.ones((11))*-42.5, final_arc[0][slice[0]:slice[1]], final_arc[1][slice[0]:slice[1]]
vertices_coil[2][0], vertices_coil[2][1], vertices_coil[2][2] = np.ones((11))*42.5, np.flip(final_arc[0][slice[0]:slice[1]]), np.flip(final_arc[1][slice[0]:slice[1]])
vertices_coil[1], vertices_coil[3] = np.zeros((3,11)), np.zeros((3,11))
vertices_coil[1][0], vertices_coil[3][0] = np.linspace(-42.5, 42.5, 11, endpoint=True), np.linspace(42.5, -42.5, 11, endpoint=True)
vertices_coil[1][1], vertices_coil[3][1] = np.ones((11))*vertices_coil[0][1][-1], np.ones((11))*vertices_coil[2][1][-1]
vertices_coil[1][2], vertices_coil[3][2] = np.ones((11))*vertices_coil[0][2][-1], np.ones((11))*vertices_coil[2][2][-1]


curr_lines = []
for line in vertices_coil:
    curr_lines.append( magpy.current.Line(100, line.transpose()))
coil = magpy.Collection(curr_lines)
coil.move((0,0,80))
# Define coil
extent_xy = 50
# Create figure
fig2 = plt.figure(figsize=[6, 5])
fig2.suptitle("Surface coil (s=9.0cm), field lines slice z = 0mm")
ax1 = fig2.add_subplot(2,2,1, projection="3d")
ax2 = fig2.add_subplot(2,2,2)
ax3 = fig2.add_subplot(2,2,3)
ax4 = fig2.add_subplot(2,2,4)
magpy.show(coil, backend='matplotlib',return_fig=False, canvas=ax1)

X, Y = np.mgrid[-extent_xy:extent_xy:100j, -extent_xy:extent_xy:100j].transpose((0,2,1))
zs = np.linspace(-extent_xy, extent_xy, 100)
grid_3D = []
for z in zs:
    grid_xy = np.stack([X, Y, np.zeros((100,100)) + z], axis=2)
    grid_3D.append(grid_xy)
grid_3D = np.array(grid_3D)
print(type(grid_3D))

x=50
grid = grid_3D[:,x,:]
B_field = magpy.getB(coil, observers=grid)
utils.show_field_lines(grid, B_field, ax=ax2, fig=fig2, slicein="x")

y=50
grid = grid_3D[y,:,:]
B_field = magpy.getB(coil, observers=grid)#.transpose(2,1,0)
print(grid.shape, B_field.shape)
utils.show_field_lines(grid, B_field, ax=ax3, fig=fig2, slicein="z")

z = 50
grid = grid_3D[:,:,z]
#grid = np.stack([np.zeros((100,100)) + z, X, Y], axis=2) #first
B_field = magpy.getB(coil, observers=grid)
utils.show_field_lines(grid, B_field, ax=ax4, fig=fig2, slicein="y")

#ax2.hlines(y=[40], xmin=[-50], xmax=[50], colors=['k'], linestyles='dashed')
#X, Y, Z = utils.plane_at("z=40")
#ax1.plot_surface(X, Y, Z, alpha=.3, label="y = 40")
# X, Y, Z = utils.plane_at("z=00")
# ax1.plot_surface(X, Y, Z, alpha=.3, label="y = 0")

ax1.set_xlabel("z (mm)")
ax1.set_ylabel("x (mm)")
ax1.set_zlabel("y (mm)")
ax2.set_xlabel("z (mm)")
ax2.set_ylabel("y (mm)")
ax3.set_xlabel("x (mm)")
ax3.set_ylabel("z (mm)")
ax4.set_xlabel("x (mm)")
ax4.set_ylabel("y (mm)")
# surface_coil.show_field_magnitude("B", vmax=100)
plt.tight_layout()
plt.show()