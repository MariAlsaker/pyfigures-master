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

# Creating circle of headcoil in 2D
r_headcoil = 135 #mm
t = np.linspace(0, 2*np.pi, 16, endpoint=False)
y_headcoil = r_headcoil* np.cos(t)
z_headcoil = r_headcoil* np.sin(t) + 100
#x_headcoil = np.zeros_like(y_headcoil)
curr_lines_headcoil = []
for pos in range(len(y_headcoil)):
    vertices = np.zeros((3,2))
    vertices[0][0], vertices[0][1] = 100, -100
    vertices[1][0], vertices[1][1] = y_headcoil[pos], y_headcoil[pos]
    vertices[2][0], vertices[2][1] = z_headcoil[pos], z_headcoil[pos]
    curr_lines_headcoil.append( magpy.current.Line(200, vertices.transpose()))
headcoil_coll = magpy.Collection(curr_lines_headcoil)

# In the MRI machine we have the coil in the x-y-plane (but really the x-z plane)
slice = (5, -4)
vertices_e_coil = np.zeros((4, 3, 11))
# negarc, negpos, posarc, posneg = vertices_coil[0], vertices_coil[1], vertices_coil[2], vertices_coil[3]
vertices_e_coil[0], vertices_e_coil[2] = np.zeros((3,11)), np.zeros((3,11)) # fra minus til pluss, fra pluss til minus
vertices_e_coil[0][0], vertices_e_coil[0][1], vertices_e_coil[0][2] = np.ones((11))*-42.5, final_arc[0][slice[0]:slice[1]], final_arc[1][slice[0]:slice[1]]
vertices_e_coil[2][0], vertices_e_coil[2][1], vertices_e_coil[2][2] = np.ones((11))*42.5, np.flip(final_arc[0][slice[0]:slice[1]]), np.flip(final_arc[1][slice[0]:slice[1]])
vertices_e_coil[1], vertices_e_coil[3] = np.zeros((3,11)), np.zeros((3,11))
vertices_e_coil[1][0], vertices_e_coil[3][0] = np.linspace(-42.5, 42.5, 11, endpoint=True), np.linspace(42.5, -42.5, 11, endpoint=True)
vertices_e_coil[1][1], vertices_e_coil[3][1] = np.ones((11))*vertices_e_coil[0][1][-1], np.ones((11))*vertices_e_coil[2][1][-1]
vertices_e_coil[1][2], vertices_e_coil[3][2] = np.ones((11))*vertices_e_coil[0][2][-1], np.ones((11))*vertices_e_coil[2][2][-1]


curr_lines_e = []
for line in vertices_e_coil:
    curr_lines_e.append( magpy.current.Line(100, line.transpose()))
coil_e = magpy.Collection(curr_lines_e)
coil_e.move((0,0,100))
# Define coil
both_coils = magpy.Collection((coil_e, headcoil_coll))
enhancing_coil = MRIcoil.MRIcoil(current=90, diameter=100, custom_coil=True, custom_coil_current_line=both_coils)
# Create figure
fig2 = plt.figure(figsize=[12, 8])
fig2.suptitle("Enhancing coil (s=9.0cm) inside head coil (16 rods)\nfield lines in slice z = 0mm, y = 40mm (plane), and x = 0mm")
ax1 = fig2.add_subplot(2,2,1, projection="3d")
ax2 = fig2.add_subplot(2,2,2)
ax3 = fig2.add_subplot(2,2,3)
ax4 = fig2.add_subplot(2,2,4)
enhancing_coil.show_coil(ax=ax1)
enhancing_coil.show_field_lines(slice="z90", ax=ax2, fig=fig2)
enhancing_coil.show_field_lines(slice="x50", ax=ax3, fig=fig2)
ax3.hlines(y=[40], xmin=[-50], xmax=[50], colors=['k'], linestyles='dashed')
enhancing_coil.show_field_lines(slice="y50", ax=ax4, fig=fig2)
ax4.hlines(y=[40], xmin=[-50], xmax=[50], colors=['k'], linestyles='dashed')
X, Y, Z = utils.plane_at("z=40")
ax1.plot_surface(X, Y, Z, alpha=.3, label="y = 40")
# X, Y, Z = utils.plane_at("z=00")
# ax1.plot_surface(X, Y, Z, alpha=.3, label="y = 0")

ax1.set_xlabel("z (mm)")
ax1.set_ylabel("x (mm)")
ax1.set_zlabel("y (mm)")
ax2.set_xlabel("z (mm)")
ax2.set_ylabel("x (mm)")
ax3.set_xlabel("z (mm)")
ax3.set_ylabel("y (mm)")
ax4.set_xlabel("x (mm)")
ax4.set_ylabel("y (mm)")
# enhancing_coil.show_field_magnitude("B", vmax=100)
plt.show()