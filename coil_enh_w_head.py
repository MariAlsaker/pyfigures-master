import my_classes.MRIcoil as MRIcoil
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.colors import to_hex
from stl import mesh
import magpylib as magpy
from scipy.optimize import curve_fit
from scipy.ndimage import rotate
import utils

def show_field_slice(field, ax, verbose:bool=False):
    """ Show a specific slice normal to the z-axis. \n 
    Parameters:
    - chosen_field: B or H
    - slice: int 0 to 100 ranging from -80mm to 80mm below/above the coil"""
    field_magn = np.sqrt(np.sum(np.square(field), axis=-1))
    field_max = np.max(field_magn)
    normalized_field_magn = field_magn/field_max *100
    img = ax.pcolormesh(normalized_field_magn, cmap="plasma", vmin=0, vmax=100)
    plt.colorbar(img, label="[%]")
    ticks = np.linspace(0, 100, 5, endpoint=True)
    # ax.set_xticks(ticks, labels=["-80", "-40", "0", "40", "80"])
    # ax.set_yticks(ticks, labels=["-80", "-40", "0", "40", "80"])
    ax.set_title(f"Magnetic flux density, B-field, z_ax = {slice*1.6-80:.2f} mm")
    ax.set_xlabel("x(mm)")
    ax.set_ylabel("y(mm)")
    
# Creating my arc in 2D
center = (0,0)
radius = 102 # Radius of coil holders underneath arc
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
r_headcoil = 141 #mm from sketch of enhancing holder (26.4 cm diam out to plastic)
t = np.linspace(0, 2*np.pi, 16, endpoint=False)
y_headcoil = r_headcoil* np.cos(t)
z_headcoil = r_headcoil* np.sin(t) #+ 100
curr_lines_headcoil = []
for pos in range(len(y_headcoil)):
    vertices = np.zeros((3,2))
    vertices[0][0], vertices[0][1] = 100, -100 # X
    vertices[1][0], vertices[1][1] = y_headcoil[pos], y_headcoil[pos] # Y
    vertices[2][0], vertices[2][1] = z_headcoil[pos], z_headcoil[pos] # Z
    curr_lines_headcoil.append( magpy.current.Line(200, vertices.transpose()))
headcoil_coll = magpy.Collection(curr_lines_headcoil)

# In the MRI machine we have the coil in the x-y-plane (but really the x-z plane)
slice = (5, -4)
vertices_e_coil = np.zeros((4, 3, 11))
# negarc, negpos, posarc, posneg 
# = vertices_coil[0], vertices_coil[1], vertices_coil[2], vertices_coil[3]
vertices_e_coil[0], vertices_e_coil[2] = np.zeros((3,11)), np.zeros((3,11)) # fra minus til pluss, fra pluss til minus
vertices_e_coil[0][0], vertices_e_coil[0][1], vertices_e_coil[0][2] = np.ones((11))*-42.5, final_arc[0][slice[0]:slice[1]], final_arc[1][slice[0]:slice[1]]
vertices_e_coil[2][0], vertices_e_coil[2][1], vertices_e_coil[2][2] = np.ones((11))*42.5, np.flip(final_arc[0][slice[0]:slice[1]]), np.flip(final_arc[1][slice[0]:slice[1]])
vertices_e_coil[1], vertices_e_coil[3] = np.zeros((3,11)), np.zeros((3,11))
vertices_e_coil[1][0], vertices_e_coil[3][0] = np.linspace(-42.5, 42.5, 11, endpoint=True), np.linspace(42.5, -42.5, 11, endpoint=True)
vertices_e_coil[1][1], vertices_e_coil[3][1] = np.ones((11))*vertices_e_coil[0][1][-1], np.ones((11))*vertices_e_coil[2][1][-1]
vertices_e_coil[1][2], vertices_e_coil[3][2] = np.ones((11))*vertices_e_coil[0][2][-1], np.ones((11))*vertices_e_coil[2][2][-1]

curr_lines_e = []
for line in vertices_e_coil:
    curr_lines_e.append( magpy.current.Line(-50, line.transpose()))
coil_e = magpy.Collection(curr_lines_e)
both_coils = magpy.Collection((coil_e, headcoil_coll))

# Create figure
fig2 = plt.figure(figsize=[12, 8])
fig2.suptitle("Enhancing coil (s=9.0cm) inside head coil (16 rods)\nfield lines in slice z = 0mm")
extent_xy, z = 150, 0
ax1 = fig2.add_subplot(1,2,1, projection="3d")
magpy.show(both_coils, backend='matplotlib', return_fig=False, canvas=ax1)
X, Y, Z = utils.plane_at("x=00", extent=extent_xy)
ax1.plot_surface(X, Y, Z, alpha=.3, label="y = 40")

ax2 = fig2.add_subplot(1,2,2)
X, Y = np.mgrid[-extent_xy:extent_xy:100j, -extent_xy:extent_xy:100j].transpose((0,2,1))
grid = np.stack([np.zeros((100,100)) + z, X, Y], axis=2)
B_field = magpy.getB(both_coils, observers=grid)
utils.show_field_lines(grid_slice=grid, B_field=B_field, ax=ax2, fig=fig2)

ax1.set_xlabel("z (mm)")
ax1.set_ylabel("x (mm)")
ax1.set_zlabel("y (mm)")
ax2.set_xlabel("x (mm)")
ax2.set_ylabel("y (mm)")
# enhancing_coil.show_field_magnitude("B", vmax=100)
plt.show()