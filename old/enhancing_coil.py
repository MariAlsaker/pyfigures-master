from matplotlib import pyplot as plt
import numpy as np
from matplotlib.colors import to_hex
from stl import mesh
import magpylib as magpy
from scipy.optimize import curve_fit

import sys
sys.path.insert(0, '/Users/marialsaker/git/pyfigures-master/my_classes') # change for your user/name
import MRIcoil
sys.path.insert(0, '/Users/marialsaker/git/pyfigures-master') # change for your user/name
import utils


# Want to create a quadratic surface loop with x = [-45, 45] and y = [-45, 45] meaning 9 cm side length
xs_curr_arc = np.linspace(-45, 45, 91)

# load stl file from online resource
stl_file_path = "/Users/marialsaker/Library/CloudStorage/OneDrive-UniversityofBergen/Medtek/Master/3Dfiles/Enhancing_holder.stl"
# create trace of mesh and fetch line of current path just below 
trace_mesh3d, curr_line_vertices = utils.trace_from_stl(stl_file_path, xs_curr_arc)
# Initiating the collection and placing it close to origo
Coil_holder = magpy.Collection(position=(0, 0, 100), style_label="'Coil holder' trace")
Coil_holder.style.model3d.add_trace(trace_mesh3d)
curr_line_vertices[:,2] = curr_line_vertices[:,2] +100 # correcting for offset in 3d stl file

# Removing points on support volume
i_to_remove = []
for i, vert in enumerate(curr_line_vertices):
    if(vert[2]<-10):
        i_to_remove.append(i)
        continue
curr_line_vertices = np.delete(curr_line_vertices, i_to_remove, axis=0)

# Removing points above fitted curve
def square(x, a, b):
    return a*x**2 + b
popt, pcov = curve_fit(square, curr_line_vertices[:,0], curr_line_vertices[:,2])
testx = np.linspace(-40, 40, 100)
i_to_remove = []
for i, vert in enumerate(curr_line_vertices):
    if square(vert[0], *popt) < vert[2]:
        i_to_remove.append(i)
curr_line_vertices = np.delete(curr_line_vertices, i_to_remove, axis=0)

# Fit a line to the chosen vertices
popt, pcov = curve_fit(square, curr_line_vertices[:,0], curr_line_vertices[:,2])
xs_curr_arc = np.linspace(-45, 45, 11) # We dont need 91 steps, so i reduce to 19
zs_curr_arc = square(xs_curr_arc, *popt)

# Define vertices for current lines to run through
vertices_curr_arc_neg = np.concatenate([xs_curr_arc.reshape(len(xs_curr_arc),1), 
                                        np.ones_like(xs_curr_arc).reshape(len(xs_curr_arc),1)*-45, 
                                        zs_curr_arc.reshape(len(xs_curr_arc),1)],
                                        axis=1)
vertices_curr_arc_pos = np.concatenate([xs_curr_arc.reshape(len(xs_curr_arc),1), 
                                        np.ones_like(xs_curr_arc).reshape(len(xs_curr_arc),1)*45, 
                                        zs_curr_arc.reshape(len(xs_curr_arc),1)],
                                        axis=1)
ys_curr_line = np.linspace(-45, 45, 11).reshape(11,1)
vertices_curr_line_neg =  np.concatenate([np.ones_like(ys_curr_line)*xs_curr_arc[0], 
                                          ys_curr_line, 
                                          np.ones_like(ys_curr_line)*square(xs_curr_arc[0], *popt)], 
                                          axis=1)
vertices_curr_line_pos =  np.concatenate([np.ones_like(ys_curr_line)*xs_curr_arc[-1], 
                                          ys_curr_line, 
                                          np.ones_like(ys_curr_line)*square(xs_curr_arc[-1], *popt)], 
                                          axis=1)
cur_arc_neg = magpy.current.Line(100, vertices_curr_arc_neg)
cur_line_pos = magpy.current.Line(100, vertices_curr_line_pos)
cur_arc_pos = magpy.current.Line(100, np.flip(vertices_curr_arc_pos, axis=0))
cur_line_neg = magpy.current.Line(100, np.flip(vertices_curr_line_neg, axis=0))
the_coil = magpy.Collection((cur_arc_neg, cur_line_pos, cur_arc_pos, cur_line_neg))

# Define coil
enhancing_loop = MRIcoil.MRIcoil(current=100, diameter=90, custom_coil=True, custom_coil_current_line=the_coil)
# # Show coil in 3D, and magnetic field strength gif - through z-axis:
# enhancing_loop.show_coil(show=True)
# enhancing_loop.show_field_magnitude("B", gif_name="special_coil_B.mp4")

# Create figure
fig2 = plt.figure(figsize=[12, 8])
fig2.suptitle("Enhancing coil (s=90mm) with field lines in slice z = 0, y = 0 and x = 0")
ax1 = fig2.add_subplot(2,2,1, projection="3d")
ax2 = fig2.add_subplot(2,2,2)
ax3 = fig2.add_subplot(2,2,3)
ax4 = fig2.add_subplot(2,2,4)
enhancing_loop.show_coil(ax=ax1)
#magpy.show(Coil_holder, backend="matplotlib", canvas=ax1)
enhancing_loop.show_field_lines(slice="z50", ax=ax2, fig=fig2)
enhancing_loop.show_field_lines(slice="x50", ax=ax3, fig=fig2)
ax3.hlines(y=[40], xmin=[-45], xmax=[45], colors=['k'], linestyles='dashed')
enhancing_loop.show_field_lines(slice="y50", ax=ax4, fig=fig2)
ax4.hlines(y=[40], xmin=[-45], xmax=[45], colors=['k'], linestyles='dashed')
X, Y, Z = utils.plane_at("z40")
ax1.plot_surface(X, Y, Z, alpha=.3, label="x = 0, slice 50")

ax1.set_xlabel("x (mm)")
ax1.set_ylabel("z (mm)")
ax1.set_zlabel("y (mm)")
ax2.set_xlabel("x (mm)")
ax2.set_ylabel("z (mm)")
ax3.set_xlabel("x (mm)")
ax3.set_ylabel("y (mm)")
ax4.set_xlabel("z (mm)")
ax4.set_ylabel("y (mm)")
plt.show()