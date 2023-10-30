import coil_field
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.colors import to_hex
from stl import mesh
import magpylib as magpy

""" SINGLE LOOP SURFACE COIL """
# single_loop_coil = coil_field.Coil_field(15, 100)
# fig1 = plt.figure(figsize=[12, 6])
# fig1.suptitle("Single loop coil (D=150mm)")
# ax1 = fig1.add_subplot(1,2,1, projection="3d")
# ax2 = fig1.add_subplot(1,2,2) 
# single_loop_coil.show_coil(ax=ax1)
# single_loop_coil.show_field_lines(slice="x50", ax=ax2, fig=fig1)
# ax2.set_title("B-field - magnetic flux")
# plt.show()
# single_loop_coil.show_field_slice("B", slice=97)
# single_loop_coil.show_field_lines(slice="y99")
# single_loop_coil.show_field("B", gif_name="norm_to_mid_val.mp4")
# single_loop_coil.show_field_slice("B", slice=50, verbose=True, filename="center_norm")
# single_loop_coil.show_field_slice("B", slice=97, verbose=True, filename="plus75mm_norm")
# single_loop_coil.show_field_slice("B", slice=2, verbose=True, filename="minus75mm_norm")


""" QUADRATURE SURFACE COIL """
# quad_coil = coil_field.Coil_field(15, 100, quadrature=True, quad_dir="out")

# # PLOTTING 3D MODEL AND 3 DIFFERENT SLICES WITH FIELD LINES #
# fig2 = plt.figure(figsize=[12, 8])
# fig2.suptitle("Quadrature coils (D=150mm) with field lines in slice z = 7.5cm, y = 0 and x = 0 \n y and x range [-(D+2)/2, (D+2)/d], z range [-2, D]")
# ax1 = fig2.add_subplot(2,2,1, projection="3d")
# ax2 = fig2.add_subplot(2,2,2)
# ax3 = fig2.add_subplot(2,2,3)
# ax4 = fig2.add_subplot(2,2,4)
# quad_coil.show_coil(ax=ax1)
# quad_coil.show_field_lines(slice="z56", ax=ax2, fig=fig2)
# quad_coil.show_field_lines(slice="x50", ax=ax3, fig=fig2)
# quad_coil.show_field_lines(slice="y50", ax=ax4, fig=fig2)
# extent = (quad_coil.diameter+2)/2
# xx = np.linspace(-extent, extent, 100)
# zz = np.linspace(-2, quad_coil.diameter, 100)
# yy = np.linspace(-extent, extent, 100)
# X, Z = np.meshgrid(xx, zz)
# Y = X*0
# #X = Y*0
# #Z = 7.5*np.ones_like(Y)
# ax1.plot_surface(X, Y, Z, alpha=.3, label="y = 0, slice 50")
# plt.legend()
# plt.show()

# # quad_coil.show_field_magnitude("B", vmax=200) #, gif_name="quad_norm_to_mid_val.mp4")
# # TODO: Still lacking a proper calculation of actual field strength in Tesla... (percentage may not be so cool, or is this really necessary?)


""" DISPLAY STL """
# def bin_color_to_hex(x):
#     """transform binary rgb into hex color"""
#     sb = f"{x:015b}"[::-1]
#     r = int(sb[:5], base=2) / 31
#     g = int(sb[5:10], base=2) / 31
#     b = int(sb[10:15], base=2) / 31
#     return to_hex((r, g, b))

# def trace_from_stl(stl_file):
#     """
#     Generates a Magpylib 3D model trace dictionary from an *.stl file.
#     backend: 'matplotlib' or 'plotly'
#     """
#     # load stl file
#     stl_mesh = mesh.Mesh.from_file(stl_file)

#     # extract vertices and triangulation
#     p, q, r = stl_mesh.vectors.shape
#     vertices, ixr = np.unique(
#         stl_mesh.vectors.reshape(p * q, r), return_inverse=True, axis=0
#     )
#     i = np.take(ixr, [3 * k for k in range(p)])
#     j = np.take(ixr, [3 * k + 1 for k in range(p)])
#     k = np.take(ixr, [3 * k + 2 for k in range(p)])
#     x, y, z = vertices.T

#     # generate and return a generic trace which can be translated into any backend
#     colors = stl_mesh.attr.flatten()
#     facecolor = np.array([bin_color_to_hex(c) for c in colors]).T
#     trace = {
#         "backend": "generic",
#         "constructor": "mesh3d",
#         "kwargs": dict(x=x, y=y, z=z, i=i, j=j, k=k, facecolor=facecolor),
#     }
#     return trace

# # load stl file from online resource
# stl_file_path = "/Users/marialsaker/Library/CloudStorage/OneDrive-UniversityofBergen/Medtek/Master/3Dfiles/Enhancing_holder.stl"
# trace_mesh3d = trace_from_stl(stl_file_path)

# coll = magpy.Collection(position=(0, -3, 0), style_label="'Mesh3d' trace")
# coll.style.model3d.add_trace(trace_mesh3d)
# # create traces for both backends

# magpy.show(coll, backend="matplotlib")