import coil_field
from matplotlib import pyplot as plt
import numpy as np

""" SINGLE LOOP SURFACE COIL """
# single_loop_coil = coil_field.Coil_field(15, 100)
# single_loop_coil.show_field_lines(slice="y99")
# single_loop_coil.show_field("B", gif_name="norm_to_mid_val.mp4")
# single_loop_coil.show_field_slice("B", slice=50, verbose=True, filename="center_norm")
# single_loop_coil.show_field_slice("B", slice=97, verbose=True, filename="plus75mm_norm")
# single_loop_coil.show_field_slice("B", slice=2, verbose=True, filename="minus75mm_norm")


""" QUADRATURE SURFACE COIL """
quad_coil = coil_field.Coil_field(15, 100, quadrature=True, quad_dir="out")

# PLOTTING 3D MODEL AND 3 DIFFERENT SLICES WITH FIELD LINES #
fig = plt.figure(figsize=[12, 8])
fig.suptitle("Quadrature coils (D=150mm) with field lines in slice z = 7.5cm, y = 0 and x = 0 \n y and x range [-(D+2)/2, (D+2)/d], z range [-2, D]")
ax1 = fig.add_subplot(2,2,1, projection="3d")
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)
quad_coil.show_coil(ax=ax1)
quad_coil.show_field_lines(slice="z56", ax=ax2, fig=fig)
quad_coil.show_field_lines(slice="x50", ax=ax3, fig=fig)
quad_coil.show_field_lines(slice="y50", ax=ax4, fig=fig)
extent = (quad_coil.diameter+2)/2
xx = np.linspace(-extent, extent, 100)
zz = np.linspace(-2, quad_coil.diameter, 100)
yy = np.linspace(-extent, extent, 100)
X, Z = np.meshgrid(xx, zz)
Y = X*0
#X = Y*0
#Z = 7.5*np.ones_like(Y)
ax1.plot_surface(X, Y, Z, alpha=.3, label="y = 0, slice 50")
plt.legend()
plt.show()

# quad_coil.show_field_magnitude("B", vmax=200) #, gif_name="quad_norm_to_mid_val.mp4")
# TODO: Still lacking a proper calculation of actual field strength in Tesla... (percentage may not be so cool)