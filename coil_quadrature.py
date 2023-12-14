import my_classes.MRIcoil as MRIcoil
from matplotlib import pyplot as plt
import numpy as np

""" QUADRATURE SURFACE COIL """
def plane_at(slice="x=00", extent=50):
    xx = np.linspace(-extent, extent, 101)
    zz = np.linspace(-extent, extent, 101)
    yy = np.linspace(-extent, extent, 101)
    if slice[0]=="x":
        Y, Z = np.meshgrid(yy, zz)
        X = np.ones_like(Y)*int(slice[-2:])
    elif slice[0]=="y":
        X, Z = np.meshgrid(xx, zz)
        Y = np.ones_like(X)*int(slice[-2:])
    elif slice[0]=="z":
        X, Y = np.meshgrid(xx, yy)
        Z = np.ones_like(X)*int(slice[-2:])
    return X, Y, Z
quad_coil = MRIcoil.MRIcoil(15, 100, quadrature=True, quad_dir="out")
# PLOTTING 3D MODEL AND 3 DIFFERENT SLICES WITH FIELD LINES #
fig2 = plt.figure(figsize=[12, 8])
fig2.suptitle("Quadrature coils (D=150mm) with field lines in slice z = 75mm, y = 0 and x = 0 \n y and x range [-(D+2)/2, (D+2)/d], z range [-2, D]")
ax1 = fig2.add_subplot(2,2,1, projection="3d")
ax2 = fig2.add_subplot(2,2,2)
ax3 = fig2.add_subplot(2,2,3)
ax4 = fig2.add_subplot(2,2,4)
quad_coil.show_coil(ax=ax1)
quad_coil.show_field_lines(slice="z56", ax=ax2, fig=fig2)
quad_coil.show_field_lines(slice="x50", ax=ax3, fig=fig2)
quad_coil.show_field_lines(slice="y50", ax=ax4, fig=fig2)
extent = (quad_coil.diameter+2)/2
X, Y, Z = plane_at("y=00")
ax1.plot_surface(X, Y, Z, alpha=.3, label="y = 0, slice 50")
plt.legend()
plt.show()
# quad_coil.show_field_magnitude("B", vmax=200) #, gif_name="quad_norm_to_mid_val.mp4")
# # TODO: Still lacking a proper calculation of actual field strength in Tesla... (percentage may not be so cool, or is this really necessary?)