import my_classes.MRIcoil as MRIcoil
from matplotlib import pyplot as plt

""" SINGLE LOOP SURFACE COIL """
single_loop_coil = MRIcoil.MRIcoil(15, 100)
fig1 = plt.figure(figsize=[12, 6])
fig1.suptitle("Single loop coil (D=150mm)")
ax1 = fig1.add_subplot(1,2,1, projection="3d")
ax2 = fig1.add_subplot(1,2,2) 
single_loop_coil.show_coil(ax=ax1)
single_loop_coil.show_field_lines(slice="x50", ax=ax2, fig=fig1)
ax2.set_title("B-field - magnetic flux")
plt.show()
fig = plt.figure(figsize=[12, 4])
ax1 = fig.add_subplot(1,3,1) 
ax2 = fig.add_subplot(1,3,2) 
ax3 = fig.add_subplot(1,3,3) 
single_loop_coil.show_field_slice("B", slice=50, ax=ax1)
single_loop_coil.show_field_slice("B", slice=97, ax=ax2)
single_loop_coil.show_field_slice("B", slice=2, ax=ax3)
plt.show()
single_loop_coil.show_field_lines(slice="y99")
single_loop_coil.show_field_magnitude("B", gif_name="norm_to_mid_val.mp4")