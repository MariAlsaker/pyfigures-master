import magpylib as magpy
import numpy as np
import matplotlib.pyplot as plt

coil = magpy.current.Loop(current=100, diameter=150)

# Display coil
fig = plt.figure(figsize=(7,9))
fig.suptitle("Magnetic view of a conducting loop")
ax1 = fig.add_subplot(2,1,1, projection="3d")
ax1.set_title("Coil illustration")
magpy.show(coil, backend='matplotlib', canvas=ax1)

# Observer grid in XY-plane
X, Y = np.mgrid[-8:8:100j, -8:8:100j].transpose((0,2,1)) #complex number means number of points to create between start and stop
grid_xy = np.stack([X, Y, np.zeros((100,100))], axis=2)

# Observer grid in XZ-plane
X, Z = np.mgrid[-8:8:100j, -8:8:100j].transpose((0,2,1))
grid_xz = np.stack([X, np.zeros((100,100)), Z], axis=2)

# CHOOSE plane and field
plane = [grid_xy, "grid_xy"] # [grid_xz, "grid_xz"] **or** [grid_xy, "grid_xy"]
B = coil.getB(plane[0])
H = coil.getH(plane[0])
field = [B, "|B| [mT]"] #[H, "|H| [kA/m]"] **or** [B, "|B| [mT]"]

# Display with pyplot
print(field[0].shape)
field_magn = np.sqrt(np.sum(field[0]**2, axis=-1))
ax2 = fig.add_subplot(2,1,2)
vis = ax2.imshow(field_magn, cmap="plasma")
vis.set_extent((-80, 80, -80, 80))
ax2.set_title(f"Magnetic flux density, {field[1]}")
ax2.set_xlabel(f"{plane[1][-2]} (mm)")
ax2.set_ylabel(f"{plane[1][-1]} (mm)")
plt.colorbar(vis)
plt.show()