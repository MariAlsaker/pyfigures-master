
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
final_arc=utils.create_2D_arc(radius=102)

# Creating circle of headcoil in 2D
r_headcoil = 141 #mm from sketch of enhancing holder (26.4 cm diam out to plastic)
t = np.linspace(0, 2*np.pi, 16, endpoint=False)
y_headcoil = r_headcoil* np.cos(t)
z_headcoil = r_headcoil* np.sin(t) #+ 100
curr_lines_headcoil = []
xs = np.linspace(0, 2*np.pi, 16)
part = np.pi*2/10
partnum = 0 # max 9
currs = 400*(2*8)*np.cos(xs+np.pi/16+partnum*part) # Changing number of part makes the field lines rotate
currs = np.flip(currs)
for pos in range(len(y_headcoil)):
    vertices = np.zeros((3,2))
    vertices[0][0], vertices[0][1] = 100, -100 # X
    vertices[1][0], vertices[1][1] = y_headcoil[pos], y_headcoil[pos] # Y
    vertices[2][0], vertices[2][1] = z_headcoil[pos], z_headcoil[pos] # Z
    this = magpy.current.Line(currs[pos], vertices.transpose())
    curr_lines_headcoil.append( this)
headcoil_coll = magpy.Collection(curr_lines_headcoil)

both=True
if both ==True:
    # Find field change and induced current.
    vals = np.zeros(shape=(10,20))
    with open("changing_field.txt", "r") as file:
        lines = file.readlines()
        num=-1
        for line in lines:
            splitted = line.split()
            if splitted[0] == "part":
                num = num+1
                i = 0
            else:
                for val_str in splitted:
                    vals[num][i] = float(val_str)
                    i = i+1
    mean_vals=np.mean(vals, axis=1)
    diffs = [mean_vals[0]-mean_vals[-1]]
    k = 1
    for mean in mean_vals: # mean oppgitt i mT = milli Tesla
        diffs.append(mean_vals[k]-mean) 
        k=k+1
        if k==len(mean_vals):break
    diffs = np.array(diffs) * 1E-3 * 0.0081 # ca areal av coil er 9*9 = 81 cm^2 = 0.81m^2

    period = 1/(33.78E6) 
    time_part = period/10
    # Faraday's law calculating induced current2
    emfs = diffs/time_part # Direction calculated by Len's law, looping at the current direction of the small loop

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
        curr_lines_e.append( magpy.current.Line(emfs[partnum], line.transpose()))
    coil_e = magpy.Collection(curr_lines_e)
    both_coils = magpy.Collection((coil_e, headcoil_coll))
    figured_coil = both_coils
else:
    figured_coil = headcoil_coll

# Create figure
fig1 = plt.figure(figsize=[5, 5])
#fig1.suptitle("Enhancing coil (s=9.0cm) inside head coil (16 rods)\nfield lines in slice z = 0mm")
extent_xy, z = 150, 0
ax1 = fig1.add_subplot(1,1,1, projection="3d")
magpy.show(figured_coil, backend='matplotlib', return_fig=False, canvas=ax1)
X, Y, Z = utils.plane_at("x=00", extent=extent_xy)
ax1.plot_surface(X, Y, Z, alpha=.3, label="y = 40")
ax1.view_init(15, 20)
ax1.set_xlabel("z (mm)")
ax1.set_ylabel("x (mm)")
ax1.set_zlabel("y (mm)")
ax1.get_legend().remove()
# plt.savefig(utils.this_path+f"birdcage.png", dpi=300)
# plt.close("all")

fig2 = plt.figure(figsize=[5, 5])
ax2 = fig2.add_subplot(1,1,1)
X, Y = np.mgrid[-extent_xy:extent_xy:100j, -extent_xy:extent_xy:100j].transpose((0,2,1))
grid = np.stack([np.zeros((100,100)) + z, X, Y], axis=2)
B_field = magpy.getB(figured_coil, observers=grid)
utils.show_field_magn(grid_slice=grid, coil=figured_coil, ax=ax2, fig=fig2, vmax=100, colorbar=False)
ax2.set_xlabel("x (mm)")
ax2.set_ylabel("y (mm)")
ticks=np.linspace(0, 100, 7)
tick_ls=np.linspace(-extent_xy, extent_xy, 7)
labels = [f"{lab:.0f}" for lab in tick_ls]
ax2.set_xticks(ticks, labels)
ax2.set_yticks(ticks, labels)
ax2.axis("off")
plt.tight_layout( pad=0)

# ax2.hlines(30, xmin=40, xmax=60) # For showing the line where flux is calculated
# plt.savefig(f"birdcage_field_part{partnum}", dpi=300)
# plt.close("all")

fig3 = plt.figure(figsize=[5, 5])
ax3 = fig3.add_subplot(1,1,1)
X, Y = np.mgrid[-extent_xy:extent_xy:100j, -extent_xy:extent_xy:100j].transpose((0,2,1))
grid = np.stack([np.zeros((100,100)) + z, X, Y], axis=2)
B_field = magpy.getB(figured_coil, observers=grid)
# with open("changing_field.txt", "a") as file:
#     file.write(f"\npart {partnum}\n")
#     file.write(np.array2string(B_field[30,40:60,2], precision=2).strip("[]"))

utils.show_field_lines(grid_slice=grid, B_field=B_field, ax=ax3, fig=fig2, slicein="x", colorbar=False)
ax3.axis("off")
plt.tight_layout( pad=0)

# plt.savefig(f"birdcage_fieldlines_part{partnum}.png", dpi=300)
# plt.close("all")

plt.show()

# FORKLARER HVORFOR FORSTERKNINGEN ER KUN PÅ EN SIDE
# Switcher polaritet og den deriverte skifter fortegn
# Da er magnetfeltet fortsatt på samme side...
