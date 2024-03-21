import scipy.io
from scipy.ndimage import convolve1d
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
import matplotlib as mpl
import numpy as np

my_cmap = "gist_gray" # "tab20", "plasma"

def show_as_video(im_array, vmax:int=1, gif_name:str=None):
    """ Show an animation of the normalized field magnitude in all slices through z-axis. \n 
    Parameters:
    - chosen_field: B or H
    - vmax: maximum value in coloring (very strong field around loop ~200)
    - gif_name: if you want to save a movie of the animation give a string: \"name.mp4\""""

    initialized = False
    imgs = []
    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot(1,1,1)
    ticks = np.linspace(0, 100, 5, endpoint=True)
    abs_array = abs(im_array)
    normalized_field_magn = abs_array/np.max(abs_array)
    for f_slc in normalized_field_magn:
        if not initialized:
            img = ax.pcolormesh(f_slc, cmap=my_cmap, vmin=0, vmax=vmax)
            plt.colorbar(img, label="[%]")
            ax.set_title(f"Magnitude of complex value")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            initialized = True
        else:
            img = ax.pcolormesh(f_slc, cmap=my_cmap, vmin=0, vmax=vmax, animated=True)
        imgs.append([img])
    ani = animation.ArtistAnimation(fig, imgs, interval=50, blit=True,
                            repeat_delay=1000)
    plt.show()
    if gif_name:
        writergif = animation.FFMpegWriter(fps=30)
        ani.save("Figures/"+gif_name, writer=writergif)

def create_circular_mask(h, w, center=None, radius=None):
    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = dist_from_center <= radius
    return mask 

def calculate_SNR(S_values, N_values):
    sd = np.std(N_values)
    signal = np.mean(S_values)
    snr = 0.655 *signal/sd
    return snr

def signal_corr_infinityTR(flipangle, tr, t1):
    num = 1-np.cos(flipangle)*np.e**(-tr/t1)
    denom = 1-np.e**(-tr/t1)
    return num/denom

def b1_field_map(im1alpha, im2alpha, alpha1, alpha2, tr, t1):
    s1 = abs(im1alpha)*signal_corr_infinityTR(flipangle=alpha1, tr=tr, t1=t1)
    s2 = abs(im2alpha)*signal_corr_infinityTR(flipangle=alpha2, tr=tr, t1=t1)
    teta = np.arccos(s2/(2*s1))
    return teta

def blurring_2D(img, kernel_size, padding=0, rep=1):
    k = np.ones((kernel_size))/kernel_size
    out = img.copy()
    if padding != 0:
        np.pad(out, pad_width=padding, mode="constant")
    for j in range(rep):
        for i in range(2):
            out = convolve1d(out, weights=k, axis=i)
    return out

def plot_signal_noise_squares(ax, img, center,offset=15, S_width = 7, N_width = 11, c="w"):
    diff = (N_width-S_width)/2
    if normalized_field_magn.shape[0] == 120:
        offset = 20
    xsS = np.array([center[0]-S_width/2, center[0]+S_width/2])
    ysS = np.array([center[1]-S_width/2, center[1]+S_width/2])
    signal_squares = img[center[1]-S_width//2:center[1]+S_width//2+1, 
                         center[0]-S_width//2:center[0]+S_width//2+1]
    xsN = np.array([xsS[0]-diff, xsS[1]+diff])
    ysN = np.array([ysS[0]-diff, ysS[1]+diff]) + offset
    center = [center[0], center[1]+offset]
    noise_squares = img[center[1]-N_width//2:center[1]+N_width//2+1, 
                        center[0]-N_width//2:center[0]+N_width//2+1]
    for j in range(2):
        ax.plot([xsS[j],xsS[j]], ysS, color = c)
        ax.plot(xsS, [ysS[j],ysS[j]], color = c)
        ax.plot([xsN[j],xsN[j]], ysN, color = c)
        ax.plot(xsN, [ysN[j],ysN[j]], color = c)
    return signal_squares, noise_squares

def norm_magn_image(image):
    magn_image = abs(image)
    max_magn = np.max(magn_image)
    return magn_image/max_magn

numpy_files_path = "/Users/marialsaker/git/pyfigures-master/MRI_data/"
coils = ["OrigSurface", "AssadiSurface", "SingleLoop", "QuadratureCoil", "Birdcage2nd", "BirdcageEnh"] # "Birdcage2nd"
readouts = ["197", "1402", "2552"]
kspace_samp = ["10", "12", "25"]
resolution = ["4.5", "3", "3"]

# Only used for localizing phantom
# radius=37/2 # found from birdcage 2552 plot
# im_centers = [[38, 40], [38, 40], [43, 40], [39, 40], [20+radius, 23+radius], [20+radius, 23+radius]]
centers = [[[58, 80], [38, 58], [38, 56]], # Orig
           [[58, 80], [38, 57], [38, 56]], # Assadi
           [[65, 77], [43, 58], [42, 55]], # Single
           [[60, 78], [38, 59], [39, 56]], # Quad
           [[58, 77], [38, 56], [38, 53]], # Birdcage
           [[58, 75], [38, 55], [38, 53]]] # Enhanced

snrs = []
calculated_snrs = []
coil_lines = [ np.zeros(shape=(len(coils), 120)),
              np.zeros(shape=(len(coils), 80)),
              np.zeros(shape=(len(coils), 80)) ]
f0s = []

for k, coil in enumerate(coils):
    with open(numpy_files_path+f"{coil}_X_optimizer.txt", "r") as optim_f:
        for line in optim_f:
            splitted = line.split(" ")
            if splitted[0] == "SNR":
                snr = float(splitted[-1].strip())
                snrs.append(snr)
            if splitted[0] == "f0:":
                f0 = int(splitted[-2])
                f0s.append(f0)
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
    axs = axs.flatten()
    fig.suptitle(f"Magnitude plots for 3D cones UTE sequence with {coil} coil,\nBloch Siegert optimization yielded: SNR={snr:.2f}, f0={f0}Hz")
    snrs_this_coil = []
    lines_this_coil = []
    for i, ros in enumerate(readouts):
        if k>3:
            im_volume = np.load(numpy_files_path+f"{coil}_{ros}_TG50.npy")
        else:
            im_volume = np.load(numpy_files_path+f"{coil}_{ros}_X.npy")
        normalized_field_magn = norm_magn_image(im_volume)
        total_len = len(normalized_field_magn)
        current_index = int(total_len/2)
        this_img = normalized_field_magn[current_index]
        img = axs[i].imshow(this_img, cmap=my_cmap, vmin=0, vmax=1) 
        signal_squares, noise_squares = plot_signal_noise_squares(ax=axs[i], img=this_img, 
                                                                  center = centers[k][i])
        axs[i].set_xlabel("x")
        axs[i].set_ylabel("y")
        axs[i].set_title(f"{ros} readouts,\n{resolution[i]}x{resolution[i]}x{resolution[i]} resolution, {kspace_samp[i]}% k-space sampling", fontsize = 10)
        snr_calc = calculate_SNR(signal_squares, noise_squares)
        snrs_this_coil.append(snr_calc)
        coil_lines[i][k] = normalized_field_magn[current_index,:,centers[k][i][0]]
        #coverage = np.sum(np.where(this_img>0.50, np.ones_like(this_img), np.zeros_like(this_img)))
        # Show coverage graphically
        # print(coil)
        # print(coverage)
    calculated_snrs.append(snrs_this_coil)
    fig.tight_layout(pad=1.0)
    fig.subplots_adjust(bottom=0.2)
    cbar_ax = fig.add_axes([0.15, 0.08, 0.75, 0.03])
    cb = fig.colorbar(img, label="Normalized", cax=cbar_ax, location="bottom")
    #fig.savefig(f"{coil}_three_readouts.png", dpi=300 ,transparent=True)
    #plt.show()
    plt.close("all")

# B1 CORRECTIONS
# image = 12
# blurred = blurring_2D(image, kernel_size=3, padding=1, rep=3)
# blurred_norm = blurred/np.max(blurred)
# blurred_norm_region = np.ma.where(blurred_norm>0.25, blurred_norm, np.ones_like(blurred_norm))
# new = this_img/blurred_norm_region
# axs[i+1].imshow(new, cmap=my_cmap, vmin=0, vmax=1)
# axs[i+1].set_xlabel("x")
# axs[i+1].set_ylabel("y")
# axs[i+1].set_title(f"{ros} ro, {resolution[i]}x{resolution[i]}x{resolution[i]} res, {kspace_samp[i]}% k samp\nB_1 inhomogeneity correction by blurring" )


fig, axs = plt.subplots(1, 2)
img1alpha = np.load(numpy_files_path+"QuadratureCoil_197_60deg.npy")
img2alpha = np.load(numpy_files_path+"QuadratureCoil_197_120deg.npy")
b1_map = b1_field_map(im1alpha=img1alpha, im2alpha=img2alpha, alpha1=60, alpha2=120, tr=100*1E-3, t1=40*1E-3)
image = np.load(numpy_files_path+"QuadratureCoil_197_X.npy")
normalized_image = norm_magn_image(image)
ind = int(len(normalized_image)/2)
axs[0].imshow(normalized_image[ind], cmap=my_cmap, vmin=0, vmax=1)
b1_map = 1 + np.cos(np.nan_to_num(b1_map, copy=True))
b1_corr_quad = normalized_image[ind]*b1_map[ind]
axs[1].imshow(b1_corr_quad, cmap=my_cmap)

fig, axs = plt.subplots(1, 2)
img1alpha = np.load(numpy_files_path+"BirdcageEnh_197_TG50-60deg.npy")
img2alpha = np.load(numpy_files_path+"BirdcageEnh_197_TG50-120deg.npy")
b1_map = b1_field_map(im1alpha=img1alpha, im2alpha=img2alpha, alpha1=60, alpha2=120, tr=100*1E-3, t1=40*1E-3)
image = np.load(numpy_files_path+"BirdcageEnh_197_TG50.npy")
normalized_image = norm_magn_image(image)
ind = int(len(normalized_image)/2)
axs[0].imshow(normalized_image[ind], cmap=my_cmap, vmin=0, vmax=1)
b1_map = 1 + np.cos(np.nan_to_num(b1_map, copy=True))
b1_corr_enh = normalized_image[ind]*b1_map[ind]
axs[1].imshow(b1_corr_enh, cmap=my_cmap)
#plt.show()
plt.close("all")

cmap = mpl.colormaps.get_cmap("plasma")
fig, axs = plt.subplots(1, 1)
axs.set_ylim([0, 90])
coil_snrs = {
    readouts[0]:[],
    readouts[1]:[],
    readouts[2]:[]
 }
for i, readout in enumerate(readouts):
    for j in range(len(coils)):
        coil_snrs[readout].append(calculated_snrs[j][i])
#print(coil_snrs)
positions = np.arange(len(coils))
width = 0.2
multiplier = 0 
colors = (cmap(0.3), cmap(0.6), cmap(0.9))
for attribute, snrs_coil in coil_snrs.items():
    offset = width*multiplier
    rects = axs.bar(positions+offset, snrs_coil, width, label=attribute, color=colors[multiplier])
    axs.bar_label(rects, padding=3, rotation="vertical")
    multiplier += 1
#axs.grid(zorder=0)
axs.set_title("SNR, Calculated ref NEMA")
axs.set_ylabel("SNR")
axs.set_xticks(positions+width, coils, rotation=25)
axs.legend(loc="upper left")
# plt.show()
plt.close("all")

cvals = np.linspace(0, 1, len(coils))
x_80 = np.linspace(0, 24, 80)
diff_80 = x_80[1]-x_80[0]
x_120 = np.linspace(0, 24, 120)
diff_120 = x_120[1]-x_120[0]
for i, lines in enumerate(coil_lines):
    fig, ax = plt.subplots(1, 1)
    if i==0: diff, extra = diff_120, 1.5/diff_120
    else: diff, extra = diff_80, 1.5/diff_80
    print(diff)
    ax.set_title(f"Intensity comparison for 3D cones {readouts[i]} scan")
    ax.vlines(0, ymin=0, ymax=1, color="k", linestyles="--")
    for j, line in enumerate(lines):
        line = np.flip(line)
        index_t = next(x[0] for x in enumerate(line) if x[1]>0.4)
        x_0 = 0
        start = int(index_t-extra)
        xs = [f*diff for f in range(len(line[start:]))]
        ax.plot(xs, line[start:], color=cmap(cvals[j]), label=coils[j])
        ax.set_xlabel("cm") # Transform from pixel to centimeter
        ax.set_ylabel("Normalized magnitude")
        plt.legend()
        plt.grid(True)
plt.show()

""" FLIP BOOK FUNCTION """

# # Function to update the plot with the next portion of data
# def update_plot(forward=True):
#     global current_index
#     step = display_range if forward else -display_range
#     current_index += step

#     if current_index < 0:
#         current_index = 0
#     elif current_index + display_range > normalized_field_magn.shape[0]-1:
#         current_index = normalized_field_magn.shape[0] - display_range

#     img.set_data(normalized_field_magn[current_index])
#     plt.draw()

# # Create buttons
# ax_next_button = plt.axes([0.81, 0.01, 0.1, 0.05])
# ax_prev_button = plt.axes([0.7, 0.01, 0.1, 0.05])

# next_button = Button(ax_next_button, 'Next')
# prev_button = Button(ax_prev_button, 'Previous')

# # Connect buttons to update functions
# next_button.on_clicked(lambda event: update_plot(forward=True))
# prev_button.on_clicked(lambda event: update_plot(forward=False))


""" FIND CIRCLE OF PHANTOM """
# im_volume = np.load(numpy_files_path+f"Birdcage_2552_X.npy")
# abs_array = abs(im_volume)
# maximum = np.max(abs_array)
# normalized_field_magn = abs_array/maximum
# total_len = len(normalized_field_magn)
# current_index = int(total_len/2)

# y = [23, 60]
# radius=(y[1]-y[0])/2
# center = [20+radius, 23+radius]
# print(center)

# len_1d = len(normalized_field_magn[current_index])
# circ_mask = create_circular_mask(h=len_1d, w=len_1d, center=center, radius=diffx/2)
# #print(circ_mask)

# img = plt.imshow(normalized_field_magn[current_index]*circ_mask, cmap=my_cmap, vmin=0, vmax=1) 
# plt.xlabel("x")
# plt.ylabel("y")
# plt.colorbar(img, label="Normalized") #cax=cbar_ax, location="bottom"
# plt.title(f"Birdcage coil,\n2552 readouts, 3x3x3, 25% k-space sampling")
# plt.show()