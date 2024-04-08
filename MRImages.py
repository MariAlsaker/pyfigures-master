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
    # Only used for localizing phantom
    # radius=37/2 # found from birdcage 2552 plot
    # im_centers = [[38, 40], [38, 40], [43, 40], [39, 40], [20+radius, 23+radius], [20+radius, 23+radius]]
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

def blurring_2D(img, kernel_size, padding=0, rep=1):
    k = np.ones((kernel_size))/kernel_size
    out = img.copy()
    if padding != 0:
        np.pad(out, pad_width=padding, mode="constant")
    for j in range(rep):
        for i in range(2):
            out = convolve1d(out, weights=k, axis=i)
    return out

def plot_signal_noise_squares(ax, img, center,offset=15, scale = 1, c="w"):
    S_width = int(7*scale)
    N_width = int(11*scale)
    offset = int(17*scale)
    diff = (N_width-S_width)/2
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
real_names = ["Original surface coil", "Assadi's surface coil", "Single loop coil", "Quadrature coil", "Birdcage coil", "Birdcage with enhancing coil"]
readouts = ["197", "1402", "2552"]
kspace_samp = ["10", "12", "25"]
resolution = ["4.5", "3", "3"]

centers3 = [[38, 55], [38, 54], [42, 54], [39, 54], [38, 54], [38, 52]]
scale = 120/80
centers = [[[int(x*scale),int(y*scale)], [x,y], [x,y]] for x,y in centers3]

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
    axs[3].text(0.1, 0.6, f"{real_names[k]}", clip_on=False, fontweight="bold", fontsize="x-large")
    axs[3].axis("off")
    snrs_this_coil = []
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
        if i == 0:
            signal_squares, noise_squares = plot_signal_noise_squares(ax=axs[i], img=this_img, scale = scale, center = centers[k][i])
        else:
            signal_squares, noise_squares = plot_signal_noise_squares(ax=axs[i], img=this_img, center = centers[k][i])
        axs[i].set_xlabel("x")
        axs[i].set_ylabel("y")
        axs[i].axis("off")
        axs[i].text(5, 5, f"{i+1}", color="k", fontweight="bold", backgroundcolor="w")
        snr_calc = calculate_SNR(signal_squares, noise_squares)
        plt.tight_layout( pad=0)
        snrs_this_coil.append(snr_calc)
        coil_lines[i][k] = normalized_field_magn[current_index,:,centers[k][i][0]]
    calculated_snrs.append(snrs_this_coil)
    cbar_ax = fig.add_axes([0.55, 0.2, 0.4, 0.03])
    cb = fig.colorbar(img, label="Normalized magnitude", cax=cbar_ax, location="bottom")
    #fig.savefig(f"{coil}_three_readouts.png", dpi=300 ,transparent=True)
#plt.show()
plt.close("all")

# B1 CORRECTION - blurred version
for coil in coils[:-2]: #[-2:]
    r="197"
    image = np.load(numpy_files_path+f"{coil}_{r}_X.npy") #_TG50
    image = norm_magn_image(image)
    ind = int(len(image)/2)
    image_slice = image[ind]
    fig, axs = plt.subplots(1, 3, figsize=(10, 4))
    axs[0].imshow(image_slice, cmap=my_cmap)
    blurred_slice = blurring_2D(image_slice, kernel_size=8, padding=1, rep=3)
    blurred_norm = blurred_slice/np.max(blurred_slice)
    blurred_norm = np.ma.where(blurred_slice<0.15, np.ones_like(blurred_slice), blurred_slice)
    blurr_div = image_slice/blurred_norm
    axs[1].imshow(blurred_norm, cmap=my_cmap)
    axs[2].imshow(blurr_div, cmap=my_cmap)
    for i,ax in enumerate(axs):
        ax.axis("off")
        ax.text(5, 8, f"{i+1}", color="k", fontweight="bold", backgroundcolor="w")
    plt.tight_layout(pad=0)
    #fig.savefig(f"{coil}_{r}_blurr_corr.png", dpi=300 ,transparent=True)
    plt.show()
plt.close("all")

# B1 CORRECTION - b1 field map
def signal_corr_infinityTR(flipangle, tr, t1):
    num = 1-np.cos(flipangle)*np.e**(-tr/t1)
    denom = 1-np.e**(-tr/t1)
    return num/denom
def b1_corr_doubleangle(img60, img120, method="multiply"):
    s1 = abs(img60)*signal_corr_infinityTR(flipangle=60, tr=100*1E-3, t1=40*1E-3)
    s2 = abs(img120)*signal_corr_infinityTR(flipangle=120, tr=100*1E-3, t1=40*1E-3)
    b1map = np.arccos(s2/(2*s1))
    b1map = np.nan_to_num(b1map, copy=True, nan=0.001)
    fa_max = np.max(b1map)
    if method=="multiply":
        # Reversing - no flip angle multiplied by one, the others less to reduce difference
        correction_map = 1-b1map/fa_max 
    elif method=="divide": 
        # Preparing for division - no flip angle is divided by one, the others more to reduce difference
        correction_map = 1+b1map/fa_max
    else:
        Exception(f"Method \"{method}\" is invalid")
    return b1map, correction_map

names_b1_corr_files = ["QuadratureCoil_197", "BirdcageEnh_197_TG50"]

for name in names_b1_corr_files:
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
    axs = axs.flatten()
    plt.tight_layout(pad=0)
    img1alpha = np.load(numpy_files_path+f"{name}_60deg.npy")
    img2alpha = np.load(numpy_files_path+f"{name}_120deg.npy")
    image = np.load(numpy_files_path+f"{name}_X.npy")
    b1map, correction_mul = b1_corr_doubleangle(img60=img1alpha, img120=img2alpha, method="multiply")
    _, correction_div = b1_corr_doubleangle(img60=img1alpha, img120=img2alpha, method="divide")
    normalized_img = norm_magn_image(image)
    ind = int(len(normalized_img)/2)
    corrected_img_mul = norm_magn_image( normalized_img[ind]*correction_mul[ind])
    corrected_img_div = norm_magn_image( normalized_img[ind]/correction_div[ind])
    axs[0].imshow(b1map[ind], cmap="plasma" ,vmin=np.pi/4, vmax=np.pi/2)
    imgs = [normalized_img[ind], corrected_img_mul, corrected_img_div]
    for i,ax in enumerate(axs):
        ax.axis("off")
        ax.text(5, 8, f"{i+1}", color="k", fontweight="bold", backgroundcolor="w")
        if i>0:
            axs[i].imshow(imgs[i-1], cmap=my_cmap, vmin=0, vmax=1)
    plt.tight_layout(pad=0)
    #plt.show()
    # fig.savefig(f"{name}_b1_DA_corr.png", dpi=300 ,transparent=True)
plt.close("all")

# Coil SNR plot
cmap = mpl.colormaps.get_cmap("plasma")
fig, axs = plt.subplots(1, 1, figsize = (7, 6))
axs.set_ylim([0, 90])
scans = ["3D cones 1", "3D cones 2", "3D cones 3"]
coil_snrs = {
    scans[0]:[],
    scans[1]:[],
    scans[2]:[]
 }
for i, scan in enumerate(scans):
    for j in range(len(coils)):
        coil_snrs[scan].append(calculated_snrs[j][i])
positions = np.arange(len(coils))
width = 0.2
multiplier = 0 
colors = (cmap(0.3), cmap(0.6), cmap(0.9))
for attribute, snrs_coil in coil_snrs.items():
    offset = width*multiplier
    rects = axs.bar(positions+offset, snrs_coil, width, label=attribute, color=colors[multiplier])
    axs.bar_label(rects, padding=3, rotation="vertical", fmt="%.0f")
    multiplier += 1
axs.set_title("SNR calculated from MR images")
axs.set_ylabel("SNR")
axs.set_xticks(positions+width, coils, rotation=25)
axs.legend(loc="upper left")
#plt.show()
#fig.savefig(f"SNRs_all_coils", dpi=300 ,transparent=True)
plt.close("all")

# Show circle of phantom
phantom_image = np.load(numpy_files_path+f"Birdcage_2552_X.npy")
abs_phantom = abs(phantom_image)
maximum = np.max(abs_phantom)
normalized_phantom = abs_phantom/maximum
total_len = len(normalized_phantom)
current_index = int(total_len/2)
y = [23, 60]
radius=(y[1]-y[0])/2
center = [20+radius, 23+radius]
len_1d = len(normalized_phantom[current_index])
circ_mask = create_circular_mask(h=len_1d, w=len_1d, center=center, radius=radius)
phantom_img_theory = np.ones_like(normalized_phantom[current_index])*circ_mask*-1
# Actual line plots
linestyle_tuple = [
     "solid",
     "dotted", 
     "dashed",
     "dashdot",
     (0, (5, 1)), # densely dashed
     (0, (10, 3)), # long dashed
     (0, (3, 1, 1, 1))] #densely dashdotted
cvals = np.linspace(0, 0.8, len(coils))
x_80 = np.linspace(0, 24, 80)
diff_80 = x_80[1]-x_80[0]
x_120 = np.linspace(0, 24, 120)
diff_120 = x_120[1]-x_120[0]
for i, lines in enumerate(coil_lines):
    fig, ax = plt.subplots(1, 1, figsize=(8,6))
    if i==0: diff, extra = diff_120, 1.5/diff_120
    else: diff, extra = diff_80, 1.5/diff_80
    ax.set_title(f"Intensity comparison for scan '3D cones {i+1}'")
    ax.vlines(0, ymin=0, ymax=1, color="k", linestyles="--")
    for j, line in enumerate(lines):
        line = np.flip(line)
        index_t = next(x[0] for x in enumerate(line) if x[1]>0.4)
        x_0 = 0
        start = int(index_t-extra)
        xs = [f*diff for f in range(len(line[start:]))]
        ax.plot(xs, line[start:], color=cmap(cvals[j]), label=coils[j], linestyle=linestyle_tuple[j])
        ax.set_xlabel("cm") # Transform from pixel to centimeter
        ax.set_ylabel("Normalized magnitude")
        phantom_d = 11.5 #cm
        offset_from_coil = 1.2
        ax.set(xlim=(0, 22), ylim=(0, 1))
        im = ax.imshow(phantom_img_theory, extent=[17,20,0.5,0.65], aspect="auto", cmap="Greys", vmin=-2, vmax=1) # (xmin, xmax, ymin, ymax)
        ax.plot([17+(20-17)/2, 17+(20-17)/2], [0.5, 0.65], "k-")
        ax.axvspan(offset_from_coil, offset_from_coil+phantom_d, facecolor="lightgray", alpha=0.1)
        plt.legend()
        #plt.savefig( f"3Dcones{i+1}_line_plots.png", dpi=300, transparent=True)
#plt.show()
plt.close("all")


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