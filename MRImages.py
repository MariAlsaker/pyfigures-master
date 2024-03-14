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

def calculate_SNR(S_values, N_values):
    sd = np.std(N_values)
    signal = np.mean(S_values)
    snr = 0.655 *signal/sd
    return snr

def gaussianKernel(size, sigma, twoDimensional=True):
    if twoDimensional:
        kernel = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma**2)) * np.e ** ((-1*((x-(size-1)/2)**2+(y-(size-1)/2)**2))/(2*sigma**2)), (size, size))
    else:
        kernel = np.fromfunction(lambda x: np.e ** ((-1*(x-(size-1)/2)**2) / (2*sigma**2)), (size,))
    return kernel / np.sum(kernel)

def blurring_2D(img, kernel_size, padding=0, rep=1, gaussian=False):
    if gaussian:
        k = gaussianKernel(kernel_size, sigma=3)
    else:
        k = np.ones((kernel_size))/kernel_size
    out = img.copy()
    if padding != 0:
        np.pad(out, pad_width=padding, mode="constant")
    for j in range(rep):
        for i in range(2):
            out = convolve1d(out, weights=k, axis=i)
    return out

numpy_files_path = "/Users/marialsaker/git/pyfigures-master/MRI_data/"
coils = ["OrigSurface", "AssadiSurface", "SingleLoop", "QuadratureCoil", "Birdcage", "BirdcageEnh"] # "Birdcage2nd"
readouts = ["197", "1402", "2552"]
centers = [[[58, 80], [38, 58], [38, 56]], 
           [[58, 80], [38, 57], [38, 56]], 
           [[65, 77], [43, 58], [42, 55]], 
           [[60, 78], [38, 59], [39, 56]], 
           [[58, 77], [38, 56], [38, 53]], 
           [[58, 75], [38, 55], [38, 53]]]
snrs = []
f0s = []
coil = coils[0]
for k, coil in enumerate(coils):
    snr = 2
    f0 = 33.78
    # with open(numpy_files_path+f"{coil}_X_optimizer.txt", "r") as optim_f:
    #     for line in optim_f:
    #         splitted = line.split(" ")
    #         if splitted[0] == "SNR":
    #             print(splitted[2])
    #             snr = float(splitted[-1].strip())
    #             snrs.append(snr)
    #         if splitted[0] == "f0:":
    #             f0 = int(splitted[-2])
    #             f0s.append(f0)
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 5))
    axs = axs.flatten()
    fig.suptitle(f"Normalized magnitude plots for 3D cones sequence with {coil} coil,\n SNR={snr:.2f}, f0={f0} (Bloch Siegert)")
    for i, ros in enumerate(readouts):
        im_volume = np.load(numpy_files_path+f"{coil}_{ros}_X.npy")
        abs_array = abs(im_volume)
        maximum = np.max(abs_array)
        normalized_field_magn = abs_array/maximum
        total_len = len(normalized_field_magn)
        display_range = 1
        current_index = int(total_len/2)
        S_width = 7
        N_width = 11
        inc =(N_width-S_width)/2
        center = centers[k][i]
        if normalized_field_magn.shape[0] == 120:
            add = 20
        else:
            add = 15
        xsS = np.array([center[0]-S_width/2, center[0]+S_width/2])
        ysS = np.array([center[1]-S_width/2, center[1]+S_width/2])
        signal_squares = normalized_field_magn[current_index][center[1]-S_width//2:center[1]+S_width//2+1, center[0]-S_width//2:center[0]+S_width//2+1]
        xsN = np.array([xsS[0]-inc, xsS[1]+inc])
        ysN = np.array([ysS[0]-inc, ysS[1]+inc]) + add
        center = [center[0], center[1]+add]
        noise_squares = normalized_field_magn[current_index][center[1]-N_width//2:center[1]+N_width//2+1, center[0]-N_width//2:center[0]+N_width//2+1]
        for j in range(2):
            axs[i].plot([xsS[j],xsS[j]], ysS, color = "w")
            axs[i].plot(xsS, [ysS[j],ysS[j]], color = "w")
            axs[i].plot([xsN[j],xsN[j]], ysN, color = "w")
            axs[i].plot(xsN, [ysN[j],ysN[j]], color = "w")

        img = axs[i].imshow(normalized_field_magn[current_index], cmap=my_cmap, vmin=0, vmax=1) 
        axs[i].set_xlabel("x")
        axs[i].set_ylabel("y")
        axs[i].set_title(f"{ros} readouts,\nmax = {maximum:.0f}, SNR = {calculate_SNR(signal_squares, noise_squares):.2f}")
        if i == len(readouts)-1 and k <4:
            blurred = blurring_2D(normalized_field_magn[current_index], kernel_size=3, padding=1, rep=5)
            blurred_norm = blurred/np.max(blurred)
            blurred_norm = np.ma.where(blurred_norm>0.25, blurred_norm, np.ones_like(blurred_norm))
            new = normalized_field_magn[current_index]/blurred_norm
            axs[i+1].imshow(new, cmap=my_cmap, vmin=0, vmax=1)
    fig.subplots_adjust(bottom=0.2)
    cbar_ax = fig.add_axes([0.15, 0.1, 0.7, 0.05])
    cb = fig.colorbar(img, label="Normalized", cax=cbar_ax, location="bottom")
    #fig.savefig(f"{coil}_three_readouts.png", dpi=300 ,transparent=True)
    plt.show()
    plt.close("all")

# cmap = mpl.colormaps.get_cmap(my_cmap)
# c_num = np.linspace(0,1,len(coils), endpoint=False)
# colors = [cmap(num) for num in c_num]
# plt.bar(coils, snrs, color=colors, zorder=3)
# plt.grid(zorder=0)
# plt.title("SNR, Bloch Siegert method")
# plt.ylabel("SNR")
# plt.show()

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


