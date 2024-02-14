import scipy.io
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
import numpy as np

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
            img = ax.pcolormesh(f_slc, cmap="plasma", vmin=0, vmax=vmax)
            plt.colorbar(img, label="[%]")
            ax.set_title(f"Magnitude of complex value")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            initialized = True
        else:
            img = ax.pcolormesh(f_slc, cmap="plasma", vmin=0, vmax=vmax, animated=True)
        imgs.append([img])
    ani = animation.ArtistAnimation(fig, imgs, interval=50, blit=True,
                            repeat_delay=1000)
    plt.show()
    if gif_name:
        writergif = animation.FFMpegWriter(fps=30)
        ani.save("Figures/"+gif_name, writer=writergif)

path = "/Users/marialsaker/MRImages/Single_loop/Exam13509/Series4/"

mat = scipy.io.loadmat(path + "ScanArchive_NO1011MR01_20240209_082731323.mat")
print(mat.keys())
im_volume = mat['bb']
im_volume = np.transpose(im_volume, (2,0,1)) # (x, y, z) = (0, 1, 2)

abs_array = abs(im_volume)
normalized_field_magn = abs_array/np.max(abs_array)

# Initial display range
display_range = 2
current_index = 55

# Create the initial plot
fig, ax = plt.subplots()
img = ax.imshow(normalized_field_magn[current_index], cmap="plasma", vmin=0, vmax=1) 
cb = plt.colorbar(img, label="Normalized")
ax.set_title(f"Magnitude of complex value")
ax.set_xlabel("x")
ax.set_ylabel("y")

# Function to update the plot with the next portion of data
def update_plot(forward=True):
    global current_index
    step = display_range if forward else -display_range
    current_index += step

    if current_index < 0:
        current_index = 0
    elif current_index + display_range > normalized_field_magn.shape[0]-1:
        current_index = normalized_field_magn.shape[0] - display_range

    img.set_data(normalized_field_magn[current_index])
    plt.draw()

# Create buttons
ax_next_button = plt.axes([0.81, 0.01, 0.1, 0.05])
ax_prev_button = plt.axes([0.7, 0.01, 0.1, 0.05])

next_button = Button(ax_next_button, 'Next')
prev_button = Button(ax_prev_button, 'Previous')

# Connect buttons to update functions
next_button.on_clicked(lambda event: update_plot(forward=True))
prev_button.on_clicked(lambda event: update_plot(forward=False))

# Show the plot
plt.show()

