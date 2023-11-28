import numpy as np
from matplotlib import pyplot as plt

folder = "/Users/marialsaker/nanovnasaver_files/"

def extract_vals_1p(folder, file):
    freqs = []
    reals = []
    ims = []
    with open(folder+file) as f_s11:
        for line in f_s11.readlines():
            if line.split(" ")[0] == "#": continue
            f, r, i = line.split(" ")
            freqs.append(float(f)*1E-6)
            reals.append(float(r))
            ims.append(float(i))

    freqs = np.array(freqs)
    reals = np.array(reals)
    ims = np.array(ims)
    magn = np.sqrt(reals**2 + ims**2)
    phase = np.arctan(ims/reals)
    return freqs, reals, ims, magn, phase

def extract_vals_2p(folder, file):
    vals = np.zeros((101, 9))
    with open(folder+file) as f_s11:
        for i, line in enumerate(f_s11.readlines()):
            if line.split(" ")[0] == "#": continue
            i = i-1
            f, r_s11, i_s11, r_s21, i_s21, _, __, ___, ____ = line.split(" ")
            vals[i][0] = float(f)*1E-6
            vals[i][1:3] = r_s11, i_s11
            vals[i][5:7] = r_s21, i_s21
    vals[:,3] = np.sqrt(vals[:,1]**2 + vals[:,2]**2)
    vals[:,4] = np.arctan(vals[:,2]/(vals[:,1]))
    vals[:,7] = np.sqrt(vals[:,5]**2 + vals[:,6]**2)
    vals[:,4] = np.arctan(vals[:,6]/(vals[:,5]))
    return vals


def plot_magn_phase(ax, freqs, magn_db, phase, color1, color2, show_phase=True, name="S11"):
    twin1 = ax.twinx()
    ax.plot(freqs, magn_db, color1+"-", label=f"|{name}|")
    if show_phase:
        twin1.plot(freqs, phase, color2+"-", label="/_ {name}")
        twin1.set_ylabel("Phase [rad/pi]")
        twin1.yaxis.label.set_color(color2)
        twin1.tick_params(axis = 'y', colors=color2)
        twin1.legend()
        ax.yaxis.label.set_color(color1)
        ax.tick_params(axis = 'y', colors=color1)
    ax.set_xlabel("Frequency [MHz]")
    ax.set_ylabel("Magnitude [dB]")
    ax.legend()
    ax.grid(True)
    return

freqs, reals, ims, magn, phase = extract_vals_1p(folder, "1-port_single_lonely_loop_elbow.s1p")
print(len(freqs))
magn_db = 20*np.log10(magn)
fig = plt.figure(figsize=[12, 8])
ax1 = fig.add_subplot(2,1,1)
plot_magn_phase(ax1, freqs, magn_db, phase, "r", "b")
ax1.set_title("S_11 plot of single lonely loop loaded with elbow")

values_right = extract_vals_2p(folder, "2-port_quad_loop_elbow_right.s2p")
values_left = extract_vals_2p(folder, "2-port_quad_loop_elbow_left.s2p")
s11_db_right = 20*np.log10(values_right[:,3])
s11_db_left = 20*np.log10(values_left[:,3])
s21_db = 20*np.log10(values_right[:,7])
ax2 = fig.add_subplot(2,1,2)
plot_magn_phase(ax2, values_right[:,0], s11_db_right, values_right[:,4], "r", "r", show_phase=False)
plot_magn_phase(ax2, values_left[:,0], s11_db_left, values_left[:,4], "g", "g", show_phase=False)
plot_magn_phase(ax2, values_right[:,0], s21_db, values_left[:,4], "b", "b", show_phase=False, name="S21")
ax2.set_title("S_11 plot of quad coil right and left, loaded with elbow")

plt.show()