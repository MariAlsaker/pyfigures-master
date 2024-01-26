import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker
import collections.abc
#hyper needs the four following aliases to be done manually.
collections.Iterable = collections.abc.Iterable
collections.Mapping = collections.abc.Mapping
collections.MutableSet = collections.abc.MutableSet
collections.MutableMapping = collections.abc.MutableMapping
#Now import hyper
#import hyper
from smithplot import SmithAxes

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
    plots = ax.plot(freqs, magn_db, label=f"|{name}|")
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    plots[0].set(color = color1)
    if show_phase:
        plots = twin1.plot(freqs, phase, label="/_ {name}")
        plots[0].set(color = color2)
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
magn_db = 20*np.log10(magn)
fig = plt.figure(figsize=[12, 6])
ax1 = fig.add_subplot(1,2,1)
c1 = "steelblue"
c2 = "orange"
plot_magn_phase(ax1, freqs, magn_db, phase, color1=c1, color2=c2, show_phase=False)
ax1.set_title("S_11 plot of single loop loaded with elbow")
min_mag = np.min(magn)
min_index = np.where(magn==min_mag)
plots = ax1.plot(freqs[min_index], magn_db[min_index])
plots[0].set(color=c2, marker = "o")
ax2 = fig.add_subplot(1,2,2, projection="smith")
vals_s11 = (reals + ims * 1j)
plots = ax2.plot(vals_s11, markevery=1, label="equipoints=11", equipoints=11, datatype=SmithAxes.S_PARAMETER)
plots[0].set(color=c1)
plots = ax2.plot(vals_s11[min_index], markevery=1, label="equipoints=11", equipoints=11, datatype=SmithAxes.S_PARAMETER)
plots[0].set(color=c2, marker="o")
plt.savefig('s11_smith.png', transparent=True)

values_right = extract_vals_2p(folder, "2-port_quad_loop_elbow_right.s2p")
values_left = extract_vals_2p(folder, "2-port_quad_loop_elbow_left.s2p")
s11_db_right = 20*np.log10(values_right[:,3])
s11_db_left = 20*np.log10(values_left[:,3])
s21_db = 20*np.log10(values_right[:,7])
ax2 = fig.add_subplot(1,1,1)
plot_magn_phase(ax2, values_right[:,0], s11_db_right, values_right[:,4], "r", "r", show_phase=False, name="S11")
plot_magn_phase(ax2, values_left[:,0], s11_db_left, values_left[:,4], "g", "g", show_phase=False, name="S22")
plot_magn_phase(ax2, values_right[:,0], s21_db, values_left[:,4], "b", "b", show_phase=False, name="S21")
ax2.set_title("S parameter plot of quad coil right (1) and left (2), loaded with elbow")

