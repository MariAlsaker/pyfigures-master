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
            linevals = line.split(" ")
            freqs.append(float(linevals[0])*1E-6)
            reals.append(float(linevals[1]))
            ims.append(float(linevals[2]))

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


def plot_magn_phase(ax, freq_lists, magn_db_lists, phase_list, magn_colors, phase_colors, show_phase=True, names="S11"):
    for i, freqs in enumerate(freq_lists):
        plots = ax.plot(freqs, magn_db_lists[i], label=f"|S11| {names[i]}")
        plots[0].set(color = magn_colors[i])
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    if show_phase:
        twin1 = ax.twinx()
        for i, freqs in enumerate(freq_lists):
            plots = twin1.plot(freqs, phase_list[i], label=f"/_ {names[i]}")
            plots[0].set(color = phase_colors[i])
        twin1.set_ylabel("Phase [rad/pi]")
        twin1.yaxis.label.set_color("blue")
        twin1.tick_params(axis = 'y', colors="blue")
        twin1.legend()
        ax.yaxis.label.set_color("black")
        ax.tick_params(axis = 'y', colors="black")
    ax.set_xlabel("Frequency [MHz]")
    ax.set_ylabel("Magnitude [dB]")
    ax.grid(True)
    return

def plot_res(ax, magn, magn_db, freqs, color):
    min_mag = np.min(magn)
    min_db = 20*np.log10(min_mag)
    min_index = np.where(magn==min_mag)
    plots = ax.plot(freqs[min_index], magn_db[min_index], label=f"min= {min_db:.0f}dB")
    plots[0].set(color=color, marker = "o")
    return min_index

freqs1, reals1, ims1, magn1, phase1 = extract_vals_1p(folder, "1-port_singleloop_elbow_load_pult_narrow.s1p")
freqs2, reals2, ims2, magn2, phase2 = extract_vals_1p(folder, "1-port_singleloop_no_load_pult_narrow.s2p")
magn1_db = 20*np.log10(magn1)
magn2_db = 20*np.log10(magn2)
fig = plt.figure(figsize=[6, 10])
ax1 = fig.add_subplot(2,1,1)
c1 = "steelblue"
c2 = "orange"
plot_magn_phase(ax1, [freqs1, freqs2], [magn1_db, magn2_db], [phase1, phase2], magn_colors=[c1,c2], phase_colors=[c1,c2], show_phase=False, names=["elbow load", "no load"])
ax2 = fig.add_subplot(2,1,2, projection="smith")
vals_s11 = (reals1 + ims1 * 1j)
plots = ax2.plot(vals_s11, markevery=1, label="equipoints=11", equipoints=11, datatype=SmithAxes.S_PARAMETER)
plots[0].set(color=c1)
vals_s11_2 = (reals2 + ims2 * 1j)
plots = ax2.plot(vals_s11_2, markevery=1, label="equipoints=11", equipoints=11, datatype=SmithAxes.S_PARAMETER)
plots[0].set(color=c2)

min_index1 = plot_res(ax=ax1, magn=magn1, magn_db=magn1_db, freqs=freqs1, color="turquoise")
plots = ax2.plot(vals_s11[min_index1], markevery=1, label="equipoints=11", equipoints=11, datatype=SmithAxes.S_PARAMETER)
plots[0].set(color="turquoise", marker="o")

min_index2 = plot_res(ax=ax1, magn=magn2, magn_db=magn2_db, freqs=freqs2, color="red")
plots = ax2.plot(vals_s11_2[min_index2], markevery=1, label="equipoints=11", equipoints=11, datatype=SmithAxes.S_PARAMETER)
plots[0].set(color="red", marker="o")

ax1.legend()
fig.suptitle("Laboratory tests of the surface loop coil \n loaded and unloaded")
ax1.set_title("|S11| magnitude")
ax2.set_title("Real and imaginary impedance (Smith chart)")
plt.savefig('s11_smith.png', transparent=True)
#plt.show()

# values_right = extract_vals_2p(folder, "2-port_quad_loop_elbow_right.s2p")
# values_left = extract_vals_2p(folder, "2-port_quad_loop_elbow_left.s2p")
# s11_db_right = 20*np.log10(values_right[:,3])
# s11_db_left = 20*np.log10(values_left[:,3])
# s21_db = 20*np.log10(values_right[:,7])
# ax2 = fig.add_subplot(1,1,1)
# plot_magn_phase(ax2, values_right[:,0], s11_db_right, values_right[:,4], "r", "r", show_phase=False, name="S11")
# plot_magn_phase(ax2, values_left[:,0], s11_db_left, values_left[:,4], "g", "g", show_phase=False, name="S22")
# plot_magn_phase(ax2, values_right[:,0], s21_db, values_left[:,4], "b", "b", show_phase=False, name="S21")
# ax2.set_title("S parameter plot of quad coil right (1) and left (2), loaded with elbow")

# Demonstrate that the quadrature coil is a reciprocal network, meaning that the S_21 = S_12