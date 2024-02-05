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
            linevals= line.split(" ") # f, r_s11, i_s11, r_s21, i_s21, _, __, ___, ____
            vals[i][0] = float(linevals[0])*1E-6
            vals[i][1:3] = linevals[1], linevals[2] 
            vals[i][5:7] = linevals[3], linevals[4]
    vals[:,3] = np.sqrt(vals[:,1]**2 + vals[:,2]**2)
    vals[:,4] = np.arctan(vals[:,2]/(vals[:,1]))
    vals[:,7] = np.sqrt(vals[:,5]**2 + vals[:,6]**2)
    vals[:,8] = np.arctan(vals[:,6]/(vals[:,5]))
    return vals # f, r_s11, i_s11, |S11|, phaseS11, r_s21, i_s21, |S21|, phaseS21


def plot_magn_phase(ax, freq_lists, magn_db_lists, phase_list, magn_colors, phase_colors, show_phase=True, names="S11"):
    for i, freqs in enumerate(freq_lists):
        plots = ax.plot(freqs, magn_db_lists[i], label=f"|S11| {names[i]}", marker=".")
        plots[0].set(color = magn_colors[i])
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    if show_phase:
        twin1 = ax.twinx()
        for i, freqs in enumerate(freq_lists):
            plots = twin1.plot(freqs, phase_list[i], label=f"/_ {names[i]}", marker=".")
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


def plot_s11_s21(ax, freq_lists, s11_db_lists, s21_lists, s11_colors, s21_colors, names):
    for i, freqs in enumerate(freq_lists):
        plots = ax.plot(freqs, s21_lists[i], label=f"|S21| {names[i]}")
        plots[0].set(color = s21_colors[i])
    for i, freqs in enumerate(freq_lists):
        plots = ax.plot(freqs, s11_db_lists[i], label=f"|S11| {names[i]}")
        plots[0].set(color = s11_colors[i])
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.set_xlabel("Frequency [MHz]")
    ax.set_ylabel("Magnitude [dB]")
    ax.grid(which="both")
    return

def plot_res(ax, magn, magn_db, freqs, color):
    min_mag = np.min(magn)
    min_index = np.where(magn==min_mag)
    min_db = magn_db[min_index]
    print(freqs[min_index], color)
    plots = ax.plot(freqs[min_index], min_db, label=f"min= {min_db[0]:.0f}dB", marker=".")
    plots[0].set(color=color, marker = "o")
    return min_index

def q_factor(freqs, magn_db, reals, ims, z0):
    """ Q factor calculated as defined in paper by J. Michael Drozd and William T. Joines 1996 """
    index_w0 = np.argmin(magn_db)
    R_nom = (1 + reals[index_w0])*(1 - reals[index_w0]) - ims[index_w0]**2
    R_denom = (1 - reals[index_w0])**2+ims[index_w0]**2
    R = z0 * R_nom / R_denom 
    n=10
    slopeRs = reals[int(index_w0-n/2):int(index_w0+n/2)]
    slopeIs = ims[int(index_w0-n/2):int(index_w0+n/2)]
    slopews = freqs[int(index_w0-n/2):int(index_w0+n/2)]
    Xs = z0* 2*slopeIs/((1-slopeRs)**2 + slopeIs**2)
    partX = (n*(np.sum(slopews*Xs)) - np.sum(slopews)*np.sum(Xs)) / (n*np.sum(slopews**2) - np.sum(slopews)**2)
    return freqs[index_w0]/(2*R+2*z0) * partX

# freqs1, reals1, ims1, magn1, phase1 = extract_vals_1p(folder, "1-port_singleloop_elbow_load_pult_narrow.s1p")
# freqs2, reals2, ims2, magn2, phase2 = extract_vals_1p(folder, "1-port_singleloop_no_load_pult_narrow.s2p")
# magn1_db = 20*np.log10(magn1)
# magn2_db = 20*np.log10(magn2)
# fig = plt.figure(figsize=[6, 10])
# ax1 = fig.add_subplot(2,1,1)
# ax1.vlines(x=[33.8], ymin=-42, ymax=1, colors="k", linestyles="--", label="f0=33.8MHz")
# c1 = "steelblue"
# c2 = "orange"
# plot_magn_phase(ax1, [freqs1, freqs2], [magn1_db, magn2_db], [phase1, phase2], magn_colors=[c1,c2], phase_colors=[c1,c2], show_phase=False, names=["elbow load", "no load"])
# ax2 = fig.add_subplot(2,1,2, projection="smith")
# vals_s11 = (reals1 + ims1 * 1j)
# plots = ax2.plot(vals_s11, markevery=1, label="equipoints=11", equipoints=11, datatype=SmithAxes.S_PARAMETER)
# plots[0].set(color=c1)
# vals_s11_2 = (reals2 + ims2 * 1j)
# plots = ax2.plot(vals_s11_2, markevery=1, label="equipoints=11", equipoints=11, datatype=SmithAxes.S_PARAMETER)
# plots[0].set(color=c2)

# min_index1 = plot_res(ax=ax1, magn=magn1, magn_db=magn1_db, freqs=freqs1, color="turquoise")
# plots = ax2.plot(vals_s11[min_index1], markevery=1, label="equipoints=11", equipoints=11, datatype=SmithAxes.S_PARAMETER)
# plots[0].set(color="turquoise", marker="o")

# min_index2 = plot_res(ax=ax1, magn=magn2, magn_db=magn2_db, freqs=freqs2, color="red")
# plots = ax2.plot(vals_s11_2[min_index2], markevery=1, label="equipoints=11", equipoints=11, datatype=SmithAxes.S_PARAMETER)
# plots[0].set(color="red", marker="o")
# ax1.legend()
# fig.suptitle("Laboratory tests of the single loop coil \n loaded and unloaded")
# ax1.set_title("|S11| magnitude")
# ax2.set_title("Real and imaginary impedance (Smith chart)")
# #plt.savefig('s11_smith_singleloop.png', transparent=True)
# print(vals_s11[min_index1])
# print(reals1[min_index1]*50+50, ims1[min_index1]*50)
# plt.show()

# Q_load = q_factor(freqs1, magn1_db, reals1, ims1, z0=50)
# Q_noload = q_factor(freqs2, magn2_db, reals2, ims2, z0=50)
# Q_ratio = Q_noload/Q_load
# print(f"Unloaded Q = {Q_noload:.4f} and loaded Q = {Q_load:.4f}\n> Q ratio = {Q_ratio:.4f}")

# Demonstrate that the quadrature coil is a reciprocal network, meaning that the S_21 = S_12
# values_first = extract_vals_2p(folder, "2-port_quadloop_no_load_pult_narrow.s2p")
# values_switch = extract_vals_2p(folder, "2-port_quadloop_no_load_pult_switch_narrow.s2p")
values_first = extract_vals_2p(folder, "2-port_quadloop_elbow_load_pult_narrow.s2p")
values_switch = extract_vals_2p(folder, "2-port_quadloop_elbow_load_pult_switch_narrow.s2p")
values_first[:,3] = 20*np.log10(values_first[:,3])
values_first[:,7] = 20*np.log10(values_first[:,7])
values_switch[:,3] = 20*np.log10(values_switch[:,3])
values_switch[:,7] = 20*np.log10(values_switch[:,7])
fig = plt.figure(figsize=[8, 6])
ax2 = fig.add_subplot(1,1,1)
ax2.vlines(x=[33.8],ymin=-55, ymax=1, colors=["k"], linestyles="--", label="f0=33.8MHz")
plot_s11_s21(ax=ax2, freq_lists=[values_first[:,0], values_switch[:,0]], 
                s11_db_lists=[values_first[:,3], values_switch[:,3]], s11_colors=["steelblue", "orange"],
                s21_lists=[values_first[:,7], values_switch[:,7]], s21_colors=["turquoise", "salmon"],
                names=["no load", "no load switch"])
min_i_first = plot_res(ax2, magn=1/20*10**values_first[:,3], magn_db=values_first[:,3], freqs=values_first[:,0], color="darkblue")
min_i_switch = plot_res(ax2, magn=1/20*10**values_switch[:,3], magn_db=values_switch[:,3], freqs=values_switch[:,0], color="orangered")
#ax2.set_title("S11 and S12 magnitudes for both connections,\nquadrature coil was loaded with elbow")
ax2.set_title("S11 and S12 magnitudes for both connections,\nquadrature coil was unloaded")
ax2.legend()
plt.savefig('s11_s21_smith_quadrature_unloaded.png', transparent=True)
#plt.savefig('s11_s21_smith_quadrature_loaded.png', transparent=True)

# f, r_s11, i_s11, |S11|, phaseS11, r_s21, i_s21, |S21|, phaseS21
print(values_first[min_i_first, 1]*50+50, values_first[min_i_first, 2]*50)
print(values_switch[min_i_switch, 1]*50+50, values_switch[min_i_switch, 2]*50)

Q_first = q_factor(values_first[0], values_first[3], values_first[1], values_first[2], z0=50)
Q_switch = q_factor(values_switch[0], values_switch[3], values_switch[1], values_switch[2], z0=50)
Q_mean = (Q_switch+Q_first)/2
print(f"Q1 = {Q_switch:.4f} and Q2 = {Q_first:.4f}\n> Q mean = {Q_mean:.4f}")
print("Q ratio based on means", -5.0104/-2.3380)
plt.show()