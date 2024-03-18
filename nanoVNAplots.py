import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker
import collections.abc
collections.Iterable = collections.abc.Iterable
collections.Mapping = collections.abc.Mapping
collections.MutableSet = collections.abc.MutableSet
collections.MutableMapping = collections.abc.MutableMapping
from smithplot import SmithAxes
import matplotlib as mpl

folder_nanovna = "/Users/marialsaker/nanovnasaver_files/"
folder_pocketvna = "/Users/marialsaker/pocketVNAfiles/"
cmap = mpl.colormaps.get_cmap("plasma")

def extract_vals_s1p(file_path):
    freqs = []
    reals = []
    ims = []
    with open(file_path) as f_s11:
        for line in f_s11.readlines():
            if line.split(" ")[0] == "#": continue
            linevals = line.split(" ")
            freqs.append(float(linevals[0])*1E-6)
            reals.append(float(linevals[1]))
            ims.append(float(linevals[2]))
    vals = np.array([freqs, reals, ims])
    return vals # f, r_s11, i_s11

def extract_vals_s2p(file_path, len_file=301):
    vals = np.zeros((len_file, 9))
    with open(file_path) as f_s11:
        miss = 0
        for i, line in enumerate(f_s11.readlines()):
            if line.split(" ")[0] == "#" or line.split(" ")[0] == "!": 
                miss = miss+1 
                continue
            i = i-miss
            linevals= line.split(" ") # f, r_s11, i_s11, r_s21, i_s21, r_s12, i_s12, r_s22, i_s22
            new_linevals = []
            for val in linevals:
                if val != "" and val != "\n":
                    new_linevals.append(val)
            linevals = np.array(new_linevals)
            linevals = np.float64(linevals)
            vals[i] = linevals
            vals[i][0] = vals[i][0]*1E-6
    return vals # f, r_s11, i_s11, r_s21, i_s21, r_s12, i_s12, r_s22, i_s22

def plot_magn_phase(ax, freq_lists, magn_db_lists, phase_list, magn_colors, phase_colors, show_phase=True, names="S11"): # Currently not in use
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

def plot_Sparams(ax, freqs, s_magns_db, colors, names):
    for i, s_magn in enumerate(s_magns_db):
        plots = ax.plot(freqs, (s_magn), label=names[i])#, marker=".")
        plots[0].set(color = colors[i])
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.set_xlabel("Frequency [MHz]")
    ax.set_ylabel("Magnitude [dB]")
    ax.grid(which="both")
    return

def plot_res(ax, magn_db, freqs, color, fillstyle="full"):
    min_index = np.argmin(magn_db)
    min_db = magn_db[min_index]
    plots = ax.plot(freqs[min_index], min_db, label=f"min=({freqs[min_index]:.3f}MHz, {min_db:.0f}dB)", marker=".", zorder=3, fillstyle=fillstyle)
    plots[0].set(color=color, marker = "o", markersize=8, fillstyle=fillstyle)
    return min_index

def q_factor(freqs, magn_db, reals, ims, z0):
    """ Q factor calculated as defined in paper by J. Michael Drozd and William T. Joines 1996 """
    index_w0 = np.argmin(magn_db)
    R_nom = (1 + reals[index_w0])*(1 - reals[index_w0]) - ims[index_w0]**2
    R_denom = (1 - reals[index_w0])**2+ims[index_w0]**2
    R = z0 * R_nom / R_denom 
    n=5 # Region of X-values for calculating derivative
    pos_slopeRs = reals[int(index_w0-n/2):int(index_w0+n/2)]
    pos_slopeIs = ims[int(index_w0-n/2):int(index_w0+n/2)]
    pos_slopews = freqs[int(index_w0-n/2):int(index_w0+n/2)]
    Xs = z0* 2*pos_slopeIs/((1-pos_slopeRs)**2 + pos_slopeIs**2)
    dX = (n*(np.sum(pos_slopews*Xs)) - np.sum(pos_slopews)*np.sum(Xs)) / (n*np.sum(pos_slopews**2) - np.sum(pos_slopews)**2)
    return freqs[index_w0]/(2*R+2*z0) * dX

def plot_smith(ax, reals, ims, c, magn_db, fillstyle="full"):
    """ ax must be projection=smith """
    vals_s11 = (reals + ims * 1j)
    min_index = np.argmin(magn_db)
    plots = ax.plot(vals_s11, markevery=1, datatype=SmithAxes.S_PARAMETER, markersize=1)
    plots[0].set(color=c)
    plots = ax.plot(vals_s11[min_index], markevery=1, datatype=SmithAxes.S_PARAMETER, markersize=10, fillstyle=fillstyle)
    plots[0].set(color=c, marker="o", fillstyle=fillstyle)

def find_mean_s11(num_files, file_path, filename, ending=".s1p", file_len = 301):
    for i in range(num_files):
        this_file = file_path+f"{filename}{i}{ending}"
        if ending==".s1p":
            vals_s1p = extract_vals_s1p(this_file)
            if i==0:
                # f, r_s11, i_s11, |S11|, phaseS11,
                data = np.zeros((num_files,5,len(vals_s1p[0])))
            data[i][0:3] = vals_s1p
            data[i][3] = 20*np.log10(np.sqrt(vals_s1p[1]**2+vals_s1p[2]**2))
            data[i][4] = np.arctan(vals_s1p[2]/vals_s1p[1])
        else:
            vals_s2p = extract_vals_s2p(this_file, len_file=file_len)
            vals_s2p = vals_s2p.transpose()
            if i==0:
                # f, r_s11, i_s11, |S11|, phaseS11, r_s21, i_s21, |S21|, phaseS21
                data = np.zeros((num_files,9,len(vals_s2p[0]))) 
            data[i][0:3] = vals_s2p[0:3]
            data[i][3] = 20*np.log10(np.sqrt(vals_s2p[1]**2+vals_s2p[2]**2))
            data[i][4] = np.arctan(vals_s2p[2]/vals_s2p[1])
    mean_data = np.mean(data, axis=0)
    std_data = np.std(data, axis=0)
    return mean_data, std_data

def find_mean_allS(num_files, file_path, filename, ending=".s2p", file_len = 301):
    magn_desibels = np.zeros((4,num_files,file_len))
    phase = np.zeros((4, num_files, file_len))
    for i in range(num_files):
        this_file = file_path+f"{filename}{i}{ending}"
        vals = extract_vals_s2p(this_file, len_file=file_len)
        vals = vals.transpose()
        if i==0:
            # f, r_s11, i_s11, r_s21, i_s21, r_s12, i_s12, r_s22, i_s22
            data = np.zeros((num_files,9,len(vals[0]))) 
        data[i] = vals 
        for j in range(4):
            magn_desibels[j][i] = 20*np.log10( np.sqrt(vals[j*2+1]**2+vals[j*2+2]**2) ) # S_22 parameter
            phase[j][i] = np.arctan(vals[j*2+2]/vals[j*2+1]) # S_11 parameter
    mean_data = np.mean(data, axis=0)
    return mean_data, np.mean(magn_desibels, axis=1), np.std(magn_desibels, axis=1), np.mean(phase, axis=1), np.std(phase, axis=1)


""" SINGLE LOOP COIL PLOTS AND MEASUREMENTS """
print_Qs_single = False

coils = [
    {"name" : "Single loop",
     "file_loc": f"{folder_pocketvna}Single_loop/",
     "ymin": -42},
    {"name" : "Assadi's surface",
     "file_loc": f"{folder_pocketvna}AssadiSurface/",
     "ymin": -30},
    {"name" : "Original surface",
     "file_loc": f"{folder_pocketvna}OrigSurface/",
     "ymin": -30},
     {"name" : "Enhancing",
     "file_loc": f"{folder_pocketvna}Enhancing/",
     "ymin": -5}
]
c_load = cmap(0.2)
c_noload = cmap(0.8)
loadfile = "siemens"
noloadfile = "noload"

for coil in coils:
    name = coil["name"]
    mean_data1, std_data1 = find_mean_s11(num_files=10, file_path=coil["file_loc"], filename=loadfile, ending=".s2p", file_len=1001)
    freqs1, reals1, ims1, magn1_db, phase1 = mean_data1[0], mean_data1[1], mean_data1[2], mean_data1[3], mean_data1[4]
    mean_data2, std_data2 = find_mean_s11(num_files=10, file_path=coil["file_loc"], filename=noloadfile, ending=".s2p", file_len=1001)
    freqs2, reals2, ims2, magn2_db, phase2 = mean_data2[0], mean_data2[1], mean_data2[2], mean_data2[3], mean_data2[4]

    fig = plt.figure(figsize=[10, 5])
    ax1 = fig.add_subplot(1,2,1)
    ax1.vlines(x=[33.78], ymin=coil["ymin"], ymax=1, colors="k", linestyles="--", label="f0=33.78MHz", zorder=0)
    err_line1 = ax1.errorbar(freqs1, magn1_db, yerr=std_data1[3], zorder=1, color=c_load)
    err_line2 = ax1.errorbar(freqs2, magn2_db, yerr=std_data2[3], zorder=1, color=c_noload)
    err_line1.set_label("loaded")
    err_line2.set_label("unloaded")
    ax1.grid(True)
    ax2 = fig.add_subplot(1,2,2, projection="smith")
    # Plotting smith charts
    min_index1 = plot_res(ax=ax1, magn_db=magn1_db, freqs=freqs1, color=c_load)
    plot_smith(ax=ax2, reals=reals1, ims=ims1, c=c_load, magn_db=magn1_db)
    min_index2 = plot_res(ax=ax1, magn_db=magn2_db, freqs=freqs2, color=c_noload, fillstyle="left")
    plot_smith(ax=ax2, reals=reals2, ims=ims2, c=c_noload, magn_db=magn2_db, fillstyle="left")
        
    ax1.legend()
    fig.suptitle(f"Laboratory tests of the {name} coil")
    ax1.set_title("|S11| magnitude")
    ax2.set_title("Real and imaginary impedance (Smith chart)")
    #plt.savefig(f's11 {name}.png', transparent=True)
    print("STD of magnitude loaded = ", std_data1[3][min_index1-1:min_index1+2])
    print("STD of magnitude unloaded = ", std_data2[3][min_index2-1:min_index2+2])

    Q_load = q_factor(freqs1, magn1_db, reals1, ims1, z0=50)
    Q_noload = q_factor(freqs2, magn2_db, reals2, ims2, z0=50)
    Q_ratio = Q_noload/Q_load
    if print_Qs_single:
        print("Q factor and ratio for single loop coil:")
        print(f"Unloaded Q = {Q_noload:.4f} and loaded Q = {Q_load:.4f}\n> Q ratio = {Q_ratio:.4f}")

#plt.show()
plt.close("all")



""" QUADRATURE COIL PLOTS AND MEASUREMENTS """
# Demonstrate that the quadrature coil is a reciprocal network, meaning that the S_21 = S_12
show_quad_plots = True
save = False
print_impedance = False
print_Qs_quad = True

values_unloaded_quad = find_mean_allS(num_files=10, file_path=folder_pocketvna+"Quad_loop/",filename=noloadfile, file_len=1001)
values_loaded_quad = find_mean_allS(num_files=10, file_path=folder_pocketvna+"Quad_loop/",filename=loadfile, file_len=1001)
# values = [mean_data, std_magn_db, mean_magn_db, mean_phase, std_phase]
# mean_data = [f, r_s11, i_s11, r_s21, i_s21, r_s12, i_s12, r_s22, i_s22]
# mean/std = [s11, s22, s21, s12]

figs = plt.figure(figsize=[7, 6]), plt.figure(figsize=[7, 6])
smithfig = plt.figure(figsize=[6,6])
smithax = smithfig.add_subplot(1,1,1, projection="smith")
Qs = []
colors = [[cmap(0.2), "silver", "dimgray", cmap(0.4)], [cmap(0.6), "silver", "dimgray", cmap(0.8)]]
loadTF = ["siemens phantom", "nothing"]
for i, values in enumerate([values_loaded_quad, values_unloaded_quad]):
    print(f"*** Load = {loadTF[i]} ***")
    ax = figs[i].add_subplot(1,1,1)
    freqs_q = values[0][0]
    s_11s = freqs_q, values[0][1], values[0][2], values[1][0], values[3][0] # f, re, im, magn_db, phase
    s_22s = freqs_q, values[0][7], values[0][8], values[1][3], values[3][3] # f, re, im, magn_db, phase
    ax.vlines(x=[33.78],ymin=-55, ymax=1, colors=["k"], linestyles="--", label="f=33.78MHz")
    if i==0:
        names = ["s_11 load", "s_21 load", "s_12 load", "s_22 load"]
        fill = "full"
    else:
        names = ["s_11 no load", "s_21 no load", "s_12 no load", "s_22 no load"]
        fill="left"
    plot_Sparams(ax=ax, freqs=freqs_q, s_magns_db=values[1], colors=colors[i], names=names)
    min_i_first = plot_res(ax, magn_db=values[1][0], freqs=freqs_q, color=colors[i][0], fillstyle=fill)
    min_i_switch = plot_res(ax, magn_db=values[1][3], freqs=freqs_q, color=colors[i][3], fillstyle=fill)
    print(f"Minimums indexes = {min_i_first}, {min_i_switch}")
    ax.set_title(f"S parameter magnitudes,\nquadrature coil was loaded with {loadTF[i]}")
    ax.legend()
    if save:    
        figs[i].savefig(f"quad_magn_{i}.png", transparent=True)
        

    if print_impedance:
        print(f"Impedance [Ohm] at resonance for quadrature coils: ")
        print(f"> ({s_11s[1, min_i_first]*50+50} + i {s_11s[2, min_i_first]*50}) Ohm")
        print(f"> ({s_22s[1, min_i_switch]*50+50} + i {s_22s[2, min_i_switch]*50}) Ohm")

    Q_first = q_factor(freqs=freqs_q, magn_db=(s_11s[3]), reals=s_11s[1], ims=s_11s[2], z0=50)
    print("q first", Q_first)
    Q_switch = q_factor(freqs=freqs_q, magn_db=(s_22s[3]), reals=s_22s[1], ims=s_22s[2], z0=50)
    print("q_switch", Q_switch)
    Qs.append((Q_switch+Q_first)/2)

    plot_smith(ax=smithax, reals=s_11s[1], ims=s_11s[2], 
               c=colors[i][0], magn_db=(s_11s[3]), fillstyle=fill)
    plot_smith(ax=smithax, reals=s_22s[1], ims=s_22s[2], 
               c=colors[i][-1], magn_db=(s_22s[3]), fillstyle=fill)
    diffs = [freqs_q[i+1]-freqs_q[i] for i in range(len(freqs_q)-1)]
    mean_diff = np.mean(diffs)
    print(f"Uncertainties for magnitude with load {loadTF[i]} at resonance: ",
          f"\n> u_f0 = {mean_diff} MHz",
          f"\n> u_s11 = {values[2][0][min_i_first]:.1f} dB ",
          f"\n> u_s22 = {values[2][3][min_i_switch]:.1f} dB")
    if save: 
        smithfig.savefig("quad_smith.png", transparent=True)

if print_Qs_quad:
    print("\nQ of quadrature coil:") 
    print(f"Q, mean loaded = {Qs[0]} \nQ, mean unloaded = {Qs[1]}") # Unloaded -11.4433
    print(f"> Q ratio = {Qs[1]/Qs[0]}")

if show_quad_plots:
    plt.show()
else:
    plt.close("all")


# """ HYBRID PLOTS AND MEASUREMENTS """

# hybrid_data = ["hybrid_setup1_isoSMA", "hybrid_setup6_isoBNC", "hybrid_setup2_through", "hybrid_setup3_through", 
#                "hybrid_setup4_through", "hybrid_setup5_through"]
# hybrid_setups = ["Hybrid, coil coax","Hybrid, MRI coax", "Hybrid through, green to same", "Hybrid through, green to opposite", 
#                "Hybrid through, blue to same", "Hybrid through, blue to opposite"]
# width, height = 9, 5
# rows, cols = 1, 2
# axs = []
# for i in range(3):
#     fig = plt.figure(figsize=[width, height])
#     ax = fig.subplots(rows, cols)
#     axs.append(ax)
# colors = [cmap(0.2), cmap(0.4), cmap(0.6), cmap(0.8)]
# labels = ["s_11", "s_21", "s_12", "s_22"]
# axs = [axs[0][0], axs[0][1], axs[1][0], axs[1][1], axs[2][0], axs[2][1]]
# prev = 0
# for i, name in enumerate(hybrid_data):
#     vals = find_mean_allS(num_files=10, file_path=folder_pocketvna+"Quad_loop/Hybrid/",filename=name, file_len=501)
#     freqs_h = vals[0][0]
#     res = next(x for x, val in enumerate(freqs_h) if val > 33.775)
#     plot_Sparams(ax=axs[i], freqs=freqs_h, s_magns_db=vals[1], colors=colors, names=labels)
#     axs[i].vlines(x=[33.78],ymin=-55, ymax=1, colors=["k"], linestyles="--", label="f=33.78MHz")
#     axs[i].set_title(hybrid_setups[i])
#     axs[i].legend()
#     print("\n", hybrid_setups[i])
#     s_sequence = ["S_11", "S_22", "S_21", "S_12"]
#     # mean_data, std_magn_db, mean_magn_db, mean_phase, std_phase
#     for i in range(4):
#         print(f"Mag of {s_sequence[i]} = {vals[1][i][res]:.5f} +/- {vals[2][i][res]:.5f}")
#         print(f"Phase of {s_sequence[i]} = {vals[3][i][res]:.5f} +/- {vals[4][i][res]:.5f}")

# print(f"\nPhase difference green s21 is {0.103-(-1.383):.3f} +/- {np.sqrt(0.004**2+0.002**2):.3f}")
# print(f"Phase difference green s12 is {0.098-(-1.378):.3f} +/- {np.sqrt(0.003**2+0.003**2):.3f}")
# print(f"Phase difference blue s21 is {0.116-(-1.370):.3f} +/- {np.sqrt(0.006**2+0.002**2):.3f}")
# print(f"Phase difference blue s12 is {0.112-(-1.368):.3f} +/- {np.sqrt(0.006**2+0.002**2):.3f}")
# plt.show()