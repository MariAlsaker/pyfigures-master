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
import matplotlib as mpl

folder_nanovna = "/Users/marialsaker/nanovnasaver_files/"
folder_pocketvna = "/Users/marialsaker/pocketVNAfiles/"
cmap = mpl.cm.get_cmap("plasma")

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
    vals = from_freim_to_five(freqs, reals, ims)
    return vals[0], vals[1], vals[2], vals[3], vals[4] # f, r_s11, i_s11, |S11|, phaseS11

def from_freim_to_five(freqs, reals, ims, db=False):
    freqs = np.array(freqs)
    reals = np.array(reals)
    ims = np.array(ims)
    if db:
        magn = 20*np.log10(np.sqrt(reals**2 + ims**2))
    else: magn = np.sqrt(reals**2 + ims**2)
    phase = np.arctan(ims/reals)
    return np.array([freqs, reals, ims, magn, phase]) # f, r_s11, i_s11, |S11|, phaseS11

def extract_vals_s2p_simple(file_path, len_file=301):
    vals = np.zeros((len_file, 9))
    with open(file_path) as f_s11:
        miss = 0
        for i, line in enumerate(f_s11.readlines()):
            if line.split(" ")[0] == "#" or line.split(" ")[0] == "!": 
                miss = miss+1 
                continue
            i = i-miss
            linevals= line.split(" ") # f, r_s11, i_s11, r_s21, i_s21, _, __, ___, ____
            new_linevals = []
            for val in linevals:
                if val != "":
                    new_linevals.append(val)
            linevals = new_linevals
            vals[i][0] = float(linevals[0])*1E-6
            vals[i][1:3] = linevals[1], linevals[2] 
            vals[i][5:7] = linevals[3], linevals[4]
    vals[:,3] = np.sqrt(vals[:,1]**2 + vals[:,2]**2)
    vals[:,4] = np.arctan(vals[:,2]/(vals[:,1]))
    vals[:,7] = np.sqrt(vals[:,5]**2 + vals[:,6]**2)
    vals[:,8] = np.arctan(vals[:,6]/(vals[:,5]))
    return vals # f, r_s11, i_s11, |S11|, phaseS11, r_s21, i_s21, |S21|, phaseS21

def extract_vals_s2p_allS(file_path, len_file=301):
    vals = np.zeros((len_file, 9))
    with open(file_path) as f_s11:
        miss = 0
        for i, line in enumerate(f_s11.readlines()):
            if line.split(" ")[0] == "#" or line.split(" ")[0] == "!": 
                miss = miss+1 
                continue
            i = i-miss
            linevals= line.split(" ") # f, r_s11, i_s11, r_s21, i_s21, _, __, ___, ____
            new_linevals = []
            for val in linevals:
                if val != "" and val != "\n":
                    new_linevals.append(val)
            linevals = np.array(new_linevals)
            linevals = np.float64(linevals)
            vals[i] = linevals
    return vals # f, r_s11, i_s11, r_s21, i_s21, r_s12, i_s12, r_s22, i_s22

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
        plots = ax.plot(freqs, s21_lists[i], label=f"|S21| {names[i]}", marker=".")
        plots[0].set(color = s21_colors[i])
    for i, freqs in enumerate(freq_lists):
        plots = ax.plot(freqs, s11_db_lists[i], label=f"|S11| {names[i]}", marker=".")
        plots[0].set(color = s11_colors[i])
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.set_xlabel("Frequency [MHz]")
    ax.set_ylabel("Magnitude [dB]")
    ax.grid(which="both")
    return

def plot_Sparams(ax, freqs, s11, s21, s12, s22, colors, names):
    for i, ss in enumerate([s11, s21, s12, s22]):
        plots = ax.plot(freqs, (ss[3]), label=names[i])#, marker=".")
        plots[0].set(color = colors[i])
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.set_xlabel("Frequency [MHz]")
    ax.set_ylabel("Magnitude [dB]")
    ax.grid(which="both")
    return

def plot_res(ax, magn_db, freqs, color):
    min_index = np.argmin(magn_db)
    min_db = magn_db[min_index]
    #print(freqs[min_index], color)
    plots = ax.plot(freqs[min_index], min_db, label=f"min= {min_db:.0f}dB, {freqs[min_index]:.3f}MHz", marker=".")
    plots[0].set(color=color, marker = "o")
    return min_index

def q_factor(freqs, magn_db, reals, ims, z0):
    """ Q factor calculated as defined in paper by J. Michael Drozd and William T. Joines 1996 """
    index_w0 = np.argmin(magn_db)
    R_nom = (1 + reals[index_w0])*(1 - reals[index_w0]) - ims[index_w0]**2
    R_denom = (1 - reals[index_w0])**2+ims[index_w0]**2
    R = z0 * R_nom / R_denom 
    n=5
    pos_slopeRs = reals[int(index_w0-n/2):int(index_w0+n/2)]
    pos_slopeIs = ims[int(index_w0-n/2):int(index_w0+n/2)]
    pos_slopews = freqs[int(index_w0-n/2):int(index_w0+n/2)]
    Xs = z0* 2*pos_slopeIs/((1-pos_slopeRs)**2 + pos_slopeIs**2)
    partX = (n*(np.sum(pos_slopews*Xs)) - np.sum(pos_slopews)*np.sum(Xs)) / (n*np.sum(pos_slopews**2) - np.sum(pos_slopews)**2)
    return freqs[index_w0]/(2*R+2*z0) * partX

def plot_smith(ax, reals, ims, c, magn_db, c_spes):
    """ ax must be projection=smith """
    vals_s11 = (reals + ims * 1j)
    min_index = np.argmin(magn_db)
    plots = ax.plot(vals_s11, markevery=1, label="equipoints=11", equipoints=11, datatype=SmithAxes.S_PARAMETER)
    plots[0].set(color=c)
    plots = ax.plot(vals_s11[min_index], markevery=1, label="equipoints=11", equipoints=11, datatype=SmithAxes.S_PARAMETER)
    plots[0].set(color=c_spes, marker="o")

def find_mean(num_files, file_path, filename, ending=".s1p", file_len = 301, allS=False):
    for i in range(num_files):
        this_file = file_path+f"{filename}{i}{ending}"
        if ending==".s1p":
            f, re, im, mag, ph = extract_vals_s1p(this_file)
            if i==0:
                # f, r_s11, i_s11, |S11|, phaseS11,
                data = np.zeros((num_files,5,len(f)))
            data[i] = np.array([f, re, im, mag, ph])
        else:
            if allS:
                vals = extract_vals_s2p_allS(this_file, len_file=file_len)
            else:
                vals = extract_vals_s2p_simple(this_file, len_file=file_len)
            vals = vals.transpose()
            if i==0:
                # f, r_s11, i_s11, |S11|, phaseS11, r_s21, i_s21, |S21|, phaseS21
                data = np.zeros((num_files,9,len(vals[0]))) 
            data[i] = vals 
    std_magn_db = 0
    mean_data = np.mean(data, axis=0)
    std_data = np.std(data, axis=0)
    return mean_data, std_data

def find_mean_allS(num_files, file_path, filename, ending=".s2p", file_len = 301):
    magn_desibels = np.zeros((4,num_files,file_len))
    phase = np.zeros((4, num_files, file_len))
    for i in range(num_files):
        this_file = file_path+f"{filename}{i}{ending}"
        vals = extract_vals_s2p_allS(this_file, len_file=file_len)
        vals = vals.transpose()
        if i==0:
            # f, r_s11, i_s11, r_s21, i_s21, r_s12, i_s12, r_s22, i_s22
            data = np.zeros((num_files,9,len(vals[0]))) 
        data[i] = vals 
        magn_desibels[0][i] = 20*np.log10( np.sqrt(vals[1]**2+vals[2]**2) ) # S_11 parameter
        magn_desibels[2][i] = 20*np.log10( np.sqrt(vals[3]**2+vals[4]**2) ) # S_21 parameter
        magn_desibels[3][i] = 20*np.log10( np.sqrt(vals[5]**2+vals[6]**2) ) # S_12 parameter
        magn_desibels[1][i] = 20*np.log10( np.sqrt(vals[7]**2+vals[8]**2) ) # S_22 parameter
        phase[0][i] = np.arctan(vals[2]/vals[1]) # S_11 parameter
        phase[2][i] = np.arctan(vals[4]/vals[3]) # S_21 parameter
        phase[3][i] = np.arctan(vals[6]/vals[5]) # S_12 parameter
        phase[1][i] = np.arctan(vals[8]/vals[7]) # S_22 parameter
    std_magn_db = [np.std(magn_desibels[0], axis=0), np.std(magn_desibels[1], axis=0), np.std(magn_desibels[2], axis=0), np.std(magn_desibels[3], axis=0)]
    mean_magn_db = [np.mean(magn_desibels[0], axis=0), np.mean(magn_desibels[1], axis=0), np.mean(magn_desibels[2], axis=0), np.mean(magn_desibels[3], axis=0)]
    mean_data = np.mean(data, axis=0)
    mean_phase = [np.mean(phase[0], axis=0), np.mean(phase[1], axis=0), np.mean(phase[2], axis=0), np.mean(phase[3], axis=0)]
    std_phase = [np.std(phase[0], axis=0), np.std(phase[1], axis=0), np.std(phase[2], axis=0), np.std(phase[3], axis=0)]
    return mean_data, std_magn_db, mean_magn_db, mean_phase, std_phase

def uncertainty_magn_db(reals, ims, std_reals, std_ims):
    magn = np.sqrt(reals**2 + ims**2)
    s_x = np.sqrt(2)*std_reals/reals
    s_y = np.sqrt(2)*std_ims/ims
    s_magn = magn * 1/2 * (np.sqrt(s_x**2+s_y**2)/(reals**2+ims**2))
    print(np.max(s_magn))
    s_magn_db = 20*0.434*(s_magn / magn)
    return s_magn_db

""" SINGLE LOOP COIL PLOTS AND MEASUREMENTS """
show_single_plots = False
print_Qs_single = False

mean_data1, std_data1 = find_mean(num_files=10, file_path = folder_pocketvna+"Single_loop/", filename="idun4na", ending=".s2p")
freqs1, reals1, ims1, magn1, phase1 = mean_data1[0], mean_data1[1], mean_data1[2], mean_data1[3], mean_data1[4]
mean_data2, std_data2 = find_mean(num_files=10, file_path = folder_pocketvna+"Single_loop/", filename="noload", ending=".s2p")
freqs2, reals2, ims2, magn2, phase2 = mean_data2[0], mean_data2[1], mean_data2[2], mean_data2[3], mean_data2[4]

magn1_db = 20*np.log10(magn1)
magn2_db = 20*np.log10(magn2)
fig = plt.figure(figsize=[6, 10])
ax1 = fig.add_subplot(2,1,1)
ax1.vlines(x=[33.78], ymin=-42, ymax=1, colors="k", linestyles="--", label="f0=33.78MHz")
c1 = "steelblue"
c2 = "orange"
plot_magn_phase(ax1, [freqs1, freqs2], [magn1_db, magn2_db], [phase1, phase2], magn_colors=[c1,c2], phase_colors=[c1,c2], show_phase=False, names=["elbow load", "no load"])
#ax1.errorbar(freqs1, magn1_db, std_magn1_db)# plt.errorbar(x, y, e, linestyle='None', marker='^')
ax2 = fig.add_subplot(2,1,2, projection="smith")
# Plotting smith charts
min_index = plot_res(ax=ax1, magn_db=magn1_db, freqs=freqs1, color="turquoise")
plot_smith(ax=ax2, reals=reals1, ims=ims1, c=c1, magn_db=magn1_db, c_spes="turquoise")
min_index = plot_res(ax=ax1, magn_db=magn2_db, freqs=freqs2, color="red")
plot_smith(ax=ax2, reals=reals2, ims=ims2, c=c2, magn_db=magn2_db, c_spes="red")
    
ax1.legend()
fig.suptitle("Laboratory tests of the single loop coil \n loaded and unloaded")
ax1.set_title("|S11| magnitude")
ax2.set_title("Real and imaginary impedance (Smith chart)")
#plt.savefig('s11_smith_singleloop.png', transparent=True)
if show_single_plots:
    plt.show()
else:
    plt.close()
Q_load = q_factor(freqs1, magn1_db, reals1, ims1, z0=50)
Q_noload = q_factor(freqs2, magn2_db, reals2, ims2, z0=50)
Q_ratio = Q_noload/Q_load
if print_Qs_single:
    print("Q factor and ratio for single loop coil:")
    print(f"Unloaded Q = {Q_noload:.4f} and loaded Q = {Q_load:.4f}\n> Q ratio = {Q_ratio:.4f}")


""" QUADRATURE COIL PLOTS AND MEASUREMENTS """
# Demonstrate that the quadrature coil is a reciprocal network, meaning that the S_21 = S_12
show_quad_plots = False
save = False
print_impedance = False
print_Qs_quad = False

values_unloaded_quad = find_mean_allS(num_files=10, file_path=folder_pocketvna+"Quad_loop/",filename="noload", file_len=1001)

values_loaded_quad = find_mean_allS(num_files=10, file_path=folder_pocketvna+"Quad_loop/",filename="idun4na", file_len=1001)

figs = plt.figure(figsize=[7, 6]), plt.figure(figsize=[7, 6])
smithfig = plt.figure(figsize=[6,6])
smithax = smithfig.add_subplot(1,1,1, projection="smith")
Qs = []
colors = [cmap(0.2), "silver", "dimgray", cmap(0.8)]
loadTF = ["ketchup + 4Na", "nothing"]
for i, values in enumerate([values_loaded_quad, values_unloaded_quad]):
    print(f"*** Load = {loadTF[i]} ***")
    ax = figs[i].add_subplot(1,1,1)
    freqs_q = values[0][0]*1E-6
    s_11s = from_freim_to_five(freqs_q, values[0][1], values[0][2], db=True)# f, r_s11, i_s11, |S11|, phaseS11
    s_21s = from_freim_to_five(freqs_q, values[0][3], values[0][4], db=True)# f, r_s21, i_s21, |S21|, phaseS21
    s_12s = from_freim_to_five(freqs_q, values[0][5], values[0][6], db=True)# f, r_s12, i_s12, |S12|, phaseS12
    s_22s = from_freim_to_five(freqs_q, values[0][7], values[0][8], db=True)# f, r_s22, i_s22, |S22|, phaseS22
    ax.vlines(x=[33.78],ymin=-55, ymax=1, colors=["k"], linestyles="--", label="f=33.78MHz")
    if i==0:
        names = ["s_11 load", "s_21 load", "s_12 load", "s_22 load"]
    else:
        names = ["s_11 no load", "s_21 no load", "s_12 no load", "s_22 no load"]
    plot_Sparams(ax=ax, freqs=freqs_q, s11=s_11s, s21=s_21s, s12=s_12s, s22=s_22s, colors=colors, names=names)
    min_i_first = plot_res(ax, magn_db=(s_11s[3]), freqs=freqs_q, color="red")
    min_i_switch = plot_res(ax, magn_db=(s_22s[3]), freqs=freqs_q, color="darkblue")
    print(f"Minimums indexes = {min_i_first}, {min_i_switch}")
    ax.set_title(f"S parameter magnitudes,\nquadrature coil was loaded with {loadTF[i]}")
    ax.legend()
    if save:    
        figs[i].savefig(f"quad_magn_{i}.png", transparent=True)
        if i ==1:
            smithfig.savefig("quad_smith.png", transparent=True)

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
               c=colors[0], magn_db=(s_11s[3]), c_spes="red")
    plot_smith(ax=smithax, reals=s_22s[1], ims=s_22s[2], 
               c=colors[-1], magn_db=(s_22s[3]), c_spes="darkblue")
    diffs = [freqs_q[i+1]-freqs_q[i] for i in range(len(freqs_q)-1)]
    mean_diff = np.mean(diffs)
    print(f"Uncertainties for magnitude with load {loadTF[i]} at resonance: ",
          f"\n> u_f0 = {mean_diff} MHz",
          f"\n> u_s11 = {values[1][0][min_i_first]:.1f} dB ",
          f"\n> u_s22 = {values[1][1][min_i_switch]:.1f} dB")

if print_Qs_quad:
    print("\nQ of quadrature coil:") 
    print(f"Q, mean loaded = {Qs[0]} \nQ, mean unloaded = {Qs[1]}") # Unloaded -11.4433
    print(f"> Q ratio = {Qs[1]/Qs[0]}")

# f, r_s11, i_s11, r_s21, i_s21, r_s12, i_s12, r_s22, i_s22
if show_quad_plots:
    plt.show()
else:
    plt.close("all")

""" HYBRID PLOTS AND MEASUREMENTS """

hybrid_data = ["hybrid_setup1_isoSMA", "hybrid_setup6_isoBNC", "hybrid_setup2_through", "hybrid_setup3_through", 
               "hybrid_setup4_through", "hybrid_setup5_through"]
hybrid_setups = ["Hybrid, coil coax","Hybrid, MRI coax", "Hybrid through, green to same", "Hybrid through, green to opposite", 
               "Hybrid through, blue to same", "Hybrid through, blue to opposite"]
width, height = 9, 5
rows, cols = 1, 2
fig1 = plt.figure(figsize=[width, height])
fig2 = plt.figure(figsize=[width, height])
fig3 = plt.figure(figsize=[width, height])
axs1 = fig1.subplots(rows, cols)
axs2 = fig2.subplots(rows, cols)
axs3 = fig3.subplots(rows, cols)
colors = [cmap(0.2), cmap(0.4), cmap(0.6), cmap(0.8)]
labels = ["s_11", "s_21", "s_12", "s_22"]
axs = [axs1[0], axs1[1], axs2[0], axs2[1], axs3[0], axs3[1]]
magn_at_res = np.zeros((len(hybrid_data), 4))
phase_at_res = np.zeros((len(hybrid_data), 4))
prev = 0
diffs = []
u_diffs = [] # HArd work ough
for i, name in enumerate(hybrid_data):
    vals = find_mean_allS(num_files=10, file_path=folder_pocketvna+"Quad_loop/",filename=name, file_len=501)
    freqs_h = vals[0][0]*1E-6
    res = next(x for x, val in enumerate(freqs_h) if val > 33.775)
    s_11s = from_freim_to_five(freqs_h, vals[0][1], vals[0][2], db=True)# f, r_s11, i_s11, |S11|, phaseS11
    s_21s = from_freim_to_five(freqs_h, vals[0][3], vals[0][4], db=True)# f, r_s21, i_s21, |S21|, phaseS21
    s_12s = from_freim_to_five(freqs_h, vals[0][5], vals[0][6], db=True)# f, r_s12, i_s12, |S12|, phaseS12
    s_22s = from_freim_to_five(freqs_h, vals[0][7], vals[0][8], db=True)# f, r_s22, i_s22, |S22|, phaseS22
    plot_Sparams(ax=axs[i], freqs=freqs_h, s11=s_11s, s21=s_21s, s12=s_12s, s22=s_22s, colors=colors, names=labels)
    axs[i].vlines(x=[33.78],ymin=-55, ymax=1, colors=["k"], linestyles="--", label="f=33.78MHz")
    axs[i].set_title(hybrid_setups[i])
    axs[i].legend()
    magn_at_res[i] = np.array((s_11s[3][res], s_21s[3][res], s_12s[3][res], s_22s[3][res]))
    phase_at_res[i] = np.array((s_11s[4][res], s_21s[4][res], s_12s[4][res], s_22s[4][res]))
    if hybrid_setups[i].split(" ")[1] == "through,":
        this = np.array([s_12s[-1], s_21s[-1]])
        if hybrid_setups[i].split(" ")[-1] == "opposite":
            diff = prev-this
            diffs.append(diff)
        prev = this
    print(hybrid_setups[i])
    s_sequence = ["S_11", "S_22", "S_21", "S_12"]
    print("Magnitude and phase") # mean_data, std_magn_db, mean_magn_db, mean_phase, std_phase
    for i in range(4):
        print(f"Mag of {s_sequence[i]} = {vals[2][i][res]} +/- {vals[1][i][res]}")
        print(f"Phase of {s_sequence[i]} = {vals[3][i][res]} +/- {vals[4][i][res]}")
diffs = np.array(diffs)

print(labels)
for i in range(len(hybrid_data)):
    print("\n", hybrid_setups[i])
    print(f"magnitude = {np.around(magn_at_res[i], decimals=2)}")
    print(f"phase = {np.around(phase_at_res[i], decimals=3)}")
print("s_12, s_21")
print(diffs[:,:,res])# /(2*np.pi)*360)
print(diffs[:,:,res] /(2*np.pi)*360)
print(f"Phase difference green s21 is {0.103-(-1.383)} +/- {np.sqrt(0.004**2+0.002**2):.3f}")
print(f"Phase difference green s12 is {0.098-(-1.378)} +/- {np.sqrt(0.003**2+0.003**2):.3f}")
print(f"Phase difference blue s21 is {0.116-(-1.370)} +/- {np.sqrt(0.006**2+0.002**2):.3f}")
print(f"Phase difference blue s12 is {0.112-(-1.368)} +/- {np.sqrt(0.006**2+0.002**2):.3f}")
plt.show()