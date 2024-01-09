from matplotlib import pyplot as plt
import numpy as np
from matplotlib import ticker

# Unicode: Ohm = \u03A9, degrees = \u00B0, mu = \u03BC
# https://pythonforundergradengineers.com/unicode-characters-in-python.html 

def parallel_RLC_Z(f, L, C, R):
    w = 2*np.pi*f
    Z_L = 1j*w*L
    Z_C = 1/(1j*w*C)
    return 1/(1/R + 1/Z_L + 1/Z_C)

def series_RLC_Z(f, L, C, R):
    w = 2*np.pi*f
    Z_L = 1j*w*L
    Z_C = 1/(1j*w*C)
    return Z_L+Z_C+R

def plot_magn_phase(ax, freqs, magns, phases, colors1, colors2, name="RLC"):
    axtwin = ax.twinx()
    for i, magn in enumerate(magns):
        plots = ax.plot(freqs, magn, label=f"Impedance {i+1}", color=colors1[i])
    ax.set_ylabel("Impedance [\u03A9]")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    #plots[0].set(color = colors1[0])
    for i, phase in enumerate(phases):
        plots = axtwin.plot(freqs, phase/np.pi*360, label=f"Phase {i+1}", color=colors2[i])
    #plots[0].set(color = colors2[0])
    axtwin.set_ylabel("Phase [\u00B0]")
    axtwin.yaxis.label.set_color(colors2[0])
    axtwin.tick_params(axis = 'y', colors=colors2[0])
    axtwin.legend(loc="upper right")
    ax.yaxis.label.set_color(colors1[0])
    ax.tick_params(axis = 'y', colors=colors1[0])
    ax.set_xlabel("Frequency [Hz]")
    ax.set_title(name)
    ax.grid(True)

frequencies = np.linspace(50, 300, 10000)
colors1 = ['blue', 'green', 'purple']
colors2 = ['red', 'orange', 'magenta']
resonance_freq = 160
Ls = np.array([100., 50., 100.])*1E-3 # Henry
Rs = [300., 300., 150.]
Cs = np.array([10.0, 20.0, 10.0])*1E-6
names = []
magns = []
phases = []
for i in range(3):
    name = f"\n R{i+1} = {Rs[i]:.1f} \u03A9, L{i+1} = {Ls[i]*1E3:.1f}mH, C{i+1} = {Cs[i]*1E6:.1f}\u03BCF, Q{i+1} = {Rs[i]*np.sqrt(Cs[i]/Ls[i]):.1f}"
    names.append(name)
    impedance = parallel_RLC_Z(frequencies, Ls[i], Cs[i], R=Rs[i])
    magnitude = np.sqrt(impedance.real**2 + impedance.imag**2)
    phase = np.arctan(impedance.imag/impedance.real)
    magns.append(magnitude)
    phases.append(phase)
fig, ax = plt.subplots(1,1, figsize=(9, 7))
plot_magn_phase(ax, frequencies, magns, phases, colors1, colors2, name=f"Impedance in a parallel RLC circuit,"+names[0]+names[1]+names[2])
plt.show()

frequencies = np.linspace(10, 100, 10000)
resonance_freq = 25
Ls = np.array([800., 200., 50.]) * 1E-3 # Henry
Cs = np.array([50., 200., 800.]) * 1E-6
Rs = [10., 10., 10.]
names = []
magns = []
phases = []
for i in range(3):
    name = f"\n R{i+1} = {Rs[i]:.1f} \u03A9, L{i+1} = {Ls[i]*1E3:.1f}mH, C{i+1} = {Cs[i]*1E6:.1f}\u03BCF, Q{i+1} = {1/Rs[i]*np.sqrt(Ls[i]/Cs[i]):.1f}"
    names.append(name)
    impedance = series_RLC_Z(frequencies, Ls[i], Cs[i], R=Rs[i])
    magnitude = np.sqrt(impedance.real**2 + impedance.imag**2)
    phase = np.arctan(impedance.imag/impedance.real)
    magns.append(magnitude)
    phases.append(phase)
fig, ax = plt.subplots(1,1, figsize=(9, 7))
plot_magn_phase(ax, frequencies, magns, phases, colors1, colors2, name=f"Impedance in a series RLC circuit,"+names[0]+names[1]+names[2])
plt.show()