from matplotlib import pyplot as plt
import numpy as np
from matplotlib import ticker
import matplotlib as mpl

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

cmap = mpl.cm.get_cmap("plasma")
# frequencies = np.linspace(50, 300, 10000)
# colors1 = [cmap(0.1), cmap(0.3), cmap(0.5)]
# colors2 = [cmap(0.7), cmap(0.8), cmap(0.9)]
# # colors1 = ['blue', 'green', 'purple']
# # colors2 = ['red', 'orange', 'magenta']
# resonance_freq = 160
# Ls = np.array([100., 50., 100.])*1E-3 # Henry
# Rs = [300., 300., 150.]
# Cs = np.array([10.0, 20.0, 10.0])*1E-6
# names = []
# magns = []
# phases = []
# for i in range(3):
#     name = f"\n R{i+1} = {Rs[i]:.1f} \u03A9, L{i+1} = {Ls[i]*1E3:.1f}mH, C{i+1} = {Cs[i]*1E6:.1f}\u03BCF, Q{i+1} = {Rs[i]*np.sqrt(Cs[i]/Ls[i]):.1f}"
#     names.append(name)
#     impedance = parallel_RLC_Z(frequencies, Ls[i], Cs[i], R=Rs[i])
#     magnitude = np.sqrt(impedance.real**2 + impedance.imag**2)
#     phase = np.arctan(impedance.imag/impedance.real)
#     magns.append(magnitude)
#     phases.append(phase)
# fig, ax = plt.subplots(1,1, figsize=(9, 7))
# plot_magn_phase(ax, frequencies, magns, phases, colors1, colors2, name=f"Impedance in a parallel RLC circuit,"+names[0]+names[1]+names[2])
# plt.show()

# frequencies = np.linspace(10, 100, 10000)
# resonance_freq = 25
# Ls = np.array([800., 200., 50.]) * 1E-3 # Henry
# Cs = np.array([50., 200., 800.]) * 1E-6
# Rs = [10., 10., 10.]
# names = []
# magns = []
# phases = []
# for i in range(3):
#     name = f"\n R{i+1} = {Rs[i]:.1f} \u03A9, L{i+1} = {Ls[i]*1E3:.1f}mH, C{i+1} = {Cs[i]*1E6:.1f}\u03BCF, Q{i+1} = {1/Rs[i]*np.sqrt(Ls[i]/Cs[i]):.1f}"
#     names.append(name)
#     impedance = series_RLC_Z(frequencies, Ls[i], Cs[i], R=Rs[i])
#     magnitude = np.sqrt(impedance.real**2 + impedance.imag**2)
#     phase = np.arctan(impedance.imag/impedance.real)
#     magns.append(magnitude)
#     phases.append(phase)
# fig, ax = plt.subplots(1,1, figsize=(9, 7))
# plot_magn_phase(ax, frequencies, magns, phases, colors1, colors2, name=f"Impedance in a series RLC circuit,"+names[0]+names[1]+names[2])
# plt.show()
    
# CURRENT THROUGH A p-n-junction DIODE
def diodeI(volts, Isat):
    e_coulomb = 1.602E-19 #C
    k = 1.380649E-23 #m^2 kg s^-2 K-1
    T = 298 # K
    curr = Isat*(np.e**(e_coulomb*volts/(T*k))-1)
    return curr

# volts = np.linspace(-0.1, 0.1, 1000)
# current = diodeI(volts, Isat=0.5)
# fig, ax = plt.subplots(1,1)
# ax.set_xlim(left=-0.11, right=0.11)
# ax.set_ylim(bottom=-2, top=20)
# ax.hlines(y=0, xmin=-1, xmax=0.5, colors="k")
# ax.vlines(x=0, ymin=-2, ymax=21, colors="k")
# ax.plot(volts, current, color=cmap(0.5))
# ax.set_title("Simple model of p-n junction behavior")
# ax.set_xlabel("Volts [V]")
# ax.set_ylabel("Current [A]")
# ax.grid(True)
# plt.show()
    
# REGROWTH OF MAGNETIZATION
def longitudinal_M(ts, M0, Mz0, T1): 
    # White matter T1 = 600 ms
    return Mz0*np.e**(-ts/T1)+M0*(1-np.e**(-ts/T1))

def transverse_M(ts, Mx0, My0, T2):
    # White matter T2 = 80
    absMxy = np.sqrt(Mx0**2+My0**2)
    return absMxy*np.e**(-ts/T2)

def M0_curie(rho0, B0, T, gyro, s):
    planck_red = 6.62607E-34/(2*np.pi) #
    k = 1.38065E-23 # J/K
    C = rho0*s*(s+1)*gyro**2*planck_red**2/(3*k)
    return C*B0/T

rho0 = 39.2E-9 * 6.02214179E23 # kg-1
B0 = 3 # Tesla
T = 310 # Kelvin
gyro = 42.58E6 # Hz/T
s = 1/2 # spin quantum number
M0 = M0_curie(rho0, B0, T, gyro, s)

ts = np.linspace(0, 3, 10000)
longitude = longitudinal_M(ts, M0=M0, Mz0=0, T1=600E-3)
transverse = transverse_M(ts, Mx0=M0, My0=0, T2=80E-3)
print(transverse[0])
print(M0)
plt.plot(ts, longitude, label="Longitudinal", color=cmap(0.3))
plt.plot(ts, transverse, label="Transverse", color=cmap(0.7)) 
plt.hlines(y=M0, xmin=0.00, xmax=3.0, colors="k", linestyles="--", label="M0")
plt.title("Theoretical relaxation in white matter")
plt.xlabel("Seconds [s]")
plt.ylabel("Magnetixation [J/T]")
plt.grid(True)
plt.legend()
plt.show()