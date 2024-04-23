import numpy as np
import matplotlib.pyplot as plt

""" Sensitivity of coil
Calculation to find a suitable radius of transmit/recieve single loop surface coil.
Ismpiration for the 35% from: 
Haase, A.; Odoj, F.; Von Kienlin, M.; Warnking, J.; Fidler, F.; Weisser, A.; Nittka, M.; Rommel, E.; Lanz, T.; Kalusche, B.; et al.
NMR probeheads for in vivo applications. Concepts Magne. Reson. 2000, 12, 361-388, doi:10.1002/1099-0534(2000)12:6. 
"""

# constants
perm = 4*np.pi*10**(-7) # H/m, permebility in vacuum/air
I = 0.3 # A
d = 0.15 # m
a = d/2

def biot_savart_axis(z):
    # Calculation from PHYS112
    num = perm*I*a**2
    den = 2*(a**2+z**2)**(3/2)
    return num/den

in_coil_plane = biot_savart_axis(0)
xs = np.linspace(-d, d, 100)
ys = biot_savart_axis(xs)/in_coil_plane
# plt.plot(xs, ys, label="Coil sensitivity", color="green")
# plt.hlines(y=[0.35], xmin=[-d], xmax=[d], label="35%",colors=["m"])
# #plt.vlines(x=[-a, a], ymin=[0,0], ymax=[1,1], label="7.5cm")
# plt.grid(True)
# plt.xlabel("Distance from z=0 [m]")
# plt.ylabel("Magnetic field strength, relative [%]")
# plt.title(f"Relative magnetic field along {d*100:.0f}cm coil axis")
# plt.legend()
# plt.show()

l = 2*np.pi*a
wire_r = 0.0013 # m = 1.3mm
strip_w = 0.010 # m = 1.0cm
strip_thickness = 0.010 # m = 10.0mm
L_wire = perm*l/(2*np.pi) * (np.log(2*l/wire_r) - 1)
L_strip = perm*l/(2*np.pi) * (np.log(2*l/strip_w + 1/2))
print(f"Circular wire: L = {L_wire}\nWire strip: L = {L_strip}")