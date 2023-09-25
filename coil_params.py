import numpy as np

#perm_air = 4*np.pi*10E-7 

w_0 = 2 *np.pi* 33.78E6 # Sodium

def single_loop_calc(loop_d, wire_d, n_caps):
    inductance_L = expected_L(loop_d, wire_d) # omitted at low fields: + perm_air*wire_d
    return cap_from_L(inductance_L)*n_caps

def single_loop_test(meas_L, n_caps):
    return cap_from_L(meas_L)*n_caps

def expected_L(loop_d, wire_d):
    wire_l = 2*np.pi*loop_d/2 # Diameter of circle
    inductance_L = 0.2*wire_l*( np.log(4*wire_l/wire_d) - 1 ) # source: NMR probes
    return inductance_L*1E-6

def cap_from_L(L):
    return 1/ (L*w_0**2)


### RUNNING CALCULATIONS ###
loop_d = 0.15 #m = 15cm
wire_d = 0.0013 #m = 1.3mm
exp_L = expected_L(loop_d, wire_d)
n_caps = 2
exp_caps = single_loop_calc(loop_d, wire_d, n_caps)
measured_L = 00
print("\nA 15cm diameter loop of 1.3mm copper wire")
print(f" - Expected inductance: {exp_L*1E6:.5f} uH")
print(f" --> This means the two caps should have ≈{exp_caps*1E12:.5f} pF\n")