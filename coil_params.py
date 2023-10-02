import numpy as np

#perm_air = 4*np.pi*10E-7 

w_0 = 2 *np.pi* 33.78E6 # Sodium

def expected_wire_L(wire_l, wire_d):
    """Inductance of a solid round wire (or tube) of diameter d and length l
    Source: NMR probeheads, equation 2.34
    Params:
    - loop_d = Loop diameter [m] - used to calculate wire length
    - wire_d = Wire diameter """
    # wire_l = 2*np.pi*loop_d/2 # [m]
    wire_L = 0.2*wire_l*( np.log(4*wire_l/wire_d) - 1 + 0.01)
    # last part of equation can be excluded (permeability * skin depth = 1 * 0.01)
    return wire_L*1E-6

def expected_loop_L(loop_d, wire_d):
    """Inductance of a circular ring of tubular cross section
    Source: NMR probeheads, equation 2.97
    Params:
    - loop_d = Loop diameter [m]
    - wire_d = Wire diameter """
    loop_L = np.pi/5*loop_d*(np.log(8*loop_d/wire_d)-2)*1E-6
    return loop_L

def cap_from_L(L):
    return 1/ (L*w_0**2)

# TODO: Add functions to calculate resistance of the system and values for matching capacitors

# ---- RUNNING CALCULATIONS ---#
loop_d = 0.15 # [m] = 15 cm
wire_d = 0.001 # [m] = 1.00 mm
n_caps = 2
measured_L = 537E-9 # Henry
meas_cap = cap_from_L(measured_L)
wire_ends_l = 0.02 # [m] length of wire from loop to SMA connector
exp_loop_L = expected_loop_L(loop_d, wire_d) + expected_wire_L(wire_l=2*wire_ends_l, wire_d=wire_d)
exp_caps = cap_from_L(exp_loop_L)
print("\nA 15cm diameter loop of 1.0mm copper wire")
print(f" - Expected loop inductance: {exp_loop_L*1E6:.5f} uH")
print(f" --> This means the total cap should be ≈ {exp_caps*1E12:.5f} pF")
print(f"    (With two caps, we should have {exp_caps*2*1E12:.5f} pF)")
print(f" - Measured inductance: {measured_L*1E6:.5f} uH")
print(f" --> This means the total cap should be ≈ {meas_cap*1E12:.5f} pF")
print(f"    (With two caps, we should have {meas_cap*2*1E12:.5f} pF)\n")
# Higher value of measured inductance - why?? maybe some inductance from the connector??