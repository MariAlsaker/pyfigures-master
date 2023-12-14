import numpy as np

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
    # Skin depth value source: Radio Engineers Handbook, Terman, 1943, page 49
    return wire_L*1E-6

def expected_loop_L(loop_d, wire_d):
    """Inductance of a circular ring of tubular cross section
    Source: NMR probeheads, equation 2.97
    Params:
    - loop_d = Loop diameter [m]
    - wire_d = Wire diameter """
    loop_L = np.pi/5*loop_d*(np.log(8*loop_d/wire_d)-2 + 0.01)*1E-6
    return loop_L

def expected_square_L(side_l, wire_d):
    """Inductance of a square loop of tubular cross section
    Source: NMR probeheads, equation 2.98
    Params:
    - side_l = Side length of square [m]
    - wire_d = Wire diameter """
    square_L = 0.8 * side_l *(np.log(2*side_l/wire_d) + wire_d/(2*side_l) - 0.774) * 1E-6
    return square_L

def cap_from_L(L):
    return 1/ (L*w_0**2)

# TODO: Add functions to calculate resistance of the system and values for matching capacitors
def expected_R(loop_d, wire_d, added_wire):
    rho_cu = 1.724E-8 # [Ohm * m] Resistivity
    mu_0 = 4E-7*np.pi # [N/A^2] Vacuum permeability
    skin_depth = np.sqrt(2*rho_cu/(w_0*mu_0))
    # Source: Elements of electromagnetics, Sadiku, eq (10.54b)
    A_eff = wire_d*skin_depth-skin_depth**2 # Hollow tube of wall thickness equal to skin depth
    wire_length = np.pi*loop_d + added_wire
    R_ac = wire_length*rho_cu/A_eff
    # Source: Elements of electromagnetics, Sadiku, eq (5.16)
    Z0 = 377 # Ohm, inductance of free space
    lamb = 3E8/(w_0/2*np.pi)
    R_rad = 8/3 * Z0 * np.pi**3 * (mu_0*np.pi*(loop_d/2)**2/lamb**2)**2
    # Source: Alex Carven - Loudet 2009
    return R_ac + R_rad

def matching_caps_from_RZ(C, resistance, impedance_line):
    C_m = C*np.sqrt(resistance/impedance_line)
    return C_m

# ---- RUNNING CALCULATIONS ---#
loop_d = 0.105 # [m] = 9 cm
square_s = 0.09 # [m] = 9 cm
wire_d = 0.0015 # [m] = 1.00 mm
n_caps = 2
measured_R = 3.89 # Ohm
wire_ends_l = 0.02 # [m] length of wire from loop to SMA connector
exp_loop_L = expected_loop_L(loop_d, wire_d) #+ expected_wire_L(wire_l=2*wire_ends_l, wire_d=wire_d)
exp_caps = cap_from_L(exp_loop_L)
print(f"\nA {wire_d*1000:.1f}mm copper wire shaped as a square with s = {square_s*100:.1f}cm, or as a loop with d = {loop_d*100:.1f}cm")
exp_R = expected_R(loop_d, wire_d, wire_ends_l*2)
print(f" - Expected square inductance: {expected_square_L(side_l=square_s, wire_d=wire_d)*1E9:.2f} nH")
print(f"   (Area = {square_s**2} m^2)")
print(f" - Expected loop inductance: {exp_loop_L*1E9:.2f} nH, Expected loop resistance: {exp_R:.3f} Ohm")
print(f"   (Area = {(loop_d/2)**2*np.pi} m^2)")
print(f" --> This means the total cap should be ≈ {exp_caps*1E12:.5f} pF")
print(f"    (With n caps, we should have {exp_caps*n_caps*1E12:.5f} pF)")
