import numpy as np

# Our alternatives
g = 0.001 # mm wire diameter
d = 0.15 # m coil diameter
N = 1 # num turns
l = 2*np.pi*d/2 # wire length
w_r = 2*np.pi*33.786*10**6 # Hz Resonance frequency
u_0 = 4*np.pi*10**(-7)
u_Cu = u_0
p_Cu = 1.678*10**(-8)
delta = np.sqrt(2*p_Cu/(w_r*u_Cu))
sigma_sample = 377 # Ohm when sample is free air
lamb = 0.26 # For H2O at 3T

# Calculations 
L_loop = u_0*d/2 * (np.log(8*d/g)-2)
L_wire = 8*10**(-7)*N*d*(2.303*np.log(16*d/g - 0.75+8*N*d*g))
A_loop = np.pi*(d/2)**2
A_conductor = np.pi*(g/2)**2
A_eff = A_conductor - np.pi*(g/2 - delta)**2
R_ac = l*p_Cu/A_eff
R_rad = 8/3*sigma_sample*np.pi**3*(N*u_0*A_loop/lamb**2)

R = R_ac+R_rad
L = L_wire+L_loop

C_tot = 1/(w_r**2 * L)
C = C_tot*4 # All capacitors in the ring


print(f"\nWe have: ", 
      f"\nR_AC = {R_ac}", f"\nR_rad = {R_rad}", f"\n -> R_tot = {R}",
      f"\nL_wire = {L_wire}",f"\nL_loop = {L_loop}", f"\n -> L_tot = {L}")
print(f"Our desired C in the loop is then 1/(w^2 * L) = {C_tot}")
print("This means that we need 4 capacitors of 4*C = ", C)

# Complex capcitance, z_tot = (x+yi)/(a+bi)
x = -3/(w_r**2 * C**2) + L/C
y = -R/(w_r*C)
a = R 
b = (-4/(w_r*C) + w_r*L)

def calculate_complex_quotient(x, y, a, b): #z_tot = (x+yi)/(a+bi)
    # z_tot = (x+yi)(a-bi)/(a^2+b^2)
    # (x+yi)(a-bi) = (xa+yb) + (ay-bx)i
    # z_tot = (xa+yb)/(a^2+b^2) + (xb-ay)/(a^2+b^2) i
    real = (a*x + b*y)/(a**2 + b**2)
    imag = (a*y - b*x)/(a**2 + b**2)
    return real, imag
z_tot_real, z_tot_im = calculate_complex_quotient(x, y, a, b)
print(f"\nThe impedance presented to matching element is: \nZ = {z_tot_real} + {z_tot_im}j")

print(f"Capacitive reactance is {1/(w_r*(C_tot))} and inductive reactance is {L*w_r}")
print(f"R={R}, L={L}, C={C}")

test_real, test_imag = calculate_complex_quotient(0, -333*1000, 1000, -333)
print(f"Test: {test_real:.0f} + {test_imag:.0f}j ohms")