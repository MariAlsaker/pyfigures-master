import numpy as np

""" This is my own everythingRF calculator (yes there is an app) """

def omega(f:float):
    """ Angular frequency from frequency in Hz """
    return 2*np.pi*f

def L(f, c):
    """ Loop inductance from measured resonant frequency with given C series capacitance """
    return 1/(omega(f)**2*c)

def C(f:float, l:float):
    """ Capacitance needed in series to achieve resonance at f with a loop having inductance l """
    return 1/(omega(f)**2*l)

def M(L_1:float, L_2:float, f_min:float, f_max:float):
    " Mutual inductance between two loops from separate loop inductances and resonant frequencies of the two coupled peaks "
    distance_f = f_max-f_min
    k = distance_f/(f_min+distance_f/2)
    # Capacitative decoupling: C = 1/(w**2 * M)
    # Inductive decoupling: L = M/k ? or we can just use overlapping
    return k*np.sqrt(L_1*L_2)  

def Q(f, bw):
    """ Q value from resonant frequency and band-width """
    # Not sure if this is the way to do it... Gosha seys no (at least not when the peak is not low enough into the decibels), Nikolai told me its good
    # Gosha would rather measure Q with the douple loop pick-up coil.
    return 2*f/bw  

def R_coil(f, L, Q):
    """ Coil resistance from resonant frequency, loop inductance and measured Q-value, preferably loaded (we only measure unloaded Q to see how bad our electric losses are) """
    return omega(f)*L/Q

def C_tot(c1, c2):
    return (c1*c2)/(c1+c2)

# Wire thickness is 1.5 mm ?
f = 69.48E6
L_test = L(f=69.48E6, c=18E-12)
print(f"Calculated L from frequency ({f}) = {L_test*1E9:.2f} nH")
f_R=11.262E6*3
C_sodium = C(f=f_R, l=L_test)
print(f"Sodium resonant freq is {f_R*1E-6:.2f} MHz")
print(f"To achieve sodium freq,  C = {C_sodium*1E12:.2f} pF")
print(f" - With 2 distributed caps we need double, {2*C_sodium*1E12:.2f} pF")
Z_0 = 50 #Ohm
Q_loaded = 200 # TODO: Measure with double pickup coil
C_m = C_sodium*np.sqrt(R_coil(f=34E6, L=L_test, Q=Q_loaded))
print(f"Our matching capacitance should be {C_m*1E12:.2f} pF")
print(f"For parallel matching circuitry, 2*Cm = {2*C_m*1E12:.2f}")
# Need to ask Nikolai about choke balun - how to make it and what to make sure of
# Also need to calculate values for hybrid. 
# What about fastening the different wires to my coil holders?? Do we need to 3D print?

C_after = C_tot(150E-12, 150E-12)
C_better = 120E-12 + 26E-12
print(f"New C = {C_better*1E12:.2f} pF")