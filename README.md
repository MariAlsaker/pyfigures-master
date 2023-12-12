## MRI coil development 
#### For sodium MRI

Python packages (unorthodox) used in this repo:
- magpylib: https://magpylib.readthedocs.io/en/latest/
    - Best used with air coils, meaning no eddy currents.
- pysmithplot: https://github.com/vMeijin/pySmithPlot 

#### MRI_coil.py 
My main simulation tool. A homemade class for simulating MRI coils using magpylib.

### nanoVNAplots.py 
Functions made to plot data from touchstonefiles in the same manner as a VNA does.
- Could be made into a class
- Preferably I would implement animations to use it as a live plot with the Signal Analyzer (Rhode&Schwarz)

#### magnetic_susceptibility.py
Stores the values used in my table of magnetic susceptibility in my masters.
Had to recalculate from imperial units into SI units. 

#### developing_calculator.py
Contain functions for calculating component values. Useful during experimental development of coils. 

