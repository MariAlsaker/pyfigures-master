# Code developed for the master's project: Surface coil development for multinuclear MRI 
### Created by Mari Maaløy Alsaker

Python packages used in this repo:
- magpylib: https://magpylib.readthedocs.io/en/latest/
- pysmithplot: https://github.com/vMeijin/pySmithPlot

#### Figures created for the theory chapter
- circularly_polarized_B.py were used to create the illustration of a circularly polarized field.
- coil_shape_compare.py contains the code for creating the figure comparing a circular and a square loop.
- magnetic_susceptibility.py Stores the values used in my table of magnetic susceptibility in my masters.
Had to recalculate from imperial units into SI units.
- background_figs.py contains the code for creating the diode law plot, the magnetic relaxation plot, and the impedance plots of a series and parallel resonator.

#### Figures created for the results chapter
- All the magnetic field shit
- nanoVNAplots.py handles laboratory measurements: Plotting S-parameter magnitudes and generate Smith charts from touchstonefiles. File: 
- MRI_images.py handles the MRI images: all displayed images, field corrections, line plots and the SNR plot. 


#### Initial work
A homemade class for simulating MRI coils using magpylib can be found in my_classes/MRI_coil.py . Calculations in this class was timeconsuming and was not used for creating the figures of static magnetic field shown in the thesis. This class together with the file coil_planning.py was heavily used for the first few moths of the project work.

developing_calculator.py contains functions for calculating component values according to the formulas presented in the thesis. Useful during experimental development of coils. 

