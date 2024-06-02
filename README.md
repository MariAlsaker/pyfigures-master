# Code developed for the master's project: Surface coil development for multinuclear MRI 
### Created by Mari Maaløy Alsaker

#### Figures created for the theory chapter
- circularly_polarized_B.py were used to create the illustration of a circularly polarized field (Fig 2.26).
- coil_shape_compare.py contains the code for creating Figure 2.12, comparing a circular and a square loop.
- magnetic_susceptibility.py Stores the values used in the table of magnetic susceptibility in the thesis (Table 2.3).
Had to recalculate from imperial units into SI units.
- background_figs.py contains the code for creating the diode law plot (Fig. 2.13), the magnetic relaxation plot (Fig. 2.4), and the impedance plots of a series (Fig. 2.16) and parallel (Fig. 2.18) resonator.
enhancing_coil_inside_birdcage.py were used to make Figure 2.22 by choosing only to have the birdcage coil lines active.

#### Figures created for the results chapter
- The static magnetic field calculations using matplotlib were done in several files. 
    - quadrature_field_rotating.py - Fig. 4.6
    - quadrature_field.py - Fig. 4.5
    - single-loop_field.py - Fig. 4.3
    - enhancing_coil_inside_birdcage.py - Used for making Figure 4.4 when the parameter "partnum" was set to 0 and all current lines active. Figure 4.7 was made by changing "partnum" through 0-9 and storing the magnitude and field line plot for each run. The values stored in "changing_field.txt" were gathered by an initial round through the 0-9 with only the birdcage active (the lines of code storing these values are commented out in the current file).
- nanoVNAplots.py handles laboratory measurements: Plotting S-parameter magnitudes and generating Smith charts from touchstonefiles. The figures numbers in the thesis are: 4.9, 4.12, 4.14, 4.15, 4.16, 4.17, 4.18, 4.19
- MRI_images.py handles the MRI images: all displayed images, field corrections, line plots, and the SNR plot. The figures numbers in the thesis are: 4.20, 4.21, 4.22, 4.23, 4.24, 4.25, 4.26, 4.27, 4.28, 4.29, 4.30, 4.31, 4.32, 4.33, 4.34, 4.35, 4.36, 4.37, 4.38, 4.39, 4.40.
- load_MRI_data.py was used to load the MRI data from the ".mat" or matrix file format into numpy arrays.


#### Initial work
A homemade class for simulating MRI coils using magpylib can be found in my_classes/MRI_coil.py. Calculations in this class were time-consuming, and the class was therefore not used to create the figures of the static magnetic field shown in the thesis. This class, together with the file coil_planning.py, was heavily used during the first few months of the project.

developing_calculator.py contains functions for calculating component values according to the formulas presented in the thesis. Useful during experimental development of coils. 

Some initial designs and calculations are in the "old" folder. These also includes the code for creating some of the figures used on the poster in Appendix C of the thesis. 

A link to the published thesis in BORA will be provided here: XX

Python packages used in this repo:
- magpylib: https://magpylib.readthedocs.io/en/latest/
- pysmithplot: https://github.com/vMeijin/pySmithPlot
- matplotlib
- numpy
- stl
- scipy
- datetime
- os
- collections
- h5py
