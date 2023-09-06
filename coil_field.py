import numpy as np
import magpylib as magpy
import matplotlib.pyplot as plt

class Coil_field:
    def __init__(self, diameter:int, current:int):
        """ Initializes a coil object for computing B and/or H field around it. \n
        Params:
        - diameter: int [mm] 
        - current: int [A]"""
        self.coil_name = f"coil_d{diameter}_I{current}"
        self.__diameter = diameter
        self.coil = magpy.current.Loop(current=current, diameter=diameter)
        self.__extent = (diameter + 2)/2
        self.__3Dgrid = self.__construct_3Dgrid()
        self.__B_field = self.__compute3D_B_field()
        self.__H_field = self.__compute3D_H_field()

    def get_name(self):
        return self.coil_name

    def show_coil(self):
        """ Show a 3D model of the coil """
        magpy.show(self.coil, backend='matplotlib')
        plt.title("3D model of conducting loop")
        plt.show()
    
    def save_visualized_field(self, chosen_field:str, folder:str):
        """ Create a folder of all field slices """
        if chosen_field=="B":
            field = self.__B_field
        elif chosen_field == "H":
            field = self.__H_field
        else:
            raise Exception(f"No field called {chosen_field}, only B or H")
        
        #NOT FINISHED
        i = 0
        f_slc = field[50]
        #for i, f_slc in enumerate(field):
        name = f"{self.get_name()}_slice{i}"
        fig = plt.figure()

        field_magn = np.sqrt(np.sum(f_slc[0]**2, axis=-1))
        plt.savefig(name)
        return 0
    
    def show_field(self, field:str, slice:tuple):
        """ Create a folder of all field slices """
        return 0

    def __construct_3Dgrid(self):
        """ Constructs a 3D grid with extent diameter**3
        \nNo return, stores grid in object.
        """
        extent = self.__extent
        X, Y = np.mgrid[-extent:extent:100j, -extent:extent:100j].transpose((0,2,1)) 
        zs = np.linspace(-extent, extent, 100)
        grid_3D = []
        for z in zs:
            grid_xy = np.stack([X, Y, np.zeros((100,100)) + z], axis=2)
            grid_3D.append(grid_xy)
        grid_3D = np.array(grid_3D)
        return grid_3D

    def __compute3D_B_field(self):
        """ Computes B-field around the coil in 3D.
        \nNo return, stores field in object.
        """
        B_field_3D = []
        for xy_grid in self.__3Dgrid:
            xy_field_B = self.coil.getB(xy_grid)
            B_field_3D.append(xy_field_B)
        B_field_3D = np.array(B_field_3D)
        return B_field_3D
    
    def __compute3D_H_field(self):
        """ Computes H-field around the coil in 3D.
        \nNo return, stores field in object.
        """
        H_field_3D = []
        for xy_grid in self.__3Dgrid:
            xy_field_H = self.coil.getB(xy_grid)
            H_field_3D.append(xy_field_H)
        H_field_3D = np.array(H_field_3D)
        return H_field_3D