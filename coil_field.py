import numpy as np
import magpylib as magpy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

PATH = "/Users/marialsaker/git/pyfigures-master"

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
        """ Show a 3D model of this coil object """
        magpy.show(self.coil, backend='matplotlib')
        plt.title("3D model of conducting loop")
        plt.show()
        
    def show_field(self, chosen_field:str, vmax:int=30, gif_name:str=None):
        """ Show an animation of all slices through z-axis. \n 
        Parameters:
        - chosen_field: B or H
        - vmax: maximum value in coloring (very strong field around loop ~200)
        - gif_name: if you want to save a movie of the animation give a string: \"name.mp4\""""
        if chosen_field=="B":
            field = self.__B_field
        elif chosen_field == "H":
            field = self.__H_field
        else:
            raise Exception(f"No field called {chosen_field}, only B or H")
        
        initialized = False
        imgs = []
        fig = plt.figure(figsize=(9,7))
        ax = fig.add_subplot(1,1,1)
        for f_slc in field:
            field_magn = np.sqrt(np.sum(f_slc**2, axis=-1))
            if not initialized:
                img = ax.pcolormesh(field_magn, cmap="plasma", vmin=0, vmax=vmax)
                plt.colorbar(img)
                initialized = True
            else:
                img = ax.pcolormesh(field_magn, cmap="plasma", vmin=0, vmax=vmax, animated=True)
            ax.set_title(f"Magnetic flux density, {chosen_field}")
            ax.set_xlabel("x(mm)")
            ax.set_ylabel("y(mm)")
            imgs.append([img])
        print(len(imgs))
        ani = animation.ArtistAnimation(fig, imgs, interval=50, blit=True,
                                repeat_delay=1000)
        plt.show()
        if gif_name:
            writergif = animation.FFMpegWriter(fps=30)
            ani.save(gif_name, writer=writergif)

    def show_field_slice(self, chosen_field:str, slice):
        if chosen_field=="B":
            field = self.__B_field
        elif chosen_field == "H":
            field = self.__H_field
        else:
            raise Exception(f"No field called {chosen_field}, only B or H")
        field_magn = np.sqrt(np.sum(field[slice]**2, axis=-1))
        fig = plt.figure(figsize=(7,7))
        ax2 = fig.add_subplot(1,1,1)
        vis = ax2.imshow(field_magn, cmap="plasma")
        vis.set_extent((-80, 80, -80, 80))
        ax2.set_title(f"Magnetic flux density, {chosen_field}")
        ax2.set_xlabel("x(mm)")
        ax2.set_ylabel("y(mm)")
        plt.colorbar(vis)
        plt.show()

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