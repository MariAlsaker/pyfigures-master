import numpy as np
import magpylib as magpy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from scipy.spatial.transform import Rotation as R

PATH = "/Users/marialsaker/git/pyfigures-master"

class MRIcoil:
    def __init__(self, current:int, diameter:int=90, quadrature:bool=False, quad_dir:str=None, custom_coil:bool=False, custom_coil_current_line=None):
        """ Initializes a coil object for computing B and/or H field around it. \n
        Params:
        - diameter: int [mm] 
        - current: int [A]
        - quadrature: boolean, if this is a quadrature coil or not
        - quad_dir: string ("out", "in", None) direction of induced flux through coils
        - custom_coil: tells if there is an input in custom_coil_current_line
        - custom_coil_current_line: a magpy.current.Line object"""
        self.diameter = diameter
        name = ""
        if custom_coil:
            name = f"custom_"
        if quadrature:
            name = f"quadrature_"
        self.coil_name = name+f"coil_d{self.diameter}_I{current}"
        self.__path = "savedcoils/"
        if quadrature:
            r = R.from_euler('zyx', [0., 135., 0.], degrees=True)
            radii  = diameter/2
            if quad_dir == "out": a=-1
            else: a=1
            coil1 = magpy.current.Loop(current=a*current, diameter=self.diameter)
            coil2 = magpy.current.Loop(current=-a*current, diameter=self.diameter, orientation=r, position=(radii+np.sqrt(radii**2/2), 0., np.sqrt(radii**2/2)))
            self.coil = magpy.Collection((coil1, coil2))
        elif custom_coil:
            self.coil = custom_coil_current_line
        else:
            self.coil = magpy.current.Loop(current=current, diameter=self.diameter)
        self.__quad = quadrature
        self.__3Dgrid = self.__construct_3Dgrid()
        self.__B_field = self.__compute3D_B_field()
        self.__H_field = self.__compute3D_H_field()

    def get_name(self):
        return self.coil_name

    def show_coil(self, ax=None, show=False):
        """ Show a 3D model of this coil object """
        fig = magpy.show(self.coil, backend='matplotlib',return_fig=True, canvas=ax)
        if show:
            plt.show()

    def show_field_lines(self, slice=str, ax=None, show=False, fig=None, colorbar=True):
        if ax == None or fig == None:
            fig, ax = plt.subplots()
        coord_axs = "xyz" # old:  "yxz"  
        flip_ax=False
        if slice[0] == "x":
            grid = self.__3Dgrid[:,int(slice[1:]),:]
            B = self.__B_field[:,int(slice[1:]),:]
            indices = (0, 2)
        elif slice[0] == 'y':
            grid = self.__3Dgrid[:,:,int(slice[1:])]
            B = self.__B_field[:,:,int(slice[1:])]
            indices = (1, 2)
        else:
            grid = self.__3Dgrid[int(slice[1:])]
            B = self.__B_field[int(slice[1:])]
            indices = (0, 1)
            flip_ax = False
        log10_norm_B = np.log10(np.linalg.norm(B, axis=2))
        splt = ax.streamplot(
            grid[:, :, indices[0]],
            grid[:, :, indices[1]],
            B[:, :, indices[0]],
            B[:, :, indices[1]],
            color=log10_norm_B,
            density=1,
            linewidth=log10_norm_B*2,
            cmap="autumn",
        )
        if colorbar:
            cb = fig.colorbar(splt.lines, ax=ax, label="|B| (mT)")
            ticks = np.array([3,10,30,100])
            cb.set_ticks(np.log10(ticks))
            cb.set_ticklabels(ticks)
        if flip_ax:
            new_i = (indices[1], indices[0])
            indices = new_i
        ax.set(
            xlabel=f"{coord_axs[indices[0]]}-position (cm)",
            ylabel=f"{coord_axs[indices[1]]}-position (cm)")
        plt.tight_layout()
        if show:
            plt.show()
        
    def show_field_magnitude(self, chosen_field:str, vmax:int=100, gif_name:str=None):
        """ Show an animation of the normalized field magnitude in all slices through z-axis. \n 
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
        ticks = np.linspace(0, 100, 5, endpoint=True)
        normalized_field_magn = self.calc_normalized_field_magn(field)
        for f_slc in normalized_field_magn:
            if not initialized:
                img = ax.pcolormesh(f_slc, cmap="plasma", vmin=0, vmax=vmax)
                plt.colorbar(img, label="[%]")
                ax.set_title(f"Magnetic flux density, {chosen_field}")
                ax.set_xlabel("x(mm)")
                ax.set_xticks(ticks, labels=["-80", "-40", "0", "40", "80"])
                ax.set_yticks(ticks, labels=["-80", "-40", "0", "40", "80"])
                ax.set_ylabel("y(mm)")
                initialized = True
            else:
                img = ax.pcolormesh(f_slc, cmap="plasma", vmin=0, vmax=vmax, animated=True)
            imgs.append([img])
        ani = animation.ArtistAnimation(fig, imgs, interval=50, blit=True,
                                repeat_delay=1000)
        plt.show()
        if gif_name:
            writergif = animation.FFMpegWriter(fps=30)
            ani.save("Figures/"+gif_name, writer=writergif)

    def calc_normalized_field_magn(self, field):
        f_coil_center = np.sqrt(np.sum(field[50][50][50]**2))
        field_magn = np.sqrt(np.sum(np.square(field), axis=-1))
        normalized_field_magn = field_magn/f_coil_center*100
        return normalized_field_magn

    def show_field_slice(self, chosen_field:str, slice, ax, verbose:bool=False):
        """ Show a specific slice normal to the z-axis. \n 
        Parameters:
        - chosen_field: B or H
        - slice: int 0 to 100 ranging from -80mm to 80mm below/above the coil"""
        if chosen_field=="B":
            field = self.__B_field
        elif chosen_field == "H":
            field = self.__H_field
        else:
            raise Exception(f"No field called {chosen_field}, only B or H")
        normalized_field_magn = self.calc_normalized_field_magn(field)
        img = ax.pcolormesh(normalized_field_magn[slice], cmap="plasma", vmin=0, vmax=100)
        plt.colorbar(img, label="[%]")
        ticks = np.linspace(0, 100, 5, endpoint=True)
        ax.set_xticks(ticks, labels=["-80", "-40", "0", "40", "80"])
        ax.set_yticks(ticks, labels=["-80", "-40", "0", "40", "80"])
        ax.set_title(f"Magnetic flux density, {chosen_field}, z_ax = {slice*1.6-80:.2f} mm")
        ax.set_xlabel("x(mm)")
        ax.set_ylabel("y(mm)")
        if verbose:
            print(f"Field magnitude in middle point is {normalized_field_magn[slice][50][50]:.2f}%")


    def __construct_3Dgrid(self):
        """ Constructs a 3D grid with extent diameter**3
        \nNo return, stores grid in object.
        """
        file_path = self.__path+self.coil_name+"__3Dgrid"
        if os.path.exists(file_path+".npy"):
            return np.load(file_path+".npy")
        else: 
            extent = (self.diameter+2)/2
            X, Y = np.mgrid[-extent:extent:100j, -extent:extent:100j].transpose((0,2,1)) # X, Y = np.mgrid[-extent:extent*1.5:150j, -extent:extent:150j].transpose((0,2,1)) 
            if self.__quad:
                zs = np.linspace(-2, self.diameter, 100)
            else:
                zs = np.linspace(-extent, extent, 100)
            grid_3D = []
            for z in zs:
                grid_xy = np.stack([X, Y, np.zeros((100,100)) + z], axis=2)
                grid_3D.append(grid_xy)
            grid_3D = np.array(grid_3D)
            grid_3D.tofile(file_path)
            return grid_3D

    def __compute3D_B_field(self):
        """ Computes B-field around the coil in 3D.
        \nNo return, stores field in object.
        """
        file_path = self.__path+self.coil_name+"__Bfield"
        if os.path.exists(file_path+".npy"):
            return np.load(file_path+".npy")
        else:
            B_field_3D = []
            for xy_grid in self.__3Dgrid:
                xy_field_B = self.coil.getB(xy_grid)
                B_field_3D.append(xy_field_B)
            B_field_3D = np.array(B_field_3D)
            B_field_3D.tofile(file_path)
            return B_field_3D
    
    def __compute3D_H_field(self):
        """ Computes H-field around the coil in 3D.
        \nNo return, stores field in object.
        """
        file_path = self.__path+self.coil_name+"__Hfield"
        if os.path.exists(file_path+".npy"):
            return np.load(file_path+".npy")
        else:
            H_field_3D = []
            for xy_grid in self.__3Dgrid:
                xy_field_H = self.coil.getB(xy_grid)
                H_field_3D.append(xy_field_H)
            H_field_3D = np.array(H_field_3D)
            H_field_3D.tofile(file_path)
            return H_field_3D
    
    def get_Bfield_vec(self):
        return self.__B_field
    
    def get_Hfield_vec(self):
        return self.__H_field