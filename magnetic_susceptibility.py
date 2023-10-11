import numpy as np

class susceptibility_list:

    def __init__(self, n_materials:int=100):
        self.__susceptibilities = {}

    def add_material(self, material_name, molar_mass, susceptibility):
        """ Converts fron susceptibility in 10^-6 cm^3/mol to 10^-8 m^3/kg
        """ 
        mass_susceptibility = susceptibility*10**-12 / (molar_mass*10**-3) * 4*np.pi
        self.__susceptibilities[material_name] = mass_susceptibility

    def print_list(self):
        for sus in self.__susceptibilities.keys():
            print(f"{sus}: {self.__susceptibilities[sus]*10**8:.3f} E-8 m^3/kg")



mylist = susceptibility_list()
mylist.add_material("Copper (Cu)", 63.546, -5.46)
mylist.add_material("Aluminum (Al)", 26.982, 16.5)
mylist.add_material("Silver (Ag)", 107.87, -19.5)
mylist.add_material("Tin (Sn)", 118.71, -37.4)
mylist.add_material("Lead (Pb)", 207.2, -23)
mylist.add_material("Water (H2O), 293K", 18.015, -12.96)

mylist.print_list()