import numpy as np
from scipy.constants import *
#Global variables for EM/QM
Ly= 50*angstrom #m, length of the nanosheet
dy= 0.25*angstrom #m
dx= 0.25*angstrom
Ny= 400
Nx= 400 #make the total simulation grid to be square, and twice as large in y as the nanosheet (100angstrom). So above nanosheet we have 25 angstrom and below also. 
Nt= int(2.5e2) 
dt= 1/(c* np.sqrt(1/dx**2 + 1/dy**2)) #s , the constraint for UCHIE is more strict than the one for schrodinger!

#EM
nx_s= Nx//4
ny_s = 4* Ny//10 #place source at the coordinates (25, 10) angstrom (see project description for axes). y axis starts from the middle, but the matrices are defined from bottom to top. 
J0= 5*1e6
tc= Nt/20* dt
sigma= tc/6
eps = 8.854 * 10**(-12) # Permittivity of free space F/m
mu = 4 * np.pi * 10**(-7) # Permeability of free space H/m

#QM
# discr_order= "second"
# nx_nano= Nx//2 #place nanosheet in the middle of the simulation domain
# kvalues= np.arange(0, Ly//dy+1) #For QM part
# N= 10**7 #1/m
# effective_mass= 0.15*electron_mass 
# omega_HO= 50*10**(14) #rad/s
# q= -elementary_charge