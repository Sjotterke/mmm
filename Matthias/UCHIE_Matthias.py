import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
from scipy.special import hankel2
import math
from scipy.constants import *
import time 
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter
from parameters import *
from Functions import *
animation =True 



# Place sensors


# Courant Number and resulting timestep

# CFL = 0.9
# dt = CFL *dy/c
tarray = np.linspace(0, Nt*dt, Nt)
# plot of source

# plt.figure()
# plt.plot(tarray, gaussian_pulse(tarray))
# plt.show()


print("dx: {}\ndy: {}".format(dx, dy))
# print("dt from 10dy/c: {}".format(dt))
print('dt from schrodinger: {:2e}'.format(2*hbar/ np.max(potential(kvalues*dy))))
print('dt from UCHIE: {:2e}'.format(dt))

# Define the fields
AD, AI, M, L = matrices_construct()

# X contains values of Ey and Bz
X = np.zeros((2*Nx+2, Ny))
# PsiR and PsiI are defined in a new simulation coordinate system spanning from 1/4 to 3/4 (where the nanosheet stretches)
PsiR= ground_state(kvalues*dy) 
PsiI= np.zeros(int(Ly//dy)+1)
Psi = PsiR + 1j* PsiI
print('initial expectation value of the position of the electron in nm: ',giga * (Ny*dy/4+np.real(np.trapz(np.conjugate(Psi)* dy*kvalues * Psi, dx=dy))))
# Periodic Boundary Conditions 1 in x direction
BC1 = np.zeros((1, 2*Nx+2))
BC1[0,0] = 1
BC1[0, Nx] = -1
M = np.vstack((M, BC1))
L = np.vstack((L, np.zeros((1,2 *Nx+2)))) 
# L = np.vstack((L, BC1))
#Periodic Boundary Conditions 2 in x direction
BC2 = np.zeros((1, 2*Nx+2))
BC2[0,Nx+1] = 1
BC2[0, -1] = -1
M = np.vstack((M, BC2))
# L = np.vstack((L, BC2))
L = np.vstack((L, np.zeros((1, 2*Nx+2))))
# check determinant of M non-zero
# detM = np.linalg.det(M)
# print("Determinant of M: {}".format(detM))

Minv = np.linalg.inv(M)
MinvL = np.matmul(Minv, L)
Ex = np.zeros((Nx+1, Ny+1)) 
# Create fig for animation
if animation == True:
    fig, ax = plt.subplots()
    artists = []
    plt.title("Bz")
    plt.xlabel("x")
    plt.ylabel("y")
    # Initialize legend outside the animation loop
    particle_legend = plt.scatter([], [], color='black', label='Particle')
    nanosheet_legend = plt.vlines([], [], [], colors='purple', label='nanosheet', linewidth=0.2)
J= []
prob_dens= np.zeros((int(Ly//dy)+1, Nt))
for it in range(Nt):
    t = it*dt
    if it%100 == 0:
        print("Iteration: {}/{}".format(it, Nt))

    Y = Ex[:-1, 1:] + Ex[1:,1:] - Ex[:-1,:-1] - Ex[1:,:-1]
    
    Y_tot = np.vstack((Y, np.zeros((2+Nx, Ny))))

    # print("Y_tot shape: {}".format(Y_tot.shape))
    X[Nx+1+nx_s, ny_s] += gaussian_pulse(t)  #add source to bz so add Nx+1 in row dimension, as X= [Ey ; Bz]
    
    middel = np.matmul(L, X) + 1/dy*Y_tot
    X = np.matmul(Minv, middel)
    Ex[:,1:-1] = (eps/dt - sigma/2)/(eps/dt + sigma/2) * Ex[:,1:-1] + 1/(eps/dt + sigma/2)*(1/(mu*dy))*(X[Nx+1:, 1:]-X[Nx+1:, :-1])
    Ex[:,0] = 0
    Ex[:,-1] = 0

    #Now, with Ey, update the QM part
    PsiR, PsiI= update_psi(PsiR, PsiI, np.transpose(X[nx_nano, Ny//4 : 3*Ny//4]), discr_order) # yvalues of Ey relevant for QM range from Ny/4 to 3Ny/4, but are reversed in the matrix (top to bottom instead of bottom to top)
    X[nx_nano, Ny//4+1 : 3*Ny//4] += current_density(PsiR,PsiI) #Ny/4+1 for dimension mismatch J[0] is always zero because of dirichlet
    J.append(current_density(PsiR,PsiI))
    Psi = PsiR + 1j* PsiI
    #make animation of the prob dens of the particle in the whole simulation domain
    prob_dens[:, it]= np.real(np.conjugate(Psi) * Psi) 
    y_expectation= np.real(np.trapz(np.conjugate(Psi)* dy*kvalues * Psi, dx=dy))
    # print((y_expectation-Ly/2)//dy)
    if animation == True:
        Plot_bz= ax.imshow(np.transpose(X[Nx+1:,:]), cmap='RdBu',animated=True)
        Plot_particle= plt.scatter(nx_nano, Ny//4+ y_expectation//dy, color='black')
        Plot_nanosheet= plt.vlines(x = Nx//2, ymin = Ny//4, ymax = 3*Ny//4, colors = 'purple', linewidth= 0.2)
        artists.append([Plot_bz, Plot_particle, Plot_nanosheet, plt.legend(handles=[particle_legend, nanosheet_legend])])


# Create an animation
if animation == True:
    ani = ArtistAnimation(fig, artists, interval=50, blit=True)
    plt.legend(handles=[particle_legend, nanosheet_legend], loc='upper right')
    plt.tight_layout()
    writer = PillowWriter(fps=20)
    ani.save('EM-QM.gif', writer=writer) 
    plt.show()


print("total time: {}".format(Nt*dt))
plt.figure()
plt.imshow(np.transpose(J), cmap= 'RdBu')
plt.title('Current Density')
plt.ylabel('y')
plt.xlabel('t')
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(prob_dens, extent=[0, Nt*dt*peta, Ly/2*giga, 3/2*Ly*giga], aspect='auto', origin='lower', cmap='terrain')
plt.xlabel('time (fs)')
plt.ylabel('Position inside the nanosheet (nm)')
plt.title('Probability density of the particle')
plt.colorbar()
plt.show()