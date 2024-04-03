import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
from sources import GaussianSource
import math
# material properties
ebs = 8.854 * 10**(-12) # Permittivity of free space F/m
mu = 4 * math.pi * 10**(-7) # Permeability of free space H/m
sigma = 0.0

# Define the grid
Nx = 50
Ny = 50
Nt = 100 
Ly = 1.0
Lx = 1.0
dx = Lx/Nx
dy = Ly/Ny
# Courant Number and resulting timestep
CL = 0.9
c = 3 * 10**8  # Speed of light m/s
dt = CL/(c*np.sqrt(1.0/dx**2))
tarray = np.linspace(0, Nt*dt, Nt)
print("Courant Number: {}\nTimestep: {}".format(CL, dt))

# Define the fields

X = np.zeros((2*Nx+2, Ny))

def matrices_construct():
    Ad = [[0 for _ in range(Nx+1)] for _ in range(Nx)]
    Ai = [[0 for _ in range(Nx+1)] for _ in range(Nx)]
    for i in range(Nx):
        for j in range(Nx+1):
            if i == j:
                Ad[i][j] = -1
                Ai[i][j] = 1
            if i+1 == j:
                Ad[i][j] = 1
                Ai[i][j] = 1
    M1 = np.hstack((1/dx*np.array(Ad), 1/dt*np.array(Ai)))
    M2 = np.hstack(((ebs/dt + sigma/2)*np.array(Ai), 1/(mu*dx)*np.array(Ad)))
    M = np.vstack((M1, M2)) 

    L1 = np.hstack((-1/dx*np.array(Ad), 1/dt*np.array(Ai)))
    L2 = np.hstack(((ebs/dt - sigma/2)*np.array(Ai), -1/(mu*dx)*np.array(Ai)))
    L = np.vstack((L1, L2))

    return np.array(Ad), np.array(Ai),M, L

matrices_construct()
AD, AI, M, L = matrices_construct()

# X contains values of Ey and Bz
X = np.zeros((2*Nx+2, Ny))

print("M shape: {}".format(M.shape))
# print("M: \n{}".format(M))

# Periodic Boundary Conditions 1
BC1 = np.zeros((1, 2*Nx+2))
BC1[0,0] = 1
BC1[0, Nx] = -1
M = np.vstack((M, BC1))
L = np.vstack((L, np.zeros((1,2 *Nx+2))))
#Periodic Boundary Conditions 2
BC2 = np.zeros((1, 2*Nx+2))
BC2[0,Nx+1] = 1
BC2[0, -1] = -1
M = np.vstack((M, BC2))
L = np.vstack((L, np.zeros((1, 2*Nx+2))))
# check determinant of M non-zero
# detM = np.linalg.det(M)
# print("Determinant of M: {}".format(detM))

Minv = np.linalg.inv(M)
MinvL = np.matmul(Minv, L)

#source specs
source = GaussianSource(tc=15, sigma=3)

Ex = np.zeros((Nx+1, Ny+1)) 
# Create fig for animation
fig, ax = plt.subplots()
artists = []
for it in range(Nt):
    t = it*dt
    print("Iteration: {}/{}".format(it, Nt))
    # Update the source
    X = np.matmul(MinvL, X)
    Y = Ex[:-1, 1:] + Ex[1:,1:] - Ex[:-1,:-1] - Ex[1:,:-1]
    print("Y shape: {}".format(Y.shape))
    X[:Nx, :] = X[:Nx, :] + 1/(dy)*Y
    X[Nx+1+Nx//2, Ny//2] += source(t)

    Ex[:,1:-1] = (ebs/dt - sigma/2)/(ebs/dt + sigma/2) * Ex[:,1:-1] + 1/(ebs/dt + sigma/2)*(1/(mu*dy))*(X[Nx+1:, 1:]-X[Nx+1:, :-1])
    Ex[:,0] = Ex[:,1]
    Ex[:,-1] = Ex[:,-2]
    artists.append([plt.imshow(X[Nx+1:], cmap='RdBu', animated=True)])

# Create an animation
ani = ArtistAnimation(fig, artists, interval=200, blit=True)
plt.show()


