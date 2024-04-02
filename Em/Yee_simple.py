import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
from sources import GaussianSource
# material properties
eps = 1.0
mu = 1.0
sigma = 0.0

# Define the grid
Nx = 100
Ny = 100
Nt = 1000
Ly = 1.0
Lx = 1.0
dx = Lx/Nx
dy = Ly/Ny
# Courant Number and resulting timestep
CL = 0.9
c = 1.0 # Speed of light
dt = CL/(c*np.sqrt(1.0/dx**2 + 1.0/dy**2))
tarray = np.linspace(0, Nt*dt, Nt)
print("Courant Number: {}\nTimestep: {}".format(CL, dt))

# Define the fields
# How do we choose what dimension bigger: pg 14 syllabus
# Ex one bigger in Nx
# Ey one bigger in Ny
# Bz one bigger in Nx and Ny
Ex  = np.zeros((Ny,Nx+1))
Ey  = np.zeros((Ny+1,Nx))
Bz = np.zeros((Ny+1,Nx+1))
print("Shape Ex: {}, Shape Ex[0] {}".format(Ex.shape, Ex[0].shape))
print(Ex)
# Place the source in the middle of the grid
source = GaussianSource(tc=15, sigma=3)
print("Source object: {}".format(source))
# Check of values of source are relevant to the grid by plot and checks
# source.plot(tarray)
source.checks()
sourcevalues = source(tarray)
# Time loop
fig, ax = plt.subplots()
artists = []
for it in range(Nt):
    # How far in simulation are we
    print("Iteration: {}/{}".format(it, Nt))

    # Update the source
    source_t = sourcevalues[it]
    Bz[Ny//2, Nx//2] += source_t
    Bz[1:-1,1:-1] = Bz[1:-1,1:-1] + dt/dy*(Ex[1:,1:-1] - Ex[:-1,1:-1]) - dt/dx*(Ey[1:-1,1:] - Ey[1:-1,:-1])
    Ex = (eps/dt - sigma/2)/(eps/dt + sigma/2)*Ex + 1/(mu*dy*(eps/dt + sigma/2))*(Bz[1:,:]- Bz[:-1,:])
    Ey = (eps/dt - sigma/2)/(eps/dt + sigma/2)*Ey - 1/(mu*dx*(eps/dt + sigma/2))*(Bz[:,1:]- Bz[:,:-1])

    artists.append([plt.imshow(Bz, cmap='RdBu', animated=True)])

# Create an animation
ani = ArtistAnimation(fig, artists, interval=50, blit=True)
plt.show()

    