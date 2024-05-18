import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
import copy
animation =True

eps = 8.854 * 10**(-12) # Permittivity of free space F/m
mu = 4 * math.pi * 10**(-7) # Permeability of free space H/m

c = 3 * 10**8  # Speed of light m/s
nx_PML = 10
ny_PML = 10
nz_PML = 10
nx =100 
ny =100 
Nx = nx_PML*2 + nx
Ny = ny_PML*2 + ny
Nt = 100
Lx = 0.1
Ly = 0.1
dx = Lx/Nx
dy = Ly/Ny
CFL = 0.9
dt = CFL *dy/c
dtau = dt * c
tarray = np.linspace(0, Nt*dt, Nt)
print("Courant Number: {}\nTimestep: {}".format(CFL, dt))
print("dx: {}\ndy: {}".format(dx, dy))
print("dt: {}".format(dt))

##############
# Z0 = 1
# sig_DC = 1
# sig_DC2 = Z0*sig_DC 
# gamma = 1
# gamma2 = c * gamma
# alpha_p = 2*gamma2/dtau + 1
# alpha_m = alpha_p - 2

m=4 
########## sigma
sig_Mx = (m + 1)/(150 *math.pi * dx)
sig_My = (m + 1)/(150 * math.pi *dy)
Sigma_x = sig_Mx*np.linspace(0,1,nx_PML)**m
Sigma_y = sig_My*np.linspace(0,1,ny_PML)**m
Sx = np.concatenate((Sigma_x[::-1], np.zeros((nx+1)), Sigma_x))
sx = Sx[1:-1].copy()
Sy = np.concatenate((Sigma_y[::-1], np.zeros((ny+1)), Sigma_y))
sy = Sy[1:-1].copy()
############ kappa 
kappa_Mx = 10
kappa_My = 10
Kappa_x = 1 + (kappa_Mx -1)*np.linspace(0,1,nx_PML)**m
Kx = np.concatenate((Kappa_x[::-1], np.ones((nx+1)), Kappa_x))
kx = Kx[1:-1].copy()
kappa_y = 1 + (kappa_My -1)*np.linspace(0,1,ny_PML)**m
Ky = np.concatenate((kappa_y[::-1], np.ones((ny+1)), kappa_y))
ky = Ky[1:-1].copy()
print("kappa_x {}".format(Kappa_x))
print("kx: {}".format(kx))


# Beta_px = Kx/dtau + Z0* Sx/2
# beta_px= kx/dtau + Z0*sx/2
# beta_mx = kx/dtau - Z0*sx/2
# Beta_mx = Kx/dtau - Z0*Sx/2
# beta_py = ky/dtau + Z0* sy/2
# Beta_py = Ky/dtau + Z0* Sy/2
# beta_my= ky/dtau - Z0*sy/2
# Beta_my = Ky/dtau-Z0*Sy/2
# print("beta_px: {} beta_mx= {} beta_py: {} beta_my {}".format(Beta_px.shape, beta_px.shape, beta_py.shape, beta_my.shape))
Beta_px =1/dtau
beta_px=1 /dtau
beta_mx = 1/dtau
Beta_mx =1/dtau
beta_py =1/dtau
Beta_py= 1/dtau
beta_my=1/dtau
Beta_my =1/dtau 
A1 = np.zeros((Nx, Nx-1))
A2 = np.zeros((Nx, Nx+1))
for ix in range(Nx):
    for iy in range(Nx-1):
        if ix == iy:
            A1[ix, iy] = 1
        if ix == iy+1:
            A1[ix, iy] = 1
for ix in range(Nx):
    for iy in range(Nx+1):
        if ix == iy:
            A2[ix, iy] = 1
        if ix == iy-1:
            A2[ix,iy]=1

print("A1 shape: {}".format(A1.shape))
# print("A1: \n{}".format(A1))
print("A2 shape {}".format(A2.shape))
# print("A2: \n{}".format(A2))
D1 = np.zeros((Nx, Nx-1))
D2 = np.zeros((Nx, Nx+1))
for ix in range(Nx):
    for iy in range(Nx-1):
        if ix == iy:
            D1[ix, iy] = 1/dx
        if ix == iy+1:
            D1[ix, iy] = -1/dx
for ix in range(Nx):
    for iy in range(Nx+1):
        if ix == iy:
            D2[ix, iy] = -1/dx
        if ix == iy-1:
            D2[ix,iy]=1/dx


print("D1 shape: {}".format(D1.shape))
# print("D1: \n{}".format(D1))
print("D2 shape {}".format(D2.shape))
# print("D2: \n{}".format(D2))
s_0= np.zeros((Nx,Nx-1))
b_0 = np.zeros((Nx, Nx+1))

######## M ################
r_1 = np.hstack((A1/dtau ,b_0, s_0, D2, s_0))
r_2 = np.hstack((s_0, Beta_px*A2, s_0,b_0, D1))
r_3 = np.hstack((-np.eye(Nx-1)/dtau, np.zeros((Nx-1, Nx+1)), 1/dtau*np.eye(Nx-1) , np.zeros((Nx-1,Nx+1)),np.zeros((Nx-1,Nx-1))))
r_4 = np.hstack((np.zeros((Nx+1, Nx-1)), -1/dtau*np.eye(Nx+1), np.zeros((Nx+1,Nx-1)), Beta_py*np.eye(Nx+1),np.zeros((Nx+1, Nx-1))))
r_5 = np.hstack((np.zeros((Nx-1, Nx-1)),np.zeros((Nx-1, Nx+1)),-beta_py*np.eye(Nx-1),np.zeros((Nx-1,Nx+1)),beta_px*np.eye(Nx-1)))
print("M r_1 shape: {}\nr_2 shape: {}\nr_3 shape: {}\nr_4 shape: {}\nr_5 shape: {}\n".format(r_1.shape,r_2.shape,r_3.shape,r_4.shape,r_5.shape ))

M = np.vstack((r_1,r_2,r_3,r_4,r_5))
print("M shape: {}".format(M.shape))

print(np.linalg.matrix_rank(M))
M_inv = np.linalg.inv(M)
########## L ################
r_1 = np.hstack((A1/dtau ,b_0, s_0, -D2, s_0))
r_2 = np.hstack((s_0, Beta_mx*A2, s_0,b_0,-D1))
r_3 = np.hstack((-np.eye(Nx-1)/dtau, np.zeros((Nx-1, Nx+1)),  1/dtau*np.eye(Nx-1) , np.zeros((Nx-1,Nx+1)),np.zeros((Nx-1,Nx-1))))
r_4 = np.hstack((np.zeros((Nx+1, Nx-1)), -1/dtau*np.eye(Nx+1), np.zeros((Nx+1,Nx-1)), Beta_my*np.eye(Nx+1),np.zeros((Nx+1, Nx-1))))
r_5 = np.hstack((np.zeros((Nx-1, Nx-1)),np.zeros((Nx-1, Nx+1)),-beta_my*np.eye(Nx-1),np.zeros((Nx-1,Nx+1)),beta_mx*np.eye(Nx-1)))

print("L r_1 shape: {}\nr_2 shape: {}\nr_3 shape: {}\nr_4 shape: {}\nr_5 shape: {}\n".format(r_1.shape,r_2.shape,r_3.shape,r_4.shape,r_5.shape ))

L = np.vstack((r_1,r_2, r_3, r_4, r_5))
print("L shape: {}".format(L.shape))

############# Construct fields ###########
X = np.zeros((3*(Nx-1)+2*(Nx+1), Ny)) #ey hz_ ey_ hz ey#
Y = np.zeros((3*(Nx-1)+2*(Nx+1), Ny)) # + matrix in for loop
print("X shape: {}".format(X.shape))
ex__ = np.zeros((Nx+1,Ny+1))
ex_ = np.zeros((Nx+1,Ny+1))
ex = np.zeros((Nx+1,Ny+1))

######### Source #############
J0 = 1
tc = Nt/5*dt
sig = tc/6 #see project why
omega = 1e12 # rad/s



def source(t):
    return J0*np.sin(omega*t)*np.exp(-(t-tc)**2/(2*sig**2))
if animation == True:
    fig, ax = plt.subplots()
    artists = []
    plt.title("Bz")
    plt.xlabel("x")
    plt.ylabel("y")


for it in range(Nt):
    t = it*dt
    print("Iteration: {}/{}".format(it, Nt))

    ######## Explicit equation #############
    #ex__old = ex__.copy()
    ex__old = copy.deepcopy(ex__)
    #ex_old = ex_.copy()
    ex_old = copy.deepcopy(ex_)
    ex__[:,1:-1] = ex__[:,1:-1] + dtau/dy*(X[3*Nx-1:4*Nx,1:]- X[3*Nx-1:4*Nx,:-1])

    ex_ = (ex_ + (ex__-ex__old))
    #ex =1/np.reshape(Beta_pz,(Ny+1, 1))*(np.reshape(Beta_mz,(Ny+1,1)) * ex + Beta_px* ex_ - Beta_mx*ex_old)
    ex = ex + (ex_ - ex_old)


    ########### Implicit equations ############
    Y[Nx:2*Nx,:]= np.matmul(A2/dy,ex[:,1:]-ex[:,:-1])
    X = np.matmul(M_inv, np.matmul(L,X)+Y)
    X[3*Nx-1+Nx//2, Ny//2] += source(t)# source is added to hz
    if animation == True:
        artists.append([plt.imshow(np.transpose(10*X[3*Nx-1:4*Nx]), cmap='viridis',vmin=-0.02*J0,vmax=0.02*J0,animated=True),
                    
                        ])

# Create an animation
if animation == True:
    ani = ArtistAnimation(fig, artists, interval=50, blit=True)
    plt.show()
print("total time: {}".format(Nt*dt))
print("X[Nx+1:] is Bz, shape: {}".format(X[Nx+1:].shape))





    