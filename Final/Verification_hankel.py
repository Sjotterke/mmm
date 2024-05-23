import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
import copy
from scipy.special import hankel2

animation = True # Show animation
plot =True # Show plot

eps = 8.854 * 10**(-12) # Permittivity of free space F/m
mu = 4 * math.pi * 10**(-7) # Permeability of free space H/m
c = 3 * 10**8  # Speed of light m/s
nx_PML =10
ny_PML =10 
nx =200
ny =200
Nx = nx_PML*2 + nx
Ny = ny_PML*2 + ny
dx = 0.25* 1e-10 # m
dy = 0.25* 1e-10# m
Nt = 200
Lx = dx * Nx 
Ly = dy * Ny
CFL = 0.9 
dt = CFL *dy/c
dtau = dt * c
tarray = np.linspace(0, Nt*dt, Nt)
print("Courant Number: {}\nTimestep: {}".format(CFL, dt))
print("dx: {}\ndy: {}".format(dx, dy))
print("dt: {} t: {}".format(dt, dt*Nt))

m=4 
########## sigma
sig_Mx = (m + 1)/(150 *math.pi * dx)
sig_My = (m + 1)/(150 * math.pi *dy)
Sigma_x = sig_Mx*np.linspace(0,1,nx_PML)**m
Sigma_y = sig_My*np.linspace(0,1,ny_PML)**m
Sx = np.concatenate((Sigma_x[::-1], np.zeros((nx+1)), Sigma_x))
Sy = np.concatenate((Sigma_y[::-1], np.zeros((ny+1)), Sigma_y))
############ kappa 
kappa_Mx = 10
kappa_My = kappa_Mx 
Kappa_x = 1 + (kappa_Mx -1)*np.linspace(0,1,nx_PML)**m
Kx = np.concatenate((Kappa_x[::-1], np.ones((nx+1)), Kappa_x))
kappa_y = 1 + (kappa_My -1)*np.linspace(0,1,ny_PML)**m
Ky = np.concatenate((kappa_y[::-1], np.ones((ny+1)), kappa_y))
Z0 = np.sqrt(mu/eps)
######## Beta 
###### x
Beta_px = Kx/dtau + Z0* Sx/2
beta_px= Kx[1:-1]/dtau + Z0*Sx[1:-1]
Bpx = np.diag(Beta_px)
bpx = np.diag(beta_px)
beta_mx = Kx[1:-1]/dtau - Z0*Sx[1:-1]
Beta_mx = Kx/dtau - Z0*Sx/2
Bmx = np.diag(Beta_mx)
bmx = np.diag(beta_mx)
####### y
beta_py = Ky[1:-1]/dtau + Z0*Sy[1:-1] 
Beta_py = Ky/dtau + Z0* Sy/2
beta_my= Ky[1:-1]/dtau - Z0*Sy[1:-1]
Beta_my = Ky/dtau-Z0*Sy/2
######## Interpolators 
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
######## Differntiators
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

######### Inverse Of M and construct L for every step in y direction and once for bulk
def M_inv_L_j(j, bulk):
    #### Check if Bulk M and L is already calculated and in case return it
    print("M_inv, L step: {} of: {}".format(j, Ny-1))
    if j>ny_PML+1 and j<Ny-ny_PML:
        return bulk[0], bulk[1], bulk
    Ssy = np.concatenate((Sigma_y[::-1], np.zeros((ny)), Sigma_y))
    Kky = np.concatenate((kappa_y[::-1], np.ones((ny)), kappa_y))
    p = Kky/dtau + Z0*Ssy/2
    m =  Kky/dtau - Z0*Ssy/2
    Bpy = np.diag(p[j]*np.ones(Ny+1))
    bpy = np.diag(p[j]*np.ones(Ny-1))
    Bmy = np.diag(m[j]*np.ones(Ny+1))
    bmy = np.diag(m[j]*np.ones(Ny-1))

    ######## M ################
    r_1 = np.hstack((A1/dtau ,                np.zeros((Nx, Nx+1)),          np.zeros((Nx, Nx-1)),            D2,                     np.zeros((Nx, Nx-1))))
    r_2 = np.hstack((np.zeros((Nx, Nx-1)),    A2@Bpx,                        np.zeros((Nx, Nx-1)) ,           np.zeros((Nx, Nx+1)),   D1))
    r_3 = np.hstack((-np.eye(Nx-1)/dtau,      np.zeros((Nx-1, Nx+1)),        1/dtau*np.eye(Nx-1) ,            np.zeros((Nx-1,Nx+1)),  np.zeros((Nx-1,Nx-1))))
    r_4 = np.hstack((np.zeros((Nx+1, Nx-1)),  -1/dtau*np.eye(Nx+1),          np.zeros((Nx+1,Nx-1)),           Bpy,                    np.zeros((Nx+1, Nx-1))))
    r_5 = np.hstack((np.zeros((Nx-1, Nx-1)),  np.zeros((Nx-1, Nx+1)),        -bpy,                            np.zeros((Nx-1,Nx+1)),  bpx))

    M = np.vstack((r_1,r_2,r_3,r_4,r_5))
    
    M_inv = np.linalg.inv(M)
    ########## L ################
    r_1 = np.hstack((A1/dtau ,               np.zeros((Nx, Nx+1)),          np.zeros((Nx, Nx-1)),           -D2,                   np.zeros((Nx, Nx-1))))
    r_2 = np.hstack((np.zeros((Nx,Nx-1)),    A2@Bmx,                        np.zeros((Nx,Nx-1)),            np.zeros((Nx,Nx+1)),   -D1))
    r_3 = np.hstack((-np.eye(Nx-1)/dtau,     np.zeros((Nx-1, Nx+1)),        1/dtau*np.eye(Nx-1) ,           np.zeros((Nx-1,Nx+1)), np.zeros((Nx-1,Nx-1))))
    r_4 = np.hstack((np.zeros((Nx+1, Nx-1)), -1/dtau*np.eye(Nx+1),          np.zeros((Nx+1,Nx-1)),          Bmy,                   np.zeros((Nx+1, Nx-1))))
    r_5 = np.hstack((np.zeros((Nx-1, Nx-1)), np.zeros((Nx-1, Nx+1)),        -bmy,                           np.zeros((Nx-1,Nx+1)), bmx))

    L = np.vstack((r_1,r_2, r_3, r_4, r_5))
    if j>ny_PML:
        bulk = [M_inv, L]
    return M_inv, L, bulk
M_invL = []
L_L = []
bulk = np.zeros(5)
for j in range(Ny):
    my_m, my_l, bulk = M_inv_L_j(j, bulk)
    M_invL.append(my_m)
    L_L.append(my_l)
    


############# Construct fields ###########
X = np.zeros((3*(Nx-1)+2*(Nx+1), Ny)) #ey hz_ ey_ hz ey#
Y = np.zeros((3*(Nx-1)+2*(Nx+1), Ny)) # + matrix in for loop
ex__ = np.zeros((Nx+1,Ny+1))
ex_ = np.zeros((Nx+1,Ny+1))
ex = np.zeros((Nx+1,Ny+1))

######### Source #############
J0 = 1
tc = Nt/5*dt
sig = tc/6 #see project why
def source(t):
    # return J0 * np.sin(omega*t)
    return J0*np.exp(-(t-tc)**2/(2*sig**2))

if animation == True:
    fig, ax = plt.subplots()
    artists = []
    plt.title("Bz")
    plt.xlabel("x")
    plt.ylabel("y")

########## Trackers  ########
Bzt1=np.zeros(Nt)
Bzt2 = np.zeros(Nt)
it1 = (Nx//2+Nx//10, Ny//2 + Ny//10)
it2 = (Nx//2+Nx//10, Ny//2)
itlist = [it1, it2]

###### Reshape for explicit update equations #######
Beta_pxR = np.reshape(Beta_px, (Nx+1,1))
Beta_mxR = np.reshape(Beta_mx, (Nx+1,1))

for it in range(Nt):
    t = it*dt
    print("Iteration: {}/{}".format(it, Nt))

    ######## Explicit equation #############
    ex__old = ex__.copy()
    ex_old = ex_.copy()
    ex__[:,1:-1] = ex__[:,1:-1] + dtau/dy*(X[3*Nx-1:4*Nx,1:]- X[3*Nx-1:4*Nx,:-1])
    ex_ = 1/Beta_py*(Beta_my*ex_ + 1/dtau*(ex__-ex__old))
    ex = ex + (Beta_pxR*ex_ - Beta_mxR*ex_old)*dtau
  
    ########### Implicit equations ############
    Y[Nx:2*Nx,:]= np.matmul(A2/dy,ex[:,1:]-ex[:,:-1])
    for j in range(Ny):
        X[:,j] = np.matmul(M_invL[j], np.matmul(L_L[j],X[:,j])+Y[:,j])
    X[3*Nx-1+Nx//2, Ny//2] += source(t)# source is added to hz

    ########### Update Trackers
    Bzt1[it] = X[3*Nx-1+it1[0], it1[1]]
    Bzt2[it] = X[3*Nx-1+it2[0], it2[1]]
    if animation == True:
        artists.append([plt.imshow(np.transpose(X[3*Nx-1:4*Nx]), cmap='viridis',vmin=-0.01*J0,vmax=0.01*J0,animated=True),
        plt.scatter(it1[0], it1[1], color = "red"),
        plt.scatter(it2[0], it2[1], color = "red")
                    ])

# Create an animation
if animation == True:
    ani = ArtistAnimation(fig, artists, interval=50, blit=True)
    plt.show()
if plot == True:
    fig, axs = plt.subplots(2, 1)

    axs[0].set_title("Trackers")
    axs[0].plot(tarray, Bzt1, label="t1")
    axs[0].plot(tarray, Bzt2, label ="t2")
    axs[0].legend()

    axs[1].set_title("Source")
    axs[1].plot(tarray, source(tarray))
    axs[1].set_xlabel("t [s]")
    axs[1].set_ylabel("j [T/s]")
    plt.show()

    ## do fourier transform of source and trackers
    n=100
    tarray = np.concatenate((tarray, np.zeros(n)))
    Bzt1 = np.concatenate((Bzt1, np.zeros(n)))
    Bzt2 = np.concatenate((Bzt2, np.zeros(n)))
    source_freq = np.fft.fftfreq(tarray.shape[-1], dt)
    source_fft = np.fft.fft(source(tarray))
    Bzt1_fft = np.fft.fft(Bzt1)
    Bzt2_fft = np.fft.fft(Bzt2)
    ana_fft = []
    for tracker in itlist:
        x = abs(Nx//2-tracker[0])*dx
        y = abs(Ny//2 - tracker[1])*dy
        ana_fft.append(J0*eps*source_freq*2*math.pi/4*hankel2(0, 2*math.pi*source_freq/(c)* np.sqrt(x**2 + y**2)))

    start=0
    end = len(source_freq)//2 # Frequencies > 0
    fig, axs = plt.subplots(2, 2)

    axs[0,0].set_title("Trackers FT")
    axs[0,0].scatter(source_freq[start: end], np.abs(ana_fft[0])[start:end], label="t1")
    axs[0,0].scatter(source_freq[start: end], np.abs(ana_fft[1])[start:end], label="t2")
    axs[0,0].set_xlabel("\u03C9 [rad/s]")
    axs[0,0].legend()

    def S_ana(rij):
        return abs(J0)*np.sqrt(2*math.pi)*sig*np.exp(-(sig*2*math.pi*rij)**2/2)
    axs[0,1].set_title("Source FT (analytical)")
    axs[0,1].scatter(source_freq[start: end], S_ana(source_freq[start:end]))
    
    axs[1,0].set_title("Tracker FT")
    axs[1,0].scatter(source_freq[start:end], np.abs(Bzt1_fft)[start:end], label="t1")
    axs[1,0].scatter(source_freq[start:end],  np.abs(Bzt2_fft)[start:end], label="t2")
    axs[1,0].set_xlabel("\u03C9 [rad/s]")
    axs[1,0].legend()

    start = 5
    end = 15
    axs[1,1].set_title("Verification")
    axs[1,1].scatter(source_freq[start:end],(dt/((Nx+Ny)*(dx+dy)))*np.abs(Bzt1_fft)[start:end]/S_ana(source_freq[start:end]),label="num 1")
    axs[1,1].plot(source_freq[start:end], np.abs(ana_fft[0])[start:end], label="analytisch 1")
    axs[1,1].scatter(source_freq[start:end],(dt/((Nx+Ny)*(dx+dy)))*np.abs(Bzt2_fft)[start:end]/(np.abs(S_ana(source_freq[start:end]))), label="num 2")
    axs[1,1].plot(source_freq[start:end], np.abs(ana_fft[1])[start:end], label="anlytisch 2")
    axs[1,0].set_xlabel("\u03C9 [rad/s]")
    axs[1,1].legend()

    plt.show()
  