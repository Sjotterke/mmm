import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
import copy
from scipy.special import hankel2
animation =True
plot =True 
omega_L = [432466666666.66656, 2e11, 3e11]

eps = 8.854 * 10**(-12) # Permittivity of free space F/m
mu = 4 * math.pi * 10**(-7) # Permeability of free space H/m

c = 3 * 10**8  # Speed of light m/s
nx_PML =30 
ny_PML =30 
nx =200
ny =200 
Nx = nx_PML*2 + nx
Ny = ny_PML*2 + ny
Nt = 1000
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
print("dt: {} t: {}".format(dt, Nt*dt))

m=4 
########## sigma
sig_Mx = (m + 1)/(150 *math.pi * dx)
sig_My = (m + 1)/(150 * math.pi *dy)
Sigma_x = sig_Mx*np.linspace(0,1,nx_PML)**m
Sigma_y = sig_My*np.linspace(0,1,ny_PML)**m
Sx = np.concatenate((Sigma_x[::-1], np.zeros((nx+1)), Sigma_x))
Sy = np.concatenate((Sigma_y[::-1], np.zeros((ny+1)), Sigma_y))
############ kappa 
kappa_Mx = 8 
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
Bpy = np.diag(Beta_py)
bpy = np.diag(beta_py)
beta_my= Ky[1:-1]/dtau - Z0*Sy[1:-1]
Beta_my = Ky/dtau-Z0*Sy/2
Bmy = np.diag(Beta_my)
bmy = np.diag(beta_my)
#print("BETA PX {}\n ZO*SY/2= {}".format(Beta_px, Z0*Sy/2))
# print("beta_px: {} beta_mx= {} beta_py: {} beta_my {}".format(Beta_px.shape, beta_px.shape, beta_py.shape, beta_my.shape))

######## PML out
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
r_1 = np.hstack((A1/dtau ,                np.zeros((Nx, Nx+1)),          np.zeros((Nx, Nx-1)),            D2,                     np.zeros((Nx, Nx-1))))
r_2 = np.hstack((np.zeros((Nx, Nx-1)),    A2@Bpx,                        np.zeros((Nx, Nx-1)) ,           np.zeros((Nx, Nx+1)),   D1))
r_3 = np.hstack((-np.eye(Nx-1)/dtau,      np.zeros((Nx-1, Nx+1)),        1/dtau*np.eye(Nx-1) ,            np.zeros((Nx-1,Nx+1)),  np.zeros((Nx-1,Nx-1))))
r_4 = np.hstack((np.zeros((Nx+1, Nx-1)),  -1/dtau*np.eye(Nx+1),          np.zeros((Nx+1,Nx-1)),           Bpy,                    np.zeros((Nx+1, Nx-1))))
r_5 = np.hstack((np.zeros((Nx-1, Nx-1)),  np.zeros((Nx-1, Nx+1)),        -bpy,                            np.zeros((Nx-1,Nx+1)),  bpx))
print("M r_1 shape: {}\nr_2 shape: {}\nr_3 shape: {}\nr_4 shape: {}\nr_5 shape: {}\n".format(r_1.shape,r_2.shape,r_3.shape,r_4.shape,r_5.shape ))

M = np.vstack((r_1,r_2,r_3,r_4,r_5))
print("1/dtau = {}\nM shape: {}".format(1/dtau,M.shape))

print(np.linalg.matrix_rank(M))
M_inv = np.linalg.inv(M)
########## L ################
r_1 = np.hstack((A1/dtau ,               np.zeros((Nx, Nx+1)),          np.zeros((Nx, Nx-1)),           -D2,                   np.zeros((Nx, Nx-1))))
r_2 = np.hstack((np.zeros((Nx,Nx-1)),    A2@Bmx,                        np.zeros((Nx,Nx-1)),            np.zeros((Nx,Nx+1)),   -D1))
r_3 = np.hstack((-np.eye(Nx-1)/dtau,     np.zeros((Nx-1, Nx+1)),        1/dtau*np.eye(Nx-1) ,           np.zeros((Nx-1,Nx+1)), np.zeros((Nx-1,Nx-1))))
r_4 = np.hstack((np.zeros((Nx+1, Nx-1)), -1/dtau*np.eye(Nx+1),          np.zeros((Nx+1,Nx-1)),          Bmy,                   np.zeros((Nx+1, Nx-1))))
r_5 = np.hstack((np.zeros((Nx-1, Nx-1)), np.zeros((Nx-1, Nx+1)),        -bmy,                           np.zeros((Nx-1,Nx+1)), bmx))

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
omega = omega_L[0] # rad/s



def source(t):
    return J0*np.sin(omega*t) *np.exp(-(t-tc)**2/(2*sig**2))
if animation == True:
    fig, ax = plt.subplots()
    artists = []
    plt.title("Bz")
    plt.xlabel("x")
    plt.ylabel("y")

########## Trackers
Bzt1=np.zeros(Nt)
Bzt2 = np.zeros(Nt)
Bzt3 = np.zeros(Nt)
it1 = (3*Nx//4, Ny//4)
it2 = (3*Nx//4, Ny//2)
it3 = (3*Nx//4, 3*Ny//4)
itlist = [it1, it2, it3]

###### Reshape for explicit update
Beta_pxR = np.reshape(Beta_px, (Nx+1,1))
Beta_mxR = np.reshape(Beta_mx, (Nx+1,1))
for it in range(Nt):
    t = it*dt
    print("Iteration: {}/{}".format(it, Nt))

    ######## Explicit equation #############
    ex__old = ex__.copy()
    ex_old = ex_.copy()
    
    ##### no PML
    # ex__[:,1:-1] = ex__[:,1:-1] + dtau/dy*(X[3*Nx-1:4*Nx,1:]- X[3*Nx-1:4*Nx,:-1])
    #ex_ =  ex_ + (ex__-ex__old)
    #ex = ex +  ex_- ex_old)

    ######## PML
    ex__[:,1:-1] = ex__[:,1:-1] + dtau/dy*(X[3*Nx-1:4*Nx,1:]- X[3*Nx-1:4*Nx,:-1])
    ex_ = 1/Beta_py*(Beta_my*ex_ + 1/dtau*(ex__-ex__old))
    ex = ex + (Beta_pxR*ex_ - Beta_mxR*ex_old)*dtau

    ########### Implicit equations ############
    Y[Nx:2*Nx,:]= np.matmul(A2/dy,ex[:,1:]-ex[:,:-1])
    X = np.matmul(M_inv, np.matmul(L,X)+Y)
    X[3*Nx-1+Nx//2, Ny//2] += source(t)# source is added to hz

    ########### Update Trackers
    if it>Nt//2:
        Bzt1[it] = X[3*Nx-1+it1[0], it1[1]]
        Bzt2[it] = X[3*Nx-1+it2[0], it2[1]]
        Bzt3[it] = X[3*Nx-1+it3[0], it3[1]]
    if animation == True:
        artists.append([plt.imshow(np.transpose(X[3*Nx-1:4*Nx]), cmap='viridis',vmin=-0.02*J0,vmax=0.02*J0,animated=True),
                    ])

# Create an animation
if animation == True:
    ani = ArtistAnimation(fig, artists, interval=50, blit=True)
    plt.show()
print("total time: {}".format(Nt*dt))
print("X[Nx+1:] is Bz, shape: {}".format(X[Nx+1:].shape))

if plot == True:
    fig, axs = plt.subplots(2, 2)

    axs[0,0].set_title("Trackers")
    axs[0,0].plot(tarray, Bzt1, label="t1")
    axs[0,0].plot(tarray, Bzt2, label ="t2")
    axs[0,0].plot(tarray, Bzt3, label="t3")
    axs[0,0].legend()

    axs[0,1].set_title("source")
    axs[0,1].plot(tarray, source(tarray))
    axs[0,1].set_xlabel("t [s]")
    axs[0,1].set_ylabel("Bz [T]")

    ############# FT Source 
    source_fft = np.fft.fft(source(tarray))
    source_freq = np.fft.fftfreq(tarray.shape[-1], dt)
   

    ana_fft = []
    for tracker in itlist:
        x = abs(Nx//2-tracker[0]) ### maal dx?
        y = abs(Ny//2 - tracker[1])
        ana_fft.append(J0*source_freq*mu/4*hankel2(0, source_freq/c * np.sqrt(x**2 + y**2)))

    axs[1,0].set_title("source freq transform")
    axs[1,0].plot(source_freq, np.abs(source_fft), label="omega: {:2e}".format(omega))
    axs[1,0].set_xlabel("omega [rad/s]")
    axs[1,0].set_ylabel("Bz [T]")
    axs[1,0].legend()

    axs[1,1].set_title("analytisch tracker")
    axs[1,1].plot(source_freq, np.abs(ana_fft[0]), label="t1")
    axs[1,1].plot(source_freq, np.abs(ana_fft[1]), label="t2")
    axs[1,1].plot(source_freq, np.abs(ana_fft[2]), label="t3")
    axs[1,1].legend()
    plt.show()
    ########## FT Trackers 
    Bzt1_fft = np.fft.fft(Bzt1)
    Bzt2_fft = np.fft.fft(Bzt2)
    Bzt3_fft = np.fft.fft(Bzt3)
    ################  
    idxm = np.abs(source_freq).argmin()
    freq1 = np.abs(source_freq-omega_L[0]).argmin()
    freq2 = np.abs(source_freq-omega_L[1]).argmin()
    freq3 = np.abs(source_freq-omega_L[2]).argmin()
    #print("minimum freq: {} Maximum freq: {}".format(np.min(np.abs(source_freq)),np.max(source_freq)))
    print("frequentie 1 index: {} Value: {} Wanted: {} Difference: {}".format(freq1, source_freq[freq1], omega_L[0], source_freq[freq1]-omega_L[0]))
    print("source_freq_shape: {}".format(source_freq.shape))
    def ana_sol(xt, yt, f):
        x = abs(xt-Nx//2)*dx ### maal dx?
        print(x)
        y = abs(yt-Ny//2)*dy
        return -J0*f*mu/4*hankel2(0, f/c * np.sqrt(x**2 + y**2))
    fig, (ax1, ax2) = plt.subplots(2,1)
    ax1.set_title("FFT")
    ax1.scatter(freq1, Bzt1_fft[freq1], color="blue", label="Tracker 1")
    ax1.scatter(freq1, source_fft[freq1], color="orange", label="Source")
    ax1.scatter(freq1, np.abs(ana_sol(it1[0],it1[1] ,freq1)), color="red", label="ana")
    ax1.legend()
    ax1.set_title("FFT")



    ax2.scatter(freq1, Bzt1_fft[freq1]/source_fft[freq1], color="blue", label="Tracker 1")
    # ax1.scatter(freq1, source_fft[freq1], color="orange", label="Source")
    ax2.scatter(freq1, np.abs(ana_sol(it1[0],it1[1] ,freq1)), color="red", label="ana")
    print(np.abs(ana_sol(it1[0],it1[1] ,freq1)))
    ax2.legend()
    plt.show()



