import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
import copy
from scipy.special import hankel2

animation = True
plot =True

eps = 8.854 * 10**(-12) # Permittivity of free space F/m
mu = 4 * math.pi * 10**(-7) # Permeability of free space H/m

c = 3 * 10**8  # Speed of light m/s
nx_PML =10
ny_PML =10 
nx =200
ny =200
Nx = nx_PML*2 + nx
Ny = ny_PML*2 + ny

dx = 1 # m
dy = 1# m
Nt = 400
Lx = dx * Nx 
Ly = dy * Ny
CFL = 1
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
Bpy = np.diag(Beta_py)
bpy = np.diag(beta_py)
beta_my= Ky[1:-1]/dtau - Z0*Sy[1:-1]
Beta_my = Ky/dtau-Z0*Sy/2
Bmy = np.diag(Beta_my)
bmy = np.diag(beta_my)

#################################### Testing
def rev_unit(n):
    a = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i==n-1-j:
                a[i,j]= 1
    return a
# sec_diag = rev_unit(Nx-1)
# Sec_diag = rev_unit(Nx+1)
# bmy = beta_my*sec_diag
# Bmy = Beta_my*Sec_diag
# print(bmy)
# bpy = beta_my*sec_diag
# Bpy = Beta_py*Sec_diag
# print(Bpy)

#print("BETA PX {}\n ZO*SY/2= {}".format(Beta_px, Z0*Sy/2))
# print("beta_px: {} beta_mx= {} beta_py: {} beta_my {}".format(Beta_px.shape, beta_px.shape, beta_py.shape, beta_my.shape))

######## PML out
# Beta_px =1/dtau
# beta_px=1 /dtau
# beta_mx = 1/dtau
# Beta_mx =1/dtau
# beta_py =1/dtau
# Beta_py= 1/dtau
# beta_my=1/dtau
# Beta_my =1/dtau 
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
def M_inv_L_j(j, bulk):
    print("j in function: {}".format(j))
    #### Check if Bulk M and L is already calculated and in case return it
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
    print("1/dtau = {}\nM shape: {}".format(1/dtau,M.shape))
    
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
print("X shape: {}".format(X.shape))
ex__ = np.zeros((Nx+1,Ny+1))
ex_ = np.zeros((Nx+1,Ny+1))
ex = np.zeros((Nx+1,Ny+1))

######### Source #############
J0 = 1
tc = Nt/5*dt
sig = tc/6 #see project why
omega = 1e18 # rad/s



def source(t):
    return J0*np.exp(-(t-tc)**2/(2*sig**2))
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
it1 = (Nx//2+Nx//10, Ny//2 + Ny//10)
it2 = (Nx//2+Nx//10, Ny//2)
it3 = (Nx//2+Nx//10, Ny//2-Ny//10)
itlist = [it1, it2, it3]

###### Reshape for explicit update
Beta_pxR = np.reshape(Beta_px, (Nx+1,1))
print(Beta_pxR)
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
    for j in range(Ny):
        X[:,j] = np.matmul(M_invL[j], np.matmul(L_L[j],X[:,j])+Y[:,j])
    X[3*Nx-1+Nx//2, Ny//2] += source(t)# source is added to hz

    ########### Update Trackers
    if it>Nt//2:
        Bzt1[it] = X[3*Nx-1+it1[0], it1[1]]
        Bzt2[it] = X[3*Nx-1+it2[0], it2[1]]
        Bzt3[it] = X[3*Nx-1+it3[0], it3[1]]
    if animation == True:
        artists.append([plt.imshow(np.transpose(10*X[3*Nx-1:4*Nx]), cmap='viridis',vmin=-0.01*J0,vmax=0.01*J0,animated=True),
        plt.scatter(it1[0], it1[1], color = "red"),
        plt.scatter(it2[0], it2[1], color = "red")
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

    ## do fourier transform of source and trackers
    np.concatenate((tarray, np.zeros(100)))
    np.concatenate((Bzt1, np.zeros(100)))
    np.concatenate((Bzt2, np.zeros(100)))
    np.concatenate((Bzt3, np.zeros(100)))
    source_fft = np.fft.fft(source(tarray))
    source_freq = np.fft.fftfreq(tarray.shape[-1], dt)
    Bzt1_fft = np.fft.fft(Bzt1)
    Bzt2_fft = np.fft.fft(Bzt2)
    Bzt3_fft = np.fft.fft(Bzt3)
    # index closest to omega and -omega plotting
    idxp = np.abs(source_freq).argmin()
    print(idxp)
    idxm = np.abs(source_freq).argmin()
    window = 30
    print("source_freq_shape: {}".format(source_freq.shape))
    startp = max(0, idxp-window)
    startm = max(0, idxm-window)
    endp = min(source_freq.shape[-1], idxp+window)
    endm = min(source_freq.shape[-1], idxm+window)
    print(startp, endp, startm, endm)
    print("source_freq window: {}".format(source_freq[startp: endp]))

    ana_fft = []
    for tracker in itlist:
        x = abs(Nx//2-tracker[0])*dx
        y = abs(Ny//2 - tracker[1])*dy
        ana_fft.append(J0*mu/4*hankel2(0, source_freq/c * np.sqrt(x**2 + y**2)))

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


    # axs[0,1].set_title("tracker fft")
    # axs[0,1].plot(source_freq, np.abs(Bzt1_fft), label="t1")
    # axs[0,1].plot(source_freq, np.abs(Bzt2_fft), label="t2")
    # axs[0,1].plot(source_freq, np.abs(Bzt3_fft), label="t3")
    # axs[0,1].legend()

    # Define the window around zero frequency to exclude
    window = 5* 1e10


    # Find the indices of frequencies within the window around zero

    exclude_indices = np.where(source_freq < window)
    filter_freq = np.delete(source_freq, exclude_indices)
    filter_source = np.delete(source_fft, exclude_indices)
    filter_ana_fft_1 = np.delete(ana_fft[0], exclude_indices)
    filter_ana_fft_2 = np.delete(ana_fft[1], exclude_indices)
    filter_ana_fft_3 = np.delete(ana_fft[2], exclude_indices)
    filter_Bzt1_fft = np.delete(Bzt1_fft, exclude_indices)
    filter_Bzt2_fft = np.delete(Bzt2_fft, exclude_indices)
    filter_Bzt3_fft = np.delete(Bzt3_fft, exclude_indices)

    idm = np.argmax(np.abs(filter_source))
    id_window = 5
    start = max(0, idm-id_window)
    end = min(filter_source.shape[-1], idm+id_window)

    fig, axs = plt.subplots(2, 2)

    axs[0,0].set_title("analytisch tracker filter")
    axs[0,0].scatter(filter_freq[start: end], np.abs(filter_ana_fft_1)[start:end], label="t1")
    axs[0,0].scatter(filter_freq[start: end], np.abs(filter_ana_fft_2)[start:end], label="t2")
    axs[0,0].scatter(filter_freq[start: end], np.abs(filter_ana_fft_3)[start:end], label="t3")
    axs[0,0].legend()

    axs[0,1].set_title("source fft filter")
    axs[0,1].scatter(filter_freq[start:end], np.abs(filter_source)[start:end])

    axs[1,0].set_title("tracker fft filter")
    print("filter_freq shape: {}\n filter_Bzt1_fft: {}".format(filter_freq.shape, filter_Bzt1_fft.shape))
    axs[1,0].scatter(filter_freq[start:end], np.abs(filter_Bzt1_fft)[start:end], label="t1")
    axs[1,0].scatter(filter_freq[start:end],  np.abs(filter_Bzt2_fft)[start:end], label="t2")
    axs[1,0].scatter(filter_freq[start:end],  np.abs(filter_Bzt3_fft)[start:end], label="t3")
    axs[1,0].legend()

    print("compare")
    a =1 
    axs[1,1].set_title("(num track / source) vs ana filter")
    axs[1,1].scatter(filter_freq[start:end],a*np.abs(filter_Bzt1_fft)[start:end]/np.abs(filter_source)[start:end],label="t1 num")
    axs[1,1].scatter(filter_freq[start:end], np.abs(filter_ana_fft_1)[start:end], label="analytisch")
    axs[1,1].scatter(filter_freq[start:end],a*np.abs(filter_Bzt2_fft)[start:end]/np.abs(filter_source)[start:end], label="t2 num")
    axs[1,1].scatter(filter_freq[start:end], np.abs(filter_ana_fft_2)[start:end], label="anlytisch")
    axs[1,1].legend()


    plt.show()
    # Bz = X[Nx+1:]
    # Bz_fft = np.fft.fft(Bz)
    # Bz_fft = np.fft.fftshift(Bz_fft)
    # freq = np.fft.fftfreq(Nt, dt)
    # plt.plot(freq, np.abs(Bz_fft))
    # plt.show()