import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
from sources import GaussianSource
from scipy.special import hankel2
import math
animation =True
plot = False
# material properties
eps = 8.854 * 10**(-12) # Permittivity of free space F/m
mu = 4 * math.pi * 10**(-7) # Permeability of free space H/m



# Define the grid
Nx = 200
Ny = 200
Nt = 200
Ly = 0.1
Lx = 0.1
# PML
nx = 30
ny = nx
Nx_T = Nx + nx
Ny_T = Ny + ny
dx = Lx/Nx_T
dy = Ly/Ny_T
c = 3 * 10**8  # Speed of light m/s
#Source specs
J0 = 1
sigma_e = 5e6
sigma_m = sigma_e/eps*mu

# Place sensors



# Courant Number and resulting timestep
CFL = 0.9
dt = CFL *dy/c
tarray = np.linspace(0, Nt*dt, Nt)
print("Courant Number: {}\nTimestep: {}".format(CFL, dt))
print("dx: {}\ndy: {}".format(dx, dy))
print("dt: {}".format(dt))

p = 1
# do a and b with list and then do a[0] *(1/nx)**p  + a[1]
def matrices_construct(a,b,c,d):
    Ad = [[0 for _ in range(Nx_T+1)] for _ in range(Nx_T)]
    Ai = [[0 for _ in range(Nx_T+1)] for _ in range(Nx_T)]
    for i in range(Nx_T):
        if i == nx-1:
            for j in range(Nx_T+1):
                if i == j:
                    Ad[i][j] = -1 *a #* (1/nx)**p ### PML coefficient for Ad
                    Ai[i][j] = 1 *b #* (1/nx)**p ### PML coefficient for Ai
                if i+1 == j:
                    Ad[i][j] = 1 * c ### NO PML coefficient for Ad
                    Ai[i][j] = 1 * d ### NO PML coefficient for Ai
        elif i == Nx_T - nx:
            for j in range(Nx_T+1):
                if i == j:
                    Ad[i][j] = -1 * c ### NO PML
                    Ai[i][j] = 1 * d  ### NO PML
                if i+1 == j:
                    Ad[i][j] = 1 *a #* (1/nx)**p ### PML
                    Ai[i][j] = 1 *b #* (1/nx)**p ### PML
        elif i < nx-1:
            for j in range(Nx_T+1):
                if i == j:
                    Ad[i][j] = -1 * a#* ((30-i)/nx)**p ###  PML
                    Ai[i][j] = 1 * b#* ((30-i)/nx)**p ### PML
                if i+1 == j:
                    Ad[i][j] = 1 * a#* ((30-i)/nx)**p  ### PML
                    Ai[i][j] = 1 * b# * ((30-i)/nx)**p ### PML
        elif i > Nx:
            for j in range(Nx_T+1):
                if i == j:
                    Ad[i][j] = -1 * a#* ((i-Nx)/nx)**p ### PML
                    Ai[i][j] = 1 * b#* ((i-Nx)/nx)**p  ### PM
                if i+1 == j:
                    Ad[i][j] = 1 * a#* ((i-Nx)/nx)**p ### PML
                    Ai[i][j] = 1 * b# * ((i-Nx)/nx)**p ### PML
        else:
            for j in range(Nx_T+1):
                if i == j:
                    Ad[i][j] = -1 * c
                    Ai[i][j] = 1 *d
                if i+1 == j:
                    Ad[i][j] = 1 * c
                    Ai[i][j] = 1 * d
    print("Ad: {}".format(Ad))
    return np.array(Ad), np.array(Ai)


MR1, MR2 = matrices_construct(1/dx,eps/dt,1/dx,eps/dt) # first Ad coef in PML, second Ai coef in PML, third Ad coef NO PML, fourth Ai coef NO PML
MR_0 = np.zeros((Nx_T, Nx_T+1))
MR_E = np.hstack((np.eye(Nx_T), np.zeros((Nx_T,1))))
MC1 = np.vstack((MR1, MR2, MR_0))
MR2, MR1 = matrices_construct(1/(mu*dx),1/dt+sigma_m/2,1/(mu*dx),1/dt)
MC2 = np.vstack((MR1, MR2, MR_0))
MR2, MR1 = matrices_construct(1/(mu*dx),0, 1/(mu*dx),0)
MC3 = np.vstack((MR1, MR2, MR_E))
M_PML = np.hstack((MC1, MC2, MC3))

LR1, LR2 = matrices_construct(-1/dx,eps/dt,-1/dx,eps/dt) # first Ad coef in PML, second Ai coef in PML, third Ad coef NO PML, fourth Ai coef NO PML

LC1 = np.vstack((LR1, LR2, MR_0))
LR2, LR1 = matrices_construct(-1/(mu*dx),1/dt-sigma_m/2,-1/(mu*dx),1/dt)
LC2 = np.vstack((LR1, LR2, MR_0))
LR2, LR1 = matrices_construct(-1/(mu*dx),0, -1/(mu*dx),0)
LC3 = np.vstack((LR1, LR2, MR_E))
L_PML = np.hstack((LC1, LC2, LC3))



print("PML M\n{}".format(M_PML))
print("M_PML shape: {}".format(M_PML.shape))
# X contains values of Ey and Bz
X = np.zeros((3*Nx_T+3, Ny))

# Periodic Boundary Conditions 1 in x direction
BC1 = np.zeros((1, 3*Nx_T+3))
BC1[0,0] = 1
BC1[0, Nx_T] = -1
M_PML = np.vstack((M_PML, BC1))
#Periodic Boundary Conditions 2 in x direction
BC2 = np.zeros((1, 3*Nx_T+3))
BC2[0,Nx_T+1] = 1
BC2[0, 2*Nx_T] = -1
M_PML = np.vstack((M_PML, BC2))
# Periodic Boundary Conditions 3 in x direction
BC3 = np.zeros((1, 3*Nx_T+3))
BC3[0, 2*Nx_T+1] = 1
BC3[0, -1] = -1
M_PML = np.vstack((M_PML, BC3))
# ALL BC for L_PML
L_PML = np.vstack((L_PML, np.zeros((3, 3*Nx_T+3))))
print("M_PML shape after BC: {}".format(M_PML.shape))

# check determinant of M non-zero
detM = np.linalg.det(M_PML)
print("Determinant of M_PML: {}".format(detM))

M_PML_inv = np.linalg.inv(M_PML)
print("M_PML_inv shape: {}".format(M_PML_inv.shape))
M_PML_invL = np.matmul(M_PML_inv, L_PML)

#source specs
tc = Nt/5*dt
sig = tc/6 #see project why
omega = 1e12 # rad/s
def source(t):
    return J0*np.sin(omega*t)*np.exp(-(t-tc)**2/(2*sig**2))
Ex = np.zeros((Nx_T+1, Ny+1)) 
# Create fig for animation
if animation == True:
    fig, ax = plt.subplots()
    artists = []
    plt.title("Bz")
    plt.xlabel("x")
    plt.ylabel("y")

### init trackers
Bzt1=np.zeros(Nt)
Bzt2 = np.zeros(Nt)
Bzt3 = np.zeros(Nt)
it1 = (3*Nx//4, Ny//4)
it2 = (3*Nx//4, Ny//2)
it3 = (3*Nx//4, 3*Ny//4)
itlist = [it1, it2, it3]

for it in range(Nt):
    t = it*dt
    print("Iteration: {}/{}".format(it, Nt))

    Y = Ex[:-1, 1:] - Ex[:-1,:-1]
    #Y = Ex[:-1, 1:] + Ex[1:,1:] - Ex[:-1,:-1] - Ex[1:,:-1]
    P = np.vstack((np.zeros((2*Nx_T, Ny)), np.zeros((0,Ny)),Y, np.zeros((3,Ny))))
    X[Nx_T+1+Nx_T//2, Ny//2] += source(t)/2
    X[2*Nx_T+2+Nx_T//2, Ny//2] += source(t)/2 

    # print("Y_tot shape: {}".format(Y_tot.shape))
    # print("L_PML shape: {}\n X shape: {}\n P shape: {}".format(L_PML.shape, X.shape, P.shape))
    
    middel = np.matmul(L_PML, X) + (dt/(dy))*P
    X = np.matmul(M_PML_inv, middel) 

    Bz = X[Nx_T+1:2*Nx_T+2]+X[2*Nx_T+2:]
     
    Ex[:nx,1:-1] = (eps/dt - sigma_e/2)/(eps/dt + sigma_e/2) * Ex[:nx,1:-1] + 1/(eps/dt + sigma_e/2)*(1/(mu*dy))*(Bz[:nx, 1:]-Bz[:nx, :-1])
    Ex[nx:Nx,1:-1] = Ex[nx:Nx,1:-1] + 1/(eps/dt)*(1/(mu*dy))*(Bz[nx:Nx, 1:]-Bz[nx:Nx, :-1])
    Ex[Nx:,1:-1] = (eps/dt - sigma_e/2)/(eps/dt + sigma_e/2) * Ex[Nx:,1:-1] + 1/(eps/dt + sigma_e/2)*(1/(mu*dy))*(Bz[Nx:, 1:]-Bz[Nx:, :-1])
    Ex[:,0] = 0
    Ex[:,-1] = 0
    
    #update trackers
    if it>Nt//2:
        Bzt1[it] = X[Nx+1+it1[0], it1[1]]
        Bzt2[it] = X[Nx+1+it2[0], it2[1]]
        Bzt3[it] = X[Nx+1+it3[0], it3[1]]
    if animation == True:
        artists.append([plt.imshow(np.transpose(Bz), cmap='viridis',vmin=-0.02*J0,vmax=0.02*J0,animated=True),
                    
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
        x = abs(Nx//2-tracker[0])
        y = abs(Ny//2 - tracker[1])
        ana_fft.append(hankel2(0, source_freq/c * np.sqrt(x**2 + y**2)))

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
    axs[0,0].plot(filter_freq[start: end], np.abs(filter_ana_fft_1)[start:end], label="t1")
    axs[0,0].plot(filter_freq[start: end], np.abs(filter_ana_fft_2)[start:end], label="t2")
    axs[0,0].plot(filter_freq[start: end], np.abs(filter_ana_fft_3)[start:end], label="t3")
    axs[0,0].legend()

    axs[0,1].set_title("source fft filter")
    axs[0,1].plot(filter_freq[start:end], np.abs(filter_source)[start:end])

    axs[1,0].set_title("tracker fft filter")
    print("filter_freq shape: {}\n filter_Bzt1_fft: {}".format(filter_freq.shape, filter_Bzt1_fft.shape))
    axs[1,0].plot(filter_freq[start:end], np.abs(filter_Bzt1_fft)[start:end], label="t1")
    axs[1,0].plot(filter_freq[start:end],  np.abs(filter_Bzt2_fft)[start:end], label="t2")
    axs[1,0].plot(filter_freq[start:end],  np.abs(filter_Bzt3_fft)[start:end], label="t3")
    axs[1,0].legend()

    print("compare")
    axs[1,1].set_title("(num track / source) vs ana filter")
    axs[1,1].plot(filter_freq[start:end],np.abs(filter_Bzt1_fft)[start:end],label="t1 num")
    axs[1,1].plot(filter_freq[start:end]*np.abs(filter_source)[start:end], np.abs(filter_ana_fft_1)[start:end], label="t1_ana*source")
    axs[1,1].plot(filter_freq[start:end],np.abs(filter_Bzt2_fft)[start:end], label="t1 num")
    axs[1,1].plot(filter_freq[start:end]*np.abs(filter_source)[start:end], np.abs(filter_ana_fft_2)[start:end], label="t1_ana*source")
    axs[1,1].legend()


    plt.show()
    # Bz = X[Nx+1:]
    # Bz_fft = np.fft.fft(Bz)
    # Bz_fft = np.fft.fftshift(Bz_fft)
    # freq = np.fft.fftfreq(Nt, dt)
    # plt.plot(freq, np.abs(Bz_fft))
    # plt.show()

