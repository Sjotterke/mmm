import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
from sources import GaussianSource
from scipy.special import hankel2
import math
animation =False
# material properties
ebs = 8.854 * 10**(-12) # Permittivity of free space F/m
mu = 4 * math.pi * 10**(-7) # Permeability of free space H/m
sigma = 0.0



# Define the grid
Nx = 300
Ny = 300
Nt = 300
Ly = 0.1
Lx = 0.1
dx = Lx/Nx
dy = Ly/Ny
c = 3 * 10**8  # Speed of light m/s
#Source specs
J0 = 1

# Place sensors


# Courant Number and resulting timestep
CFL = 0.9
dt = CFL *dy/c
tarray = np.linspace(0, Nt*dt, Nt)
print("Courant Number: {}\nTimestep: {}".format(CFL, dt))
print("dx: {}\ndy: {}".format(dx, dy))
print("dt: {}".format(dt))
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
    print("M {}".format(M))
    L1 = np.hstack((-1/dx*np.array(Ad), 1/dt*np.array(Ai)))
    L2 = np.hstack(((ebs/dt - sigma/2)*np.array(Ai), -1/(mu*dx)*np.array(Ad)))
    L = np.vstack((L1, L2))
    print("L {}".format(L))
    return np.array(Ad), np.array(Ai),M, L

matrices_construct()
AD, AI, M, L = matrices_construct()

# X contains values of Ey and Bz
X = np.zeros((2*Nx+2, Ny))

print("M shape: {}".format(M.shape))
# print("M: \n{}".format(M))

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

#source specs
tc = Nt/5*dt
sig = tc/6 #see project why
omega = 1e12 # rad/s
def source(t):
    return J0*np.sin(omega*t)*np.exp(-(t-tc)**2/(2*sig**2))
Ex = np.zeros((Nx+1, Ny+1)) 
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

    Y = Ex[:-1, 1:] + Ex[1:,1:] - Ex[:-1,:-1] - Ex[1:,:-1]
    
    Y_tot = np.vstack((Y, np.zeros((2+Nx, Ny))))

    # print("Y_tot shape: {}".format(Y_tot.shape))
    X[Nx+1+Nx//2, Ny//2] += source(t) 
    middel = np.matmul(L, X) + 1/dy*Y_tot
    X = np.matmul(Minv, middel)
    Ex[:,1:-1] = (ebs/dt - sigma/2)/(ebs/dt + sigma/2) * Ex[:,1:-1] + 1/(ebs/dt + sigma/2)*(1/(mu*dy))*(X[Nx+1:, 1:]-X[Nx+1:, :-1])
    Ex[:,0] = 0
    Ex[:,-1] = 0
    
    #update trackers
    if it>Nt//2:
        Bzt1[it] = X[Nx+1+it1[0], it1[1]]
        Bzt2[it] = X[Nx+1+it2[0], it2[1]]
        Bzt3[it] = X[Nx+1+it3[0], it3[1]]
    if animation == True:
        artists.append([plt.imshow(np.transpose(X[Nx+1:,:]), cmap='viridis',vmin=-0.02*J0,vmax=0.02*J0,animated=True),
                    
                        ])

# Create an animation
if animation == True:
    ani = ArtistAnimation(fig, artists, interval=50, blit=True)
    plt.show()
print("total time: {}".format(Nt*dt))
print("X[Nx+1:] is Bz, shape: {}".format(X[Nx+1:].shape))


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

