import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import *
from PIL import Image, ImageDraw
from io import BytesIO
import time 
from parameters import *

def potential(y):
    return effective_mass* omega_HO**2 * (y-Ly/2)**2 /2

def ground_state(y):
    return ((effective_mass*omega_HO)/(np.pi*hbar))**(1/4) * np.exp(-(effective_mass*omega_HO / (2*hbar)) * (y-Ly/2)**2)

def gaussian_pulse(t):
    return J0 * np.exp(-(t-tc)**2 / (2*sigma**2))


def second_order_laplacian(Psi):
    return (Psi[2:] - 2* Psi[1:-1] + Psi[:-2]) / dy**2 

def fourth_order_laplacian(Psi):
    return (-Psi[4:] + 16*Psi[3:-1] - 30*Psi[2:-2] + 16*Psi[1:-3] - Psi[:-4]) / (12*dy**2)

def current_density(PsiR, PsiI):
    return N*q*hbar/(effective_mass*dy) * (PsiR[:-1]* PsiI[1:] - PsiR[1:]*PsiI[:-1])


def update_psi(PsiR, PsiI, Ey,  discr_order):
    if discr_order == "second":
        PsiR[1:-1]= PsiR[1:-1] - hbar*dt/(2*effective_mass) * second_order_laplacian(PsiI) - dt/hbar * (q*(kvalues[1:-1]*dy - Ly/2)* Ey[1:-1] - potential(kvalues[1:-1]*dy)) * PsiI[1:-1]
        PsiI[1:-1]= PsiI[1:-1] + hbar*dt/(2*effective_mass) * second_order_laplacian(PsiR) + dt/hbar * (q*(kvalues[1:-1]*dy - Ly/2) * Ey[1:-1] - potential(kvalues[1:-1]*dy)) * PsiR[1:-1]
            
    elif discr_order == "fourth":
        PsiR[2:-2]= PsiR[2:-2] - hbar*dt/(2*effective_mass) * fourth_order_laplacian(PsiI) - dt/hbar * (q*(kvalues[2:-2]*dy - Ly/2) * Ey[2:-2] - potential(kvalues[2:-2]*dy)) * PsiI[2:-2]
        PsiI[2:-2]= PsiI[2:-2] + hbar*dt/(2*effective_mass) * fourth_order_laplacian(PsiR) + dt/hbar * (q*(kvalues[2:-2]*dy - Ly/2) * Ey[2:-2] - potential(kvalues[2:-2]*dy))* PsiR[2:-2]
    return PsiR, PsiI

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
    M2 = np.hstack(((eps/dt + sigma/2)*np.array(Ai), 1/(mu*dx)*np.array(Ad)))
    M = np.vstack((M1, M2)) 
    L1 = np.hstack((-1/dx*np.array(Ad), 1/dt*np.array(Ai)))
    L2 = np.hstack(((eps/dt - sigma/2)*np.array(Ai), -1/(mu*dx)*np.array(Ad)))
    L = np.vstack((L1, L2))
    return np.array(Ad), np.array(Ai),M, L


