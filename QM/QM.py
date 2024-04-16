
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import *
from scipy.integrate import quad
from PIL import Image, ImageDraw
from io import BytesIO
import time 


Number_of_timesteps= int(1e5) 
Ly= 50*10**(-9) #m
dy= 0.125*10**(-9) #m
kvalues= np.arange(0, Ly//dy+1)
dt= 4.1696*10**(-18) #s

tc= Number_of_timesteps/4* dt
sigma_t= tc/6

e_mass= 9.109383*10**(-31) #kg
omega_HO= 50*10**(12) #rad/s
q= -elementary_charge

def create_plot(probdens):
    y = kvalues*dy
    plt.plot(y, probdens)
    plt.xlabel('y')
    plt.ylabel('Probability Density')
    plt.grid(True)
    # Save the plot to memory buffer
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf

def potential(y):
    return e_mass* omega_HO**2 * (y-Ly/2)**2 /2

def ground_state(y):
    return ((e_mass*omega_HO)/(np.pi*hbar))**(1/4) * np.exp(-(e_mass*omega_HO / (2*hbar)) * (y-Ly/2)**2)

def gaussian_pulse(t, amplitude= 1e8):
    return amplitude * np.exp(-(t-tc)**2 / (2*sigma_t**2))



# Initialize arrays to store data
prob_dens_all = []
expectation_pos_all = []
expectation_mom_all= []
expectation_kinetic_energy_all= []
total_energy_all= []
potential_energy_all= []
for discr_order in ["second", "fourth"]:
    start= time.time()
    PsiR= np.zeros(int(Ly//dy)+1)
    PsiI= np.zeros(int(Ly//dy)+1)
    prob_dens= np.zeros((int(Ly//dy)+1, Number_of_timesteps))
    expectation_pos= np.zeros(Number_of_timesteps, dtype= 'complex')
    expectation_mom= np.zeros(Number_of_timesteps, dtype= 'complex')
    expectation_kinetic_energy= np.zeros(Number_of_timesteps, dtype= 'complex')
    total_energy= np.zeros(Number_of_timesteps, dtype= 'complex')
    potential_energy= np.zeros(Number_of_timesteps, dtype= 'complex')
    # Set initial wavefunction
    for k in range(len(PsiR)):
        PsiR[k]= ground_state(k*dy)
        PsiR[0]= 0
        PsiR[-1]= 0
    Psi= PsiR + 1j* PsiI

    # Calculate probability density and expectation values
    prob_dens[:,0]= np.real(np.conjugate(Psi) * Psi)
    expectation_pos[0]= np.sum(dy * kvalues * prob_dens[:,0]*dy) #prob density times dy gives prob so we get y*P(y)
    expectation_mom[0]= np.sum(-1j* hbar* np.conj(Psi) * np.gradient(Psi, dy))
    expectation_kinetic_energy[0]=  expectation_mom[0]**2/(2*e_mass)
    total_energy[0]= np.sum(-1j* hbar* np.conj(Psi) * np.gradient(Psi, dy))**2/ (2*e_mass) 

    for n in range(1,Number_of_timesteps):
        # Apply appropriate discretization
        if discr_order == "second":
            PsiR[1:-1]= PsiR[1:-1] - hbar*dt/(2*e_mass) * (PsiI[2:] - 2* PsiI[1:-1] + PsiI[:-2]) / dy**2 - dt/hbar * (q*(kvalues[1:-1]*dy - Ly/2)* gaussian_pulse(n*dt) - potential(kvalues[1:-1]*dy)) * PsiI[1:-1]
            PsiI[1:-1]= PsiI[1:-1] + hbar*dt/(2*e_mass) * (PsiR[2:] - 2* PsiR[1:-1] + PsiR[:-2]) / dy**2 + dt/hbar * (q*(kvalues[1:-1]*dy - Ly/2) * gaussian_pulse(n*dt) - potential(kvalues[1:-1]*dy)) * PsiR[1:-1]
        elif discr_order == "fourth":
            PsiR[2:-2]= PsiR[2:-2] - hbar*dt/(2*e_mass) * (-PsiI[4:] + 16*PsiI[3:-1] - 30*PsiI[2:-2] + 16*PsiI[1:-3] - PsiI[:-4]) / (12*dy**2) - dt/hbar * (q*(kvalues[2:-2]*dy - Ly/2) * gaussian_pulse(n*dt) - potential(kvalues[2:-2]*dy)) * PsiI[2:-2]
            PsiI[2:-2]= PsiI[2:-2] + hbar*dt/(2*e_mass) * (-PsiR[4:] + 16*PsiR[3:-1] - 30*PsiR[2:-2] + 16*PsiR[1:-3] - PsiR[:-4]) / (12*dy**2) + dt/hbar * (q*(kvalues[2:-2]*dy - Ly/2) * gaussian_pulse(n*dt) - potential(kvalues[2:-2]*dy))* PsiR[2:-2]

        Psi= PsiR + 1j* PsiI
        prob_dens[:,n]= np.real(np.conjugate(Psi) * Psi)
        expectation_pos[n]= np.trapz(np.conjugate(Psi)* dy*kvalues * Psi, dx=dy)
        
        expectation_mom[n] =  np.trapz( -1j * hbar * np.conjugate(Psi) * np.gradient(Psi, dy), dx=dy)
        expectation_kinetic_energy[n]=  np.trapz( - hbar**2 / (2*e_mass)* np.conjugate(Psi) * np.gradient(np.gradient(Psi, dy), dy), dx=dy)
        potential_energy[n]= np.trapz( np.conjugate(Psi)* potential(kvalues*dy)* Psi, dx=dy )
        total_energy[n] = expectation_kinetic_energy[n] + potential_energy[n]
        
    
    # Append probability density and expectation value of position for this case
    prob_dens_all.append(prob_dens)
    expectation_pos_all.append(expectation_pos)
    expectation_mom_all.append(expectation_mom)
    expectation_kinetic_energy_all.append(expectation_kinetic_energy)
    total_energy_all.append(total_energy)
    potential_energy_all.append(potential_energy)
    end= time.time()
    print("Time elapsed during {}-order calculations:".format(discr_order), end - start)


# Calculate the range for the colorbar
min_prob_dens = min(np.min(prob_dens_all[0]), np.min(prob_dens_all[1]))
max_prob_dens = max(np.max(prob_dens_all[0]), np.max(prob_dens_all[1]))

# Plotting
plt.figure(figsize=(16, 8))

# Plot for discr_order = "second"
plt.subplot(1, 2, 1)
plt.imshow(prob_dens_all[0], extent=[0, Number_of_timesteps*dt*10**15, 0, Ly*10**9], aspect='auto', origin='lower', cmap='terrain', vmin=min_prob_dens, vmax=max_prob_dens)
plt.plot(np.arange(Number_of_timesteps)*dt* 10**15, expectation_pos_all[0]*10**9, label='Expectation Value of Position', color= 'black', linewidth= 0.5)
plt.legend()
plt.xlabel('Time (fs)')
plt.ylabel('Position (nm)')
plt.colorbar()
plt.title('Second Order Discretization')
# Plot for discr_order = "fourth"
plt.subplot(1, 2, 2)
plt.imshow(prob_dens_all[1], extent=[0, Number_of_timesteps*dt*10**15, 0, Ly*10**9], aspect='auto', origin='lower', cmap='terrain', vmin=min_prob_dens, vmax=max_prob_dens)
plt.plot(np.arange(Number_of_timesteps)*dt* 10**15, expectation_pos_all[1]*10**9, label='Expectation Value of Position', color= 'black', linewidth= 0.5)
plt.legend()
plt.xlabel('Time (fs)')
plt.ylabel('Position (nm)')
plt.title('Fourth Order Discretization')
plt.colorbar(label='Probability Density')
plt.tight_layout()


plt.figure(figsize=(16, 8))
plt.plot(np.arange(Number_of_timesteps)*dt* 10**15, expectation_kinetic_energy_all[0] , label='Kinetic energy', color= 'red', linewidth= 0.5)
plt.plot(np.arange(Number_of_timesteps)*dt* 10**15, potential_energy_all[0], label='Potential energy', color= 'black', linewidth= 0.5)
plt.plot(np.arange(Number_of_timesteps)*dt* 10**15, - q*expectation_pos_all[0]* gaussian_pulse(np.arange(Number_of_timesteps)*dt), label='Pulse energy', color= 'yellow', linewidth= 0.5)
plt.plot(np.arange(Number_of_timesteps)*dt* 10**15, total_energy_all[0] , label='Total energy (2nd order)', color= 'green', linewidth= 0.5)
plt.plot(np.arange(Number_of_timesteps)*dt* 10**15, total_energy_all[1] , label='Total energy (4th order)', color= 'blue', linewidth= 0.5)

plt.legend()
plt.xlabel('Time (fs)')
plt.ylabel('Energy')
plt.title('Conservation of energy')
plt.tight_layout()


plt.show()

