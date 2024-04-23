
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import *
from PIL import Image, ImageDraw
from io import BytesIO
import time 
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter

Number_of_timesteps= int(1e5) 
Ly= 50*angstrom #m
dy= 0.25*angstrom #m
kvalues= np.arange(0, Ly//dy+1)

make_gif= False

dt= dy/speed_of_light #s
N= 10**7 #1/m


tc= Number_of_timesteps/4* dt
sigma_t= tc/6


effective_mass=0.15 * electron_mass
omega_HO= 50*10**(14) #rad/s
q= -elementary_charge

def potential(y):
    return effective_mass* omega_HO**2 * (y-Ly/2)**2 /2

def ground_state(y):
    return ((effective_mass*omega_HO)/(np.pi*hbar))**(1/4) * np.exp(-(effective_mass*omega_HO / (2*hbar)) * (y-Ly/2)**2)

def gaussian_pulse(t, amplitude= 1e10):
    return amplitude * np.exp(-(t-tc)**2 / (2*sigma_t**2))


def second_order_laplacian(Psi):
    return (Psi[2:] - 2* Psi[1:-1] + Psi[:-2]) / dy**2 

def fourth_order_laplacian(Psi):
    return (-Psi[4:] + 16*Psi[3:-1] - 30*Psi[2:-2] + 16*Psi[1:-3] - Psi[:-4]) / (12*dy**2)

def current_density(PsiR, PsiI):
    return N*q*hbar/(effective_mass*dy) * (PsiR[:-1]* PsiI[1:] - PsiR[1:]*PsiI[:-1])
def animate(n):
    plt.cla()
    plt.plot(kvalues*dy* giga, J[:,n], label= "{:d}".format(n))
    plt.xlabel("y [nm]")
    plt.ylabel("Current density [A/m²]")
    plt.legend(title= "Timestep")
    plt.xlim(0,Ly*giga)
    plt.ylim(np.min(J), np.max(J))
def continuity(J, rho):
    return (rho[1:,1:] + rho[:-1,1:] - rho[1:,:-1] - rho[:-1,:-1])/(2*dt) + (J[1:,1:] - J[:-1,1:]) / dy

print("dt for Maxwell: {:.2e}".format(dy/speed_of_light))
print("dt for Schrodinger: {:.2e}".format(1/10   *2*hbar/np.max(potential(kvalues*dy))))
print("1 Period of HO: {:.2e}".format(2*np.pi/omega_HO) )
# Initialize arrays to store data
prob_dens_all = []
expectation_pos_all = []
expectation_mom_all= []
expectation_kinetic_energy_all= []
total_energy_all= []
potential_energy_all= []

plt.figure()
for discr_order in ["second", "fourth"]:
    start= time.time()
    # Set initial wavefunction
    PsiR= ground_state(kvalues*dy)
    PsiI= np.zeros(int(Ly//dy)+1)
    Psi= PsiR+ 1j*PsiI

    prob_dens= np.zeros((int(Ly//dy)+1, Number_of_timesteps))
    expectation_pos= np.zeros(Number_of_timesteps, dtype= 'complex')
    expectation_mom= np.zeros(Number_of_timesteps, dtype= 'complex')
    expectation_kinetic_energy= np.zeros(Number_of_timesteps, dtype= 'complex')
    total_energy= np.zeros(Number_of_timesteps, dtype= 'complex')
    potential_energy= np.zeros(Number_of_timesteps, dtype= 'complex')
    J=  np.zeros((int(Ly//dy)+1, Number_of_timesteps))

    # Calculate probability density and expectation values
    prob_dens[:,0]= np.real(np.conjugate(Psi) * Psi)
    expectation_pos[0]= np.trapz(np.conjugate(Psi)* dy*kvalues * Psi, dx=dy) #prob density times dy gives prob so we get y*P(y)

    expectation_kinetic_energy[0]=   - hbar**2 / (2*effective_mass) * np.trapz(np.conjugate(Psi[1:-1]) * second_order_laplacian(Psi), dx=dy)
    potential_energy[0]= np.trapz( np.conjugate(Psi)* potential(kvalues*dy)* Psi, dx=dy )
    total_energy[0] = expectation_kinetic_energy[0] + potential_energy[0]
    J[:-1,0]= current_density(PsiR,PsiI)

    
    for n in range(1,Number_of_timesteps):
        
        # Apply appropriate discretization
        if discr_order == "second":
            PsiR[1:-1]= PsiR[1:-1] - hbar*dt/(2*effective_mass) * second_order_laplacian(PsiI) - dt/hbar * (q*(kvalues[1:-1]*dy - Ly/2)* gaussian_pulse(n*dt) - potential(kvalues[1:-1]*dy)) * PsiI[1:-1]
            PsiI[1:-1]= PsiI[1:-1] + hbar*dt/(2*effective_mass) * second_order_laplacian(PsiR) + dt/hbar * (q*(kvalues[1:-1]*dy - Ly/2) * gaussian_pulse(n*dt) - potential(kvalues[1:-1]*dy)) * PsiR[1:-1]
            
        elif discr_order == "fourth":
            PsiR[2:-2]= PsiR[2:-2] - hbar*dt/(2*effective_mass) * fourth_order_laplacian(PsiI) - dt/hbar * (q*(kvalues[2:-2]*dy - Ly/2) * gaussian_pulse(n*dt) - potential(kvalues[2:-2]*dy)) * PsiI[2:-2]
            PsiI[2:-2]= PsiI[2:-2] + hbar*dt/(2*effective_mass) * fourth_order_laplacian(PsiR) + dt/hbar * (q*(kvalues[2:-2]*dy - Ly/2) * gaussian_pulse(n*dt) - potential(kvalues[2:-2]*dy))* PsiR[2:-2]

        Psi= PsiR + 1j* PsiI
        prob_dens[:,n]= np.real(np.conjugate(Psi) * Psi)
        expectation_pos[n]= np.trapz(np.conjugate(Psi)* dy*kvalues * Psi, dx=dy)
        charge_dens= q*prob_dens[:,n]

        if discr_order == "second":
            # expectation_mom[n] =  np.trapz( -1j * hbar * np.conjugate(Psi) * np.gradient(Psi, dy), dx=dy)
            expectation_kinetic_energy[n]=  - hbar**2 / (2*effective_mass) * np.trapz(np.conjugate(Psi[1:-1]) * second_order_laplacian(Psi), dx=dy)
            potential_energy[n]= np.trapz( np.conjugate(Psi)* potential(kvalues*dy)* Psi, dx=dy )
            total_energy[n] = expectation_kinetic_energy[n] + potential_energy[n]
        elif discr_order == "fourth":
            # expectation_mom[n] =  np.trapz( -1j * hbar * np.conjugate(Psi) * np.gradient(Psi, dy), dx=dy)
            expectation_kinetic_energy[n]=  - hbar**2 / (2*effective_mass) * np.trapz(np.conjugate(Psi[2:-2]) * fourth_order_laplacian(Psi), dx=dy)
            potential_energy[n]= np.trapz( np.conjugate(Psi)* potential(kvalues*dy)* Psi, dx=dy )
            total_energy[n] = expectation_kinetic_energy[n] + potential_energy[n]
        J[:-1,n]= current_density(PsiR,PsiI)

    if make_gif==True:
        fig= plt.figure()
        ani= FuncAnimation(fig, animate, frames= np.arange(0, Number_of_timesteps, 1000))
        ani.save("D:\School\Master2\MMM\Current_Density_{:s}.gif".format(discr_order), dpi=300,
             writer=PillowWriter(fps=10))
        plt.close()

    
    
        
    


    # Append probability density and expectation value of position for this case
    prob_dens_all.append(prob_dens)
    expectation_pos_all.append(expectation_pos)
    expectation_mom_all.append(expectation_mom)
    expectation_kinetic_energy_all.append(expectation_kinetic_energy)
    total_energy_all.append(total_energy)
    potential_energy_all.append(potential_energy)

    ##Check the continuity equation 
    
    
    plt.plot(np.arange(1,Number_of_timesteps)*dt* peta, np.trapz(continuity(J,rho= prob_dens*q), axis= 0, dx=dy), label="{:s} order".format(discr_order))
    
    end= time.time()
    print("Time elapsed during {}-order calculations:".format(discr_order), end - start)

plt.title("Continuity equation integrated along y should equal 0")
plt.legend()


# Calculate the range for the colorbar
min_prob_dens = min(np.min(prob_dens_all[0]), np.min(prob_dens_all[1]))
max_prob_dens = max(np.max(prob_dens_all[0]), np.max(prob_dens_all[1]))

# Plotting
plt.figure(figsize=(16, 8))

# Plot for discr_order = "second"
plt.subplot(1, 2, 1)
plt.imshow(prob_dens_all[0], extent=[0, Number_of_timesteps*dt*peta, 0, Ly*giga], aspect='auto', origin='lower', cmap='terrain', vmin=min_prob_dens, vmax=max_prob_dens)
plt.plot(np.arange(Number_of_timesteps)*dt* peta, expectation_pos_all[0]*giga, label='Expectation Value of Position', color= 'black', linewidth= 0.5)
plt.legend()
plt.xlabel('Time (fs)')
plt.ylabel('Position (nm)')
plt.colorbar()
plt.title('Second Order Discretization')
# Plot for discr_order = "fourth"
plt.subplot(1, 2, 2)
plt.imshow(prob_dens_all[1], extent=[0, Number_of_timesteps*dt*peta, 0, Ly*giga], aspect='auto', origin='lower', cmap='terrain', vmin=min_prob_dens, vmax=max_prob_dens)
plt.plot(np.arange(Number_of_timesteps)*dt* peta, expectation_pos_all[1]*giga, label='Expectation Value of Position', color= 'black', linewidth= 0.5)
plt.legend()
plt.xlabel('Time (fs)')
plt.ylabel('Position (nm)')
plt.title('Fourth Order Discretization')
plt.colorbar(label='Probability Density')
plt.tight_layout()


fig, ax= plt.subplots(figsize=(16, 8))
plt.plot(np.arange(Number_of_timesteps)*dt*peta, - q*expectation_pos_all[0]* gaussian_pulse(np.arange(Number_of_timesteps)*dt)* physical_constants['joule-electron volt relationship'][0], label='Pulse energy', color= 'red', linewidth= 0.5)
plt.plot(np.arange(Number_of_timesteps)*dt*peta, total_energy_all[0]* physical_constants['joule-electron volt relationship'][0] , label='Kinetic energy + Potential energy (2nd order)', color= 'green', linewidth= 0.5)
plt.plot(np.arange(Number_of_timesteps)*dt*peta, total_energy_all[1]* physical_constants['joule-electron volt relationship'][0] , label='Kinetic energy + Potential energy (4th order)', color= 'blue', linewidth= 0.5)
# plt.xlim(left=Number_of_timesteps*dt * 10**15/2) #apply limits when only looking at the total energies for 2nd and fourth order
# plt.ylim(bottom= 0.6)
plt.legend()
plt.xlabel('Time (fs)')
plt.ylabel('Energy (eV)')
plt.title('Conservation of energy')
plt.tight_layout()
sub_axes = plt.axes([.6, .6, .25, .25]) 
# sub_axes= zoomed_inset_axes(ax,1, loc=1)
# plot the zoomed portion
sub_axes.plot(np.arange(Number_of_timesteps/2, Number_of_timesteps)*dt*peta, total_energy_all[0][int(Number_of_timesteps/2) : int(Number_of_timesteps)]* physical_constants['joule-electron volt relationship'][0], label='Total energy (2nd order)', color= 'green', linewidth= 0.5) 
sub_axes.plot(np.arange(Number_of_timesteps/2, Number_of_timesteps)*dt*peta, total_energy_all[1][int(Number_of_timesteps/2) : int(Number_of_timesteps)]* physical_constants['joule-electron volt relationship'][0] , label='Total energy (4th order)', color= 'blue', linewidth= 0.5)

sub_axes.set_xlim(Number_of_timesteps/2*dt*peta, Number_of_timesteps*dt*peta)
mark_inset(ax, sub_axes, loc1=3, loc2=4, fc="none", ec="0.5")
plt.draw()


plt.show()
