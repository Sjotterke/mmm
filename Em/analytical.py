import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hankel2

J0 = 1
omega = 1
mu0 = 1
c = 1
k0 = omega /c
Nx =101
Ny = 101 
Lx = 1
Ly = 1

x0 =Lx/2
y0 = Ly/2
# index of the source
ix = round(x0/Lx * Nx)
iy =round(y0/Ly * Ny)
x = np.linspace(0, Lx, Nx) - x0
y = np.linspace(0, Ly, Ny) - y0
X, Y = np.meshgrid(x, y)
print("X: \n{} \n Y: \n{}".format(X, Y))

hz = -J0*omega*mu0/4 * hankel2(0, k0 * np.sqrt((X)**2 + (Y)**2))
# dom imshow of real part and imaginary part
fig, ax =plt.subplots(1, 2, figsize=(10, 5))

plt.title("Analytical Solution of H_z imaginery")
# plot the sources on it
ax[0].scatter(ix, iy, color="red", label="Source")
ax[0].imshow(hz.imag)
ax[0].set_xlabel("x")
ax[0].set_ylabel("y")
ax[0].set_title("Imaginary part of H_z")

ax[1].scatter(ix, iy, color="red", label="Source")
ax[1].imshow(hz.real)
ax[1].set_xlabel("x")
ax[1].set_ylabel("y")
ax[1].set_title("Real part of H_z")
plt.show()

print("hz: \n{}\nwith shape: {}".format(hz, hz.shape))
