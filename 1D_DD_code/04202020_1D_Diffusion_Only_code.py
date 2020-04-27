import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
from scipy.special import erf
mesh_cells = 500  # mesh step size
time_steps = 100000  # time steps
x = np.linspace(0, 14e-6, mesh_cells)  # 1D mesh
dx = 14e-6/mesh_cells  # Spacing between mesh
q = 1.602e-19  # C
epsilon = 25 * 8.854e-12  # C/V*m
kb = 1.38e-23  # J/K
T = 298  # K

p = np.zeros(mesh_cells)
# p = np.ones(mesh_cells)
# p[230:260] = 1e20
p = 1e25 * x
# n = 1e25 * -x + 14e-6*1e25
# p = p*1e20

sol_p = np.empty([mesh_cells,time_steps])
# sol_n = np.empty_like(sol_p)

sol_p[:,0] = p  # Initial conditions

Dp = 1e-14 # m^2/s  Diffusivity
# Dn = 1e-14 # m^2/s  Diffusivity

timestep = dx**2 / (2*Dp)

##### DIFFUSION ONLY #####
for k in range(time_steps):
    for i in range(mesh_cells):

        if i == 0:
            if k == 0:
                pass
            else:
                sol_p[i,k] = sol_p[i, k-1] + Dp*timestep/(dx**2)*(-2*sol_p[i, k-1] + 2*sol_p[i+1, k-1])

        elif i == 499:
            if k == 0:
                pass
            else:
                sol_p[i,k] = sol_p[i, k-1] + Dp*timestep/(dx**2)*(-2*sol_p[i, k-1] + 2*sol_p[i-1, k-1])

        elif k == 0:
            pass

        else:
            sol_p[i,k] = sol_p[i, k-1] + Dp*timestep/(dx**2) * (sol_p[i+1, k-1] - 2*sol_p[i, k-1] + sol_p[i-1, k-1])

# np.savetxt("sol_p test1.csv", sol_p)


print(np.sum(sol_p[:,0]))
print(np.sum(sol_p[:,-1]))
plt.figure()
plt.plot(x, sol_p[:,0], color='blue')
plt.plot(x, sol_p[:,-1], color='red')
plt.title("Charge Density")
plt.xlabel("Distance (m)")
plt.ylabel("Charge Density (m^-3)")

plt.show()
