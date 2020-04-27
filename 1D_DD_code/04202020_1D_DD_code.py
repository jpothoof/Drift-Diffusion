import numpy as np
import matplotlib.pyplot as plt


mesh_cells = 500  # mesh step size
time_steps = 7000  # time steps
x = np.linspace(0, 14e-6, mesh_cells)  # 1D mesh
dx = 14e-6/mesh_cells  # Spacing between mesh
q = 1.602e-19  # C
epsilon = 25 * 8.854e-12  # C/V*m
kb = 1.38e-23  # J/K
T = 298  # K

# p = np.zeros(mesh_cells)
p = np.ones(mesh_cells)
# p[230:260] = 1e20
# p = 1e25 * x
# n = 1e25 * -x + 14e-6*1e25
p = p*1e20

# pot = np.zeros(500) # Potential
Efield = np.zeros(500)

for i in range(len(Efield)):
    if i > 0:
        Efield[i] = (q/epsilon) * dx * p[i-1] + Efield[i-1]
    else:
        pass
Efield = Efield - np.mean(Efield)

# for j in range(len(pot)):
#     if j > 0:
#         pot[j] = dx * -Efield[j-1] + pot[j-1]
#     else:
#         pass

# plt.figure()
# plt.plot(x, p)
# plt.plot(x, n)
# plt.figure()
# plt.plot(x,Efield)
# plt.figure()
# plt.plot(x, pot)
# plt.show()

sol_p = np.empty([mesh_cells,time_steps])
# sol_n = np.empty_like(sol_p)
sol_Efield = np.empty_like(sol_p)
# sol_pot = np.empty([mesh_cells,time_steps])

sol_p[:,0] = p  # Initial conditions
# sol_n[:,0] = n
sol_Efield[:,0] = Efield
sol_Efield[0,1:] = 0
# sol_pot[:,0] = pot
# sol_pot[0,:] = 0
mu_p = 1e-13
# mu_n = 1e13
# Dp = 1e-14 # m^2/s  Diffusivity
Dp = mu_p * kb * T / q
# Dn = mu_n * kb * T / q

timestep = dx**2 / (2*Dp)

##### DRIFT-DIFFUSION #####
for k in range(time_steps):
    for i in range(mesh_cells):

        if i == 0:  #Neumann Boundary Condition - reflective
            if k == 0:  #Skip initial conditions
                pass
            else:
                sol_p[i,k] = sol_p[i, k-1] + Dp*timestep/(dx**2)*(-2*sol_p[i, k-1] + 2*sol_p[i+1, k-1])
                # sol_n[i,k] = sol_n[i, k-1] + Dn*timestep/(dx**2)*(-2*sol_n[i, k-1] + 2*sol_n[i+1, k-1])

        elif i == 499:  #Neumann Boundary Condition - reflective
            if k == 0:  #Skip initial conditions
                pass
            else:
                sol_p[i,k] = sol_p[i, k-1] + Dp*timestep/(dx**2)*(-2*sol_p[i, k-1] + 2*sol_p[i-1, k-1])
                # sol_n[i,k] = sol_n[i, k-1] + Dn*timestep/(dx**2)*(-2*sol_n[i, k-1] + 2*sol_n[i-1, k-1])
                sol_Efield[i, k] = (q / epsilon) * dx * sol_p[i - 1, k] + sol_Efield[i - 1, k]

        elif k == 0: # Skip initial conditions
            pass

        else:  # Calculate for interior cells
            sol_p[i,k] = sol_p[i, k-1] - mu_p * sol_Efield[i,k-1] * timestep / (2*dx) * (sol_p[i+1,k-1] - sol_p[i-1,k-1]) \
                         + Dp*timestep/(dx**2) * (sol_p[i+1, k-1] - 2*sol_p[i, k-1] + sol_p[i-1, k-1])

            # sol_n[i,k] = sol_n[i, k-1] - mu_n * sol_Efield[i,k-1] * timestep / (2*dx) * (sol_n[i+1,k-1] - sol_n[i-1,k-1]) \
            #              + Dn*timestep/(dx**2) * (sol_n[i+1, k-1] - 2*sol_n[i, k-1] + sol_n[i-1, k-1])

            sol_Efield[i, k] = (q / epsilon) * dx * sol_p[i - 1, k] + sol_Efield[i - 1, k]

    sol_Efield[:,k] = sol_Efield[:,k] - np.mean(sol_Efield[:,k])

            # if i > 0:
            #     sol_pot[i,k] = dx * -Efield[i-1,k] + pot[i-1,k]
            # else:
            #     pass

# np.savetxt("sol_p test1.csv", sol_p)
# np.savetxt("sol_Efield test1.csv")

print(np.sum(sol_p[:,0]))
print(np.sum(sol_p[:,-1]))
plt.figure()
plt.plot(x, sol_p[:,0], color='blue')
plt.plot(x, sol_p[:,-1], color='red')
plt.title("Charge Density")
plt.xlabel("Distance (m)")
plt.ylabel("Charge Density (m^-3)")
# plt.plot(x, sol_n[:,0], color='red')
# plt.plot(x, sol_n[:,-1], color='red')
plt.figure()
plt.plot(x, sol_Efield[:,0], color='blue')
plt.plot(x, sol_Efield[:,-1], color='red')
plt.title("Electric Field")
plt.xlabel("Distance (m)")
plt.ylabel("Electric Field (V/m)")
# print(sol_Efield[:,-1])
# plt.plot(x, sol_Efield[:,2])
# plt.figure()
# plt.plot(x, sol_pot[:,0])
# plt.plot(x, sol_pot[:,-1])

plt.show()
