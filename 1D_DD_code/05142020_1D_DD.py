import numpy as np
import matplotlib.pyplot as plt

steps = 10000
N = 500
l = 14e-6
dx = l/N
x = np.linspace(0,l,N)  # x for plotting chargedensity and Efield
x2 = np.linspace(0,l,N+1)  # x for plotting current density
# p_ini = np.genfromtxt('p_density_endtime_2.txt')
# n_ini = np.genfromtxt('n_density_endtime_2.txt')

q = 1.602e-19  # C
epsilon = 25 * 8.854e-12  # C/V*m
kb = 1.38e-23  # J/K
T = 298  # K
mu_p = 1e-13  # m^2/Vs
mu_n = mu_p  # m^2/Vs
Dp = mu_p * kb * T / q  # m^2/s
Dn = mu_n * kb * T / q  # m^2/s
V = 9  # external potential
E_ext = V/l  # external Efield

p = np.zeros([N, steps])
n = np.zeros([N, steps])
Efield = np.zeros([N, steps])
Jp = np.zeros([N+1, steps])
Jn = np.zeros([N+1, steps])

# Initial Charge density
p[:,0] = 1e20  # m^-3
# p[240:260,0] = 1e20
n[:,0] = 1e20  # m^-3

# Solving for Initial Efield
for i_ in range(N+1):
    if i_ == 0:
        pass
    elif i_ == N:
        pass
    else:
        # Efield[i_, 0] = q / epsilon * dx * (p[i_, 0] - n[i_, 0]) + Efield[i_-1, 0]
        # Efield[i_, 0] = q / epsilon * dx * p[i_, 0] + Efield[i_-1, 0]
        Efield[i_, 0] = q / epsilon * dx * -n[i_, 0] + Efield[i_-1, 0]

# Subtract out average to get actual Efield and add external Efield
Efield[:,0] = Efield[:,0] - np.mean(Efield[:,0])
Efield[:,0] = Efield[:,0] + E_ext

# Solving for Initial Current Density
# Have to use average Efield and charge density for drift term?? Otherwise asymmetric current density that breaks
# simulation.
for j_ in range(N+1):
    if j_ == 0:
        pass
    elif j_ == N:
        pass
    else:
        # Jp[j_, 0] = mu_p * (Efield[j_-1, 0] + Efield[j_,0])/2 * (p[j_-1, 0]+p[j_,0])/2 - Dp / dx * (p[j_, 0] - p[j_ - 1, 0])
        Jn[j_, 0] = mu_n * (Efield[j_-1, 0] + Efield[j_,0])/2 * (n[j_-1, 0]+n[j_,0])/2 + Dn / dx * (n[j_, 0] - n[j_ - 1, 0])

# PLOTTING INITIAL CONDITIONS
# fig, (ax1, ax2) = plt.subplots(1,2, figsize=[12,6])
# ax1.plot(x, p[:,0])
# ax2.plot(x2, Jp[:,0])
# print(Jp[:,0])
# plt.figure()
# plt.plot(x, Efield[:, 0])
# plt.show()


dt = dx**2 / (2*Dp)  # Timestep

for k in range(steps):  # steps in time
    if k == 0:  # skip initial condtions
        pass
    else:
        for i in range(N+1):
            if i == 0:
                # p[i,k] = -dt/dx * (Jp[i+1,k-1]-Jp[i,k-1]) + p[i,k-1]
                n[i,k] = dt/dx * (Jn[i+1,k-1]-Jn[i,k-1]) + n[i,k-1]

            elif i == N:
                pass

            else:
                # p[i,k] = -dt/dx * (Jp[i+1,k-1]-Jp[i,k-1]) + p[i,k-1]
                n[i,k] = dt/dx * (Jn[i+1,k-1]-Jn[i,k-1]) + n[i,k-1]
                # Efield[i, k] = q / epsilon * dx * (p[i, k] - n[i, k]) + Efield[i - 1, k]
                # Efield[i, k] = q / epsilon * dx * p[i, k] + Efield[i - 1, k]
                Efield[i, k] = q / epsilon * dx * -n[i, k] + Efield[i-1, k]

        Efield[:, k] = Efield[:, k] - np.mean(Efield[:, k])
        Efield[:, k] = Efield[:, k] + E_ext

        for j in range(N+1):
            if j == 0:
                pass
            elif j == N:
                pass
            else:
                # Jp[j, k] = mu_p * (Efield[j-1, k] + Efield[j, k]) / 2 * (p[j-1, k] + p[j, k]) / 2 - Dp / dx * (p[j, k] - p[j-1, k])
                Jn[j, k] = mu_n * (Efield[j-1, k] + Efield[j, k]) / 2 * (n[j-1, k] + n[j, k]) / 2 + Dn / dx * (n[j, k] - n[j-1, k])

# print(np.sum(p[:,0]))
# print(np.sum(p[:,-1]))
# print(np.sum(n[:,0]))
# print(np.sum(n[:,-1]))
# print(p[:, 0])
# print(p[:,10000])
# print(p[:,-1])
# print(n[:,-1])
# print(Jp[:,1000])
# print(p[:, 0])
# print(n[:,0])
# print(p[:,-1])
# print(n[:,-1])
# print(Jp[:,-1])
# print(Jn[:,-1])
# print(Jp[:,0])
# print(Jp[:,50])

# np.savetxt('n_density_endtime_3.txt', n[:,-1])
# np.savetxt('p_density_endtime_3.txt', p[:,-1])

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2,figsize=[12,12])
plt.subplots_adjust(hspace=0.5)
# ax1.plot(x,p[:,0], color='blue', label='initial')
# ax1.plot(x,p[:,-1], color='green', label='end')
ax1.plot(x,n[:,-1], color='red', label='end')
ax1.plot(x,n[:,0], color='purple', label='initial')
ax1.set_title("Charge Density")
ax1.set_xlabel("Distance (m)")
ax1.set_ylabel("Charge Density (m^-3)")

# ax2.plot(x2,Jp[:,0], color='blue')
ax2.plot(x2,Jn[:,0], color='purple')
# ax2.plot(x2,Jp[:,-1], color='green')
ax2.plot(x2,Jn[:,-1], color='red')
ax2.set_title("Current Density")
ax2.set_xlabel("Distance (m)")
ax2.set_ylabel("Current Density (A/m^2)")

ax3.plot(x, Efield[:,0], color='purple')
ax3.plot(x, Efield[:,-1], color='red')
ax3.set_title("Electric Field")
ax3.set_xlabel("Distance (m)")
ax3.set_ylabel("Electric Field (V/m)")

# ax4.plot(x,p[:,0], color='blue', label='initial')
# ax4.plot(x,p[:,100], color='green')
# ax4.plot(x,p[:,500], color='purple')
# ax4.plot(x,n[:,-1], color='red', label='end')
# ax4.set_title("Charge Density")
# ax4.set_xlabel("Distance (m)")
# ax4.set_ylabel("Charge Density (m^-3)")
ax1.legend()
plt.show()