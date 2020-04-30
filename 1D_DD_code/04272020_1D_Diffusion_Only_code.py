import numpy as np
import matplotlib.pyplot as plt

steps = 50000
N = 500
l = 14e-6
dx = l/N
x = np.linspace(0,l,N)
x2 = np.linspace(0,l,N+1)

q = 1.602e-19  # C
epsilon = 25 * 8.854e-12  # C/V*m
kb = 1.38e-23  # J/K
T = 298  # K
Dp = 1e-14  # m^2/s

p = np.zeros([N, steps])
Jp = np.zeros([N+1, steps])

p[240:260,0] = 1e20  # m^-3

for j in range(N+1):
    if j == 0:
        pass
    elif j == N:
        pass
    else:
        Jp[j, 0] = -Dp / dx * (p[j, 0] - p[j-1, 0])

# fig, (ax1, ax2) = plt.subplots(1,2, figsize=[12,6])
# ax1.plot(x, p[:,0])
# ax2.plot(x2, Jp[:,0])
# plt.show()

dt = dx**2 / (2*Dp)

for k in range(steps):
    if k == 0:
        pass

    else:
        for i in range(N+1):
            if i == 0:
                p[i,k] = -dt/dx * (Jp[i+1, k-1]-Jp[i, k-1]) + p[i, k-1]
            elif i == N:
                pass
            else:
                p[i,k] = -dt/dx * (Jp[i+1, k-1]-Jp[i, k-1]) + p[i, k-1]
                Jp[i, k] = -Dp/dx * (p[i, k] - p[i-1, k])

print(np.sum(p[:,0]))
print(np.sum(p[:,-1]))
# print(Jp[:,-1])

fig, (ax1, ax2) = plt.subplots(1,2, figsize=[12,6])
ax1.plot(x,p[:,0], color='blue', label='initial')
ax1.plot(x,p[:,10000], color='green')
ax1.plot(x,p[:,-1], color='red', label='end')
ax1.set_title("Charge Density")
ax1.set_xlabel("Distance (m)")
ax1.set_ylabel("Charge Density (m^-3)")

ax2.plot(x2,Jp[:,0], color='blue')
ax2.plot(x2,Jp[:,10000], color='green')
ax2.plot(x2,Jp[:,-1], color='red')
ax2.set_title("Current Density")
ax2.set_xlabel("Distance (m)")
ax2.set_ylabel("Current Density (A/m^2)")
ax1.legend()
plt.show()