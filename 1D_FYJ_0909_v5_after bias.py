import numpy as np
import matplotlib.pyplot as plt
from fipy import Variable, FaceVariable, CellVariable, Grid1D, ExplicitDiffusionTerm, TransientTerm, DiffusionTerm, \
    Viewer, ImplicitSourceTerm, ConvectionTerm
from scipy import special
import pandas as pd

l =1e-6  # Length of system in m
nx = 100  # Number of cells in system
dx = l / nx  # Length of each cell in m
x = np.linspace(0, l, nx)  # Array to calculate initial values in functions below

q = 1.602e-19  # Elementary Charge
epsilon_r = 25  # Relative permittivity of system
epsilon = epsilon_r * 8.854e-12  # Permittivity of system  C/V*m
kb = 1.38e-23  # J/K
# kb=8.61e-5                   #eV/K
T = 298  # K
f = kb * T / q  # Volts
mu_n = 1.1e-09 / 10000  # m^2/V*s
mu_p = 1.1e-09 / 10000  # m^2/V*s
Dn = f * mu_n  # m^2/s
Dp = f * mu_p  # m^2/s

# k_rec = q*(mu_n+mu_p)/(2*epsilon)
k_rec = 0.

##### The first type of functions######
y01 = np.zeros(nx)
y01[0:10] = 1e21

y02 = np.zeros(nx)
y02[90:100] = 1e21


#### Using numpy to read-out the data####
# Pion_data = np.genfromtxt('Pion.txt')
# choose_index = 500        ### Choose a specific line as the initial carrier distribution
# P_initial = Pion_data[:,choose_index]
#
# Nion_data = np.genfromtxt('Nion.txt')
# N_initial = Nion_data[:,choose_index]


mesh = Grid1D(dx=dx, nx=nx)  # Establish mesh in how many dimensions necessary


Pion = CellVariable(mesh=mesh, name='Positive ion Charge Density', value=y01)
Nion = CellVariable(mesh=mesh, name='Negative ion Charge Density', value=y02)
potential = CellVariable(mesh=mesh, name='Potential')


#### Equations set-up ####

# In English:  dPion/dt = -1/q * divergence.Jp(x,t) - k_rec * Nion(x,t) * Pion(x,t) where
#             Jp = q * mu_p * E(x,t) * Pion(x,t) - q * Dp * grad.Pion(x,t)         and     E(x,t) = -grad.potential(x,t)
# Continuity Equation
Pion_equation = TransientTerm(coeff=1., var=Pion) == mu_p * ConvectionTerm(coeff=potential.faceGrad, var=Pion) + Dp * DiffusionTerm(coeff=1., var=Pion) - k_rec * Pion * Nion

# In English:  dNion/dt = 1/q * divergence.Jn(x,t) - k_rec * Nion(x,t) * Pion(x,t)   where
#             Jn = q * mu_n * E(x,t) * Nion(x,t) - q * Dn * grad.Nion(x,t)         and     E(x,t) = -grad.potential(x,t)
# Continuity Equation
Nion_equation = TransientTerm(coeff=1., var=Nion) == -mu_n * ConvectionTerm(coeff=potential.faceGrad, var=Nion) + Dn * DiffusionTerm(coeff=1., var=Nion) - k_rec * Pion * Nion

# In English:  d^2potential/dx^2 = -q/epsilon * Charge_Density      and     Charge Density= Pion-Nion
# Poisson's Equation
potential_equation = DiffusionTerm(coeff=1., var=potential) == -(q / epsilon) * (Pion - Nion)

### Boundary conditions ###
# Fipy is defaulted to be no-flux, so we only need to constrain potential
potential.constrain(0., where=mesh.exteriorFaces)

### Solve Equations in a coupled manner ###
eq = Pion_equation & Nion_equation & potential_equation

steps = 1000
timestep = 0.1

#create empty matrix to save data
potential_grad_save = np.empty([nx,steps])
potential_save = np.empty_like(potential_grad_save)
Pion_save = np.empty_like(potential_grad_save)
Nion_save = np.empty_like(potential_grad_save)



for step in range(steps):
    eq.solve(dt=timestep)

    if (step % 1000) == 0:

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))    ### Create 3 subplots, ax1: x--E(x,t), ax1_2 : x--V(x,t),  ax2: x--Pion/Nion(x,t),  ax3: t--Pion(x=0, t)
        ax1.set_ylabel('Electric Field intensity', c='grey')
        ax1.tick_params(axis='y', labelcolor='grey')
        ax1.ticklabel_format(axis='y', style='sci', scilimits=(0, 1))
        ax1.axhline(y=0)

        ax1_2 = ax1.twinx()
        ax1_2.ticklabel_format(axis='y', style='sci', scilimits=(0, 1))
        ax1_2.set_ylabel('Potential', c='tab:green')
        ax1_2.tick_params(axis='y', labelcolor='tab:green')

        ax2.set_ylabel('Concentration')
        ax3.set_ylabel("Pion at x=0")


# for l= 1e-6
        ax1.set_ylim([-1e5, 1e5])
        ax1_2.set_ylim(-0.02, 0.02)
        ax2.set_ylim([-1e10, 2e21])
#
# for l=10e-6
#         ax1.set_ylim([-1e6, 1e6])
#         ax1_2.set_ylim(-0.5, 0.5)
#         ax2.set_ylim([-1e10, 7e21])
#         ax3.set_xlim(0, 1000)

    l1 = ax1.plot(x, potential.grad()[0], c='grey')
    l2 = ax1_2.plot(x, potential.value, c='tab:green')
    l3 = ax2.plot(x, Pion.value,label='Positive',  c='tab:red')
    l4 = ax2.plot(x, Nion.value,label='Negative',  c='tab:blue')

    ax3.scatter(step, Pion.value[0], c='r', s=1)

    # potential_grad_save[:, step] = potential.grad()[0]
    # potential_save[:, step] = potential.value
    # Pion_save[:, step] = Pion.value
    # Nion_save[:, step] = Nion.value

    ax2.legend(loc="upper right")


    fig.suptitle('Step: ' + str(step))
    fig.show()
    plt.pause(0.05)

    for l in [l1, l2, l3, l4]:
        l[0].remove()

    # np.savetxt('Pion.txt', Pion_save, delimiter='\t')
    # np.savetxt('Nion.txt', Nion_save, delimiter='\t')








