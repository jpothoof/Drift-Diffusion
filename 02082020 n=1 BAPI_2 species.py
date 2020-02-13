import numpy as np
import matplotlib.pyplot as plt
from fipy import Variable, FaceVariable, CellVariable, Grid1D, ExplicitDiffusionTerm, TransientTerm, DiffusionTerm, \
    Viewer, ImplicitSourceTerm, ConvectionTerm
from scipy import special
import pandas as pd
l = 14e-6  # Length of system in m
nx = 500  # Number of cells in system
dx = l / nx  # Length of each cell in m
x = np.linspace(0, l, nx)  # Array to calculate initial values in functions below
q = 1.602e-19  # Elementary Charge
epsilon_r = 6  # Relative permittivity of system
epsilon = epsilon_r * 8.854e-12  # Permittivity of system  C/V*m
kb = 1.38e-23  # J/K
# kb=8.61e-5                   #eV/K
T = 298  # K
f = kb * T / q  # Volts
mu_n_1 = 9e-14  # m^2/V*s
mu_p_1 = 9e-14  # m^2/V*s
# mu_n_2 = 3e-14  # m^2/V*s
# mu_p_2 = 1e-13  # m^2/V*s
Dn_1 =f * mu_n_1  # m^2/s
Dp_1 = f * mu_p_1  # m^2/s
# Dn_2 =f * mu_n_2
# Dp_2 = f * mu_p_2
k_rec = q*(mu_n_1+mu_p_1)/(2*epsilon)
# k_rec = 0
##### The first type of functions######
# y01 = np.zeros(nx)
# y01[0:nx] = 2.5e21 # Positive ion density
# y02 = np.zeros(nx)
# y02[0:nx] = 2.5e21 # Negative ion density
# y03 = np.zeros(nx)
# y03[0:nx] = 2.5e21 # Electron density
# y04 = np.zeros(nx)
# y04[0:nx] = 2.5e21 # Hole density

pini = 4 * 1e+20 #C/m^3
nini = 4 * 1e+20 #C/m^3
k1 = 2.0
p1 = 19
k2 = 19
p2 = 2.0

def y01(x):
    """Initial positive ion charge density"""
    return pini*((special.gamma(k1+p1))/(special.gamma(k1)*special.gamma(p1))*((x/l)**(k1-1))*(1-(x/l))**(p1-1))/7.3572

def y02(x):
    """"Initial negative ion charge density"""
    return nini*((special.gamma(k2+p2))/(special.gamma(k2)*special.gamma(p2))*((x/l)**(k2-1))*(1-(x/l))**(p2-1))/7.3572

# def y03(x):
#     """Initial positive ion charge density"""
#     return pini*((special.gamma(k1+p1))/(special.gamma(k1)*special.gamma(p1))*((x/l)**(k1-1))*(1-(x/l))**(p1-1))/7.3572
#
# def y04(x):
#     """"Initial negative ion charge density"""
#     return nini*((special.gamma(k2+p2))/(special.gamma(k2)*special.gamma(p2))*((x/l)**(k2-1))*(1-(x/l))**(p2-1))/7.3572
# plt.plot(x, y01(x))
# plt.plot(x, y03(x))
# plt.show()

mesh = Grid1D(dx=dx, nx=nx)  # Establish mesh in how many dimensions necessary
Pion = CellVariable(mesh=mesh, name='Positive ion Charge Density', value=y01(x))
Nion = CellVariable(mesh=mesh, name='Negative ion Charge Density', value=y02(x))
# Hole = CellVariable(mesh=mesh, name='Hole Density',value=y03(x))
# Electron = CellVariable(mesh=mesh, name='Electron Density',value=y04(x))
potential = CellVariable(mesh=mesh, name='Potential')

#### Equations set-up ####
# In English:  dPion/dt = -1/q * divergence.Jp(x,t) - k_rec * Nion(x,t) * Pion(x,t) where
#             Jp = q * mu_p * E(x,t) * Pion(x,t) - q * Dp * grad.Pion(x,t)         and     E(x,t) = -grad.potential(x,t)
# Continuity Equation
Pion_equation = TransientTerm(coeff=1., var=Pion) == mu_p_1 * ConvectionTerm(coeff=potential.faceGrad, var=Pion) + Dp_1 * DiffusionTerm(coeff=1., var=Pion) - k_rec * Pion * Nion
# In English:  dNion/dt = 1/q * divergence.Jn(x,t) - k_rec * Nion(x,t) * Pion(x,t)   where
#             Jn = q * mu_n * E(x,t) * Nion(x,t) - q * Dn * grad.Nion(x,t)         and     E(x,t) = -grad.potential(x,t)
# Continuity Equation
Nion_equation = TransientTerm(coeff=1., var=Nion) == -mu_n_1 * ConvectionTerm(coeff=potential.faceGrad, var=Nion) + Dn_1 * DiffusionTerm(coeff=1., var=Nion) - k_rec * Pion * Nion
# Electron_equation = TransientTerm(coeff=1., var=Electron) == -mu_n_2 * ConvectionTerm(coeff=potential.faceGrad, var=Electron) + Dn_2 * DiffusionTerm(coeff=1., var=Electron) - k_rec * Electron * Hole
# Hole_equation = TransientTerm(coeff=1., var=Hole) == mu_p_2 * ConvectionTerm(coeff=potential.faceGrad, var=Hole) + Dp_2 * DiffusionTerm(coeff=1., var=Hole) - k_rec * Electron * Hole
# In English:  d^2potential/dx^2 = -q/epsilon * Charge_Density      and     Charge Density= Pion-Nion
# Poisson's Equation
potential_equation = DiffusionTerm(coeff=1., var=potential) == -(q / epsilon) * (Pion - Nion)
# potential_equation = DiffusionTerm(coeff=1., var=potential) == -(q / epsilon) * (Pion - Nion + Hole - Electron)
### Boundary conditions ###
# Fipy is defaulted to be no-flux, so we only need to constrain potential

potential.constrain(0., where=mesh.exteriorFaces)

# potential.constrain(9., where=mesh.facesLeft)
# potential.constrain(0., where=mesh.facesRight)
# potential.constrain(0., where=mesh.exteriorFaces)
### Solve Equations in a coupled manner ###
eq = Pion_equation & Nion_equation & potential_equation


steps = 2000
timestep = 0.2

Efield_save = np.empty([nx, steps])
potential_save = np.empty_like(Efield_save)
# Pion_save = np.empty_like(Efield_save)
# Nion_save = np.empty_like(Efield_save)

for step in range(steps):
    eq.solve(dt=timestep)
    #
    if (step % 2000) == 0:
        fig, (ax1, ax2, ax3,ax4) = plt.subplots(4, 1, figsize=(6, 12))
        ax1.set_ylabel('Electric Field', c='grey')
        ax1.tick_params(axis='y', labelcolor='grey')
        ax1.ticklabel_format(axis='y', style='sci', scilimits=(0, 1))
        ax1.axhline(y=0)
        ax1_2 = ax1.twinx()
        ax1_2.ticklabel_format(axis='y', style='sci', scilimits=(0, 1))
        ax1_2.set_ylabel('Potential', c='tab:green')
        ax1_2.tick_params(axis='y', labelcolor='tab:green')
        ax2.set_ylabel('Concentration')
# for l= 10e-6
        ax1.set_ylim([-4e6,4e6])
        ax1_2.set_ylim(-8,8)
        ax2.set_ylim([0, 2e23])
        ax3.set_ylim([0,2e23])
        ax3.set_ylabel("Nion_1 at x=0")
        ax4.set_ylabel("Potential_1 at x=0")

        ax1.axhline(y=0)

    Efield = -potential.grad()[0]
    l1 = ax1.plot(x, Efield, c='grey')
    l2 = ax1_2.plot(x, potential.value, c='tab:green')
    l3 = ax2.plot(x, Pion.value,label='Positive_1',  c='tab:red')
    l4 = ax2.plot(x, Nion.value,label='Negative_1',  c='tab:blue')
    # l5 = ax3.plot(x, Hole.value, label='Positive_2', c='red')
    # l6 = ax3.plot(x, Electron.value, label='Negative_2', c='green')



    # ax3.plot(x, Electron.value,label='Electron', c='tab:blue')
    # ax3.plot(x, Hole.value,label="Hole", c='tab:red')
    ax3.scatter(step, Nion.value[0], c='r', s=1)
    # ax4.scatter(step, Hole.value[0], c='b', s=1)

    Efield_save[:, step] = Efield
    potential_save[:, step] = potential.value
    # Pion_save[:, step] = Pion.value
    # Nion_save[:, step] = Nion.value


    ax2.legend(loc="upper right")
    fig.suptitle('Time: ' + str(step*timestep)[:4])
    fig.show()
    plt.pause(0.00000000001)
    # print(np.sum(Nion.value))
    for l in [l1, l2, l3, l4]:
        l[0].remove()

# np.savetxt('Pion_1.txt', Pion_save, delimiter='\t')
# np.savetxt('Nion_1.txt', Nion_save, delimiter='\t')
# np.savetxt('Electrical field intensity_1.txt',Efield_save, delimiter='\t')
np.savetxt('BAPI_potential_two_species_6.txt', potential_save, delimiter='\t')