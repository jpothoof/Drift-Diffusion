import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from fipy import Variable, FaceVariable, CellVariable, Grid1D, ExplicitDiffusionTerm, TransientTerm, DiffusionTerm, Viewer, ImplicitSourceTerm, ConvectionTerm
from fipy.tools import numerix

##########################################################
# ################''' SET-UP PARAMETERS '''###############
##########################################################

# a1,b1,c1,a2,b2,c2 = [1.07114255e+00,  6.50014631e+05, -4.51527221e+00,  1.04633414e+00,
#  1.99708312e+05, -1.52479293e+00]
# Parameters for sum of sines fit (Potential fit)

# a = -3930.03590805
# b, c = 3049.38274411, -4.01434474
# Parameters for exponential fit (Charge Density)  Not used yet

q = 1.602e-19                # Elementary Charge

pini = 154.1581560721245/q   # m^-3

nini = -134.95618729/q       # m^-3


k1 = 1.8

p1 = 17

k2 = 17

p2 = 1.8
# Parameters for charge density fit (Susi's fit)

l = 1e-6 # Length of system in m

nx = 100                  # Number of cells in system

dx = l/nx                 # Length of each cell in m

x = np.linspace(0,l,nx)   # Array to calculate initial values in functions below


epsilon_r = 25                # Relative permittivity of system
epsilon_0 = 8.854e-12
epsilon = epsilon_r*epsilon_0 # Permittivity of system  C/V*m

kb = 1.38e-23                 # J/K

T = 298                       # K

f = kb*T/q                    # Volts

mu_n = 1.1e-09/10000          # m^2/V*s

mu_p = 1.1e-09/10000          # m^2/V*s

Dn = f * mu_n                 # m^2/s

Dp = f * mu_p                 # m^2/s

# k_rec = q*(mu_n+mu_p)/(2*epsilon)*10
k_rec = 0

# pini*np.exp(a*x)
# nini*np.exp(b*x+c)           # Equations for exponential charge density fits (Not Used Yet)





##################################################################
# #############''' INITIAL CONDITION FUNCTIONS '''################
##################################################################
sigma = 1
u = 0.5*l
def y01(x):
    """Initial positive ion charge density"""
    # return pini*((special.gamma(k1+p1))/(special.gamma(k1)*special.gamma(p1))*((x/l)**(k1-1))*(1-(x/l))**(p1-1))/7.3572
    return 1/100*x*epsilon*1/q*1e21
    # return 1e13*1/sigma*np.sqrt(2*np.pi) * np.exp(-np.power(x - u, 2.) / (2 * np.power(sigma, 2.)))-1e15*1/sigma

# def y02(x):
#    """"Initial negative ion charge density"""
#    return nini*((special.gamma(k2+p2))/(special.gamma(k2)*special.gamma(p2))*((x/l)**(k2-1))*(1-(x/l))**(p2-1))/7.3572

def y03(x):
    """Initial potential"""
#    return a1*np.sin(b1*x+c1) + a2*np.sin(b2*x+c2)
    return -1/600*1e21*x**3
    # return 1e13*(np.sqrt(2 / np.pi) / np.exp(0.5 * (-u + x) ** 2) + (-u + x) * special.erf((-u + x) / np.sqrt(2)))-7978845608028.654

#print(np.min(y03(x)))
plt.plot(x,y01(x))
plt.show()

plt.plot(x,y03(x))
plt.show()

mesh = Grid1D(dx=dx, nx=nx) # Establish mesh in how many dimensions necessary





##############################################################################
# ################''' SETUP CELLVARIABLES AND EQUATIONS '''####################
##############################################################################

# CellVariable - defines the variables that you want to solve for:

'''Initial value can be established when defining the variable, or later using 'var.value =' 
   Value defaults to zero if not defined'''

# sigma = 1
# u = 0.5*l

Pion = CellVariable(mesh=mesh, name='Positive ion Charge Density', value=y01(x))

# Nion = CellVariable(mesh=mesh, name='Negative ion Charge Density', value=y02(x))

potential = CellVariable(mesh=mesh, name='Potential', value=0.)

# EQUATION SETUP BASIC DESCRIPTION
'''Equations to solve for each variable must be defined:
  -TransientTerm = dvar/dt
  -ConvectionTerm = dvar/dx
  -DiffusionTerm = d^2var/dx^2
  -Source terms can be described as they would appear mathematically
Notes:  coeff = terms that are multiplied by the Term.. must be rank-1 FaceVariable for ConvectionTerm
        "var" must be defined for each Term if they are not all the variable being solved for,
        otherwise will see "fipy.terms.ExplicitVariableError: Terms with explicit Variables cannot mix with Terms with implicit Variables." '''

# In English:  dPion/dt = -1/q * divergence.Jp(x,t) - k_rec * Nion(x,t) * Pion(x,t) where
#             Jp = q * mu_p * E(x,t) * Pion(x,t) - q * Dp * grad.Pion(x,t)         and     E(x,t) = -grad.potential(x,t)
# Continuity Equation

# Pion.equation = TransientTerm(coeff=1, var=Pion) == mu_p * (ConvectionTerm(coeff=potential.faceGrad,var=Pion) + Pion * potential.faceGrad.divergence) + DiffusionTerm(coeff=Dp,var=Pion) - k_rec*Pion*Nion
Pion.equation = TransientTerm(coeff=1, var=Pion) == mu_p * (ConvectionTerm(coeff=potential.faceGrad, var=Pion) + Pion * potential.faceGrad.divergence)

# In English:  dNion/dt = 1/q * divergence.Jn(x,t) - k_rec * Nion(x,t) * Pion(x,t)   where
#             Jn = q * mu_n * E(x,t) * Nion(x,t) - q * Dn * grad.Nion(x,t)         and     E(x,t) = -grad.potential(x,t)
# Continuity Equation

# Nion.equation = TransientTerm(coeff=1, var=Nion) == -mu_n * (ConvectionTerm(coeff=potential.faceGrad,var=Nion) + Nion * potential.faceGrad.divergence) + DiffusionTerm(coeff=Dn,var=Nion) - k_rec*Pion*Nion


# In English:  d^2potential/dx^2 = -q/epsilon * Charge_Density      and     Charge Density = Pion + Nion
# Poisson's Equation

# potential.equation = DiffusionTerm(coeff=1, var=potential) == (-q/epsilon)*(Pion + Nion)
potential.equation = DiffusionTerm(coeff=1, var=potential) == (-q/epsilon)*Pion


################################################################
# #################''' BOUNDARY CONDITIONS '''###################
################################################################

Pion.faceGrad.constrain(0., where=mesh.exteriorFaces)  # dPion/dx = 0 at the exterior faces of the mesh
# Nion.faceGrad.constrain(0., where=mesh.exteriorFaces)  # dNion/dx = 0 at the exterior faces of the mesh
potential.constrain(0., where=mesh.facesLeft)      # potential = 0 at the exterior faces of the mesh
# Ef.constrain(0., where=mesh.exteriorFaces)





################################################################
# ################''' SOLVE EQUATIONS '''########################
################################################################

# eq = Pion.equation & Nion.equation & potential.equation  #Couple all of the equations together
eq = Pion.equation & potential.equation

steps = 100  # How many time steps to take
dt = 0.01      # How long each time step is in seconds

if __name__ == "__main__":
    # viewer = Viewer(vars=(potential,))  #Sets up viewer for the potential with y-axis limits
     viewer = Viewer(vars=(Pion,),datamin=0,datamax=2e22)        # Sets up viewer for negative ion density with y-axis limits
    # viewer = Viewer(vars=(Nion,),datamin=-1e21,datamax=0)       # Sets up viewer for positive ion density  with y-axis limits

for steps in range(steps):   # Time loop to step through
    eq.solve(dt=dt)          # Solves all coupled equation with timestep dt

    if __name__ == '__main__':
        viewer.plot()        # Plots results using matplotlib
        plt.pause(1)         # Pauses each frame for n amount of time
        plt.autoscale()      # Autoscale axes if necessary