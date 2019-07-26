import numpy as np
import matplotlib.pyplot as plt
from scipy import special
import fipy as fp
from fipy import Variable, FaceVariable, CellVariable, Grid1D, ExplicitDiffusionTerm, TransientTerm, DiffusionTerm, Viewer, ImplicitSourceTerm, ConvectionTerm
from fipy.tools import numerix

a1,b1,c1,a2,b2,c2 = [ 1.04633244e+00,  1.99709309e+03, -1.52480044e+00,  1.07114122e+00,
  6.50014924e+03, -4.51527387e+00]
# Parameters for sum of sines fit

pini = 9.634885e14    #Initial peak positive charge density
nini = -8.434762e14   #Initial peak negative charge density
k1 = 1.8
p1 = 17               #Parameters to fit charge density equations
k2 = 17
p2 = 1.8
l = 0.00134901960784314 #Length of system

nx = 134
dx = l/nx
#x = np.linspace(0,l,nx)

q = 1.602e-19  #Elementary Charge
epsilon_r = 25 #Relative permittivity of system
epsilon = epsilon_r*8.854e-14 #Permittivity of system
kb = 1.38E-21
T = 303.3
f = kb*T/q
mu_n = 1.1e-09
mu_p = 1.1e-09
Dn = f * mu_n
Dp = f * mu_p
k_rec = q*(mu_n+mu_p)/(2*epsilon)*10

#k_rec = 1 #Ion recombination term, actually q*(mu_n+mu_p)/(2*epsilon)
#mu_p = 1  #Mobility of positive ion species
#mu_n = 1  #Mobility of negative ion species
#Dp = 1    #Diffusion coefficient of positive ion species
#Dn = 1    #Diffusion coefficient of negative ion species

def y01(x):
    """Initial positive ion charge density"""
    return pini*((special.gamma(k1+p1))/(special.gamma(k1)*special.gamma(p1))*((x/l)**(k1-1))*(1-(x/l))**(p1-1))/7.3572

def y02(x):
    """"Initial negative ion charge density"""
    return nini*((special.gamma(k2+p2))/(special.gamma(k2)*special.gamma(p2))*((x/l)**(k2-1))*(1-(x/l))**(p2-1))/7.3572

def y03(x):
    """Initial potential"""
    return a1*np.sin(b1*x+c1) + a2*np.sin(b2*x+c2)

mesh = Grid1D(dx=dx, nx=nx)

Pion = CellVariable(mesh=mesh, name='Positive ion Charge Density', value=y01(mesh.x),hasOld=True)
Nion = CellVariable(mesh=mesh, name='Negative ion Charge Density', value=y02(mesh.x),hasOld=True)
potential = CellVariable(mesh=mesh, name='Potential', value=y03(mesh.x),hasOld=True)


Jp.value = -mu_p * Pion.harmonicFaceValue * potential.faceGrad + Dp * Pion.faceGrad
Jn.value = -mu_n * Nion.harmonicFaceValue * potential.faceGrad + Dn * Pion.faceGrad

Pion.equation = TransientTerm(coeff=1, var=Pion) == -numerix.gradient(Jp)
Nion.equation = TransientTerm(coeff=1, var=Nion) == numerix.gradient(Jn)
potential.equation = DiffusionTerm(coeff=epsilon, var=potential) == Pion - Nion

Pion.constrain(0., where=mesh.facesLeft)
Pion.constrain(0., where=mesh.facesRight)
Nion.constrain(0., where=mesh.facesLeft)
Nion.constrain(0., where=mesh.facesRight)
potential.constrain(0., where=mesh.facesLeft)
potential.constrain(0., where=mesh.facesRight)

eq = Pion.equation & Nion.equation & potential.equation

steps = 253
dt = 0.1

if __name__ == "__main__":
    viewer = Viewer(vars=(Nion,))

for steps in range(steps):
    eq.solve(dt=dt)

    if __name__ == '__main__':
        viewer.plot()
        plt.pause(1)
        plt.autoscale()