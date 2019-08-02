import numpy as np
import matplotlib.pyplot as plt
from scipy import special
import fipy as fp
from fipy import Variable, FaceVariable, CellVariable, Grid1D, ExplicitDiffusionTerm, TransientTerm, DiffusionTerm, Viewer, ImplicitSourceTerm, ConvectionTerm
from fipy.tools import numerix

a1,b1,c1,a2,b2,c2 = [ 1.04633244e+00,  1.99709309e+03, -1.52480044e+00,  1.07114122e+00,
  6.50014924e+03, -4.51527387e+00]
# Parameters for sum of sines fit

pini = 154.1581560721245 #C/m^3
nini = -134.95618729 #C/m^3
a = -3930.03590805
b, c = 3049.38274411, -4.01434474
k1 = 1.8
p1 = 17
k2 = 17
p2 = 1.8

l = 0.00134901960784314 #Length of system

nx = 134
dx = l/nx
x = np.linspace(0,l,134)

q = 1.602e-19  #Elementary Charge
epsilon_r = 25 #Relative permittivity of system
epsilon = epsilon_r*8.854e-12 #Permittivity of system  C/V*m
kb = 8.617e-5 #eV/K
T = 303.3 #K
f = kb*T/q #Volts
mu_n = 1.1e-09/10000 #m^2/V*s
mu_p = 1.1e-09/10000 #m^2/V*s
Dn = f * mu_n #m^2/s
Dp = f * mu_p #m^2/s
k_rec = q*(mu_n+mu_p)/(2*epsilon)*10
#pini*np.exp(a*x)
#nini*np.exp(b*x+c)

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

Pion = CellVariable(mesh=mesh, name='Positive ion Charge Density', value=y01(x))
Nion = CellVariable(mesh=mesh, name='Negative ion Charge Density', value=y02(x))
potential = CellVariable(mesh=mesh, name='Potential', value=y03(x))
print(potential.value)

#Jp = -mu_p * Pion.harmonicFaceValue * potential.faceGrad + Dp * Pion.faceGrad
#Jn = -mu_n * Nion.harmonicFaceValue * potential.faceGrad + Dn * Nion.faceGrad


Pion.equation = TransientTerm(coeff=1, var=Pion) == -mu_p * (ConvectionTerm(coeff=potential.faceGrad,var=Pion) + Pion *
                potential.faceGrad.divergence) + DiffusionTerm(coeff=Dp,var=Pion) - k_rec*Pion*Nion

Nion.equation = TransientTerm(coeff=1, var=Nion) == mu_n  * (ConvectionTerm(coeff=potential.faceGrad,var=Nion) + Nion *
                potential.faceGrad.divergence) + DiffusionTerm(coeff=Dn,var=Nion) - k_rec*Pion*Nion

potential.equation = DiffusionTerm(coeff=1, var=potential) == (-q/epsilon)*(Pion + Nion)

Pion.constrain(0., where=mesh.exteriorFaces)
Nion.constrain(0., where=mesh.exteriorFaces)
potential.constrain(0., where=mesh.exteriorFaces)
#Jp.constrain(0., where=mesh.exteriorFaces)
#Jn.constrain(0., where=mesh.exteriorFaces)

eq = Pion.equation & Nion.equation & potential.equation

steps = 253
dt = 0.0000000000001

if __name__ == "__main__":
    viewer = Viewer(vars=(potential,))

for steps in range(steps):
    eq.solve(dt=dt)

    if __name__ == '__main__':
        viewer.plot()
        plt.pause(1)
        plt.autoscale()