import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy import special
import fipy as fp
from fipy import Variable, FaceVariable, CellVariable, Grid1D, ExplicitDiffusionTerm, TransientTerm, DiffusionTerm, Viewer
from fipy.tools import numerix

a1,b1,c1,a2,b2,c2 = [ 1.04633244e+00,  1.99709309e+03, -1.52480044e+00,  1.07114122e+00,
  6.50014924e+03, -4.51527387e+00]
pini = 9.634885e14
nini = -8.434762e14
k1 = 1.8
p1 = 17
k2 = 17
p2 = 1.8
l = 0.00134901960784314
x = np.linspace(0,l,170)

n = 170
q = 1.602e-19
epsilon_r = 25
epsilon = epsilon_r*8.854e-14
kb = 1.38e-21
T = 303.3
f = kb*T/q
mu_n = 1.1e-9
mu_p = 1.1e-9
D_n = f * mu_n
D_p = f * mu_p
r_rec = q*(mu_n+mu_p)/(2*epsilon)*10


y01 = pini*((special.gamma(k1+p1))/(special.gamma(k1)*special.gamma(p1))*((x/l)**(k1-1))*(1-(x/l))**(p1-1))/7.3572
# Initial positive ion charge density
y02 = nini*((special.gamma(k2+p2))/(special.gamma(k2)*special.gamma(p2))*((x/l)**(k2-1))*(1-(x/l))**(p2-1))/7.3572
# Initial negative ion charge density
y03 = a1*np.sin(b1*x+c1) + a2*np.sin(b2*x+c2)
# Initial potential
y0 = [y01,y02,y03]
y0 = np.asarray(y0) #  Array containing initial values

dy01 = np.zeros(y01.shape)
dy02 = np.zeros(y02.shape)
dy03 = np.zeros(y03.shape)
range_ = range(170)
for nums in range_:
    dy01[nums] = (y01[nums] - y01[nums - 1]) / (x[nums] - x[nums - 1])
    dy02[nums] = (y02[nums] - y02[nums - 1]) / (x[nums] - x[nums - 1])
    dy03[nums] = (y03[nums]-y03[nums-1])/(x[nums]-x[nums-1])
Jp0 = q*mu_p*dy03*y01+q*D_p*dy01
Jn0 = q*mu_n*dy03*y02+q*D_n*dy02



Pp = np.ones(n)*y01  #Positive ion charge density
Pn = np.ones(n)*y02  #Negative ion charge density
V = np.ones(n)*y03   #Potential
Jp = np.ones(n)*Jp0  #Positive ion current density
Jn = np.ones(n)*Jn0  #Negative ion current desnity
dPp_dt = np.empty(n)
dPn_dt = np.empty(n)
d2V_dx2 = np.empty(n)
dJp_dx = np.empty(n)
dJn_dx = np.empty(n)



dx = l/n
x_ = np.linspace(dx/2, l*dx/2, n)
t_final = 1500
dt = 10
t = np.arange(0,t_final,dt)

for j in range(1,len(t)):
  for nums in range_:
    dJp_dx[nums] = (Jp[nums] - Jp[nums - 1]) / (x[nums] - x[nums - 1])
    dJn_dx[nums] = (Jn[nums] - Jn[nums - 1]) / (x[nums] - x[nums - 1])

    for i in range(1,n-1):
    dPp_dt[i] = -1/q*dJp_dx-r_rec*Pn*Pp
    dPn_dt[i] = 1/q*dJn_dx-r_rec*Pn*Pp
dPp_dt[0] = 0
dPp_dt[n-1] = 0
dPn_dt[0] = 0
dPn_dt[n-1] = 0
Pp = Pp + dPp_dt*dt
Pn = Pn + dPn_dt*dt