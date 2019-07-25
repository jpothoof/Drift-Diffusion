import numpy as np
import matplotlib.pyplot as plt
from scipy import special

a1,b1,c1,a2,b2,c2 = [ 1.04633244e+00,  1.99709309e+03, -1.52480044e+00,  1.07114122e+00,
  6.50014924e+03, -4.51527387e+00]
# Parameters for sum of sines fit

pini = 9.634885e14    #Initial peak positive charge density
nini = -8.434762e14   #Initial peak negative charge density
k1 = 1.8
p1 = 17               #Parameters to fit charge density equations
k2 = 17
p2 = 1.8
l = 0.00134901960784314

x = np.linspace(0,l,134)

y01 = pini*((special.gamma(k1+p1))/(special.gamma(k1)*special.gamma(p1))*((x/l)**(k1-1))*(1-(x/l))**(p1-1))/7.3572
# Initial positive ion charge density

y02 = nini*((special.gamma(k2+p2))/(special.gamma(k2)*special.gamma(p2))*((x/l)**(k2-1))*(1-(x/l))**(p2-1))/7.3572
# Initial negative ion charge density

y03 = a1*np.sin(b1*x+c1) + a2*np.sin(b2*x+c2)
# Initial potential

plt.figure(1)
plt.subplot(2,1,1)
plt.plot(x,y03)
plt.xlabel('Position / cm')
plt.ylabel('Potential / V')
plt.title('Potential Fit')

plt.subplot(2,2,3)
plt.plot(x,y01)
plt.xlabel('Position / cm')
plt.ylabel('Charge Density / cm^-3')
plt.title('Positive Charge Density Fit',y=1.08)

plt.subplot(2,2,4)
plt.plot(x,y02)
plt.xlabel('Position / cm')
plt.ylabel('Charge Density / cm^-3')
plt.title('Negative Charge Density Fit',y=1.08)
plt.subplots_adjust(wspace=0.5,hspace=0.6)

plt.show()