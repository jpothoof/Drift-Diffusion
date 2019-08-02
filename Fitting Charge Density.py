import matplotlib.pyplot as plt
import pandas as pd
from scipy import optimize
import numpy as np
from scipy import signal
from scipy import special

LED_SKPM = pd.read_excel(r'C:\Users\justi\Desktop\Research 2019\3March\PEAPbI 3.22.19\3.22.19 PEAPbI LED.xlsx',sheet_name='Potential')
Position = pd.read_excel(r'C:\Users\justi\Desktop\Research 2019\3March\PEAPbI 3.22.19\3.22.19 PEAPbI Position.xlsx')

#print(LED_SKPM)
#print(Position)

Charging = LED_SKPM.iloc[:,3:45]
Discharging = LED_SKPM.iloc[:,46:]
Initial_potential = LED_SKPM.iloc[:,46]
initial_potential_junction = Initial_potential[65:199]
ipj_array = initial_potential_junction.values
position_junction = Position[65:199]
posj_array = position_junction.values

#with pd.option_context('display.max_rows', None, 'display.max_columns', None):
 # print(posj_array)

ipj_array_smoothed = signal.savgol_filter(ipj_array,15,3)
#dy = np.gradient(ipj_array,np.arange(ipj_array.size))
#dy = signal.savgol_filter(dy,13,4)
#dy2 = np.gradient(dy,np.arange(dy.size))

dy = np.zeros(ipj_array_smoothed.shape)
range_ = range(134)
for nums in range_:
    dy[nums] = (ipj_array_smoothed[nums]-ipj_array_smoothed[nums-1])/(posj_array[nums]-posj_array[nums-1])
dy = signal.savgol_filter(dy,7,3)

dy2 = np.zeros(dy.shape)
for nums in range_:
    dy2[nums] = (dy[nums]-dy[nums-1])/(posj_array[nums]-posj_array[nums-1])

nperov = 25
nvac = 8.854187817e-12
q = 1.6e-19
dy2 = dy2*1e12*nperov*nvac
#plt.plot(posj_array,dy)
#plt.plot(posj_array,dy2)
#plt.show()
#pini1 = max(dy2)
#print(dy2)

df = pd.DataFrame(posj_array,dy2)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
  print(dy2)

pini = 9.634885e14
pini1 = 154.1581560721245
nini1 = -134.95618729
nini = -8.434762e14
k1 = 1.8
p1 = 17
k2 = 17
p2 = 1.8
l = 0.00134901960784314
x_p = np.linspace(0,l,51)
x_n = np.linspace(0,l,53)
x = np.linspace(0,l,134)

u01 = pini1*((special.gamma(k1+p1))/(special.gamma(k1)*special.gamma(p1))*((x/l)**(k1-1))*(1-(x/l))**(p1-1))/7.3572
u02 = nini1*((special.gamma(k2+p2))/(special.gamma(k2)*special.gamma(p2))*((x/l)**(k2-1))*(1-(x/l))**(p2-1))/7.3572
plt.plot(posj_array,u01)
plt.plot(posj_array,u02)
plt.show()
'''pdy2 = dy2[39:90]
ndy2 = dy2[52:105]

def model(x, a):
    return pini*np.exp(a*x)
def n_model(x,b,c):
    return nini*np.exp(b*x+c)

params, param_covariance = optimize.curve_fit(model, x_p, pdy2)
print(params)
a = params

n_params, n_param_covariance = optimize.curve_fit(n_model,x_n,ndy2)
print(n_params)
b, c = n_params

plt.figure(1)
plt.subplot(2,2,1)
plt.plot(x_p,pini*np.exp(a*x_p))
plt.plot(x_p,pdy2)
plt.title('Positive Charge Density fit', y=1.08)

plt.subplot(2,2,2)
plt.plot(x_n,ndy2)
plt.plot(x_n,nini*np.exp(b*x_n+c))
plt.title('Negative Charge Density fit', y=1.08)

plt.subplot(2,2,3)
plt.plot(x_p,pini*np.exp(a*x_p))
plt.plot(x_n,nini*np.exp(b*x_n+c))

plt.show()

#plt.plot(x,u01)
#plt.plot(x,u02)

#u01_max = np.amax(u01)
#print(u01_max)
#u02_min = np.amin(u02)
#print(u02_min)
#print(u01_max/pini)
#print(u02_min/nini)
#with pd.option_context('display.max_rows', None, 'display.max_columns', None):
  #print(dy2)'''