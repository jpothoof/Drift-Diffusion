import matplotlib.pyplot as plt
import pandas as pd
from scipy import optimize
import numpy as np

LED_SKPM = pd.read_excel(r'C:\Users\justi\Desktop\Research 2019\3March\PEAPbI 3.22.19\3.22.19 PEAPbI LED.xlsx',sheet_name='Potential')
Position = pd.read_excel(r'C:\Users\justi\Desktop\Research 2019\3March\PEAPbI 3.22.19\3.22.19 PEAPbI Position.xlsx')

#print(LED_SKPM)
#print(Position)

Charging = LED_SKPM.iloc[:,3:45]
Discharging = LED_SKPM.iloc[:,46:]
Initial_potential = LED_SKPM.iloc[:,46]
#with pd.option_context('display.max_rows', None, 'display.max_columns', None):
  #print(Initial_potential)
initial_potential_junction = Initial_potential[65:199]
ipj_array = initial_potential_junction.values
position_junction = Position[65:199]
posj_array = position_junction.values
#print(initial_potential_junction)

x = np.linspace(0,0.00134901960784314,134)
#+ a3 * np.sin(b3 * x + c3).flatten()
def test_func(x, a1, b1,c1,a2,b2,c2):
  return a1 * np.sin(b1 * x + c1).flatten() + a2*np.sin(b2*x+c2).flatten()

params, params_covariance = optimize.curve_fit(test_func, x, ipj_array)

print(params)

a1,b1,c1,a2,b2,c2 = params

plt.figure(1)
#plt.plot(posj_array, ipj_array)
#plt.plot(posj_array,test_func(posj_array,a1,b1,c1,a2,b2,c2,a3,b3,c3))
plt.plot(x,ipj_array)
plt.plot(x,test_func(x,a1,b1,c1,a2,b2,c2))

plt.show()

#######################################################################################################################
