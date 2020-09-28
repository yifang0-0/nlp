from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
 
def func(x, a, b, c):
    return a * np.exp(-b * x) + c
 
xdata = [1,2,3,4]
ydata= [6140, 1994, 979, 648]
plt.plot(xdata,ydata,'b-')
popt, pcov = curve_fit(func, xdata, ydata)
#popt数组中，三个值分别是待求参数a,b,c
x=np.linspace(1,4,20)
y2 = [func(i, popt[0],popt[1],popt[2]) for i in x]
plt.plot(x,y2,'r--')
print(popt)
plt.show()