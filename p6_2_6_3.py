# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 19:28:00 2024

@author: Juan
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from scipy import interpolate as inter
from scipy import stats as st
from puystats import linregress_w as linr

x=np.array([10,20,30,40,50,60,70,80,90,100,110])#Frecuencia Hz
y=np.array([16,45,64,75,70,115,142,167,183,160,221])#Voltaje mV
dy=np.array([5,5,5,5,30,5,5,5,5,30,5])

"""
Regresión con pesos
"""

lin_reg = linr(x,y,dy)
f= lambda x: lin_reg.slope()*x+lin_reg.intercept()



x_p=np.linspace(min(x),max(x),500)

fig, ax= plt.subplots(2,1, figsize=(6,8))

"""
Gráfica regresión con pesos
"""

ax[0].set_title("Gráfica de frecuencia contra voltaje")
ax[0].set_xlabel("Frecuencia (Hz)")
ax[0].set_ylabel("Voltaje (mV)")
ax[0].grid()
ax[0].scatter(x,y, c="tomato")
ax[0].plot(x_p,f(x_p),c="k",linestyle="--",label="y = ("+str(round(lin_reg.slope(),2))+"$\pm$ "
        +str(round(lin_reg.err_slope(),2))+")x + ("+str(round(lin_reg.intercept(),0))+"$\pm$"+
         str(round(lin_reg.err_intercept(),0)) +")")
ax[0].legend(loc="upper left")
ax[0].errorbar(x, y, yerr= dy, ecolor="k",capsize=2,linestyle="")

print(lin_reg.slope())
print("Error m de y vs x:",lin_reg.err_slope())
print(lin_reg.intercept())
print("Error b de y vs x:",lin_reg.err_intercept())

res=(y-f(x))/dy

aCu=np.sqrt(1/(len(y)-2)*np.sum((y-lin_reg.slope()*x-lin_reg.intercept())**2))
D=len(y)*np.sum(x**2)-(np.sum(x))**2
print(aCu)
print(aCu*np.sqrt(len(y)/D))

"""
Gráfica residuales normalizados
"""

ax[1].scatter(x,res,c="red")
ax[1].grid()
ax[1].set_xlabel("Frecuencia (Hz)")
ax[1].set_ylabel("Residuales $R_i$ ")
ax[1].set_title("Gráfica de residuales normalizados")
ax[1].legend()
plt.tight_layout()

print(np.average(res))


"""
Drubin-Watson
"""
Ri=y-f(x)
Drb_Wts=np.sum((Ri[1:]-Ri[:-1])**2)/np.sum(Ri**2)

print("Durbin-Watson="+str(Drb_Wts))

plt.figure(2)
currentAxis = plt.gca()
currentAxis.add_patch(Rectangle((0-2, 0-2), 4, 4, facecolor="none",ec="k",lw=2))
plt.scatter(res[:-1],res[1:])
plt.grid()
plt.title("Lag-Plot para los residuales normalizados")
plt.xlabel("Residual normalizado (i-1)")
plt.ylabel("Resdiual normalizado (i)")


