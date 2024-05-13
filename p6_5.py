# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 09:42:39 2024

@author: Juan
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from scipy import interpolate as inter
from scipy import stats as st
from puystats import linregress_w as linr
from scipy.optimize import minimize as mini
from scipy.optimize import newton as nwt

x=np.array([0.05,0.25,0.45,0.65,0.85,1.05,1.25,1.45,1.65,1.85])#Desplazamiento m
y=np.array([0.00,0.21,0.44,0.67,0.88,1.1,1.3,1.5,2.0,2.24])#Fase rad
dy=np.array([0.05,0.05,0.05,0.05,0.09,0.1,0.2,0.5,0.1,0.07])

#print(len(x),len(y),len(dy))

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

ax[0].set_title("Gráfica de fase contra desplazamiento")
ax[0].set_xlabel("Desplazamiento (m)")
ax[0].set_ylabel("Fase (rad)")
ax[0].grid()
ax[0].scatter(x,y, c="tomato")
ax[0].plot(x_p,f(x_p),c="k",linestyle="--",label="y = ("+str(round(lin_reg.slope(),2))+"$\pm$ "
        +str(round(lin_reg.err_slope(),2))+")x + ("+str(round(lin_reg.intercept(),2))+"$\pm$"+
         str(round(lin_reg.err_intercept(),2)) +")")
ax[0].legend(loc="upper left")
ax[0].errorbar(x, y, yerr= dy, ecolor="k",capsize=2,linestyle="")

print("m = "+str(lin_reg.slope()))
print("Error m:",lin_reg.err_slope())
print("c= "+str(lin_reg.intercept()))
print("Error c:",lin_reg.err_intercept())

res=(y-f(x))/dy

"""
aCu=np.sqrt(1/(len(y)-2)*np.sum((y-lin_reg.slope()*x-lin_reg.intercept())**2))
D=len(y)*np.sum(x**2)-(np.sum(x))**2
print(aCu)
print(aCu*np.sqrt(len(y)/D))
"""

"""
Gráfica residuales normalizados
"""

ax[1].scatter(x,res,c="red")
ax[1].grid()
ax[1].set_xlabel("Desplazamiento (m)")
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
currentAxis.add_patch(Rectangle((0-1, 0-1), 2, 2, facecolor="none",ec="k",lw=2))
plt.scatter(res[:-1],res[1:])
plt.grid()
plt.title("Lag-Plot para los residuales normalizados")
plt.xlabel("Residual normalizado (i-1)")
plt.ylabel("Resdiual normalizado (i)")


