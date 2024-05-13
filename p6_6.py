# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 10:33:07 2024

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

"""
parte 1: Datos dados
"""


x=np.array([1,2,3,4,5,6,7,8,9,10])
y=np.array([51,103,150,199,251,303,347,398,452,512])
dy=np.array([1,1,2,2,3,3,4,5,6,7])

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

ax[0].set_title("Gráfica de y contra x")
ax[0].set_xlabel("x")
ax[0].set_ylabel("y")
ax[0].grid()
ax[0].scatter(x,y, c="tomato")
ax[0].plot(x_p,f(x_p),c="k",linestyle="--",label="y = ("+str(round(lin_reg.slope(),1))+"$\pm$ "
        +str(round(lin_reg.err_slope(),1))+")x + ("+str(round(lin_reg.intercept(),0))+"$\pm$"+
         str(round(lin_reg.err_intercept(),0)) +")")
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
ax[1].set_xlabel("x")
ax[1].set_ylabel("Residuales $R_i$ ")
ax[1].set_title("Gráfica de residuales normalizados")
ax[1].legend()
plt.tight_layout()

print(np.average(res))





"""
parte 2: pesos iguales
"""

dy=4*np.ones(len(y))

"""
Regresión con pesos
"""

lin_reg = linr(x,y,dy)
f= lambda x: lin_reg.slope()*x+lin_reg.intercept()


x_p=np.linspace(min(x),max(x),500)

fig2, ax2= plt.subplots(2,1, figsize=(6,8))

"""
Gráfica regresión con pesos
"""

ax2[0].set_title("Gráfica de y contra x")
ax2[0].set_xlabel("x")
ax2[0].set_ylabel("y")
ax2[0].grid()
ax2[0].scatter(x,y, c="tomato")
ax2[0].plot(x_p,f(x_p),c="k",linestyle="--",label="y = ("+str(round(lin_reg.slope(),1))+"$\pm$ "
        +str(round(lin_reg.err_slope(),1))+")x + ("+str(round(lin_reg.intercept(),0))+"$\pm$"+
         str(round(lin_reg.err_intercept(),0)) +")")
ax2[0].legend(loc="upper left")
ax2[0].errorbar(x, y, yerr= dy, ecolor="k",capsize=2,linestyle="")

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

ax2[1].scatter(x,res,c="red")
ax2[1].grid()
ax2[1].set_xlabel("x")
ax2[1].set_ylabel("Residuales $R_i$ ")
ax2[1].set_title("Gráfica de residuales normalizados")
ax2[1].legend()
plt.tight_layout()

print(np.average(res))






"""
parte 3: pesos dispares
"""

dy[0]=1
dy[-1]=1
dy[1:-1]=8

"""
Regresión con pesos
"""

lin_reg = linr(x,y,dy)
f= lambda x: lin_reg.slope()*x+lin_reg.intercept()


x_p=np.linspace(min(x),max(x),500)

fig3, ax3= plt.subplots(2,1, figsize=(6,8))

"""
Gráfica regresión con pesos
"""

ax3[0].set_title("Gráfica de y contra x")
ax3[0].set_xlabel("x")
ax3[0].set_ylabel("y")
ax3[0].grid()
ax3[0].scatter(x,y, c="tomato")
ax3[0].plot(x_p,f(x_p),c="k",linestyle="--",label="y = ("+str(round(lin_reg.slope(),1))+"$\pm$ "
        +str(round(lin_reg.err_slope(),1))+")x + ("+str(round(lin_reg.intercept(),0))+"$\pm$"+
         str(round(lin_reg.err_intercept(),0)) +")")
ax3[0].legend(loc="upper left")
ax3[0].errorbar(x, y, yerr= dy, ecolor="k",capsize=2,linestyle="")

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

ax3[1].scatter(x,res,c="red")
ax3[1].grid()
ax3[1].set_xlabel("x")
ax3[1].set_ylabel("Residuales $R_i$ ")
ax3[1].set_title("Gráfica de residuales normalizados")
ax3[1].legend()
plt.tight_layout()

print(np.average(res))






