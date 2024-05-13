# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 20:39:56 2024

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

x=np.array([10,20,30,40,50,60,70,80,90,100,110])#Frecuencia Hz
y=np.array([16,45,64,75,70,115,142,167,183,160,221])#Voltaje mV
dy=np.array([5,5,5,5,30,5,5,5,5,30,5])



m=np.linspace(-1,5,1000)
c=np.linspace(-4,4,1000)

mm,cc = np.meshgrid(m,c)
def X_2(X,arg=(x,y)):
    m,c=X[0],X[1]
    x,y=arg[0],arg[1]
    X_2=np.sum((y-m*x-c)**2/dy**2)
    return X_2

p_min=mini(X_2,(2.1,-0.5))
m_min,c_min=p_min.x[0],p_min.x[1]



"""
Método iterativo
"""
X_2_min= X_2((m_min,c_min))

"""
Se definen las ecuaciones para resolver por Newton-Raphson
"""
def f_m(m,c=c_min,x=x,y=y,alpha=dy,contor=1):  
    return np.sum((y-(m)*x-c)**2/(alpha)**2)-X_2_min-contor

def f_c(c,m=m_min,x=x,y=y,alpha=dy,contor=1):  
    return np.sum((y-(m)*x-c)**2/(alpha)**2)-X_2_min-contor

"""
Metodo para incertidumbre de la pendiente m
"""

def Met_iter_m(m=m_min,c=c_min,x=x,y=y,alpha=dy,gmm=0.05,err=1e-7,contor=1):
    
    cj_1=c
    d=1
    mj=m
    itera=0
    
    while d>err:
        mj_1=mj
        """
        Si se quiere que vaya a la derecha cambiar signo del 0.02 en el segundo argumento
        de la función nwt() de la línea de abajo por un +.
        Si en cambio se quiere ir a la izquierda cambiar el signo de la línea de abjo por -
        """
        mj=nwt(f_m,mj-0.02,args=(cj_1,x,y,alpha,contor))
        D_cj=2*gmm*np.sum((y-(mj)*x-cj_1)/(alpha)**2)
        
        
        cj_1 += D_cj
        
        diff=np.abs(mj-mj_1)
        d = diff 
        itera +=1
        #print(mj,cj_1,D_cj)
    
    return mj, itera

"""
Metodo para incertidumbre del intercepto c
"""
def Met_iter_c(m=m_min,c=c_min,x=x,y=y,alpha=dy,gmm=0.0001,err=1e-7,contor=1):
    
    mj_1=m
    d=1
    cj=c
    itera=0
    
    while d>err:
        cj_1=cj
        """
        Si se quiere que vaya a la derecha cambiar signo del 0.02 en el segundo argumento
        de la función nwt() de la línea de abajo por un +.
        Si en cambio se quiere ir a la izquierda cambiar el signo de la línea de abjo por -
        """
        cj=nwt(f_c,cj-0.01,args=(mj_1,x,y,alpha,contor))
        D_mj=2*gmm*np.sum(x*(y-(mj_1)*x-cj)/(alpha)**2)
        
        
        mj_1 += D_mj
        
        diff=np.abs(cj-cj_1)
        d = diff 
        itera +=1
        #print(cj,mj_1,D_mj)
    
    return cj, itera

"""
Para evaluar los distintos valores que se piden en el problema ya sea
para el contorno de Delta X^2=1,4 o 9, cambiar el argumento contor. Esta puesto
en 9 ya que ese fue el último que se evaluó
"""
alpha_m,iteracion= Met_iter_m(contor=9)
print(alpha_m,iteracion)
print((alpha_m-m_min)/3)
alpha_c,iteracion_c= Met_iter_c(contor=9)
print(alpha_c,iteracion_c)
print((alpha_c-c_min)/3)



