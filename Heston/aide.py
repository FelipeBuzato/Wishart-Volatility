#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 16:22:20 2020

@author: louisgallais
"""


import numpy as np
from scipy.optimize import broyden1

X0 = 100
V0 = 0.05
r = 0.01
kappa = 5
theta=0.05
lambd=0.5
rho=-0.8

def heston(kappa,theta,lambd,T,K):
     I=complex(0,1)
     P, umax, N = 0, 1000, 10000
     du=umax/N
     aa= theta*kappa*T/lambd**2
     bb= -2*theta*kappa/lambd**2
     for i in range (1,N) :
         u2=i*du
         u1=complex(u2,-1)
         a1=rho*lambd*u1*I
         a2=rho*lambd*u2*I
         d1=np.sqrt((a1-kappa)**2+lambd**2*(u1*I+u1**2))
         d2=np.sqrt((a2-kappa)**2+lambd**2*(u2*I+u2**2))
         g1=(kappa-a1-d1)/(kappa-a1+d1)
         g2=(kappa-a2-d2)/(kappa-a2+d2)
         b1=np.exp(u1*I*(np.log(X0/K)+r*T))*( (1-g1*np.exp(-d1*T))/(1-g1) )**bb
         b2=np.exp(u2*I*(np.log(X0/K)+r*T))*( (1-g2*np.exp(-d2*T))/(1-g2) )**bb
         phi1=b1*np.exp(aa*(kappa-a1-d1)\
         +V0*(kappa-a1-d1)*(1-np.exp(-d1*T))/(1-g1*np.exp(-d1*T))/lambd**2)
         phi2=b2*np.exp(aa*(kappa-a2-d2)\
         +V0*(kappa-a2-d2)*(1-np.exp(-d2*T))/(1-g2*np.exp(-d2*T))/lambd**2)
         P+= ((phi1-phi2)/(u2*I))*du
     return K*np.real((X0/K-np.exp(-r*T))/2+P/np.pi)
# Example of usage of heston()
T, K = 0.5, 100
call = heston(kappa,theta,lambd,T,K)
print("call = ",call, " put = ", call-X0+K*np.exp(-r*T))
# example of calibration
price1=heston(kappa,theta,lambd,T,90)
price2=heston(kappa,theta,lambd,T,105)
price3=heston(kappa,theta,lambd,T,110)





def F(x):
     return [(price1-heston(x[0],x[1],x[2],T,90)), \
             (price2-heston(x[0],x[1],x[2],T,105)), \
                 (price3-heston(x[0],x[1],x[2],T,110))]
x = broyden1(F, [1.4,0.03,0.5], f_tol=1e-14)
print("[kappa,theta,lambda] =",x)











#%%
import matplotlib.pyplot as plt

A = np.array([19.148, 6.326, 1.445, 1.2943, 0.516, 0.507, 0.506, 0.474,
              0.2454, 0.2435, 0.199, 0.1978, 0.192, 0.1496, 0.09315, 0.0926, 0.070,
              0.066])

plt.plot(A)



