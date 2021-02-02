#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 14:34:34 2020

@author: louisgallais
"""



## Importation de ce dont j'ai besoin
import numpy as np
import matplotlib.pyplot as plt
import time 

import corePricer.IndexOptions.Heston.Calibration as calib
import corePricer.IndexOptions.Heston.Pricer as pricer

from corePricer.NappeDeVol.TracageNappeDeVol import (MatVol_Call, MatVol_Put)
#import BaseDeDonnées.lienBaseWeb as lien

#%%


if __name__ == '__main__':
    
    
    #1.60785111, 0.17811325, 0.33561584, 0.99489447
    tfdq_prixPut_SX5E, tfdq_prixCall_SX5E, tdf_VolCall_SX5E, S0, r, Tf, div, T, TP, KPut, PutPrice, PutVol, KCall, CallPrice, CallVol = recuperation_des_données_SX5E()    
    print('ok on est bon')
    mu = -0.006
   # V0 = 0.2
    #S0 = 3300
    N_months = 60
    initial_Simplex = np.array([[4.03017093,  0.02819646,  0.17141997,  3.31162045, -0.1873663],
                           [4.02556173,  0.02817285,  0.17126889,  3.30615883, -0.18641558],
                           [4.02679301,  0.02819958,  0.17148738,  3.31110502, -0.18767019],
                           [1.6, 0.04, 0.04, 0.04, 0.8],
                           [1.15, 0.04, 0.3, 0.3, -0.6],
                           [2.15, 0.04, 0.3, 0.3, -0.6]])
    
    NappedeVol = calib.HestonCalibration(T, KCall, CallPrice, mu, S0)
    print('ok on est bon')
    a = NappedeVol.nelder_mead(NappedeVol.lossFonction2, initial_Simplex)
    a = a[0]
    
    print(a)
    kappa, theta, V0, sigma, rho = a[0], a[1], a[2], a[3], a[4]
    kappa, theta, V0, sigma, rho =  2.33017746e+01,  1.87723797e-02,  3.29151687e-01,  1.65150579e+00, -9.99999985e-01
    Diffusion = pricer.HestonPricer(mu, kappa, theta, V0, sigma, rho)
    
    D = Diffusion.DiffusionHeston(S0, N_months)
    
    
    K = KCall
    T = T
    HestonPrice = np.zeros((len(K), len(T)))
    HestonVol = np.zeros((len(K), len(T)))
    for i in range(K.shape[0]):
        for j in range(T.shape[0]):
            HestonPrice[i, j] = Diffusion.HestonFormula_Call(K[i], S0, T[j])
            
            
    HestonVol = MatVol_Call(T, K, HestonPrice, S0, r, div)

 
    
#%%
X0 = 137.14


S0 = 137.14
r = 0.001   
div = 0.0068
#V0 = 0.0337
K = np.array([120, 125, 130, 135, 140, 145, 150])    
T = np.array([0.1233, 0.2685, 0.7151, 0.9534])


CallPrice =  np.array([[17.5399, 18.4781, 21.1350, 22.3635],
                      [12.8889,	 14.1227, 17.1876, 18.5477],
                      [8.5359, 10.0404,	 13.5115, 15.0476],
                      [4.6903, 6.4381, 10.1650, 11.7645],
                      [1.7960, 3.5134, 7.2126, 8.8694],
                      [0.3665, 1.5057,	4.7473,	6.3808],
                      [0.0654, 0.4821, 2.8619, 4.3398]])




CallVol = np.array([[0.2780, 0.2638,  0.2532, 0.2518],
                      [0.2477, 0.2402,  0.2364, 0.2369],
                      [0.2186, 0.2158, 0.2203, 0.2239],
                      [0.1878, 0.1930,  0.2047, 0.2098],
                      [0.1572, 0.1712, 0.1894, 0.1970],
                      [0.1334, 0.1517, 0.1748, 0.1849],
                      [0.1323, 0.1373, 0.1618, 0.1736]])
    
    
CallVol = MatVol_Call(T, K, CallPrice, S0, np.array([r]), np.array([div]))  



initial_Simplex = np.array([[0.7, 0.5, 0.03, 0.3, -0.7],
                           [1.7, 0.04, 0.2, 0.4, -0.5],
                           [1.32, 0.03, 0.15, 0.4, -0.8],
                           [1.6, 0.04, 0.04, 0.4, -0.8],
                           [1.15, 0.04, 0.25, 0.3, -0.6],
                           [2, 0.03, 0.2, 0.9, -0.7]])


NappedeVol = calib.HestonCalibration(T, K, CallPrice, r, S0)
print('ok on est bon')
a = NappedeVol.nelder_mead(NappedeVol.lossFonction2, initial_Simplex)
print('ok c est bon')
print(a)


kappa, theta, V0, sigma, rho = 2.33017746e+01,  1.87723797e-02,  3.29151687e-01,  1.65150579e+00, -9.99999985e-01
print(NappedeVol.lossFonction2(np.array([kappa, theta, V0, sigma, rho])))
    
Hestontest = pricer.HestonPricer(r, kappa, theta, V0, sigma, rho)

HestonPrice = np.zeros((len(K), len(T)))
HestonLouis = np.zeros((len(K), len(T)))
HestonPricePut = np.zeros((len(K), len(T)))
for i in range(K.shape[0]):
    for j in range(T.shape[0]):
        #HestonPrice[i, j] = Hestontest.Call_MonteCarlo(S0, K[i], T[j])
        HestonLouis[i, j] = Hestontest.HestonFormula_Call(K[i], S0, T[j])
        
        #HestonPricePut[i, j] = Hestontest.HestonFormula_Put(K[i], S0, r, T[j])
     
HestonVol = MatVol_Call(T, K, HestonPrice, S0, np.array([r]), np.array([div]))        
HestonVolLouis = MatVol_Call(T, K, HestonLouis, S0, np.array([r]), np.array([div]))        
 
print(HestonPrice)
    
    
#%%
ax = plt.axes(projection='3d')

TI, KI = np.meshgrid(T, K)
ax.plot_surface(KI, TI, CallPrice, rstride=1, cstride=1, edgecolor='none')  
TI, KI = np.meshgrid(T, K)
#ax.plot_surface(KI, TI, HestonPrice, rstride=1, cstride=1, edgecolor='none')
ax.plot_surface(KI, TI, HestonLouis, rstride=1, cstride=1, edgecolor='none')

plt.title("courbes des prix")
#%%    
    
    
ax = plt.axes(projection='3d')

TI, KI = np.meshgrid(T, K)
ax.plot_surface(KI, TI, CallVol*100, rstride=1, cstride=1, edgecolor='none')  
TI, KI = np.meshgrid(T, K)
#ax.plot_surface(KI, TI, HestonVol*100, rstride=1, cstride=1, edgecolor='none')
ax.plot_surface(KI, TI, HestonVol*100, rstride=1, cstride=1, edgecolor='none')

plt.title("courbes des vol")
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    