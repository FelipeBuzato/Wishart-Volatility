#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 14:23:43 2020

@author: louisgallais
"""


import math
import numpy as np
from scipy.integrate import quad




class HestonPricer :


    def __init__(self, mu,  kappa, theta, V0, sigma, rho):
        self.mu = mu #tendance
        self.kappa = kappa #retour a la moyenneH
        self.theta = theta #variance a l'infini
        self.sigma = sigma # vol de la vol
        self.rho = rho #corrélation entre vol et indice
        self.V0 = V0

    def MonteCarloSimulation(self, S, v, N_years, N_simulations = 5000):
        """S : stock price,
        v : volatilité,
        N_years : nombre d'années,
        N_simulations : nombre de simulations (par défaut 5000),
        return S_t prix actualisé et v_t volatilité actualisé"""
        #je suis pas convaincu par le nom de la function !!!
        # Not very happy with that because we only have one step per year
        S_t = np.zeros((N_years, N_simulations))
        v_t = np.zeros((N_years, N_simulations))
        S_t[0] = S
        v_t[0] = v
        for year in range(N_years - 1): #je pense pas que ca aille jusqu a N_years
            W_t1 = np.random.normal(0,1, N_simulations)
            W_t2 = self.rho * W_t1 + np.sqrt(1 - self.rho ** 2) * np.random.normal(0,1, N_simulations)

            S_t[year + 1] += S_t[year] * self.mu  + np.sqrt(v_t[year]) * S_t[year] * W_t1
            v_t[year + 1] += self.kappa * (self.theta - v_t[year]) + self.sigma * np.sqrt(v_t[year]) * W_t2

        return S_t, v_t



    def DiffusionHeston(self, S, N_months, N_sim = 500000):

        """S : stock price,
        v : volatilité,
        N_months : nombre de mois,
        N_simulations : nombre de simulations (par défaut 50000),
        return S_t prix actualisé et v_t volatilité au temps final
        fait un gros nombre de trajectoires différentes
        suivant un schéma d'Euler
        """

        S_t = np.zeros((N_months + 1, N_sim))
        v_t = np.zeros((N_months + 1, N_sim))

        S_t[0] = S #sur la ligne c'est la meme date
        v_t[0] = self.V0
        duree = 1/12
        for month in range(N_months):
            W_t1 = np.random.normal(0, 1, N_sim )
            W_t2 = self.rho * W_t1 + np.sqrt(1 - self.rho ** 2) * np.random.normal(0,1, N_sim)
            #les lois normales sont corrélées
            #dt = 1/12 ??? c'est pas très clair
            S_t[month + 1] = (S_t[month] + S_t[month] * self.mu * duree +
                              np.sqrt(v_t[month]) * np.sqrt(duree) *S_t[month] * W_t1)

            v_t[month + 1] = (v_t[month] + self.kappa * (self.theta - v_t[month]) * 1/12
                              + self.sigma * np.sqrt(v_t[month]) * np.sqrt(duree) * W_t2)
        return S_t




    def HestonFormula_Call(self, K, S0, T):
        """Formule pour call européen"""
        r, V0,  kappa, theta, lambd, rho = self.mu, self.V0, self.kappa, self.theta, self.sigma, self.rho
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
            b1=np.exp(u1*I*(np.log(S0/K)+r*T))*( (1-g1*np.exp(-d1*T))/(1-g1) )**bb
            b2=np.exp(u2*I*(np.log(S0/K)+r*T))*( (1-g2*np.exp(-d2*T))/(1-g2) )**bb
            phi1=b1*np.exp(aa*(kappa-a1-d1)\
            +V0*(kappa-a1-d1)*(1-np.exp(-d1*T))/(1-g1*np.exp(-d1*T))/lambd**2)
            phi2=b2*np.exp(aa*(kappa-a2-d2)\
            +V0*(kappa-a2-d2)*(1-np.exp(-d2*T))/(1-g2*np.exp(-d2*T))/lambd**2)
            P+= ((phi1-phi2)/(u2*I))*du
        return K*np.real((S0/K-np.exp(-r*T))/2+P/np.pi)



    def HestonFormula_Put(self, K, S0, T):
        r = self.mu
        V0 = self.V0
        callPrice = self.HestonFormula_Call(K, S0, T)
        return callPrice - S0 + K *np.exp(-r*T)


    def callTPayoff(self, spot, strike):
       return np.maximum(spot - strike, 0.0)




    def Call_MonteCarlo(self, S0, K, T, div = 0.0068, M=100000, N=25):
        dt = T / N
        
        path = np.zeros(N)
        var = np.zeros(N)
        callT = np.zeros(M)
        z1 = np.random.normal(size=(M,N))
        z2 = np.random.normal(size=(M,N))
        
        for i in range(M):
            var[0] = self.V0
            path[0] = S0
            
            for j in range(1, N):
                #simulate variance equation first
                var[j] = var[j-1] + self.kappa * (self.theta - var[j-1]) * dt + self.sigma * np.sqrt(var[j-1] * dt) * z1[i,j]
                
                # use truncation method
                if var[j] <= 0.0:
                    var[j] = np.maximum(var[j], 0.0)
                    
                # simulation of price path
                path[j] = path[j-1] * np.exp((self.mu - div - 0.5 * var[j]) * dt + np.sqrt(var[j] * dt) * z2[i,j])
                
            callT[i] = self.callTPayoff(path[-1], K)
        
        result = callT.mean() * np.exp(-self.mu * T)
        return result

