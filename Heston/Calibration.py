#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 14:23:09 2020

@author: louisgallais
"""



import numpy as np
import corePricer.IndexOptions.Heston.Pricer as Pricer
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import minimize
from corePricer.IndexOptions.BlackScholes.BlackScholesPricer import BlackNScholesClosedFormula, VegaBS
import copy


#%%


class HestonCalibration :

    def __init__(self, T, K, MktPrice, mu, S0):
        """T tableau de maturité
        K tableau de strike
        MktPrice tableau correspondant
        S0 ok
        V0 ok 
        mu va pas bouger aussi : égal a ... """
        self.T = T
        self.K = K
        self.MktPrice = MktPrice #tableaux de calls...
        self.S0 = S0
        self.mu = mu
        self.Nk = len(K)
        self.Nt = len(T)
        #self.V0 = V0



    def lossFonction(self, x) :
        kappa, theta, V0, sigma, rho = x[0], x[1], x[2], x[3], x[4]
        errors = np.zeros((self.Nk, self.Nt))
        monProduit = Pricer.HestonPricer(self.mu, kappa, theta, V0, sigma, rho)
        for i in range(self.Nk):
            strike = self.K[i]
            for j in range(self.Nt):
                Mat = self.T[j] # Maturity
                marketPrice = self.MktPrice[i, j]
                modelPrice = monProduit.HestonFormula_Call(strike, self.S0, Mat)
                errors[i, j] = (marketPrice - modelPrice)**2 / VegaBS(monProduit.theta,
                                                                      strike, self.mu , Mat, self.S0)**2   
                #print(modelPrice)
        if (kappa <= 0 or theta <= 0 or sigma <= 0 or rho < -1 or rho > 1):
            errorLoss = 1e9    #1milliard
        else :
            #print(errors)
            errorLoss = np.sum(errors) / (self.Nt * self.Nk)
        return errorLoss








    def lossFonction2(self, x) :
        kappa, theta, V0, sigma, rho = x[0], x[1], x[2], x[3], x[4]
        errors = np.zeros((self.Nk, self.Nt))
        monProduit = Pricer.HestonPricer(self.mu, kappa, theta, V0, sigma, rho)
        for i in range(self.Nk):
            strike = self.K[i]
            for j in range(self.Nt):
                Mat = self.T[j] # Maturity
                marketPrice = self.MktPrice[i, j]
                modelPrice = monProduit.HestonFormula_Call(strike, self.S0, Mat)
                errors[i, j] = (marketPrice - modelPrice)**2
                #print(modelPrice)
        if (kappa <= 0 or theta <= 0 or sigma <= 0 or rho < -1 or rho > 1):
            errorLoss = 1e9    #1milliard
        else :
            #print(errors)
            errorLoss = np.sum(errors) / (self.Nt * self.Nk)
        return errorLoss
    
    

    
    def nelder_mead(self, f, x_start,
                    step=0.1, no_improve_thr=10e-6,
                    no_improv_break=100, max_iter=0,
                    alpha=1., gamma=2., rho=-0.5, sigma=0.5):
        """
            @param f (function): function to optimize, must return a scalar score
                and operate over a numpy array of the same dimensions as x_start
            @param x_start (numpy array): initial position
            @param step (float): look-around radius in initial step
            @no_improv_thr,  no_improv_break (float, int): break after no_improv_break iterations with
                an improvement lower than no_improv_thr
            @max_iter (int): always break after this number of iterations.
                Set it to 0 to loop indefinitely.
            @alpha, gamma, rho, sigma (floats): parameters of the algorithm
                (see Wikipedia page for reference)
                
            return: tuple (best parameter array, best score)
        """
    
        # init
        dim = len(x_start[0])
        prev_best = f(x_start[0])
        no_improv = 0
        res = [[x_start[0], prev_best]]
    
        for i in range(1, dim + 1):
            score = f(x_start[i])
            res.append([x_start[i], score])
        
        # simplex iter
        iters = 0
        while 1:
            # order
            res.sort(key=lambda x: x[1])
            best = res[0][1]
            print(res)
            # break after max_iter
            if max_iter and iters >= max_iter:
                return res[0]
            iters += 1
    
            # break after no_improv_break iterations with no improvement
            print ('...best so far:', best)
    
            if best < prev_best - no_improve_thr:
                no_improv = 0
                prev_best = best
            else:
                no_improv += 1
    
            if no_improv >= no_improv_break:
                return res[0]
    
            # centroid
            x0 = [0.] * dim
            for tup in res[:-1]:
                for i, c in enumerate(tup[0]):
                    x0[i] += c / (len(res)-1)
    
            # reflection
            xr = x0 + alpha*(x0 - res[-1][0])
            rscore = f(xr)
            if res[0][1] <= rscore < res[-2][1]:
                del res[-1]
                res.append([xr, rscore])
                continue
    
            # expansion
            if rscore < res[0][1]:
                xe = x0 + gamma*(x0 - res[-1][0])
                escore = f(xe)
                if escore < rscore:
                    del res[-1]
                    res.append([xe, escore])
                    continue
                else:
                    del res[-1]
                    res.append([xr, rscore])
                    continue
    
            # contraction
            xc = x0 + rho*(x0 - res[-1][0])
            cscore = f(xc)
            if cscore < res[-1][1]:
                del res[-1]
                res.append([xc, cscore])
                continue
    
            # reduction
            x1 = res[0][0]
            nres = []
            for tup in res:
                redx = x1 + sigma*(tup[0] - x1)
                score = f(redx)
                nres.append([redx, score])
            res = nres
    
#%%
            #!/usr/bin/env python
