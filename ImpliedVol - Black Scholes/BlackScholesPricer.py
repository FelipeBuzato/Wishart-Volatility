# -*- coding: utf-8 -*-


"""
ici il s'agit de prendre en compte les dividendes continus
"""



import numpy as np
from scipy.stats import norm



def BlackNScholesClosedFormula(sigma, K, r, d, T, S0):
    """sigma : 0.2 signifie 20% de volatilite
        K :
        r : taux, pas en pourcents 0.02 = 2% par an
        T :
        S0 : prix initial
        return le prix d'un call selon la méthode de BlackAndScholes"""

    if (K == 0):
        return S0
    d1 = (1 / (sigma * np.sqrt(T))) * (np.log(S0 / K) + (r - d + (1 / 2) * sigma ** 2) * T)
    d2 = d1 - sigma * np.sqrt(T)

    # Gives the price of a call
    return S0 * np.exp(-d*T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def BlackNScholesClosedFormula_put(sigma, K, r, d, T, S0):
    """sigma :
       K :
       r : taux
       T :
       S0 : prix initial
       return le prix d'un put selon la méthode de BlackAndScholes
       put européen ou américain ? mieux vaut deux fonctions ou juste un if ?"""

    if (K == 0):
        return S0
    d1 = (1 / (sigma * np.sqrt(T))) * (np.log(S0 / K) + (r - d + (1 / 2) * sigma ** 2) * T)
    d2 = d1 - sigma * np.sqrt(T)

    # Gives the price of a put
    return -S0 * np.exp(-d*T) * norm.cdf(-d1) + K * np.exp(-r * T) * norm.cdf(- d2)


def VegaBS(sigma, K , r, d, T, S0):
    """ """
    d1 =  (1 / (sigma * np.sqrt(T))) * (np.log(S0 / K) + (r - d + (1 / 2) * sigma ** 2) * T)
    phi_d1  = np.exp(- 0.5* d1**2 )/ np.sqrt(2  * np.pi)

    return S0 * phi_d1 * np.sqrt(T)


def CallFromCallPutParity(put_price, K, r, d, T, S):
    
    return put_price + S * np.exp(-d * T) - np.exp(- r * T) * K







