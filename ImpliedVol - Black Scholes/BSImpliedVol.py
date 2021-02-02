# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""


@author: Louis Gallais
FAUT RAJOUTER LES DIVIDENDES
"""




import numpy as np

import pdb

from corePricer.IndexOptions.BlackScholes.BlackScholesPricer import (BlackNScholesClosedFormula, 
                                                                     VegaBS, 
                                                                     CallFromCallPutParity,
                                                                     BlackNScholesClosedFormula_put)



def impliedVolComputation_withStartingPoint(price, implied_guess, K, r, d, T, S0):
    """ This is supposed to be a Newton's method"""

    threshold = 1e-3
    difference  = abs(price  - BlackNScholesClosedFormula(implied_guess, K, r, d, T, S0))
    sigma = implied_guess

    while(difference > threshold):

        price_estimate = BlackNScholesClosedFormula(sigma, K, r, d, T, S0)
        sigma =  sigma - (price_estimate - price) / VegaBS(sigma,  K, r, d, T , S0)

        sigma = max(sigma, 2e-2)

        difference = abs(BlackNScholesClosedFormula(sigma, K, r, d, T, S0) - price)

    return sigma




def impliedVolComputation(price, K, r, d, T, S0):

    starting_point  = implied_vol_estimate(price, K, r, d, T, S0)

    return impliedVolComputation_withStartingPoint(price, starting_point, K, r, d, T, S0)




def implied_vol_estimate(price, K, r, d, T, S0):
    estimate = np.sqrt(2 * np.pi / T) * (price / S0)
    
    a = max(estimate , 0.2)
    return a



# We now write all the same things but with puts

def implied_vol_estimate_put(price, K, r, d, T, S0):
    call_price = CallFromCallPutParity(price, K, r, d, T, S0)
    print(call_price)
    return implied_vol_estimate(call_price, K,r,d,T,S0)


def impliedVolComputationPut(put_price, K,r,d,T,S0):

    #pdb.set_trace()

    call_price = CallFromCallPutParity(put_price, K, r, d, T, S0)

    implied_guess = implied_vol_estimate_put(put_price, K, r, d, T, S0)
    threshold = 1e-3

    difference  = abs(put_price  - BlackNScholesClosedFormula_put(implied_guess, K, r, d, T, S0))

    sigma = implied_guess
    
    while(difference > threshold):

        price_estimate = BlackNScholesClosedFormula_put(sigma, K, r, d, T, S0)
        if VegaBS(sigma,  K, r, d, T , S0) < 1e-4 :
            
            #print(put_price, K, r, T, S0)
            
            
            return 0.0
        
        #print(sigma)
        sigma =  sigma - (price_estimate - put_price) / VegaBS(sigma,  K, r, d, T , S0)
        sigma = max(sigma, 2e-2)
        
        difference = abs(BlackNScholesClosedFormula_put(sigma, K, r, d, T, S0) - put_price)

    return sigma



def impiedvolPutDicho(put_price, K,r,d, T,S0):
    Pr = put_price
    vmax = 1
    vmin = 0.001
    while vmax-vmin > 1e-4:
        Pmax = BlackNScholesClosedFormula_put(vmax, K, r,d,  T, S0)
        Pmin = BlackNScholesClosedFormula_put(vmin, K, r,d, T, S0)
        vm = 0.5*(vmax + vmin)
        Pm = BlackNScholesClosedFormula_put(vm, K, r,d, T, S0)
        if Pmax < Pr:
            print("c'est faux")
            break
        elif Pmin > Pr :
            print("j'en ai marre")
            break
        
        elif Pm < Pr:
            print("tout va bien")
            vmin = vm
        else :
            print("tout va bien")
            vmax = vm
    return vmax, vmin, Pmax, Pmin , Pr
            

def impiedvolCallDicho(call_price, K,r,d,T,S0):
    Pr = call_price
    vmax = 1
    vmin = 0.001
    while vmax-vmin > 1e-4:
        Pmax = BlackNScholesClosedFormula(vmax, K, r,d, T, S0)
        Pmin = BlackNScholesClosedFormula(vmin, K, r,d, T, S0)
        vm = 0.5*(vmax + vmin)
        Pm = BlackNScholesClosedFormula(vm, K, r, d,T, S0)
        if Pmax < Pr:
            print("c'est faux")
            break
        elif Pmin > Pr :
            print("j'en ai marre")
            break
        elif Pm < Pr:
            vmin = vm
        else :
            vmax = vm
    return vmax, vmin, Pmax, Pmin , Pr       
            
            
            