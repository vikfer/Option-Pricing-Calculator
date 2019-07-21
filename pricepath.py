#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 12:48:56 2019

@author: vpfernandez
"""

import matplotlib.pyplot as plt
import numpy as np
import math
#import pandas as pd
import scipy.stats as si
#from mibian import BS
import time

import psutil

from qol import notify

"################################### Formulas #################################"

def gbm(μ, σ, x0, t): 
    return math.exp(np.log(x0) + (μ - 1 / 2 * σ ** 2) * t + σ * np.random.normal(0,1,1))

def mean(liste):
    return sum(liste)/len(liste)

def payoff_c(paths,K):
    array = paths.transpose()[-1] - K
    zeros = (np.sign(array) + 1)/2
    array2 = array*zeros
    val = mean(array2)
    return val

def payoff_p(paths,K):
    array = K - paths.transpose()[-1]
    zeros = (np.sign(array) + 1)/2
    array2 = array*zeros
    val = mean(array2)
    return val  

#NOT WORKING
#payoff_c_vect = np.vectorize(payoff_c, excluded=['paths'])
#payoff_p_vect = np.vectorize(payoff_p, excluded=['paths'])

def payoff_lookback_c(paths):
    mini = np.amin(paths, axis=1)
    array = paths.transpose()[-1]-mini
    val = mean(array)
    return val

def payoff_lookback_p(paths):
    maxi = np.amax(paths, axis=1)
    array = maxi-paths.transpose()[-1]
    val = mean(array)
    return val


def BS_C(S, K, T, r, sigma):    
    d1 = (np.log(S/K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S/K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    call = ((S*si.norm.cdf(d1,0,1)) - (K*np.exp(-r*T)*si.norm.cdf(d2,0,1)))
    return call

def BS_P(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S/K) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))    
    put = ((K*np.exp(-r*T) * si.norm.cdf(-d2,0,1)) - (S*si.norm.cdf(-d1,0,1)))
    return put

# VECTORIZE
BS_C_vect = np.vectorize(BS_C)
BS_P_vect = np.vectorize(BS_P)



def impliedcalc_c(val, S, K, T, r, sigmin=0.15, sigmax=0.40, precision=0.0001):
    testval = np.arange(sigmin+precision,sigmax,precision)
    c = BS_C_vect(S, K, T, r, testval)
    test_c = abs(c-val)

    implied_c = testval[np.argmin(test_c)]    
    return implied_c

    
def impliedcalc_p(val, S, K, T, r, sigmin=0.15, sigmax=0.40, precision=0.0001):
    testval = np.arange(sigmin+precision,sigmax,precision)
    p = BS_P_vect(S, K, T, r, testval)
    test_p = abs(p-val)

    implied_p = testval[np.argmin(test_p)]    
    return implied_p


"##################################### Paths ##################################"

S0 = 10
sigma = 0.20
K = 10
r = 0/100
t = 1/4

n = 5_000_000
steps = 10

jump_size = 0.00 #
jump_freq = 1 #per period
jump_direc = -1

memory_limit = 0.75


"##############################################################################"

vol_t = math.sqrt(t)*sigma
r_t = ((1+r)**t)-1
r_t = r * t

dt = 1/steps
wt = math.sqrt(dt)
r_subt = ((1+r_t)**dt)-1
r_subt = r_t*dt

"##############################################################################"

paths = np.zeros((1,1))
jump = np.zeros((1,1))
bytes_used = 8*((steps+8)*n)+2*99999
availiable_memory = psutil.virtual_memory()[1]

if bytes_used > availiable_memory*memory_limit:
    raise MemoryError("Insufficient Memory to complete the requested action: \nRequired: {} Go\nAvailiable: {} Go\nConsider reducing main matrices by a factor of {}".format(round(bytes_used/10**9,2),round((availiable_memory*memory_limit)/10**9,2),round(bytes_used/(availiable_memory*memory_limit),2)))
else:
    paths = np.zeros((steps+1,n))

start = time.time()
paths[0] = S0

for m in range(1,steps+1,1):
    
    jump = np.random.poisson(jump_freq/steps,n) * np.random.exponential(jump_size,n) * jump_direc
    
    paths[m] = paths[m-1] * (1+(r_t*dt) + jump + (vol_t*np.random.normal(0,1,n)*wt))

   
    if m > 0 and (m/steps)*(100/5) == np.floor((m/steps) *(100/5)):
        tick = time.time() - start
        remaining = round(tick*(steps/m - 1),2)
        if remaining > 60:
            print("{}% (remaining: {}m {}s)".format(int((m/steps) *100), np.floor(tick*(steps/m - 1) / 60), round(remaining - np.floor(tick*(steps/m - 1) / 60)*60,2)))
        else:
            print("{}% (remaining: {}s)".format(int((m/steps) *100),remaining))



sample = 50
for rr in range(sample):
    plt.plot(paths.transpose()[rr])
plt.title("Price path, Sample (n={})".format(sample))
plt.show()

print("Normalization")
for i in range(len(paths)):        
    paths[i] = paths[i] - r_t*i
paths = paths - (paths[-1].mean()-S0)
print("Done!\n")

print("Transposition")
paths = np.transpose(paths)
print("Done!\n")

tick = time.time() - start
print("Done in {} seconds".format(np.round(tick,2)))

"#################################### SMILE ###################################"

mink = 10 - 1
maxk = 10 + 1
step_smile = 0.5
rangek = np.arange(mink,maxk+step_smile,step_smile)

implied_smile_c = np.zeros((len(rangek)))
implied_smile_p = np.zeros((len(rangek)))


for z, k in enumerate(rangek):
   
    K = np.round(k,2)
    
    val_c = payoff_c(paths,K)
    val_p = payoff_p(paths,K)    
    print("Strike = {} \nVal call = {} \nVal put = {}".format(K,np.round(val_c,4),np.round(val_p,4)))
    
    bsc_c = BS_C(S0, K, t, r, sigma)
    bsc_p = BS_P(S0, K, t, r, sigma)    
    print("BS call = {} \nBS put = {}".format(np.round(bsc_c,4),np.round(bsc_p,4)))
    
    impl_c = impliedcalc_c(val_c, S0, K, t, r, sigmin=0.15, sigmax=0.3, precision=0.0001)
    impl_p = impliedcalc_p(val_p, S0, K, t, r, sigmin=0.15, sigmax=0.3, precision=0.0001)    
    print("Implied: {}% / {}% \n".format(np.round(impl_c,4)*100,np.round(impl_p,4)*100))

    implied_smile_c[z] = impl_c
    implied_smile_p[z] = impl_p
    
plt.plot(rangek,implied_smile_c)
plt.plot(rangek,implied_smile_p)
axes = plt.gca()
axes.set_ylim([round(min(min(implied_smile_c),min(implied_smile_p)),3)-0.001,
               round(max(max(implied_smile_c),max(implied_smile_p)),3)+0.001])
axes.set_yticklabels(['{:,.2%}'.format(x) for x in axes.get_yticks()])
plt.legend(("Calls","Puts"))
plt.title("Volatility Smile")
#plt.title("Volatility Smile: Jump {}% / P(x) = {} per 3-month".format(int(jumps*100),int(times)))
plt.show()










print("Payoff from Floating Lookback options (C): {} / Expected: {}".format(round(payoff_lookback_c(paths),4),round(0.773217,4)))
print("Payoff from Floating Lookback options (P): {} / Expected: {}".format(round(payoff_lookback_p(paths),4),round(0.823217,4)))

tick = time.time() - start
print("Done in {} seconds".format(np.round(tick,2)))

notify("Done","Done in {} seconds".format(np.round(tick,2)))