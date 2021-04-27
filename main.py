#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 08:01:14 2021

@author: gilto
"""

import numpy as np
import scipy.optimize as optimize
import matplotlib.pyplot as plt

from numba import njit

AWESOME_PURPLE = 0.55, 0.42, 1

m = 0.02592
sigma = 0.06164

K = 12*30
S = 100000
W0 = 1

R = 5
omega = (1 + R)*W0

@njit(nogil = True)
def A(phi):
    return 1 + m*phi

@njit(nogil = True)
def B(phi):
    return A(phi)**2 + sigma**2*phi**2

@njit(nogil = True)
def Objective(phi):
    objective = W0**2*np.power(B(phi), K)
    objective -= 2*omega*W0*np.power(A(phi), K)
    objective += omega**2
    return objective

@njit(nogil = True)
def computeRiskyReturns(K, S):
    np.random.seed(0)
    mu = np.zeros(shape = (S, K))
    for s in range(S):
        for k in range(K):
            mu[s, k] = np.random.normal(loc = m, scale = sigma)
    return mu

@njit(nogil = True)
def computeWealth(phi, W0, mu):
    S, K = mu.shape
    W = np.zeros(shape = S)
    for s in range(S):
        WK = W0
        for k in range(K):
            risky_returns = mu[s, k]*phi
            WK *= np.maximum(0, 1 + risky_returns[0])
        W[s] = WK
    return W

@njit(nogil = True)
def computeTarget(phi, W0, mu, R):
    return (computeWealth(phi, W0, mu) > omega).mean()

mu = computeRiskyReturns(K, S)
res = optimize.minimize(Objective, 0, bounds = [[0, 1]], method = "SLSQP")

phi_optim = res.x.copy()[0]
goal_proba = computeTarget(np.array([phi_optim]), W0, mu, R)
print(f"Optimal allocation: {phi_optim:.4f}")
print(f"Proba to achieve goal: {goal_proba:.4f}")

WK = computeWealth(np.array(K*[phi_optim]), W0, mu)
proba, bins, _ = plt.hist(WK, bins = 1000, density = True, cumulative = True)
plt.close()
bins = .5*(bins[1:] + bins[:-1])

plt.figure(figsize = (10, 5))
plt.plot(bins, 1 - proba, color = AWESOME_PURPLE)
plt.xlabel(r"$R$")
plt.ylabel(r"Probability of at least $R$ returns")
plt.title(f"Monte Carlo Simulations (K = {K} and S = {S})")



