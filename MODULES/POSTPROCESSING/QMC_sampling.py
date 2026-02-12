# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 16:29:39 2026

@author: anafd
"""

import numpy as np

from scipy.special import roots_legendre
from scipy.stats import qmc

class QuadratureMethod(object):
    def __init__(self, gdim, seed=42):
        self.gdim = gdim
        self.RNG = np.random.default_rng(seed)
        self.qmc_sampler = qmc.Sobol(d=gdim, scramble=True, seed=self.RNG)

    def MC(self, n):
        xtrain = self.RNG.uniform(size=(self.gdim, n))
        weights = np.ones((n,))/n
        return np.array(xtrain), weights

    def QMC(self, n):
        m = int(np.ceil(np.log2(n)))
        xtrain = self.qmc_sampler.random_base2(m=m).reshape((self.gdim,-1))
        self.qmc_sampler.reset()
        self.qmc_sampler._scramble()
        weights = np.ones((2**m,))*(2.**-m)
        return np.array(xtrain), weights

    def FixedGaussQuad(self, n):
        gdim = self.gdim
        x,w = roots_legendre(n)
        x = (x+1)/2
        w /= 2
        if gdim > 1:
            X = np.meshgrid(*[x]*gdim)
            x = np.vstack([item.flatten() for item in X])
            w = np.outer(w,w).flatten() if gdim == 2 else np.einsum('i,j,k->ijk', *[w]*3).flatten()
        else:
            x = x.reshape((1,-1))
        return np.array(x),np.array(w)

