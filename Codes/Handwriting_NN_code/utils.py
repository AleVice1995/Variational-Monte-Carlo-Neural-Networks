# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 07:23:56 2026

@author: Ale
"""
#utils.py

import numpy as np

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

def ReLU(z):
    return np.maximum(0,z)

def ReLU_prime(z):
    bool_vect=z>0
    return bool_vect.astype(int)

def Cost_quadratic(y, guess):
    diff=(y-guess)**2
    return 0.5*(np.sum(diff))

def Cost_cross_entropy(y, guess):
    term1=y*np.log(guess)
    term2=(1-y)*np.log((1-guess))
    return np.sum(np.nan_to_num(-(term1+term2)))