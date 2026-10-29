# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 14:18:15 2026

@author: Ale
"""

import torch

def autocorrelation_length(local_energies, autocorr_terms=100):
    
    c_autocorr=torch.zeros(autocorr_terms, 
                           device=local_energies.device, dtype=local_energies.dtype)
    samples=local_energies.shape[0]
    energy_mean=local_energies.mean()
    en_std_dev=((local_energies**2).mean()-(energy_mean**2)).sqrt()
    cont=0
    if autocorr_terms<samples:
        with torch.no_grad():
            for i in range(autocorr_terms, samples):
                cont+=1
                for j in range(autocorr_terms):
                    c_autocorr[j]+=local_energies[i,0]*local_energies[i-j,0]
            
            print(f"cont: {cont:d}")
            c_autocorr=c_autocorr/cont
            c_autocorr=(c_autocorr-energy_mean**2)/(en_std_dev**2)
        return c_autocorr
    else:
        print("Number of autocorrelations terms less than MC samples")