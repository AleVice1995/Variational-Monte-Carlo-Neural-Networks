# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 10:20:10 2026

@author: Ale
"""

import torch
import torch.nn as nn
import numpy as np
from torch.func import vmap, jacrev, functional_call, hessian
from torch.func import jvp



def HO_potential(x, HO_factor_molt=0.5, *args, **kwargs):
    return HO_factor_molt*torch.sum(x**2, dim=1, keepdim=True)
    
# def Lennard_jones_potential(x, sigma, eps, pbc_cond=False, length_box_pbc=0,r_tilde=0.0, *args, **kwargs):
    
#     L=length_box_pbc
#     r_cut=length_box_pbc/2
#     x=x.reshape(x.shape[0],-1,3)
#     n_particles=x.shape[1]
    
#     V = torch.zeros(
#         (x.shape[0], 1),
#         dtype=x.dtype,
#         device=x.device
#     )
#     for i in range(n_particles):
#         for j in range(i+1,n_particles):
#             coords=x[:,i,:]-x[:,j,:]
#             if pbc_cond:
#                 coords=coords-L*(coords/L).round()
#             r_squared=torch.sum(coords**2, dim=1, keepdim=True) #this is the modulus square of distance
#             r_squared=torch.where(r_squared<r_tilde**2, r_tilde**2, r_squared)
#             if pbc_cond:
#                 particles_counted=r_squared.sqrt()<=r_cut
#             else:
#                 particles_counted=r_squared.bool()
#             attr_term=(sigma**6)/(r_squared**3)
#             rep_term=attr_term**2
#             v_pair=4*eps*(rep_term-attr_term)
#             # torch.where(condizione, valore_se_vera, valore_se_falsa)
#             V=V+torch.where(particles_counted, v_pair, torch.zeros_like(v_pair))
            
    
    
#     return V
            
def V_tail(n_particles, density, length_box_pbc, sigma, eps, *args, **kwargs):
    
    r_cut=length_box_pbc/2
    pot=(n_particles*8*torch.pi*eps*density*(sigma**6)*(sigma**6-3*(r_cut**6)))/(9*(r_cut**9))
    return pot
            
def distance(x):
    x=x.reshape(x.shape[0],-1,3)
    dr=x[:,0,:]-x[:,1,:]
    r_squared=torch.sum(dr**2, dim=1, keepdim=True)
    
    return r_squared.sqrt()




def Lennard_jones_potential(x, sigma, eps, pbc_cond=False, length_box_pbc=0.0 ,r_tilde=0.0, *args, **kwargs):
    
    L=length_box_pbc
    r_cut=length_box_pbc/2
    x=x.reshape(x.shape[0],-1,3)
    n_particles=x.shape[1]
    
    V = torch.zeros(
        (x.shape[0], 1),
        dtype=x.dtype,
        device=x.device
    )
    
    ij = torch.triu_indices(
        n_particles,
        n_particles,
        offset=1,
        device=x.device
    )
    i, j = ij[0], ij[1]

    # Differenze di posizione per tutte le coppie
    dr = x[:, i, :] - x[:, j, :]

    # Minimum-image convention
    if pbc_cond:
        dr = dr - L * torch.round(dr / L)

    # Distanze
    r_squared = torch.sum(dr**2, dim=-1)
    r = torch.sqrt(r_squared.clamp_min(1e-12))
    r=torch.where(r<r_tilde, r_tilde, r)
    if pbc_cond:
        particles_counted=r<=r_cut
    else:
        particles_counted=r.bool()
    
    attr_term=(sigma**6)/(r**6)
    rep_term=attr_term**2
    v_pair=4*eps*(rep_term-attr_term)
    # torch.where(condizione, valore_se_vera, valore_se_falsa)
    V=torch.where(particles_counted, v_pair, torch.zeros_like(v_pair)).sum(dim=1, keepdim=True)
    return V