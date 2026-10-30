# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 09:40:42 2026

@author: Ale
"""

import torch
import torch.nn as nn
import numpy as np
from torch.func import vmap, jacrev, functional_call, hessian
from torch.func import jvp
import time

def compute_energy_local(model, particle_configurations, kinetic_factor, potentials,
                         exp_function, n_particles):
    # T=0
    # V=0
    start_kin=time.time()
    print("Computing kinetic term:")
    T=compute_kinetic_energy_local(model, particle_configurations, kinetic_factor, exp_function)
    end_kin=time.time()
    elapsed=end_kin-start_kin
    hours, remainder=divmod(elapsed, 3600)
    minutes, seconds=divmod(remainder, 60)
    print(f"\nElapsed time: {hours:.0f} hh {minutes:.0f} mm {seconds:.2f} ss\n")
    
    start_pot=time.time()
    print("Computing potential term:")
    V=compute_potential_energy(potentials, particle_configurations)
    end_pot=time.time()
    elapsed=end_pot-start_pot
    hours, remainder=divmod(elapsed, 3600)
    minutes, seconds=divmod(remainder, 60)
    print(f"\nElapsed time: {hours:.0f} hh {minutes:.0f} mm {seconds:.2f} ss\n")
    
    local_energies=T+V
    energy_mean=local_energies.mean()
    T_mean=T.mean()
    V_mean=V.mean()
    energy_var=((local_energies**2).mean()-(energy_mean**2))
    energy_std_dev=energy_var.sqrt()
    samples=len(local_energies)
    if samples>1:
        energy_stoch_err=(energy_var/(samples-1)).sqrt()
    else:
        energy_stoch_err=energy_std_dev
        
    print(f"T_mean: {T.mean()/n_particles: .04f}")
    print(f"V_mean: {V.mean()/n_particles: .04f}")
    print(f"Energy mean: {energy_mean/n_particles: .04f}")
    print(f"Energy std_dev: {energy_std_dev/n_particles: .04f}")
    print(f"Energy stoch_err: {energy_stoch_err/n_particles: .04f}")
    
    return local_energies, energy_mean, energy_std_dev, energy_stoch_err, T_mean, V_mean
    
    
    
def compute_kinetic_energy_local(model, particle_configurations, kinetic_factor, exp_function):
    configurations=particle_configurations
    if not configurations.requires_grad:
        configurations.requires_grad_(True)
    f_NN=model(configurations)
    
    #Generate S=f_NN*torc.One_like and then derive the sclar obtained for the
    # N*m_cooridnates of configurations, obtaining the [N, m_cooridnates] Jacobian
    df_NN_dx=torch.autograd.grad(outputs=f_NN, inputs=configurations,
                                 grad_outputs=torch.ones_like(f_NN),
                                 create_graph=True)[0]
    hessian_diag_terms=[]
    num_coordinates=configurations.shape[1]
    
    for i in range(num_coordinates):
        d2f_NN_dx2=torch.autograd.grad(outputs=df_NN_dx[:,i], inputs=configurations,
                                       grad_outputs=torch.ones_like(df_NN_dx[:,i]),
                                       create_graph=False,
                                       retain_graph=True)[0]
        hessian_diag_terms.append(d2f_NN_dx2[:,i])
    hessian_pure_der=torch.stack(hessian_diag_terms, dim=1)
    laplacian=torch.sum(hessian_pure_der, dim=1, keepdim=True)
    if exp_function:
        df_NN_dx_squared_summed=torch.sum(df_NN_dx**2, dim=1, keepdim=True)
        T=-kinetic_factor*(df_NN_dx_squared_summed+laplacian)
    else:
        T=-kinetic_factor*(laplacian/f_NN)
    
    T=T.detach()
    configurations.requires_grad_(False)
    
    return T

def compute_potential_energy(potentials, particle_configurations):
    with torch.no_grad():
        configurations=particle_configurations
        
        V=0
        for pot, info in potentials:
            pot_energy=pot(x=configurations, **info)
            V+=pot_energy
        return V





