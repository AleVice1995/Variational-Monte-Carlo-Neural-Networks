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
import textwrap

def compute_energy_local(model, particle_configurations, kinetic_factor, potentials,
                         exp_function, n_particles, file=None):
    
    # T=0
    # V=0
    # start_kin=time.time()
    # print("Computing kinetic term:")
    T, T_grad, T_JF=compute_kinetic_energy_local(model, particle_configurations, kinetic_factor, exp_function)
    # end_kin=time.time()
    # elapsed=end_kin-start_kin
    # hours, remainder=divmod(elapsed, 3600)
    # minutes, seconds=divmod(remainder, 60)
    # print(f"\nElapsed time: {hours:.0f} hh {minutes:.0f} mm {seconds:.2f} ss\n")
    
    # start_pot=time.time()
    # print("Computing potential term:")
    V=compute_potential_energy(potentials, particle_configurations)
    # end_pot=time.time()
    # elapsed=end_pot-start_pot
    # hours, remainder=divmod(elapsed, 3600)
    # minutes, seconds=divmod(remainder, 60)
    # print(f"\nElapsed time: {hours:.0f} hh {minutes:.0f} mm {seconds:.2f} ss\n")
    V_mean=V.mean()
    T_mean=T.mean()
    T_mean_grad=T_grad.mean()
    T_mean_JF=T_JF.mean()
    
    local_energies, energy_mean, energy_std_dev, energy_stoch_err=statistics(T, V)
    local_energies_grad, energy_mean_grad, energy_std_dev_grad, energy_stoch_err_grad=statistics(T_grad, V)
    local_energies_JF, energy_mean_JF, energy_std_dev_JF, energy_stoch_err_JF=statistics(T_JF, V)
    
    
    print(f"System of {n_particles: d} particles\n")
    print("Total Energy of system:")
    print(f"T_mean: {T_mean: .04f}")
    print(f"V_mean: {V_mean: .04f}")
    print(f"E mean: {energy_mean: .04f}")
    print(f"E std_dev: {energy_std_dev: .04f}")
    print(f"E stoch_err: {energy_stoch_err: .04f}")
    
    print(f"T_mean_grad: {T_mean_grad: .04f}")
    print(f"E mean grad: {energy_mean_grad: .04f}")
    print(f"E std_dev grad: {energy_std_dev_grad: .04f}")
    print(f"E stoch_err grad: {energy_stoch_err_grad: .04f}")
    
    print(f"T_mean_JF: {T_mean_JF: .04f}")
    print(f"E mean JF: {energy_mean_JF: .04f}")
    print(f"E std_dev JF: {energy_std_dev_JF: .04f}")
    print(f"E stoch_err JF: {energy_stoch_err_JF: .04f}\n")
    
    
    if n_particles!=1:
        print("Energy per particle:")
        print(f"T_mean: {T.mean()/n_particles: .04f}")
        print(f"V_mean: {V.mean()/n_particles: .04f}")
        print(f"E mean: {energy_mean/n_particles: .04f}")
        print(f"E std_dev: {energy_std_dev/n_particles: .04f}")
        print(f"E stoch_err: {energy_stoch_err/n_particles: .04f}")
        
        print(f"T_mean_grad: {T_mean_grad/n_particles: .04f}")
        print(f"E mean grad: {energy_mean_grad/n_particles: .04f}")
        print(f"E std_dev grad: {energy_std_dev_grad/n_particles: .04f}")
        print(f"E stoch_err grad: {energy_stoch_err_grad/n_particles: .04f}")
        
        print(f"T_mean_JF: {T_mean_JF/n_particles: .04f}")
        print(f"E mean JF: {energy_mean_JF/n_particles: .04f}")
        print(f"E std_dev JF: {energy_std_dev_JF/n_particles: .04f}")
        print(f"E stoch_err JF: {energy_stoch_err_JF/n_particles: .04f}\n")
    
    if file is not None:
        text=textwrap.dedent(f"""\
            System of {n_particles: d} particles\n
            Total Energy of system:
            T_mean: {T_mean: .04f}
            V_mean: {V_mean: .04f}
            E mean: {energy_mean: .04f}
            E std_dev: {energy_std_dev: .04f}
            E stoch_err: {energy_stoch_err: .04f}
            T_mean_grad: {T_mean_grad: .04f}
            E mean grad: {energy_mean_grad: .04f}
            E std_dev grad: {energy_std_dev_grad: .04f}
            E stoch_err grad: {energy_stoch_err_grad: .04f}
            T_mean_JF: {T_mean_JF: .04f}
            E mean JF: {energy_mean_JF: .04f}
            E std_dev JF: {energy_std_dev_JF: .04f}
            E stoch_err JF: {energy_stoch_err_JF: .04f}\n
            """)
        if n_particles!=1:
            text+=textwrap.dedent(f"""\
                Energy per particle:
                T_mean: {T.mean()/n_particles: .04f}
                V_mean: {V.mean()/n_particles: .04f}
                E mean: {energy_mean/n_particles: .04f}
                E std_dev: {energy_std_dev/n_particles: .04f}
                E stoch_err: {energy_stoch_err/n_particles: .04f}
                T_mean_grad: {T_mean_grad/n_particles: .04f}
                E mean grad: {energy_mean_grad/n_particles: .04f}
                E std_dev grad: {energy_std_dev_grad/n_particles: .04f}
                E stoch_err grad: {energy_stoch_err_grad/n_particles: .04f}
                T_mean_JF: {T_mean_JF/n_particles: .04f}
                E mean JF: {energy_mean_JF/n_particles: .04f}
                E std_dev JF: {energy_std_dev_JF/n_particles: .04f}
                E stoch_err JF: {energy_stoch_err_JF/n_particles: .04f}\n
                """)
        file.write(text)
        
    return local_energies, energy_mean, energy_std_dev, energy_stoch_err, V_mean, T_mean,\
        T_mean_grad, energy_mean_grad, T_mean_JF, energy_mean_JF
    
    
    
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
        T_grad=kinetic_factor*df_NN_dx_squared_summed
        T_JF=-0.5*kinetic_factor*laplacian
    else:
        grad_psi_over_psi = df_NN_dx / f_NN
        grad_psi_over_psi_squared = torch.sum(grad_psi_over_psi**2, dim=1, keepdim=True)
        lap_psi_over_psi = laplacian / f_NN
        
        T=-kinetic_factor*lap_psi_over_psi
        T_grad=kinetic_factor*grad_psi_over_psi_squared
        T_JF=-0.5*kinetic_factor*(lap_psi_over_psi-grad_psi_over_psi_squared)
    
    T=T.detach()
    T_grad=T_grad.detach()
    T_JF=T_JF.detach()
    configurations.requires_grad_(False)
    
    return T, T_grad, T_JF

def compute_potential_energy(potentials, particle_configurations):
    with torch.no_grad():
        configurations=particle_configurations
        
        V=0
        for pot, info in potentials:
            pot_energy=pot(x=configurations, **info)
            V+=pot_energy
        return V



def statistics(T, V):
    local_energies=T+V
    samples=len(local_energies)
    energy_mean=local_energies.mean()   
    if samples>1:
        energy_var=local_energies.var(correction=1)
        energy_std_dev=local_energies.std(correction=1)
    else:
        energy_var=local_energies.var(correction=0)
        energy_std_dev=local_energies.std(correction=0)
        
    energy_stoch_err=(energy_var/samples).sqrt()
        
    return local_energies, energy_mean, energy_std_dev, energy_stoch_err

