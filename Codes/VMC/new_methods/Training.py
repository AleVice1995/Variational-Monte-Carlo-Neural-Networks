# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 12:25:59 2026

@author: Ale
"""

import torch
from VMC.new_methods.Energy import compute_energy_local
from VMC.new_methods.Optimizers import minSR
import time


def training_NN_VMC(model, sampler, epochs, 
                    starting_config, delta, energy_info,
                    MC_samples, burn_samples,
                    start_delta_finder, 
                    optimizer, scheduler, diag_shift):
    
    min_acc_ratio=sampler.min_accep_ratio
    max_acc_ratio=sampler.max_accep_ratio
    acceptance_ratio=0.0
    acceptance_ratio_list=[]
    delta_list=[]
    
    
    exp_function=sampler.exp_function
    n_particles=sampler.n_particles
    
    
    kinetic_factor=energy_info[0]
    potentials=energy_info[1]
    energy_mean_list=[]
    energy_std_dev_list=[]
    energy_stoch_err_list=[]
    T_mean_list=[]
    V_mean_list=[]
    grad_list=[]
    lr_list=[]
    
    
    
    
    for epoch in range(epochs):
        print(f"Epoch: {epoch+1:d}")
        current_lr=optimizer.param_groups[0]["lr"]
        print(f"Learning rate: {current_lr:.04e}\n")
        condition_delta_finder=(start_delta_finder is not None) and (epoch>=start_delta_finder)
        if (condition_delta_finder and (acceptance_ratio<min_acc_ratio or acceptance_ratio>max_acc_ratio)):
            delta=sampler.delta_finder(model, starting_config, delta)
        delta_list.append(delta)
        start_conf=time.time()
        print("Sampling:")
        current_config, acceptance_ratio=sampler.metropolis_sampling(model, starting_config, delta, 
                                MC_samples, burn_samples, save_data=True)
        end_conf=time.time()
        elapsed=end_conf-start_conf
        hours, remainder=divmod(elapsed, 3600)
        minutes, seconds=divmod(remainder, 60)
        print(f"\nElapsed time: {hours:.0f} hh {minutes:.0f} mm {seconds:.2f} ss\n")
        
        acceptance_ratio_list.append(acceptance_ratio)
        start_en=time.time()
        print("Computing Energy:")
        data=compute_energy_local(model, current_config, kinetic_factor, potentials,
                                  exp_function, n_particles)
        end_en=time.time()
        elapsed=end_en-start_en
        hours, remainder=divmod(elapsed, 3600)
        minutes, seconds=divmod(remainder, 60)
        print(f"\nElapsed time: {hours:.0f} hh {minutes:.0f} mm {seconds:.2f} ss\n")
        
        local_energies=data[0]
        energy_mean=data[1]
        
        start_minSR=time.time()
        print("Applying minSR::")
        grad_norm=minSR(model, current_config, n_particles,
                  local_energies, energy_mean, 
                  optimizer, exp_function, diag_shift)
        end_minSR=time.time()
        elapsed=end_minSR-start_minSR
        hours, remainder=divmod(elapsed, 3600)
        minutes, seconds=divmod(remainder, 60)
        print(f"\nElapsed time: {hours:.0f} hh {minutes:.0f} mm {seconds:.2f} ss\n")
        
        
        
        energy_mean_list.append(energy_mean)
        energy_std_dev_list.append(data[2])
        energy_stoch_err_list.append(data[3])
        T_mean_list.append(data[4])
        V_mean_list.append(data[5])
        grad_list.append(grad_norm)
        starting_config=current_config[-1]
        lr_list.append(current_lr)
        
        
        
        if scheduler is not None:
            # scheduler.step(energy_mean)
            scheduler.step()
        print("")
        
    return local_energies, energy_mean_list, energy_std_dev_list, energy_stoch_err_list,\
            T_mean_list, V_mean_list, delta_list, acceptance_ratio_list,\
                current_config, grad_list, lr_list

        
        