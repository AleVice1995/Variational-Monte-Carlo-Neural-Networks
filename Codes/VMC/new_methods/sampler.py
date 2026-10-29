# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 09:58:32 2026

@author: Ale
"""

import torch
import torch.nn as nn
import numpy as np
from torch.func import vmap, jacrev, functional_call, hessian
from torch.func import jvp
# from torch.utils.data import DataLoader
# from torch.utils.data import TensorDataset

class Sampler():
    def __init__(self, n_particles, dimension,
                 pbc_cond=False, length_box_pbc=0.0,
                 exp_function=True, single_particle_move=False,
                 min_accep_ratio=0.40, max_accep_ratio=0.60):
        
        self.n_particles=n_particles
        self.dimension=dimension
        self.pbc_cond=pbc_cond
        self.length_box_pbc=length_box_pbc
        self.single_particle_move=single_particle_move
        self.exp_function=exp_function
        self.min_accep_ratio=min_accep_ratio
        self.max_accep_ratio=max_accep_ratio
        
        
    # def 
    
    
    
    def metropolis_sampling(self, model, current_config, delta, 
                            new_samples, burn_samples, save_data):
        
        total_samples=new_samples+burn_samples
        
        new_config_list=[]
        accepted_new_samples=0
        # cont_burn=0
        
        # current_config=current_config.reshape(-1, self.dimension)
        if self.pbc_cond:
            L=self.length_box_pbc
        
        with torch.no_grad():
            i=0
            while i<total_samples:
                # cont
                new_config=current_config.clone()
                if self.single_particle_move:
                    particle=i%self.n_particles
                    
                    
                    idx_start = self.dimension * particle
                    idx_end = idx_start + self.dimension
                    
                    displacement=delta*(torch.rand(self.dimension)-0.5)
                    
                    new_config[idx_start:idx_end]+=displacement
                else:
                    displacement=delta*(torch.rand_like(current_config)-0.5)
                    new_config+=displacement
                    
                if self.pbc_cond:
                    new_config=new_config-L*torch.round((new_config/L))
                
                y_new=model(new_config)
                if i==0:
                    y_current=model(current_config)
                    
                if self.exp_function:
                    log_ratio=torch.clamp(2*(y_new-y_current), max=0.0)
                    p=torch.exp(log_ratio)
                else:
                    p=(y_new/y_current).square()
                    p=torch.clamp(p, max=1.0)
                    
                if torch.rand((), device=p.device)<=p:
                    current_config=new_config
                    y_current=y_new
                    if i>=burn_samples:
                        accepted_new_samples+=1
                if save_data and i>=burn_samples:
                    new_config_list.append(current_config.clone())
                
                i+=1
            acceptance_ratio=accepted_new_samples/new_samples
            
            if save_data:
                print(f"delta: {delta:.04f}")
                print(f"Acceptance ratio: {acceptance_ratio:.04f}\n")
                return torch.stack(new_config_list), acceptance_ratio
            return acceptance_ratio    
    
    def delta_finder(self, model, current_config, delta):
        print("Finding new best delta parameter...")
        samples=1000
        delta_list=[]
        func_temp=0
        func_temp_list=[]
        min_acc_ratio=self.min_accep_ratio
        max_acc_ratio=self.max_accep_ratio
        
        
        for i in range(2):
            delta=self.modify_delta(delta, func_temp)
            acceptance_ratio=self.metropolis_sampling(model, current_config, delta, samples, burn_samples=0, save_data=False)
            
            if acceptance_ratio>min_acc_ratio and acceptance_ratio<max_acc_ratio:
                break
            
            delta_list.append(delta)
            func_temp=acceptance_ratio-0.5
            func_temp_list.append(func_temp)
        
        i=0
        num_of_evaluation=20
        while (acceptance_ratio<min_acc_ratio or acceptance_ratio>max_acc_ratio) and i<num_of_evaluation:
            if (np.sign(func_temp_list[0])==-np.sign(func_temp_list[1])):
                delta=(delta_list[0]+delta_list[1])/2
                print(f"Testing delta: {delta:.04f}...")
                acceptance_ratio=self.metropolis_sampling(model, current_config, delta, samples, burn_samples=0, save_data=False)
                print(f"Acceptance ratio: {acceptance_ratio: .04f}")
                func_temp=acceptance_ratio-0.5
                if (np.sign(func_temp)==np.sign(func_temp_list[0])):
                    delta_list[0]=delta
                    func_temp_list[0]=func_temp
                else:
                    delta_list[1]=delta
                    func_temp_list[1]=func_temp
            else:
                # m=(delta_value_list[1]-delta_value_list[0])/(f_temp_list[1]-f_temp_list[0])
                # Delta=delta_value_list[1]-m*f_temp_list[1]
                delta=self.modify_delta(delta_list[-1], func_temp)
                print(f"Testing Delta: {delta:.04f}...")
                acceptance_ratio=self.metropolis_sampling(model, current_config, delta, samples, burn_samples=0, save_data=False)
                print(f"Acceptance ratio: {acceptance_ratio: .04f}")
                func_temp=acceptance_ratio-0.5
                delta_list[0]=delta_list[1]
                delta_list[1]=delta
                func_temp_list[0]=func_temp_list[1]
                func_temp_list[1]=func_temp
            i+=1
        
        print("Search complete\n")
        # acceptance_ratio=self.new_config(curr_config, Delta, 10000, exp_function, save_data=False)
        # print(f"Delta: {Delta:.04f}")
        # print(f"Acceptance ratio: {acceptance_ratio:.04f}\n")
        return delta
    
    def modify_delta(self, Delta, f):
        if f>0:
            increment=1.5
        else:
            increment=1.9
        return Delta*(1+f*increment)