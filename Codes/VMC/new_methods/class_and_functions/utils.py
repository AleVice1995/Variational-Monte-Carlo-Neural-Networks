# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 14:18:15 2026

@author: Ale
"""

import torch
import numpy as np
import textwrap

def autocorrelation_length(local_energies, autocorr_terms=100):
    
    #Autocorrelation function: R(tau)=<F(x_i)*F(x_{i+tau})>
    #Auto-covariance function: C(tau)=<[F(x_i)-F_mean]*[F(x_{i+tau})-F_mean]>=R(tau)-<F_mean>^2
    #Normlaized Auto-correlation function: rho(tau)=<[F(x_i)-F_mean]*[F(x_{i+tau})-F_mean]>/Var(F)=C(tau)/C(0)
    
    # Here we are computing the nomalized autoccrelation functions
    c_autocorr=torch.zeros(autocorr_terms, 
                           device=local_energies.device, dtype=local_energies.dtype)
    samples=local_energies.shape[0]
    energy_mean=local_energies.mean()
    deviations_en=local_energies-energy_mean
    en_var=(deviations_en**2).mean()
    if autocorr_terms<samples:
        with torch.no_grad():
            for j in range(autocorr_terms):
                        # the following correspond to: [F(x_i)-F_mean]*[F(x_{i+tau})-F_mean]
                        prod_terms=(deviations_en[:,0])*(deviations_en[:,0]) if j==0 else ((deviations_en[j:,0])*(deviations_en[:-j,0]))
                        # The following correspond to: <[F(x_i)-F_mean]*[F(x_{i+tau})-F_mean]> so here we compute the mean of the factor omputed before
                        c_autocorr[j]=prod_terms.mean()
            # Here in the following, we normalized with respect to the variance: <[F(x_i)-F_mean]*[F(x_{i+tau})-F_mean]>/Var(F)=C(tau)/C(0)
            c_autocorr=(c_autocorr)/en_var
        return c_autocorr
    else:
        print("Number of autocorrelations terms less than MC samples")        
        

def autocorrelation_length_Pearson(local_energies, autocorr_terms=100):
# =============================================================================
#     The pearson method here defined should give the same result as the previous one
#     for a wide-sense stationary (WSS) process.
#     
#     A WSS (Wide-Sense Stationary) process, or weakly stationary process,
#     is a stochastic process whose fundamental statistical properties do not change over time.
#     In particular, it must satisfy three conditions.
#     ### 1. Constant mean over time
#     The mean value of the process must not depend on the time instant:
#         <F(t)>=mu=constant
#     2. Constant variance over time
#     The variance must also be independent of time:
#         Var(F(t))=sigma^2=constant
#     
#     3. Autocovariance depends only on the time difference
#     The autocovariance between two time instants must depend only on their time separation, not on the individual time instants:
#         
#         C(t_1,t_2)=C(t_2-t_1)
#         
#     For example, the correlation between F(10) and F(15) must be equal to the correlation between F(100) and F(105),
#     because in both cases the time separation is 5.
#     
#     the previous uses mean and variance of all the samples for every j
#     while here we calculate dthe mean and std for every 2 sets.
#     When the number of samples is large and much longer than the correlation time 
#     then the two of them should coincide
#     Autocorrelation function: R(tau)=<F(x_i)*F(x_{i+tau})>
#     Auto-covariance function: C(tau)=<[F(x_i)-F_mean]*[F(x_{i+tau})-F_mean]>=R(tau)-<F_mean>^2
#     Normlaized Auto-correlation function: rho(tau)=<[F(x_i)-F_mean]*[F(x_{i+tau})-F_mean]>/Var(F)=C(tau)/C(0)
# =============================================================================
    
    # Here we are computing the nomalized autoccrelation functions
    # c_autocorr=torch.zeros(autocorr_terms, 
    #                        device=local_energies.device, dtype=local_energies.dtype)
    c_autocorr=[]
    samples=local_energies.shape[0]
    # energy_mean=local_energies.mean()
    # en_std_dev=((local_energies**2).mean()-(energy_mean**2)).sqrt()
    # deviations_en=local_energies-energy_mean
    
    local_energies=local_energies.numpy()
    # cont=0
    if autocorr_terms<samples:
        for j in range(autocorr_terms):            
            if j==0:
                en_mean=np.mean(local_energies)
                c_0=np.mean((local_energies-en_mean)**2)/np.var(local_energies)
                c_autocorr.append(c_0)
            else:
                en_mean_set1=np.mean(local_energies[j:])
                en_std_set1=np.std(local_energies[j:])
                en_dev_set1=local_energies[j:]-en_mean_set1
                
                
                en_mean_set2=np.mean(local_energies[:-j])
                en_std_set2=np.std(local_energies[:-j])
                en_dev_set2=local_energies[:-j]-en_mean_set2
                
                c_j=np.mean(en_dev_set1*en_dev_set2)/(en_std_set1*en_std_set2)
                c_autocorr.append(c_j)   
        return np.array(c_autocorr)
    else:
        print("Number of autocorrelations terms less than MC samples")




#WRONG METHOD
def autocorrelation_length_old(local_energies, autocorr_terms=100):
    
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
                    c_autocorr[j]+=(local_energies[i,0])*(local_energies[i-j,0])
            
            print(f"cont: {cont:d}")
            c_autocorr=c_autocorr/cont
            c_autocorr=(c_autocorr-energy_mean**2)/(en_std_dev**2)
        return c_autocorr
    else:
        print("Number of autocorrelations terms less than MC samples")
        

def print_values_on_log(problem_under_study, n_particles, dimension, starting_config, energy_info,
                        layers, activations, epochs, samples, batch, sweep_per_save, diag_shift, reg_param, 
                        exp_function, time_control, pbc_cond=False, density=None, Length_box=None, switch_power=1.0,
                        file=None):
    if Length_box is not None and isinstance(Length_box, torch.Tensor):
        Length_box=Length_box.item()
    text=textwrap.dedent(f"""\
                         {problem_under_study}
                         
                         Parameter values of the run:
                         Number of particles: {n_particles: d}
                         Number of dimensions: {dimension: d}
                         
                         Total Epochs of training: {epochs: d}
                         Monte Carlo samples: {samples: d}
                         Batch utilized: {batch: d}
                         Configurations swept per save: {sweep_per_save: d}
                         
                         MinSR diagonalization parameter: {diag_shift: .04e}
                         Regularization parameter SGD: {reg_param: .04e}
                         
                         f_NN=log(Psi): {exp_function}
                         Energy clipping controlled by epoch: {time_control}
                         
                         
                         """)
    if pbc_cond:
       text+=textwrap.dedent(f"""\
                             System studied under periodic boundary conditions: {pbc_cond}
                             Density of system: {density: .04f} A^-3
                             Length of periodic virtual box: {Length_box: .04f} A
                             Power of switch function: {switch_power: d}
                             
                             
                             """)
    print(text)
    
    if file is not None:
        file.write(text)
    
    data= (n_particles, dimension, starting_config, energy_info,\
           layers, activations, epochs, samples, batch, sweep_per_save,\
        diag_shift, reg_param, exp_function, time_control)
    if pbc_cond:
        data=(*data, pbc_cond, density, Length_box, switch_power)
        
    return data

def compute_der_for_plots(x, model, norm_const):
    
    x=x.requires_grad_(True)
    
    f=model(x)-model(norm_const)
    f=torch.exp(f)
    f_der=torch.autograd.grad(outputs=f.sum(), inputs=x, create_graph=True)[0]
    f_sec_der=torch.autograd.grad(outputs=f_der.sum(), inputs=x, create_graph=False)[0]
    x=x.requires_grad_(False)
    return f.detach(), f_der.detach(), f_sec_der.detach()
    