# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 21:27:46 2026

@author: Ale
"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from VMC.new_methods.class_and_functions.Neural_Networks_models import  Neural_Network_VMC_HO
from VMC.new_methods.class_and_functions.utils import autocorrelation_length
from VMC.new_methods.class_and_functions.sampler import Sampler
from VMC.new_methods.class_and_functions.Energy import compute_energy_local
device=torch.device("cpu")

checkpoint = torch.load("VMC/new_methods/harmonic_oscillators/checkpoint_HO.pth", map_location=device, weights_only=False)
input_data=checkpoint["input_data"]
output_data=checkpoint["output_data"]


(n_particles, dimension, starting_config, energy_info,\
 layers, activations, epochs, samples, batch, sweep_per_save,\
     diag_shift, reg_param, exp_function, time_control)=input_data
    
coordinates=n_particles*dimension
[kinetic_factor, potentials]=energy_info

model_HO=Neural_Network_VMC_HO(layers, activations)
model_HO.load_state_dict(checkpoint["model_state_dict"])

local_energies_last_epoch, energy_mean_list, energy_std_dev_list, energy_stoch_err_list,\
        T_mean_list, V_mean_list, delta_list, acceptance_ratio_list,\
            current_config, grad_list, lr_list,\
                fraction_nonfinite_values_list, fraction_clipped_values_list, control_list,\
                    T_mean_grad_list, energy_mean_grad_list,\
                        T_mean_JF_list, energy_mean_JF_list=output_data
                        
                        
                        
                        

#Plotting Data
iterations=np.arange(1, epochs+1)
plt.close('all')
fig_num=0
if coordinates==1:
    x_batch_tensors=current_config
    x_data_tensors=torch.linspace(-4, 4, 2000).unsqueeze(1)
    
    
    with torch.no_grad():
        y_tensors=model_HO(starting_config)-model_HO(torch.tensor([0.0,]))
        y_guess_on_batch_tensors=model_HO(x_batch_tensors)-model_HO(torch.tensor([0.0,]))
        y_guess_tensors=model_HO(x_data_tensors)-model_HO(torch.tensor([0.0,]))
    
    x=starting_config.detach().numpy()    
    x_batch=x_batch_tensors.detach().numpy()
    x_data=x_data_tensors.detach().numpy()
    f_NN_start=y_tensors.detach().numpy()
    f_NN_guess=y_guess_tensors.detach().numpy()
    f_NN_guess_on_batch=y_guess_on_batch_tensors.detach().numpy()
    
    
    # print(model_HO)
    
    psi_model_HO=np.exp(-(x_data**2)/2)
    psi2_model_HO=np.exp(-(x_data**2))
    
    
    if exp_function:
        psi_start=np.exp(f_NN_start)
        psi_guess_on_batch=np.exp(f_NN_guess_on_batch)
        psi_guess=np.exp(f_NN_guess)
        
        
        
        plt.figure(num=(fig_num:=fig_num+1), dpi=120)
        plt.plot(x_data, f_NN_guess, marker='.', linestyle='None', label='general behaviour of NN')
        plt.plot(x_batch, f_NN_guess_on_batch, marker='.', linestyle='None', label='Metropoli config')
        plt.plot(x, f_NN_start, marker='+', ms=10, linestyle='None', label='starting point')
        plt.plot(x_data, -(x_data**2)/2, marker='None', label='model')
        plt.xlabel('Position')
        plt.ylabel(r"$f_{NN}$")
        plt.grid(True)
        plt.legend()
        plt.show()
    else:
        plt.close('all')
        psi_start=f_NN_start
        psi_guess_on_batch=f_NN_guess_on_batch
        psi_guess=f_NN_guess
    
    
    plt.figure(num=(fig_num:=fig_num+1), dpi=120)
    plt.plot(x_data, psi_guess, marker='.', linestyle='None', label='general behaviour of NN')
    plt.plot(x_batch, psi_guess_on_batch, marker='.', linestyle='None', label='Metropoli config')
    plt.plot(x, psi_start, marker='+', ms=10, linestyle='None', label='starting point')
    plt.plot(x_data, psi_model_HO, marker='None', label='model')
    plt.xlabel('Position')
    plt.ylabel(r"$\Psi$")
    plt.grid(True)
    plt.legend()
    plt.show()
    
    
    
    # plt.close('all')
    plt.figure(num=(fig_num:=fig_num+1), dpi=120)
    plt.plot(x_data, psi_guess**2, marker='.', linestyle='None', label='general behaviour of NN')
    plt.plot(x_batch, psi_guess_on_batch**2, marker='.', linestyle='None', label='Metropoli config')
    plt.plot(x, psi_start**2, marker='+', ms=10, linestyle='None', label='starting point')
    plt.plot(x_data, psi2_model_HO, marker='None', label='model')
    plt.xlabel('Position')
    plt.ylabel(r"$|\Psi|^2$")
    plt.grid(True)
    plt.legend()
    plt.show()


potential_energies=np.array(V_mean_list)/n_particles
energies=np.array(energy_mean_list)/n_particles
en_std_devs=np.array(energy_std_dev_list)/n_particles
en_errors=np.array(energy_stoch_err_list)/n_particles
kinetic_energies=np.array(T_mean_list)/n_particles
kinetic_grad_energies=np.array(T_mean_grad_list)/n_particles
energies_grad=np.array(energy_mean_grad_list)/n_particles
kinetic_JF_energies=np.array(T_mean_JF_list)/n_particles
energies_JF=np.array(energy_mean_JF_list)/n_particles
# min_en=(min(energies))
# min_en_std=en_std_devs(energies.index(min_en))
# print(f"Minimum energy: {min(energies):.04f}")
plt.figure(num=(fig_num:=fig_num+1), dpi=120)
plt.plot(iterations, potential_energies, marker='None', linestyle='-', zorder=3, label="NN V")
plt.plot(iterations, kinetic_energies, marker='None', linestyle='-', zorder=3, label="NN T")
plt.plot(iterations, energies, marker='None', linestyle='-', zorder=3, label="NN E")
plt.plot(iterations, kinetic_grad_energies, marker='None', linestyle='-', label=r"NN $T_{\nabla}$")
plt.plot(iterations, energies_grad, marker='None', linestyle='-', label=r"NN $E_{\nabla}$")
plt.plot(iterations, kinetic_JF_energies, marker='None', linestyle='-', label=r"NN $T_{JF}$")
plt.plot(iterations, energies_JF, marker='None', linestyle='-', label=r"NN $E_{JF}$")
# plt.errorbar(iterations, model_HO.energy_list, yerr=model_HO.en_std_dev_list, fmt='o-', capsize=4, markersize=5, label="Energia VMC")
plt.axhline(y=0.5, color='red', linestyle='--', label="Exact energy HO")
plt.xlabel("Epoch")
plt.ylabel("Energy")
plt.ylim([-0.1*energies[-1],1.7*energies[-1]])
plt.grid(True)
plt.legend()
plt.show()

plt.figure(num=(fig_num:=fig_num+1), dpi=120)
plt.plot(iterations, lr_list, marker='.', linestyle='None')
plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
plt.xlabel("Epoch")
plt.ylabel("Learning rate")
plt.grid(True)
plt.show()

plt.figure(num=(fig_num:=fig_num+1), dpi=120)
plt.plot(iterations, delta_list, marker='.', linestyle='-')
plt.xlabel("Epoch")
plt.ylabel("Delta")
plt.grid(True)
plt.show()



plt.figure(num=(fig_num:=fig_num+1), dpi=120)
plt.plot(iterations, acceptance_ratio_list, marker='.', linestyle='-')
plt.xlabel("Epoch")
plt.ylabel("Acceptance ratio [%]")
plt.grid(True)
plt.show()

plt.figure(num=(fig_num:=fig_num+1), dpi=120)
plt.plot(iterations, grad_list, marker='.', linestyle='-')
plt.xlabel("Epoch")
plt.ylabel(r"||$\nabla$||")
plt.grid(True)
plt.show()

plt.figure(num=(fig_num:=fig_num+1), dpi=120)
plt.plot(iterations, fraction_nonfinite_values_list, marker='.', linestyle='-', label="non-finite values")
plt.plot(iterations, fraction_clipped_values_list, marker='.', linestyle='-', label="clipped values")
plt.xlabel("Epoch")
plt.ylabel(r"Percentage [%]")
plt.grid(True)
plt.legend()
plt.show()

plt.figure(num=(fig_num:=fig_num+1), dpi=120)
plt.plot(iterations, control_list, marker='.', linestyle='-')
plt.xlabel("Epoch")
plt.ylabel(r"Control function value")
plt.grid(True)
# plt.legend()
plt.show()

# loss=model_HO.loss_list
# plt.figure(num=4, dpi=120)
# plt.plot(iterations, loss, marker='.', linestyle='-')
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.grid(True)
# plt.show()

#%%
sampler=Sampler(n_particles=n_particles, dimension=dimension, 
                exp_function=exp_function)
MC_samples=5000
batch_autocorr=1
current_config_autocorr, acceptance_ratio_autocorr=sampler.metropolis_sampling_parallel_walkers_batch(model_HO, current_config, delta_list[-1], 
                        MC_samples, sweep_per_save=10, batch=batch_autocorr, burned_sweep=3, save_data=True)

data_autocorr=compute_energy_local(model_HO, current_config_autocorr, kinetic_factor, potentials,
                          exp_function, n_particles)
local_energies_autocorr=data_autocorr[0]
local_energies_autocorr=local_energies_autocorr.reshape(-1,batch_autocorr,1)

#%%
saved_samples_per_walker=local_energies_autocorr.shape[0]
autocorr_terms=25
C_tau_terms_per_walker=[]
if autocorr_terms<saved_samples_per_walker:
    for i in range(batch_autocorr):

        C_tau_terms_tens=autocorrelation_length(local_energies_autocorr[:,i], autocorr_terms=autocorr_terms)
        # C_tau_terms=C_tau_terms_tens.detach().cpu().numpy()
        C_tau_terms_per_walker.append(C_tau_terms_tens)
        
    C_tau_terms_per_walker=torch.stack(C_tau_terms_per_walker)
    C_tau_terms_mean=torch.mean(C_tau_terms_per_walker, dim=0)
    C_tau_terms_mean=C_tau_terms_mean.detach().cpu().numpy()
    C_tot=0.5+np.sum(C_tau_terms_mean[1:])
    # C_tot=np.sum(C_tau_terms_mean)
    tau=np.arange(autocorr_terms)
    
    x_data_exp_model_HO=np.linspace(0,autocorr_terms-1, 1000)
    y_data_exp_model_HO=np.exp(-x_data_exp_model_HO/(C_tot))
    
    plt.close(fig=fig_num)
    plt.figure(num=(fig_num:=fig_num+1), dpi=120)
    plt.plot(tau, C_tau_terms_mean, marker='.', linestyle='-', label='data')
    plt.plot(x_data_exp_model_HO, y_data_exp_model_HO, marker='None', linestyle='-', label='model')
    plt.ylabel(r"$c(E)_{\tau}$")
    plt.xlabel(r"$\tau$")
    plt.grid(True)
    plt.legend()
    plt.show()

#%%

weights = model_HO.layers[0].weight.detach().numpy().flatten()
plt.figure(num=(fig_num:=fig_num+1), dpi=120)
plt.hist(weights, bins=10, density=True, edgecolor="black")
plt.xlabel("Valore del peso")
plt.ylabel("Frequenza")
plt.title("Distribuzione dei pesi del primo layer")
plt.grid(True, alpha=0.3)
plt.show()

