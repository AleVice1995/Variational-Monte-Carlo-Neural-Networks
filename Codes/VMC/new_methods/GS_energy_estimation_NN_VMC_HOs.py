# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 12:28:34 2026

@author: Ale
"""

import torch
from VMC.new_methods.Neural_Networks_models import Neural_Network_VMC_HO
import torch.nn as nn
from VMC.new_methods.sampler import Sampler
from VMC.new_methods.Training import training_NN_VMC
from VMC.new_methods.Potential_energy_list import HO_potential
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from VMC.new_methods.utils import autocorrelation_length




torch.set_default_dtype(torch.float64)
# device = torch.device("cuda")
device=torch.device("cpu")
if device.type == "cuda":
    torch.cuda.init()
    

print(f"Dispositivo utilizzato: {device}")

start_datetime=datetime.now()
start=time.time()


    



n_particles=1
dimension=1
coordinates=dimension*n_particles
diag_shift=1e-3
start_delta_finder=5
MC_samples=5000
burn_samples=1000
epochs=300
delta=13.0
learning_rate=1e-2
reg_param=0.0

single_particle_move=True
exp_function=True
pbc_cond=False
length_box_pbc=0.0
kinetic_factor=0.5

HO_factor_molt=0.5
info_HO={
    "HO_factor_molt": 0.5}
pot_HO_info=[HO_potential, info_HO]
potentials=[pot_HO_info]
energy_info=[kinetic_factor, potentials]

starting_config=2*(torch.rand(coordinates, device=device)-0.5)








layers=([coordinates, 64, 64, 1])
activations=[nn.Softplus(), nn.Softplus(), nn.Identity()]

model_HO=Neural_Network_VMC_HO(layers, activations)

sampler=Sampler(n_particles=n_particles, dimension=dimension,
                pbc_cond=pbc_cond, length_box_pbc=length_box_pbc, 
                exp_function=exp_function, single_particle_move=single_particle_move)
optimizer=torch.optim.SGD(model_HO.parameters(), lr=learning_rate, weight_decay=reg_param)
# scheduler=None
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=50,  # Every 50 epochs
    gamma=0.7      # Multiply LR by 0.1
)
data=training_NN_VMC(model_HO, sampler, epochs, 
                    starting_config, delta, energy_info,
                    MC_samples, burn_samples,
                    start_delta_finder, 
                    optimizer, scheduler, diag_shift)





end_datetime=datetime.now()
end=time.time()
elapsed=end-start
hours, remainder=divmod(elapsed, 3600)
minutes, seconds=divmod(remainder, 60)
print(f"\nRun started at: {start_datetime.strftime('%d/%m/%Y %H:%M:%S')}")
print(f"Run finished at: {end_datetime.strftime('%d/%m/%Y %H:%M:%S')}")
print(f"\nElapsed time: {hours:.0f} hh {minutes:.0f} mm {seconds:.2f} ss\n")


#%%
local_energies_last_epoch, energy_mean_list, energy_std_dev_list, energy_stoch_err_list,\
        T_mean_list, V_mean_list, delta_list, acceptance_ratio_list,\
            current_config, grad_list, lr_list=data



#Plotting Data
iterations=np.arange(1, epochs+1)
plt.close('all')
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
        
        
        
        plt.figure(num=0, dpi=120)
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
    
    
    plt.figure(num=1, dpi=120)
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
    plt.figure(num=2, dpi=120)
    plt.plot(x_data, psi_guess**2, marker='.', linestyle='None', label='general behaviour of NN')
    plt.plot(x_batch, psi_guess_on_batch**2, marker='.', linestyle='None', label='Metropoli config')
    plt.plot(x, psi_start**2, marker='+', ms=10, linestyle='None', label='starting point')
    plt.plot(x_data, psi2_model_HO, marker='None', label='model')
    plt.xlabel('Position')
    plt.ylabel(r"$|\Psi|^2$")
    plt.grid(True)
    plt.legend()
    plt.show()



energies=np.array(energy_mean_list)/n_particles
en_std_devs=np.array(energy_std_dev_list)/n_particles
en_errors=np.array(energy_stoch_err_list)/n_particles
kinetic_energies=np.array(T_mean_list)/n_particles
potential_energies=np.array(V_mean_list)/n_particles
# min_en=(min(energies))
# min_en_std=en_std_devs(energies.index(min_en))
# print(f"Minimum energy: {min(energies):.04f}")
plt.figure(num=3, dpi=120)
plt.plot(iterations, kinetic_energies, marker='.', linestyle='-', label="NN Kinetic")
plt.plot(iterations, potential_energies, marker='.', linestyle='-', label="NN Potential")
plt.plot(iterations, energies, marker='.', linestyle='-', label="NN Energy")
# plt.errorbar(iterations, model_HO.energy_list, yerr=model_HO.en_std_dev_list, fmt='o-', capsize=4, markersize=5, label="Energia VMC")
plt.axhline(y=0.5, color='red', linestyle='--', label="Exact energy HO")
plt.xlabel("Epoch")
plt.ylabel("Energy")
plt.ylim([0,1])
plt.grid(True)
plt.legend()
plt.show()

plt.figure(num=2, dpi=120)
plt.plot(iterations, lr_list, marker='.', linestyle='None')
plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
plt.xlabel("Epoch")
plt.ylabel("Learning rate")
plt.grid(True)
plt.show()

plt.figure(num=4, dpi=120)
plt.plot(iterations, delta_list, marker='.', linestyle='-')
plt.xlabel("Epoch")
plt.ylabel("Delta")
plt.grid(True)
plt.show()



plt.figure(num=5, dpi=120)
plt.plot(iterations, acceptance_ratio_list, marker='.', linestyle='-')
plt.xlabel("Epoch")
plt.ylabel("Acceptance ratio [%]")
plt.grid(True)
plt.show()

plt.figure(num=6, dpi=120)
plt.plot(iterations, grad_list, marker='.', linestyle='-')
plt.xlabel("Epoch")
plt.ylabel(r"||$\nabla$||")
plt.grid(True)
plt.show()

# loss=model_HO.loss_list
# plt.figure(num=4, dpi=120)
# plt.plot(iterations, loss, marker='.', linestyle='-')
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.grid(True)
# plt.show()

autocorr_terms=7

if autocorr_terms<MC_samples:
    C_tau_terms_tens=autocorrelation_length(local_energies_last_epoch, autocorr_terms=autocorr_terms)
    C_tau_terms=C_tau_terms_tens.detach().cpu().numpy()
    C_tot=np.sum(C_tau_terms)
    tau=np.arange(autocorr_terms)
    
    x_data_exp_model_HO=np.linspace(0,autocorr_terms-1, 1000)
    y_data_exp_model_HO=np.exp(-x_data_exp_model_HO/C_tot)
    
    plt.figure(num=7, dpi=120)
    plt.plot(tau, C_tau_terms, marker='.', linestyle='-', label='data')
    plt.plot(x_data_exp_model_HO, y_data_exp_model_HO, marker='None', linestyle='-', label='model')
    plt.ylabel(r"$c(E)_{\tau}$")
    plt.xlabel(r"$\tau$")
    plt.grid(True)
    plt.legend()
    plt.show()

#%%

weights = model_HO.layers[0].weight.detach().numpy().flatten()
plt.figure(num=8, dpi=120)
plt.hist(weights, bins=10, density=True, edgecolor="black")
plt.xlabel("Valore del peso")
plt.ylabel("Frequenza")
plt.title("Distribuzione dei pesi del primo layer")
plt.grid(True, alpha=0.3)
plt.show()
#%%


torch.save({
    "model_state_dict": model_HO.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "epochs_completed": epochs,
    "learning_rate": learning_rate,
}, "VMC/checkpoint_HO_minSR.pth")





