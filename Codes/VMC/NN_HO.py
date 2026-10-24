# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 16:11:13 2026

@author: Ale
"""
import time
from datetime import datetime
from VMC.NN_class_for_VMC import Neural_Network_VMC as NNVMC
from VMC.NN_class_for_VMC import HO_potential
from VMC.NN_class_for_VMC import tot_para_NN
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np

# device = torch.device("cuda")
device=torch.device("cpu")
if device.type == "cuda":
    torch.cuda.init()
    
print(f"Dispositivo utilizzato: {device}")

start_datetime=datetime.now()
start=time.time()

# seed = 1445#123#123423#8

# print(f"Seed: {seed:d}\n")
# torch.manual_seed(seed)
# np.random.seed(seed)
start_delta_finder=10
N_particles=1
single_particle_move=False
MC_samples=2200#600#1200
epochs=3000
learning_rate=1e-2
reg_param=5.0/MC_samples
exp_function=True #False
patience=50
burn_data=600

use_minSR=True
diag_shift=1e-3

input_neurons=1
starting_config=2*(torch.rand(input_neurons, device=device)-0.5)#torch.tensor([0.9472])torch.tensor([0.5],)
Delta=13.0#22.0



layers=([1,64, 64,  1])
activations=(nn.Softplus(), nn.Softplus(), nn.Identity())  
tot_param=tot_para_NN(layers)

# layers=([input_neurons, first_hidden_layer_neurons, second_hidden_layer_neurons,
#          third_hidden_layer_neurons, fourth_hidden_layer_neurons, ouput_neurons])
# activations=(nn.Softplus(), nn.Softplus(), nn.Softplus(), nn.Softplus(), nn.Identity())
model_HO=NNVMC(layers, activations).to(device)
optimizer=torch.optim.SGD(model_HO.parameters(), lr=learning_rate, weight_decay=reg_param)
scheduler=None
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
#                                                         mode='min',
#                                                         factor=0.5,
#                                                         patience=patience,
#                                                         threshold=0.01,
#                                                         eps=10**(-12))
# model_HO.load_state_dict(
#     torch.load("VMC/trained_model_HO.pth", weights_only=True)
# )
# model_HO.load_state_dict(
#     torch.load("VMC/trained_model_2_part.pth", weights_only=True)
# )
# model_HO.pbc_cond=False
# model_HO.MC_samples=MC_samples
# model_HO.particle_config.append(starting_config)
# model_HO.MC_new_config(Delta, exp_function, burn_data)
model_HO.particles=N_particles
model_HO.Training_NN_VMC([HO_potential], starting_config, Delta, MC_samples,
                    epochs, learning_rate, reg_param, exp_function=exp_function,
                    optimizer=optimizer,scheduler=scheduler, burn_data=burn_data,
                    single_particle_move=single_particle_move, start_delta_finder=start_delta_finder,
                    use_minSR=use_minSR, diag_shift=diag_shift)

end_datetime=datetime.now()
end=time.time()
elapsed=end-start
hours, remainder=divmod(elapsed, 3600)
minutes, seconds=divmod(remainder, 60)
print(f"\nRun started at: {start_datetime.strftime('%d/%m/%Y %H:%M:%S')}")
print(f"Run finished at: {end_datetime.strftime('%d/%m/%Y %H:%M:%S')}")
print(f"\nElapsed time: {hours:.0f} hh {minutes:.0f} mm {seconds:.2f} ss\n")







#%%

#Plotting Data
iterations=np.arange(1, epochs+1)
plt.close('all')
if input_neurons==1:
    x_batch_tens=model_HO.particle_config
    x_data_tens=torch.linspace(-4, 4, 2000).unsqueeze(1)
    
    
    with torch.no_grad():
        y_tens=model_HO(starting_config)-model_HO(torch.tensor([0.0,]))
        y_guess_on_batch_tens=model_HO(x_batch_tens)-model_HO(torch.tensor([0.0,]))
        y_guess_tens=model_HO(x_data_tens)-model_HO(torch.tensor([0.0,]))
    
    x=starting_config.detach().numpy()    
    x_batch=x_batch_tens.detach().numpy()
    x_data=x_data_tens.detach().numpy()
    f_NN_start=y_tens.detach().numpy()
    f_NN_guess=y_guess_tens.detach().numpy()
    f_NN_guess_on_batch=y_guess_on_batch_tens.detach().numpy()
    
    
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

energies=np.array(model_HO.energy_list)#/N_particles
en_std_devs=np.array(model_HO.en_std_dev_list)
en_errors=np.array(model_HO.en_err_list)#/N_particles
min_en=(min(energies))
# min_en_std=en_std_devs(energies.index(min_en))
print(f"Minimum energy: {min(energies):.04f}")
plt.figure(num=3, dpi=120)
plt.plot(iterations, model_HO.energy_list, marker='.', linestyle='-')
# plt.errorbar(iterations, model_HO.energy_list, yerr=model_HO.en_std_dev_list, fmt='o-', capsize=4, markersize=5, label="Energia VMC")
plt.axhline(y=0.5, color='red', linestyle='--', label="Energia esatta HO")
plt.xlabel("Epochs")
plt.ylabel("Energy")
plt.ylim([0,1])
plt.grid(True)
plt.show()



# loss=model_HO.loss_list
# plt.figure(num=4, dpi=120)
# plt.plot(iterations, loss, marker='.', linestyle='-')
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.grid(True)
# plt.show()

plt.figure(num=5, dpi=120)
plt.plot(iterations, model_HO.acc_ratio_list, marker='.', linestyle='-')
plt.xlabel("Epochs")
plt.ylabel("Acceptance ratio [%]")
plt.grid(True)
plt.show()

plt.figure(num=6, dpi=120)
plt.plot(iterations, model_HO.Delta_list, marker='.', linestyle='-')
plt.xlabel("Epochs")
plt.ylabel("Delta")
plt.grid(True)
plt.show()

autocorr_terms=7

if autocorr_terms<MC_samples:
    C_tau_terms_tens=model_HO.autocorrelation_length(HO_potential, exp_function, autocorr_terms=autocorr_terms)
    C_tau_terms=C_tau_terms_tens.detach().cpu().numpy()
    C_tot=np.sum(C_tau_terms)
    tau=np.arange(autocorr_terms)
    
    x_data_exp_model_HO=np.linspace(0,autocorr_terms-1, 1000)
    y_data_exp_model_HO=np.exp(-x_data_exp_model_HO/3)
    
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
