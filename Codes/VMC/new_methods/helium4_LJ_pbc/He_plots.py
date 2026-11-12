# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 22:30:00 2026

@author: Ale
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from VMC.new_methods.class_and_functions.Neural_Networks_models import  Neural_Network_VMC_He_0K
from VMC.new_methods.class_and_functions.utils import autocorrelation_length, compute_der_for_plots
from VMC.new_methods.class_and_functions.sampler import Sampler
from VMC.new_methods.class_and_functions.Energy import compute_energy_local


torch.set_default_dtype(torch.float64)
device=torch.device("cpu")

# file_name="VMC/new_methods/helium4_LJ_pbc/checkpoint_He_pbc_run1_switch_power_1.pth"
# file_name="VMC/new_methods/helium4_LJ_pbc/checkpoint_He_pbc_run2_switch_power_2.pth"
file_name="VMC/new_methods/helium4_LJ_pbc/checkpoint_He_pbc_run3_switch_power_3.pth"
checkpoint = torch.load(file_name, map_location=device, weights_only=False)
input_data=checkpoint["input_data"]
output_data=checkpoint["output_data"]


(n_particles, dimension, starting_config, energy_info,\
 layers, activations, epochs, samples, batch, sweep_per_save,\
     diag_shift, reg_param, exp_function, time_control,\
         pbc_cond, density, Length_box, switch_power)=input_data
    
coordinates=n_particles*dimension
[kinetic_factor, potentials]=energy_info

model_He=Neural_Network_VMC_He_0K(layers, activations, Length_box, dimension, switch_power)
model_He.load_state_dict(checkpoint["model_state_dict"])

local_energies_last_epoch, energy_mean_list, energy_std_dev_list, energy_stoch_err_list,\
        T_mean_list, V_mean_list, delta_list, acceptance_ratio_list,\
            current_config, grad_list, lr_list,\
                fraction_nonfinite_values_list, fraction_clipped_values_list, control_list,\
                    T_mean_grad_list, energy_mean_grad_list,\
                        T_mean_JF_list, energy_mean_JF_list=output_data
                        
                        
                        
                        
                        
fig_num=0
# inspect_configuration(model_He.particle_config[0], Length_box)
conf_part=starting_config
plt.close('all')
fig=plt.figure(num=(fig_num:=fig_num+1), dpi=120)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(conf_part[:,0],conf_part[:,1],conf_part[:,2], label='model_He')
# ax.scatter(data_x1_x2[::chosen_points,0], data_x1_x2[::chosen_points,1], y_guess_final[::chosen_points], label='trained NN')
# ax.scatter(data_x1_x2[::chosen_points,0], data_x1_x2[::chosen_points,1], y_guess_initial[::chosen_points], label='initial NN')
ax.set_xlim((-Length_box/2.0, Length_box/2.0))
ax.set_ylim((-Length_box/2.0, Length_box/2.0))
ax.set_zlim((-Length_box/2.0, Length_box/2.0))
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.legend()
plt.show()



box_vertex_max_dist=np.sqrt(3)*Length_box/2
r_12=torch.linspace(0.0,box_vertex_max_dist*3.5,100001)
r_12=r_12.unsqueeze(1)

f_12, f_12_der, f_12_sec_der=compute_der_for_plots(r_12, model_He.phi, torch.tensor([Length_box/2]))
# model_He.eval()
# with torch.no_grad():
#     f_12=model_He.phi(r_12.unsqueeze(1))
#     # f_12=f_12-torch.max(f_12)
#     f_12=f_12-model_He.phi(torch.tensor([Length_box/2]))
#     f_12=torch.exp(f_12).squeeze()
#     f_12_der=torch.gradient(f_12, spacing=(r_12.squeeze(),))[0]
#     f_12_sec_der=torch.gradient(f_12_der, spacing=(r_12.squeeze(),))[0]

model=torch.exp(-(2.95/r_12)**5)
plt.figure(num=(fig_num:=fig_num+1), dpi=120)
plt.plot(r_12, f_12, marker='None', linestyle='-', zorder=4, label="NN")
plt.plot(r_12, f_12_der, marker='None', linestyle='-.', zorder=3, label="NN first derivative")
plt.plot(r_12, f_12_sec_der, marker='None', linestyle=':', zorder=3, label="NN second derivative")
plt.plot(r_12, model, marker='None', linestyle='-', zorder=3, label="model")
plt.axvline(Length_box/2, color='red', linestyle='--', label=r"$\dfrac{L}{2}$")
plt.axvline(Length_box, color='purple', linestyle='--', label=r"L")
plt.axvline(np.sqrt(3)*Length_box/2, color='blue', linestyle='--', label=r"$\dfrac{\sqrt{3}}{2}L$")
plt.xlabel(r"$r_{12}$")
plt.ylabel(r"Jastrow NN")
plt.ylim([-1,1.2*f_12.max()])
plt.grid(True)
plt.legend()
plt.show()


iterations=np.arange(1, epochs+1)
potential_energies=np.array(V_mean_list)/n_particles
energies=np.array(energy_mean_list)/n_particles
en_std_devs=np.array(energy_std_dev_list)/n_particles
en_errors=np.array(energy_stoch_err_list)/n_particles
kinetic_energies=np.array(T_mean_list)/n_particles
kinetic_grad_energies=np.array(T_mean_grad_list)/n_particles
energies_grad=np.array(energy_mean_grad_list)/n_particles
kinetic_JF_energies=np.array(T_mean_JF_list)/n_particles
energies_JF=np.array(energy_mean_JF_list)/n_particles
#minimum energy
index_min_en=np.argmin(energies)
print("Minimum Energy:")
print(f"Epoch: {index_min_en: d}")
print(f"T_mean: {T_mean_list[index_min_en].item()/n_particles: .04f}")
print(f"V_mean: {V_mean_list[index_min_en].item()/n_particles: .04f}")
print(f"Energy mean: {energy_mean_list[index_min_en].item()/n_particles: .04f}")
print(f"Energy std_dev: {energy_std_dev_list[index_min_en].item()/n_particles: .04f}")
print(f"Energy stoch_err: {energy_stoch_err_list[index_min_en].item()/n_particles: .04f}")
print(f"Gradient norm before clip: {grad_list[index_min_en]:.6f}\n")
#minimum energy std dev
index_min_en_std_dev=np.argmin(en_std_devs)
print("Minimum Energy std dev:")
print(f"Epoch: {index_min_en_std_dev: d}")
print(f"T_mean: {T_mean_list[index_min_en_std_dev].item()/n_particles: .04f}")
print(f"V_mean: {V_mean_list[index_min_en_std_dev].item()/n_particles: .04f}")
print(f"Energy mean: {energy_mean_list[index_min_en_std_dev].item()/n_particles: .04f}")
print(f"Energy std_dev: {energy_std_dev_list[index_min_en_std_dev].item()/n_particles: .04f}")
print(f"Energy stoch_err: {energy_stoch_err_list[index_min_en_std_dev].item()/n_particles: .04f}")
print(f"Gradient norm before clip: {grad_list[index_min_en_std_dev]:.6f}\n")
#minimum grad
index_min_grad=np.argmin(np.array(grad_list))
print("Minimum Grad:")
print(f"Epoch: {index_min_grad: d}")
print(f"T_mean: {T_mean_list[index_min_grad].item()/n_particles: .04f}")
print(f"V_mean: {V_mean_list[index_min_grad].item()/n_particles: .04f}")
print(f"Energy mean: {energy_mean_list[index_min_grad].item()/n_particles: .04f}")
print(f"Energy std_dev: {energy_std_dev_list[index_min_grad].item()/n_particles: .04f}")
print(f"Energy stoch_err: {energy_stoch_err_list[index_min_grad].item()/n_particles: .04f}")
print(f"Gradient norm before clip: {grad_list[index_min_grad]:.6f}\n")
plt.figure(num=(fig_num:=fig_num+1), dpi=120)
plt.plot(iterations, potential_energies, marker='None', linestyle='-', zorder=3, label="NN V")
plt.plot(iterations, kinetic_energies, marker='None', linestyle='-', zorder=3, label="NN T")
plt.plot(iterations, energies, marker='None', linestyle='-', zorder=3, label="NN E")
plt.plot(iterations, kinetic_grad_energies, marker='None', linestyle='-', label=r"NN $T_{\nabla}$")
plt.plot(iterations, energies_grad, marker='None', linestyle='-', label=r"NN $E_{\nabla}$")
plt.plot(iterations, kinetic_JF_energies, marker='None', linestyle='-', label=r"NN $T_{JF}$")
plt.plot(iterations, energies_JF, marker='None', linestyle='-', label=r"NN $E_{JF}$")
# plt.errorbar(iterations, energies, yerr=en_std_devs, fmt='o-', capsize=4, markersize=5, label="Energia VMC")
plt.axhline(y=-6., color='red', linestyle='--', label="valore da raggiungere")
plt.xlabel("Epochs")
plt.ylabel("Energy")
plt.ylim([-50,40])
plt.grid(True)
plt.legend()
plt.show()

plt.figure(num=(fig_num:=fig_num+1), dpi=120)
plt.plot(iterations, en_std_devs, marker='.', linestyle='None')
# plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
plt.xlabel("Epoch")
plt.ylabel("Standard deviation")
plt.ylim([0,2.5*en_std_devs[-1]])
plt.grid(True)
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
plt.xlabel("Epochs")
plt.ylabel("Delta")
plt.grid(True)
plt.show()

plt.figure(num=(fig_num:=fig_num+1), dpi=120)
plt.plot(iterations, acceptance_ratio_list, marker='.', linestyle='-')
plt.xlabel("Epochs")
plt.ylabel("Acceptance ratio [%]")
plt.grid(True)
plt.show()


plt.figure(num=(fig_num:=fig_num+1), dpi=120)
plt.plot(iterations, grad_list, marker='.', linestyle='-')
plt.xlabel("Epochs")
plt.ylabel(r"||$\delta \theta$||")
plt.grid(True)
plt.show()

plt.figure(num=(fig_num:=fig_num+1), dpi=120)
plt.plot(iterations, fraction_nonfinite_values_list, marker='.', linestyle='-', label="non-finite values")
plt.plot(iterations, fraction_clipped_values_list, marker='.', linestyle='-', label="clipped_values")
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
#%%
sampler=Sampler(n_particles=n_particles, dimension=dimension, 
                exp_function=exp_function, pbc_cond=pbc_cond,
                length_box_pbc=Length_box)
MC_samples=1000
batch_autocorr=10
current_config_autocorr, acceptance_ratio_autocorr=sampler.metropolis_sampling_parallel_walkers_batch(model_He, current_config, delta_list[-1], 
                        MC_samples, sweep_per_save=10, batch=batch_autocorr, burned_sweep=3, save_data=True)

data_autocorr=compute_energy_local(model_He, current_config_autocorr, kinetic_factor, potentials,
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
    print(f"C_tot: {C_tot: .04f}")
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
# from VMC.new_methods.utils import autocorrelation_length, autocorrelation_length_Pearson
# autocorr_terms=29

# if autocorr_terms<MC_samples:
#     C_tau_terms=autocorrelation_length(local_energies_last_epoch/n_particles, autocorr_terms=autocorr_terms)
#     if isinstance(C_tau_terms, torch.Tensor):
#         C_tau_terms=C_tau_terms.detach().cpu().numpy()
#     C_tot=np.sum(C_tau_terms)
#     tau=np.arange(autocorr_terms)
    
#     x_data_exp_model=np.linspace(0,autocorr_terms-1, 1000)
#     y_data_exp_model=np.exp(-x_data_exp_model/C_tot)
#     # plt.close(fig=7)
#     plt.figure(num=7, dpi=120)
#     plt.plot(tau, C_tau_terms, marker='.', linestyle='-', label='data')
#     plt.plot(x_data_exp_model, y_data_exp_model, marker='None', linestyle='-', label='model')
#     plt.ylabel(r"$c(E)_{\tau}$")
#     plt.xlabel(r"$\tau$")
#     plt.grid(True)
#     plt.legend()
#     plt.show()

# loss=model_He.loss_list
# plt.figure(num=2, dpi=120)
# plt.plot(iterations, loss, marker='.', linestyle='-')
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.grid(True)
# plt.show()
