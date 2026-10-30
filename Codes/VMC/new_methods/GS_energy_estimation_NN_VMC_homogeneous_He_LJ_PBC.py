# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 17:17:12 2026

@author: Ale
"""

import time
from datetime import datetime
from VMC.new_methods.Neural_Networks_models import Neural_Network_VMC_He_0K
from VMC.new_methods.sampler import Sampler
from VMC.new_methods.Training import training_NN_VMC
from VMC.new_methods.Potential_energy_list import Lennard_jones_potential, V_tail
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np
from VMC.FCC_lattice_He import FCC
from VMC.new_methods.utils import autocorrelation_length
import gc



torch.set_default_dtype(torch.float64)
# device = torch.device("cuda")
device=torch.device("cpu")
if device.type == "cuda":
    torch.cuda.init()
    

print(f"Dispositivo utilizzato: {device}")

start_datetime=datetime.now()
start=time.time()


h=6.626e-34
m_He=3.73e9
PM_He=4.002
Na=6.022e23
m_He=(PM_He/Na)/1000
k_B=1.38e-23
from_met_to_Angs=1.0e10

kinetic_factor=((h*from_met_to_Angs)**2)/(((2*np.pi)**2)*2*m_He*k_B)

a0=5
factor_HO=kinetic_factor/(a0**4)


# seed = 13#123#1445#123#123423#8
# print(f"Seed: {seed:d}\n")
# torch.manual_seed(seed)

density=0.02
n_particles=32
dimension=3
coordinates=dimension*n_particles
diag_shift=1e-2
start_delta_finder=50
MC_samples=3000
burn_samples=3000
epochs=1
delta=1.0
learning_rate=2e-3
reg_param=0.0

single_particle_move=True
exp_function=True
pbc_cond=True

conf_part, Length_box=FCC(density, n_particles)
Length_box_numpy=Length_box.numpy()
Length_box_half_numpy=Length_box_numpy/2.0
starting_config=conf_part.reshape(-1)


info_LJ={
    "sigma": 2.556,
    "eps": 10.4,
    "pbc_cond": pbc_cond,
    "length_box_pbc": Length_box
    }

pot_LJ_info=[Lennard_jones_potential, info_LJ]

info_V_tail={
    "n_particles": n_particles,
    "density": density,
    "length_box_pbc": Length_box,
    "sigma": 2.556,
    "eps": 10.4,
    }
pot_V_tail=[V_tail, info_V_tail]

potentials=[pot_LJ_info, pot_V_tail]
energy_info=[kinetic_factor, potentials]

V=Lennard_jones_potential(conf_part.unsqueeze(0), sigma=2.556, eps=10.4, pbc_cond=True, length_box_pbc=Length_box, r_tilde=0.0)
# inspect_configuration(starting_config_NN, Length_box)
v_tail_contr=V_tail(n_particles, density, Length_box, 2.556, 10.4)
print(f"V_LJ: {V.item(): .04f}")
print(f"V_tail: {v_tail_contr: .010f}")

layers=([1,16, 16,  1])
activations=(nn.Softplus(), nn.Softplus(),  nn.Identity()) 

model_He=Neural_Network_VMC_He_0K(layers, activations, Length_box)


sampler=Sampler(n_particles=n_particles, dimension=dimension, 
                pbc_cond=pbc_cond, length_box_pbc=Length_box,
                exp_function=exp_function, single_particle_move=single_particle_move)



optimizer=torch.optim.SGD(model_He.parameters(), lr=learning_rate, weight_decay=reg_param)
scheduler=None
# scheduler = torch.optim.lr_scheduler.StepLR(
#     optimizer,
#     step_size=50,  # Every 50 epochs
#     gamma=0.7      # Multiply LR by 0.1
# )
data=training_NN_VMC(model_He, sampler, epochs, 
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
            

# inspect_configuration(model_He.particle_config[0], Length_box)
plt.close('all')
fig=plt.figure(num=0, dpi=120)
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

iterations=np.arange(1, epochs+1)
energies=np.array(energy_mean_list)/n_particles
en_std_devs=np.array(energy_std_dev_list)/n_particles
en_errors=np.array(energy_stoch_err_list)/n_particles
kinetic_energies=np.array(T_mean_list)/n_particles
potential_energies=np.array(V_mean_list)/n_particles
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
plt.figure(num=1, dpi=120)
plt.plot(iterations, kinetic_energies, marker='.', linestyle='-', label="NN Kinetic")
plt.plot(iterations, potential_energies, marker='.', linestyle='-', label="NN Potential")
plt.plot(iterations, energies, marker='.', linestyle='-', label="NN Energy")
# plt.errorbar(iterations, energies, yerr=en_std_devs, fmt='o-', capsize=4, markersize=5, label="Energia VMC")
plt.axhline(y=-6., color='red', linestyle='--', label="valore da raggiungere")
plt.xlabel("Epochs")
plt.ylabel("Energy")
plt.ylim([-30,100])
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


plt.figure(num=3, dpi=120)
plt.plot(iterations, delta_list, marker='.', linestyle='-')
plt.xlabel("Epochs")
plt.ylabel("Delta")
plt.grid(True)
plt.show()

plt.figure(num=4, dpi=120)
plt.plot(iterations, acceptance_ratio_list, marker='.', linestyle='-')
plt.xlabel("Epochs")
plt.ylabel("Acceptance ratio [%]")
plt.grid(True)
plt.show()


plt.figure(num=5, dpi=120)
plt.plot(iterations, grad_list, marker='.', linestyle='-')
plt.xlabel("Epochs")
plt.ylabel(r"||$\delta \theta$||")
plt.grid(True)
plt.show()

#%%
from VMC.new_methods.utils import autocorrelation_length, autocorrelation_length_Pearson
autocorr_terms=29

if autocorr_terms<MC_samples:
    C_tau_terms=autocorrelation_length(local_energies_last_epoch/n_particles, autocorr_terms=autocorr_terms)
    if isinstance(C_tau_terms, torch.Tensor):
        C_tau_terms=C_tau_terms.detach().cpu().numpy()
    C_tot=np.sum(C_tau_terms)
    tau=np.arange(autocorr_terms)
    
    x_data_exp_model=np.linspace(0,autocorr_terms-1, 1000)
    y_data_exp_model=np.exp(-x_data_exp_model/C_tot)
    # plt.close(fig=7)
    plt.figure(num=7, dpi=120)
    plt.plot(tau, C_tau_terms, marker='.', linestyle='-', label='data')
    plt.plot(x_data_exp_model, y_data_exp_model, marker='None', linestyle='-', label='model')
    plt.ylabel(r"$c(E)_{\tau}$")
    plt.xlabel(r"$\tau$")
    plt.grid(True)
    plt.legend()
    plt.show()

# loss=model_He.loss_list
# plt.figure(num=2, dpi=120)
# plt.plot(iterations, loss, marker='.', linestyle='-')
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.grid(True)
# plt.show()

