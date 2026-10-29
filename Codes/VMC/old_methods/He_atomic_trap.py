# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 03:16:34 2026

@author: Ale
"""

import time
from datetime import datetime
from VMC.old_methods.NN_class_for_VMC import Neural_Network_VMC as NNVMC
from VMC.old_methods.NN_class_for_VMC import HO_potential
from VMC.old_methods.NN_class_for_VMC import tot_para_NN
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np
from VMC.old_methods.NN_class_for_VMC import Lennard_jones_potential
import gc




gc.collect()

# Se utilizzi CUDA, libera la cache della GPU
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("Pulizia della memoria completata.")


# device = torch.device("cuda")
device=torch.device("cpu")
if device.type == "cuda":
    torch.cuda.init()


h=6.626e-34
m_He=3.73e9
PM_He=4.002
Na=6.022e23
m_He=(PM_He/Na)/1000
k_B=1.38e-23
from_met_to_Angs=1.0e10

kinetic_factor=np.float32(((h*from_met_to_Angs)**2)/(((2*np.pi)**2)*2*m_He*k_B))
a0=5
factor_HO=kinetic_factor/(a0**4)




start_datetime=datetime.now()
start=time.time()

# seed = 13#123#1445#123#123423#8
# print(f"Seed: {seed:d}\n")
# torch.manual_seed(seed)
# np.random.seed(seed)
N_particles=2
start_delta_finder=20#None#1
single_particle_move=True
PBC_NN=False
# Jastrow_appl=False
MC_samples=2000
epochs=600
learning_rate=1e-2
reg_param=0.0
exp_function=True
patience=20
burn_data=1000
Delta=20

use_minSR=True
diag_shift=1e-3

input_neurons=(1+int(PBC_NN))*3*N_particles
# starting_config=2*(torch.rand(input_neurons, device=device)-0.5)
starting_config=torch.tensor(([0,0,2.0,0,0,-2.0]))
conf_part=starting_config.reshape(1,-1,3)
dr=conf_part[0,0,:]-conf_part[0,1,:]
r=torch.sum(dr**2, dim=0, keepdim=True).sqrt()
V_LG=Lennard_jones_potential(conf_part, 2.556, 10.4)
V_HO=HO_potential(starting_config.unsqueeze(0), HO_factor_molt=factor_HO)
print(f"distance He-He: {r.item(): .04f}")
print(f"V_HO He-He: {V_HO.item(): .04f}")
print(f"V_LJ He-He: {V_LG.item(): .04f}")

layers=([input_neurons,64, 64, 1])
tot_param=tot_para_NN(layers)
activations=(nn.ELU(), nn.Softplus(), nn.Identity()) 

model_HeHe=NNVMC(layers, activations)
optimizer=torch.optim.SGD(model_HeHe.parameters(), lr=learning_rate, weight_decay=reg_param)
scheduler=None
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
#                                                         mode='min',
#                                                         factor=0.5,
#                                                         patience=patience,
#                                                         threshold=0.01,
#                                                         eps=10**(-12))
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=50,  # Every 50 epochs
    gamma=0.8      # Multiply LR by 0.1
)
# checkpoint = torch.load("VMC/checkpoint_HeHe2.pth", map_location=device, weights_only=True)
# model_HeHe.load_state_dict(checkpoint["model_state_dict"])
# optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
# scheduler.load_state_dict(checkpoint["scheduler_state_dict"])


model_HeHe.kinetic_factor_multiplier=kinetic_factor
model_HeHe.HO_pot_factor=factor_HO
model_HeHe.particles=N_particles
model_HeHe.Training_NN_VMC([HO_potential, Lennard_jones_potential], starting_config, Delta, MC_samples,
                    epochs, learning_rate, reg_param, exp_function=exp_function,
                    optimizer=optimizer, scheduler=scheduler, burn_data=burn_data,
                    single_particle_move=single_particle_move,start_delta_finder=start_delta_finder,
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

final_config=model_HeHe.particle_config
final_config_dist=final_config.reshape(MC_samples,-1,3)
dr=final_config_dist[:,0,:]-final_config_dist[:,1,:]
r=torch.sum(dr**2, dim=1, keepdim=True).sqrt()
V_LG=Lennard_jones_potential(final_config, sigma=2.556, eps=10.4)
V_HO=HO_potential(x=final_config, HO_factor_molt=factor_HO)
r, V_LG, V_HO
# print(f"V_HO He-He: {V_HO: .04f}")
# print(f"V_LJ He-He: {V_LG.item(): .04f}")
#%%

plt.close('all')
fig=plt.figure(num=0, dpi=120)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(conf_part[0,:,0],conf_part[0,:,1],conf_part[0,:,2], label='model_He')
# ax.scatter(data_x1_x2[::chosen_points,0], data_x1_x2[::chosen_points,1], y_guess_final[::chosen_points], label='trained NN')
# ax.scatter(data_x1_x2[::chosen_points,0], data_x1_x2[::chosen_points,1], y_guess_initial[::chosen_points], label='initial NN')
# ax.set_xlim((-Length_box/2.0, Length_box/2.0))
# ax.set_ylim((-Length_box/2.0, Length_box/2.0))
# ax.set_zlim((-Length_box/2.0, Length_box/2.0))
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

# plt.legend()
plt.show()

iterations=np.arange(1, epochs+1)
energies=np.array(model_HeHe.energy_list)#/N_particles
en_std_devs=np.array(model_HeHe.en_std_dev_list)#/N_particles
en_errors=np.array(model_HeHe.en_err_list)#/N_particles
plt.figure(num=1, dpi=120)
plt.plot(iterations, model_HeHe.energy_list, marker='.', linestyle='-')
# plt.plot(iterations, energy_minSR, marker='.', linestyle='-', label='MinSR')

# plt.errorbar(iterations, energies, yerr=en_std_devs, fmt='o-', capsize=4, markersize=5, label="Energia VMC")
# plt.axhline(y=0.5, color='red', linestyle='--', label="Energia esatta HO")
plt.xlabel("Epochs")
plt.ylabel("Energy")
# plt.legend()
plt.ylim([0,10])
plt.grid(True)
plt.show()

# loss=model_HeHe.loss_list
# plt.figure(num=2, dpi=120)
# plt.plot(iterations, loss, marker='.', linestyle='-')
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.grid(True)
# plt.show()

plt.figure(num=3, dpi=120)
plt.plot(iterations, model_HeHe.acc_ratio_list, marker='.', linestyle='-')
plt.xlabel("Epochs")
plt.ylabel("Acceptance ratio [%]")
plt.grid(True)
plt.show()

plt.figure(num=4, dpi=120)
plt.plot(iterations, model_HeHe.Delta_list, marker='.', linestyle='-')
plt.xlabel("Epochs")
plt.ylabel("Delta")
plt.grid(True)
plt.show()

plt.figure(num=5, dpi=120)
plt.plot(iterations, model_HeHe.grad_list, marker='.', linestyle='-')
plt.xlabel("Epochs")
plt.ylabel(r"||$\Delta$||")
plt.grid(True)
plt.show()



r_12=torch.linspace(0.1,20,10000)
r_12=r_12.unsqueeze(1)

model_HeHe.eval()
with torch.no_grad():
    f_12=model_HeHe.psi(r_12)
    f_12=f_12-torch.max(f_12)
    f_12=torch.exp(f_12)

plt.figure(num=6, dpi=120)
plt.plot(r_12, f_12, marker='.', linestyle='-')
plt.xlabel(r"$r_{12}$")
plt.ylabel(r"BOH")
plt.grid(True)
plt.show()
# with torch.no_grad():
#     model_HeHe.eval()
#     data_2part=5*(torch.rand([2000,6])-0.5)
#     f_12=model_HeHe(data_2part)
#     f_12=f_12-torch.max(f_12)
#     f_12=torch.exp(f_12).squeeze(1).numpy()
#     data_2part=data_2part.reshape(data_2part.shape[0], -1,3)
#     r1=torch.sqrt(torch.sum(data_2part[:,0,:]**2, dim=-1)).numpy()
#     r2=torch.sqrt(torch.sum(data_2part[:,1,:]**2, dim=-1)).numpy()
#     R1, R2=np.meshgrid(r1, r2)
# fig=plt.figure(num=6, dpi=120)
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(r1, r2, f_12, label='NN')
# # ax.plot_surface(X1, X2, Z)
# # ax.scatter(data_x1_x2[::chosen_points,0], data_x1_x2[::chosen_points,1], y_guess_final[::chosen_points], label='trained NN')
# # ax.scatter(data_x1_x2[::chosen_points,0], data_x1_x2[::chosen_points,1], y_guess_initial[::chosen_points], label='initial NN')
# ax.set_xlabel("x1")
# ax.set_ylabel("x2")
# ax.set_zlabel("f(x1, x2)")
# # plt.legend()
# plt.show()
#%%
energy_minSR=model_HeHe.energy_list
#%%
# torch.save({
#     "model_state_dict": model_HeHe.state_dict(),
#     "optimizer_state_dict": optimizer.state_dict(),
#     "scheduler_state_dict": scheduler.state_dict(),
#     "epochs_completed": epochs,
#     "learning_rate": learning_rate,
# }, "VMC/checkpoint_HeHe.pth")

torch.save({
    "model_state_dict": model_HeHe.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    # "scheduler_state_dict": scheduler.state_dict(),
    "epochs_completed": epochs,
    "learning_rate": learning_rate,
}, "VMC/checkpoint_HeHe_minSR.pth")