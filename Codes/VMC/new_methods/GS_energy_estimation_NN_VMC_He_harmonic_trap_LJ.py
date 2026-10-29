# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 11:58:41 2026

@author: Ale
"""

import torch
from VMC.new_methods.Neural_Networks_models import Neural_Network_VMC_HeHe
import torch.nn as nn
from VMC.new_methods.sampler import Sampler
from VMC.new_methods.Training import training_NN_VMC
from VMC.new_methods.Potential_energy_list import HO_potential
from VMC.new_methods.Potential_energy_list import Lennard_jones_potential
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt



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

kinetic_factor=np.float32(((h*from_met_to_Angs)**2)/(((2*np.pi)**2)*2*m_He*k_B))
a0=5.0
factor_HO=kinetic_factor/(a0**4)

info_HO={
    "HO_factor_molt": factor_HO}
pot_HO_info=[HO_potential, info_HO]

info_LJ={
    "sigma": 2.556,
    "eps": 10.4,
    }
pot_LJ_info=[Lennard_jones_potential, info_LJ]
potentials=[pot_HO_info, pot_LJ_info]


energy_info=[kinetic_factor, potentials]

n_particles=2
dimension=3
coordinates=dimension*n_particles
diag_shift=1e-3
start_delta_finder=20
MC_samples=2000
burn_samples=3000
epochs=1000
delta=20.0
learning_rate=1e-2
reg_param=0.0

single_particle_move=True
exp_function=True


starting_config=torch.tensor(([0, 0, 2.0, 0, 0, -2.0]))

layers_pos=([1*dimension, 32, 32, 1])
activations_pos=(nn.Softplus(), nn.Softplus(), nn.Identity())
layers_dist=([1, 32, 32, 1])
activations_dist=(nn.Softplus(), nn.Softplus(), nn.Identity())

model_HeHe=Neural_Network_VMC_HeHe(layers_pos=layers_pos, activations_pos=activations_pos,
                                   layers_dist=layers_dist, activations_dist=activations_dist)

sampler=Sampler(n_particles=n_particles, dimension=dimension, 
                exp_function=exp_function, single_particle_move=single_particle_move)
optimizer=torch.optim.SGD(model_HeHe.parameters(), lr=learning_rate, weight_decay=reg_param)
# scheduler=None
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=50,  # Every 50 epochs
    gamma=0.8      # Multiply LR by 0.1
)
data=training_NN_VMC(model_HeHe, sampler, epochs, 
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
            


conf_part=starting_config.reshape(1,-1,3)

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
energies=np.array(energy_mean_list)/n_particles
en_std_devs=np.array(energy_std_dev_list)/n_particles
en_errors=np.array(energy_stoch_err_list)/n_particles
kinetic_energies=np.array(T_mean_list)/n_particles
potential_energies=np.array(V_mean_list)/n_particles
plt.figure(num=1, dpi=120)
plt.plot(iterations, kinetic_energies, marker='.', linestyle='-', label="NN Kinetic")
plt.plot(iterations, potential_energies, marker='.', linestyle='-', label="NN Potential")
plt.plot(iterations, energies, marker='.', linestyle='-', label="NN Energy")
# plt.errorbar(iterations, energies, yerr=en_std_devs, fmt='o-', capsize=4, markersize=5, label="Energia VMC")
plt.axhline(y=1.17/2, color='red', linestyle='--', label="Exact energy HO")
plt.xlabel("Epoch")
plt.ylabel("Energy")
plt.ylim([-2,5])
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
plt.plot(iterations, acceptance_ratio_list, marker='.', linestyle='-')
plt.xlabel("Epoch")
plt.ylabel("Acceptance ratio [%]")
plt.grid(True)
plt.show()

plt.figure(num=4, dpi=120)
plt.plot(iterations, delta_list, marker='.', linestyle='-')
plt.xlabel("Epoch")
plt.ylabel("Delta")
plt.grid(True)
plt.show()

plt.figure(num=5, dpi=120)
plt.plot(iterations, grad_list, marker='.', linestyle='-')
plt.xlabel("Epoch")
plt.ylabel(r"||$\nabla$||")
plt.grid(True)
plt.show()



r_12=torch.linspace(0.1,20,10000)
r_12=r_12.unsqueeze(1)

model_HeHe.eval()
with torch.no_grad():
    f_12=model_HeHe.dist(r_12)
    # f_12=f_12-torch.max(f_12)
    f_12=f_12-model_HeHe.dist(torch.tensor([a0]))
    f_12=torch.exp(f_12)

plt.figure(num=6, dpi=120)
plt.plot(r_12, f_12, marker='.', linestyle='-')
plt.axvline(a0, color='red', linestyle='--', label=r"$a_0$")
plt.xlabel(r"$r_{12}$")
plt.ylabel(r"Jastrow NN")
plt.grid(True)
plt.legend()
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




# loss=model_HeHe.loss_list
# plt.figure(num=2, dpi=120)
# plt.plot(iterations, loss, marker='.', linestyle='-')
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.grid(True)
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











