# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 21:13:28 2026

@author: Ale
"""

import time
from datetime import datetime
from VMC.old_methods.NN_class_for_VMC import Neural_Network_VMC as NNVMC
from VMC.old_methods.NN_class_for_VMC import HO_potential
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np
from VMC.FCC_lattice_He import FCC
from VMC.old_methods.NN_class_for_VMC import Lennard_jones_potential
from VMC.old_methods.NN_class_for_VMC import V_tail
from VMC.old_methods.NN_class_for_VMC import count_small_distances
from VMC.old_methods.NN_class_for_VMC import tot_para_NN
import gc


def inspect_configuration(x, L, sigma=2.556, eps=10.4):
    positions = x.reshape(-1, 3)
    n = positions.shape[0]

    print("Box length:", L.item() if torch.is_tensor(L) else L)

    for i in range(n):
        for j in range(i + 1, n):
            dr = positions[i] - positions[j]
            dr = dr - L * torch.round(dr / L)
            r = torch.linalg.vector_norm(dr)

            V = 4 * eps * (
                (sigma / r)**12 - (sigma / r)**6
            )

            print(
                f"Pair ({i}, {j}): "
                f"r = {r.item():.6f} Å, "
                f"V_LJ = {V.item():.6f} K"
            )
# # Elimina il modello e i tensori associati
# if model_He:
#     del model_He

# Raccoglie gli oggetti Python non più referenziati
gc.collect()

# Se utilizzi CUDA, libera la cache della GPU
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("Pulizia della memoria completata.")


# device = torch.device("cuda")
device=torch.device("cpu")
if device.type == "cuda":
    torch.cuda.init()
    
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

seed = 13#123#1445#123#123423#8
print(f"Seed: {seed:d}\n")
torch.manual_seed(seed)
# np.random.seed(seed)
density=0.02#0.02#0.009#0.001#0.021#0.04#0.021559#0.03#0.021559#32
N_particles=32
start_delta_finder=50#None#300#None#1
single_particle_move=True
PBC_NN=True
Jastrow_appl=False#True
MC_samples=100#600#600#1200
epochs=300#1000
learning_rate=1e-3
reg_param=0.0#0.01#0.0
exp_function=True #False
patience=50
burn_data=3000
Delta=1#22.0
is_pot_hard_core=False

use_minSR=True
diag_shift=1e-2
use_continuation=False#True


# torch.empty_like()
conf_part, Length_box=FCC(density, N_particles)
Length_box=np.float32(Length_box)
Length_box_half=Length_box/2.0
starting_config_NN=conf_part.reshape(-1)
# conf_part_large=conf_part#.unsqueeze(0)

# count_small_distances(torch.stack([conf_part_large, conf_part_large]), True, Length_box, alpha=2.34)

V=Lennard_jones_potential(conf_part.unsqueeze(0), sigma=2.556, eps=10.4, pbc_cond=True, length_box_pbc=Length_box, r_tilde=0.0)
# inspect_configuration(starting_config_NN, Length_box)
v_tail_contr=V_tail(N_particles, density, Length_box, 2.556, 10.4)
print(f"V_LJ: {V.item(): .04f}")
print(f"V_tail: {v_tail_contr: .010f}")


input_neurons=1#3*N_particles*(1+int(PBC_NN))
layers=([input_neurons,16, 16,  1])
activations=(nn.Softplus(), nn.Softplus(),  nn.Identity()) 
tot_param=tot_para_NN(layers)




model_He=NNVMC(layers, activations)
optimizer=torch.optim.SGD(model_He.parameters(), lr=learning_rate, weight_decay=reg_param)
scheduler=None
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
#                                                        mode='min',
#                                                        factor=0.5,
#                                                        patience=patience,
#                                                        threshold=0.01,
#                                                        eps=10**(-12))

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=50,  # Every 50 epochs
    gamma=0.7      # Multiply LR by 0.1
)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
#                                                         mode='min',
#                                                         factor=0.5,
#                                                         patience=patience,
#                                                         threshold=0.01,
#                                                         eps=10**(-12))

# checkpoint = torch.load("VMC.old_methods/checkpoint_He_minSR_3.pth", map_location=device, weights_only=True)
# model_He.load_state_dict(checkpoint["model_state_dict"])
# optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
# scheduler.load_state_dict(checkpoint["scheduler_state_dict"])


# model_He.MC_samples=MC_samples
# model_He.particle_config.append(starting_config_NN)
# model_He.MC_new_config(Delta, exp_function, burn_data)
model_He.kinetic_factor_multiplier=kinetic_factor
model_He.use_Jastrow=Jastrow_appl
model_He.particles=N_particles
model_He.density=density
model_He.length_box_pbc=Length_box

# model_He.MC_samples=MC_samples
# model_He.pbc_cond=PBC_NN
# model_He.min_acc_ratio=0.48
# model_He.max_acc_ratio=0.52
# model_He.particle_config.append(starting_config_NN)
# Delta=model_He.delta_finder(Delta,  exp_function, samples=1000, single_particle_move=single_particle_move)
# model_He.MC_new_config(Delta, exp_function, burn_data, single_particle_move=single_particle_move, is_pot_hard_core=is_pot_hard_core)
# local_energies = model_He.Energy_local([Lennard_jones_potential, V_tail], exp_function, alpha=0.0)

model_He.Training_NN_VMC([Lennard_jones_potential, V_tail], starting_config_NN, Delta, MC_samples,
                    epochs, learning_rate, reg_param, exp_function=exp_function,
                    optimizer=optimizer, scheduler=scheduler, burn_data=burn_data,
                    pbc_cond=PBC_NN, length_box_pbc=Length_box,
                    single_particle_move=single_particle_move,start_delta_finder=start_delta_finder,
                    is_pot_hard_core=is_pot_hard_core,
                    use_minSR=use_minSR, diag_shift=diag_shift, use_continuation=use_continuation)


end_datetime=datetime.now()
end=time.time()
elapsed=end-start
hours, remainder=divmod(elapsed, 3600)
minutes, seconds=divmod(remainder, 60)
print(f"\nRun started at: {start_datetime.strftime('%d/%m/%Y %H:%M:%S')}")
print(f"Run finished at: {end_datetime.strftime('%d/%m/%Y %H:%M:%S')}")
print(f"\nElapsed time: {hours:.0f} hh {minutes:.0f} mm {seconds:.2f} ss\n")

#%%
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
energies=np.array(model_He.energy_list)/N_particles
en_std_devs=np.array(model_He.en_std_dev_list)/N_particles
en_errors=np.array(model_He.en_err_list)/N_particles
plt.figure(num=1, dpi=120)
plt.plot(iterations, energies, marker='.', linestyle='-')
# plt.errorbar(iterations, energies, yerr=en_std_devs, fmt='o-', capsize=4, markersize=5, label="Energia VMC")
# plt.axhline(y=0.5, color='red', linestyle='--', label="Energia esatta HO")
plt.xlabel("Epochs")
plt.ylabel("Energy")
plt.ylim([-10,100])
plt.grid(True)
plt.show()

# loss=model_He.loss_list
# plt.figure(num=2, dpi=120)
# plt.plot(iterations, loss, marker='.', linestyle='-')
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.grid(True)
# plt.show()

plt.figure(num=3, dpi=120)
plt.plot(iterations, model_He.acc_ratio_list, marker='.', linestyle='-')
plt.xlabel("Epochs")
plt.ylabel("Acceptance ratio [%]")
plt.grid(True)
plt.show()

plt.figure(num=4, dpi=120)
plt.plot(iterations, model_He.Delta_list, marker='.', linestyle='-')
plt.xlabel("Epochs")
plt.ylabel("Delta")
plt.grid(True)
plt.show()

plt.figure(num=5, dpi=120)
plt.plot(iterations, model_He.grad_list, marker='.', linestyle='-')
plt.xlabel("Epochs")
plt.ylabel(r"||$\Delta$||")
plt.grid(True)
plt.show()

#%%

# torch.save({
#     "model_state_dict": model_He.state_dict(),
#     "optimizer_state_dict": optimizer.state_dict(),
#     # "scheduler_state_dict": scheduler.state_dict(),
#     "epochs_completed": epochs,
#     "learning_rate": learning_rate,
# }, "VMC.old_methods/checkpoint_He.pth")



# torch.save({
#     "model_state_dict": model_He.state_dict(),
#     "optimizer_state_dict": optimizer.state_dict(),
#     # "scheduler_state_dict": scheduler.state_dict(),
#     "epochs_completed": epochs,
#     "learning_rate": learning_rate,
# }, "VMC.old_methods/checkpoint_He_continuation2.pth")


# torch.save({
#     "model_state_dict": model_He.state_dict(),
#     "optimizer_state_dict": optimizer.state_dict(),
#     # "scheduler_state_dict": scheduler.state_dict(),
#     "epochs_completed": epochs,
#     "learning_rate": learning_rate,
# }, "VMC.old_methods/checkpoint_He_minSR.pth")

torch.save({
    "model_state_dict": model_He.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    # "scheduler_state_dict": scheduler.state_dict(),
    "epochs_completed": epochs,
    "learning_rate": learning_rate,
}, "VMC.old_methods/checkpoint_He_minSR_3.pth")