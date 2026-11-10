# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 11:58:41 2026

@author: Ale
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime

from VMC.new_methods.class_and_functions.Neural_Networks_models import Neural_Network_VMC_HeHe
from VMC.new_methods.class_and_functions.sampler import Sampler
from VMC.new_methods.class_and_functions.Training import training_NN_VMC
from VMC.new_methods.class_and_functions.Potential_energy_list import HO_potential, Lennard_jones_potential
from VMC.new_methods.class_and_functions.utils import autocorrelation_length, print_values_on_log
from VMC.new_methods.class_and_functions.Energy import compute_energy_local


torch.set_default_dtype(torch.float64)

file=open("VMC/new_methods/helium4_harmonic_trapped_LJ/GS_En_est_NN_VMC_He_Ho_trap_LJ_log_file.txt", "w")
problem_under_study="VMC neural network estimation of the ground state energy of helium-4 atoms in harmonic trap with LJ interactions"

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
batch=500
sweep_per_save=4
epochs=1000
delta=20.0
learning_rate=1e-2
reg_param=0.0

single_particle_move=True
exp_function=True
time_control=True


starting_config=torch.tensor(([0, 0, 2.0, 0, 0, -2.0]))

layers_pos=([1*dimension, 32, 32, 1])
activations_pos=(nn.Softplus(), nn.Softplus(), nn.Identity())
layers_dist=([1, 32, 32, 1])
activations_dist=(nn.Softplus(), nn.Softplus(), nn.Identity())

input_data=print_values_on_log(problem_under_study,n_particles, dimension, starting_config, energy_info,\
                               [layers_pos, layers_dist], [activations_pos, activations_dist], epochs, MC_samples, batch, sweep_per_save,\
                                   diag_shift, reg_param, exp_function, time_control=time_control, file=file)





model_HeHe=Neural_Network_VMC_HeHe(layers_pos=layers_pos, activations_pos=activations_pos,
                                   layers_dist=layers_dist, activations_dist=activations_dist,
                                   dimension=dimension)

sampler=Sampler(n_particles=n_particles, dimension=dimension, 
                exp_function=exp_function, single_particle_move=single_particle_move)
optimizer=torch.optim.SGD(model_HeHe.parameters(), lr=learning_rate, weight_decay=reg_param)
# scheduler=None
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=50,  # Every 50 epochs
    gamma=0.8      # Multiply LR by 0.1
)

# checkpoint = torch.load("VMC/new_methods/checkpoint_HeHe.pth", map_location=device, weights_only=True)
# model_HeHe.load_state_dict(checkpoint["model_state_dict"])


output_data=training_NN_VMC(model_HeHe, sampler, epochs, 
                    starting_config, delta, energy_info,
                    MC_samples, batch,sweep_per_save,
                    start_delta_finder, time_control, 
                    optimizer, scheduler, diag_shift, file)





end_datetime=datetime.now()
end=time.time()
elapsed=end-start
hours, remainder=divmod(elapsed, 3600)
minutes, seconds=divmod(remainder, 60)
print(f"\nRun started at: {start_datetime.strftime('%d/%m/%Y %H:%M:%S')}")
print(f"Run finished at: {end_datetime.strftime('%d/%m/%Y %H:%M:%S')}")
print(f"\nElapsed time: {hours:.0f} hh {minutes:.0f} mm {seconds:.2f} ss\n")
if file is not None:
    file.write(f"\nRun started at: {start_datetime.strftime('%d/%m/%Y %H:%M:%S')}\n")
    file.write(f"Run finished at: {end_datetime.strftime('%d/%m/%Y %H:%M:%S')}\n")
    file.write(f"\nElapsed time: {hours:.0f} hh {minutes:.0f} mm {seconds:.2f} ss\n")
    file.close()
#%%
torch.save({
    "model_state_dict": model_HeHe.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    # "scheduler_state_dict": scheduler.state_dict(),
    "epochs_completed": epochs,
    "learning_rate": learning_rate,
    "input_data": input_data,
    "output_data": output_data,
}, "VMC/new_methods/helium4_harmonic_trapped_LJ/checkpoint_HeHe.pth")









