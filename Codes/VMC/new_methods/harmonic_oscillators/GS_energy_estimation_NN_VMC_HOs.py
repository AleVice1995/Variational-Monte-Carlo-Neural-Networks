# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 12:28:34 2026

@author: Ale
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime

from VMC.new_methods.class_and_functions.Training import training_NN_VMC
from VMC.new_methods.class_and_functions.Neural_Networks_models import Neural_Network_VMC_HO
from VMC.new_methods.class_and_functions.sampler import Sampler
from VMC.new_methods.class_and_functions.Potential_energy_list import HO_potential
from VMC.new_methods.class_and_functions.utils import autocorrelation_length, print_values_on_log
from VMC.new_methods.class_and_functions.Energy import compute_energy_local






torch.set_default_dtype(torch.float64)

file=open("VMC/new_methods/harmonic_oscillators/GS_En_est_NN_VMC_HO_log_file.txt", "w")
problem_under_study="VMC neural network estimation of the ground state energy of Harmonic oscillators"
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
batch=500
sweep_per_save=4
epochs=1000
delta=13.0
learning_rate=1e-2
reg_param=0.0

single_particle_move=True
exp_function=True
pbc_cond=False
time_control=True

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

input_data=print_values_on_log(problem_under_study,n_particles, dimension, starting_config, energy_info,\
                               layers, activations, epochs, MC_samples, batch, sweep_per_save,\
                                   diag_shift, reg_param, exp_function, time_control=time_control, file=file)







model_HO=Neural_Network_VMC_HO(layers, activations)

sampler=Sampler(n_particles=n_particles, dimension=dimension,
                pbc_cond=pbc_cond, length_box_pbc=length_box_pbc, 
                exp_function=exp_function, single_particle_move=single_particle_move)
optimizer=torch.optim.SGD(model_HO.parameters(), lr=learning_rate, weight_decay=reg_param)
# scheduler=None
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=100,  # Every 50 epochs
    gamma=0.7      # Multiply LR by 0.1
)


# checkpoint = torch.load("VMC/new_methods/checkpoint_HO.pth", map_location=device, weights_only=True)
# model_HO.load_state_dict(checkpoint["model_state_dict"])

output_data=training_NN_VMC(model_HO, sampler, epochs, 
                    starting_config, delta, energy_info,
                    MC_samples, batch, sweep_per_save,
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
    "model_state_dict": model_HO.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    # "scheduler_state_dict": scheduler.state_dict(),
    "epochs_completed": epochs,
    "learning_rate": learning_rate,
    "input_data": input_data,
    "output_data": output_data,
}, "VMC/new_methods/harmonic_oscillators/checkpoint_HO.pth")




