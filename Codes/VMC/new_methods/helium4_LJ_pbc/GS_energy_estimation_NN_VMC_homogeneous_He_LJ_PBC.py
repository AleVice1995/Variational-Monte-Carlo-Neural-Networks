# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 17:17:12 2026

@author: Ale
"""

import time
from datetime import datetime
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from VMC.FCC_lattice_He import FCC
from VMC.new_methods.class_and_functions.Neural_Networks_models import Neural_Network_VMC_He_0K
from VMC.new_methods.class_and_functions.sampler import Sampler
from VMC.new_methods.class_and_functions.Training import training_NN_VMC
from VMC.new_methods.class_and_functions.Potential_energy_list import Lennard_jones_potential, V_tail
from VMC.new_methods.class_and_functions.utils import autocorrelation_length, print_values_on_log
from VMC.new_methods.class_and_functions.Energy import compute_energy_local
import gc



torch.set_default_dtype(torch.float64)
# file=None
switch_power="3"
file_changer="_run3_switch_power_"+switch_power
file_name="VMC/new_methods/helium4_LJ_pbc/GS_En_est_NN_VMC_Homog_He_LJ_PBC_log_file" +file_changer + ".txt"
checkpoint_name="VMC/new_methods/helium4_LJ_pbc/checkpoint_He_pbc" + file_changer + ".pth"

file=open("VMC/new_methods/helium4_LJ_pbc/GS_En_est_NN_VMC_Homog_He_LJ_PBC_log_file" +file_changer + ".txt", "w")
problem_under_study="VMC Neural Network Estimation of the Ground State Energy of Liquid Helium-4"
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
start_delta_finder=20
MC_samples=600
batch=500
sweep_per_save=4
epochs=60
delta=1.0
learning_rate=2e-3
reg_param=0.0
switch_power=int(switch_power)

single_particle_move=True
exp_function=True
pbc_cond=True
time_control=True

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
# print(f"V_LJ: {V.item(): .04f}")
# print(f"V_tail: {v_tail_contr: .010f}")

layers=([1,16, 16,  1])
activations=(nn.Softplus(), nn.Softplus(),  nn.Identity()) 


input_data=print_values_on_log(problem_under_study,n_particles, dimension, conf_part, energy_info,\
                               layers, activations, epochs, MC_samples, batch, sweep_per_save,\
                                   diag_shift, reg_param, exp_function, time_control=time_control, pbc_cond=pbc_cond,
                    density=density, Length_box=Length_box, switch_power=switch_power, file=file)






model_He=Neural_Network_VMC_He_0K(layers, activations, Length_box, dimension, switch_power)


sampler=Sampler(n_particles=n_particles, dimension=dimension, 
                pbc_cond=pbc_cond, length_box_pbc=Length_box,
                exp_function=exp_function, single_particle_move=single_particle_move)



optimizer=torch.optim.SGD(model_He.parameters(), lr=learning_rate, weight_decay=reg_param)
# scheduler=None
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=100,  # Every 50 epochs
    gamma=0.7      # Multiply LR by 0.1
)

# checkpoint = torch.load("VMC/new_methods/checkpoint_He_pbc.pth", map_location=device, weights_only=True)
# model_He.load_state_dict(checkpoint["model_state_dict"])

output_data=training_NN_VMC(model_He, sampler, epochs, 
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


torch.save({
    "model_state_dict": model_He.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "epochs_completed": epochs,
    "learning_rate": learning_rate,
    "input_data": input_data,
    "output_data": output_data,
}, checkpoint_name)


print(f"file name:\n{file_name}\n")
print(f"checkpoint name:\n{checkpoint_name}\n")