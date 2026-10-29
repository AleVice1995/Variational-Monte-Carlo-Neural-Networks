# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 12:21:51 2026

@author: Ale
"""
import torch
from VMC.Energy import compute_potential_energy
from VMC.Potential_energy_list import HO_potential
from VMC.Potential_energy_list import V_tail
from VMC.Potential_energy_list import Lennard_jones_potential
from VMC.FCC_lattice_He import FCC

density=0.02
N_particles=32
conf_part, Length_box=FCC(density, N_particles)
configurations=conf_part.reshape(1,-1)
# configurations=10*(torch.rand((10,6))-0.5)

HO_factor_molt=0.5
info_HO={
    "HO_factor_molt": 0.5}
pot_HO_info=[HO_potential, info_HO]

info_LJ={
    "sigma": 2.556,
    "eps": 10.4,
    "pbc_cond": True,
    "length_box_pbc": Length_box,
    }
pot_LJ_info=[Lennard_jones_potential, info_LJ]

info_V_tail={
    "n_particles": N_particles,
    "density": density,
    "length_box_pbc": Length_box,
    "sigma": 2.556,
    "eps": 10.4,
    }
pot_V_tail=[V_tail, info_V_tail]


potentials=[pot_HO_info, pot_LJ_info, pot_V_tail]
V=compute_potential_energy(potentials, configurations)

print(f"V: {V[0,0].numpy():.04f}")