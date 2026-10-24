# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 00:30:53 2026

@author: Ale
"""

import numpy as np
import torch

def set_particles_in_cell(n_particles, length_cell):
    
    particles_in_cell=[]#np.zeros((n_particles,3))
    for i in range(n_particles):
        x_cont=y_cont=z_cont=0
        if i==1:
            x_cont=y_cont=1
        elif i==2:
                x_cont=z_cont=1
        elif i==3:
            y_cont=z_cont=1
        x_coord=length_cell*x_cont/2
        y_coord=length_cell*y_cont/2
        z_coord=length_cell*z_cont/2
        
        particles_in_cell.append([x_coord, y_coord, z_coord])
        
    return torch.tensor(particles_in_cell)


def FCC(density, N_particles):
    with torch.no_grad():
        particles_per_full_cell=4
        n_cells=int(np.ceil(N_particles/particles_per_full_cell))
        cells_per_direction=int(np.ceil(np.cbrt(n_cells)))
        length_box=np.cbrt(N_particles/density)
        length_per_cell=length_box/cells_per_direction
        
        n_cells_full=int(np.floor(N_particles/particles_per_full_cell))
        remaining_particles_cell_not_full=N_particles%particles_per_full_cell
        
        if N_particles<=particles_per_full_cell:
            particle_config=set_particles_in_cell(N_particles, length_per_cell)
        else:
            cell_config=set_particles_in_cell(particles_per_full_cell, length_per_cell)
            particle_config=[]
            for i in range(n_cells):
                x_cont=i%cells_per_direction
                y_cont=int(i/cells_per_direction)%cells_per_direction
                z_cont=int(i/(cells_per_direction**2))%cells_per_direction
                if i<n_cells_full:
                    new_cell=cell_config
                else:
                    new_cell=set_particles_in_cell(remaining_particles_cell_not_full, length_per_cell)
                
                translation=length_per_cell*torch.tensor([x_cont,y_cont,z_cont])
                new_cell=new_cell+translation
                # particle_config=torch.vstack((particle_config, new_cell))
                particle_config.append(new_cell)
            particle_config = torch.cat(particle_config, dim=0)
            # particle_config = torch.stack(particle_config)
        particle_config=particle_config-length_box/2.0
    
    return particle_config.float(), torch.tensor(length_box)

