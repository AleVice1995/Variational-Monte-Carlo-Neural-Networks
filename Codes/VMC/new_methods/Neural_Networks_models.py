# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 08:55:08 2026

@author: Ale
"""

# import numpy as np
import torch
import torch.nn as nn
import numpy as np
from torch.func import vmap, jacrev, functional_call, hessian
from torch.func import jvp
# from torch.utils.data import DataLoader
# from torch.utils.data import TensorDataset


class Neural_Network_VMC_HO(nn.Module):
    
    def __init__(self, layers, activations):
        super().__init__()
        self.num_of_layers=len(layers)
        self.layers=nn.ModuleList()
        self.activations=nn.ModuleList(activations)
        
        for i in range(self.num_of_layers-1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
            nn.init.xavier_uniform_(self.layers[i].weight)
            # nn.init.zeros_(self.layers[i].bias)
            
    
        nn.init.normal_(self.layers[-1].weight, mean=0.0, std=1e-3)
        # nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)
        
    def forward(self, x):
        for i in range(self.num_of_layers-1):
            z=self.layers[i](x)
            x=self.activations[i](z)
            
        return x
    
    
     
        
        
class Neural_Network_VMC_HeHe(nn.Module):
    
    def __init__(self, layers_pos, activations_pos, layers_dist, activations_dist):
        super().__init__()
        
        #initializing POS subnet
        self.num_of_layers_pos=len(layers_pos)
        self.layers_pos=nn.ModuleList()
        self.activations_pos=nn.ModuleList(activations_pos)
        
        for i in range(self.num_of_layers_pos-1):
            self.layers_pos.append(nn.Linear(layers_pos[i], layers_pos[i+1]))
            nn.init.xavier_uniform_(self.layers_pos[i].weight)
            # nn.init.zeros_(self.layers_pos[i].bias)
        
        nn.init.normal_(self.layers_pos[-1].weight, mean=0.0, std=1e-3)
        # nn.init.zeros_(self.layers_pos[-1].weight)
        nn.init.zeros_(self.layers_pos[-1].bias)
        
        #initializing DIST subnet
        self.num_of_layers_dist=len(layers_dist)
        self.layers_dist=nn.ModuleList()
        self.activations_dist=nn.ModuleList(activations_dist)
        
        for i in range(self.num_of_layers_dist-1):
            self.layers_dist.append(nn.Linear(layers_dist[i], layers_dist[i+1]))
            nn.init.xavier_uniform_(self.layers_dist[i].weight)
            # nn.init.zeros_(self.layers_dist[i].bias)
            
        nn.init.normal_(self.layers_dist[-1].weight, mean=0.0, std=1e-3)
        # nn.init.zeros_(self.layers_dist[-1].weight)
        nn.init.zeros_(self.layers_dist[-1].bias)
        
    def forward(self, x):
        single=(x.ndim==1)
        
        if single:
            x=x.unsqueeze(0)
        
        x=x.reshape(x.shape[0],-1,3)
        n_particles=x.shape[1]
        
        ij=torch.triu_indices(n_particles,
                              n_particles,
                              offset=1,
                              device=x.device)
        i, j=ij[0], ij[1]
        
        dr=x[:, i, :]- x[:, j,:]
        
        r2=torch.sum(dr**2, dim=-1, keepdim=True)
        r=(r2.sqrt()).clamp_min(1e-12)
        pos_out=self.pos(x).sum(1)
        dist_out=self.dist(r).sum(1)
        # feat=torch.cat([pos_out, dist_out], dim=-1)
        # out=self.rho(feat)
        out=pos_out+dist_out
        return out.squeeze(0) if single else out
    
    def pos(self, x):
        for i in range(self.num_of_layers_pos-1):
            z=self.layers_pos[i](x)
            x=self.activations_pos[i](z)
            
        return x
    
    def dist(self, x):
        for i in range(self.num_of_layers_dist-1):
            z=self.layers_dist[i](x)
            x=self.activations_dist[i](z)
            
        return x
    

'introdurre il length box in qualche maniera'
class Neural_Network_VMC_He_0K(nn.Module):
    
    def __init__(self, layers, activations, length_box_pbc):
        super().__init__()
        self.num_of_layers=len(layers)
        self.layers=nn.ModuleList()
        self.activations=nn.ModuleList(activations)
        self.length_box_pbc=length_box_pbc
        
        for i in range(self.num_of_layers-1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
            nn.init.xavier_uniform_(self.layers[i].weight)
            # nn.init.zeros_(self.layers[i].bias)
            
    
        nn.init.normal_(self.layers[-1].weight, mean=0.0, std=1e-3)
        # nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)
    
    def forward(self, x):
        
        single=(x.ndim==1)
        
        if single:
            x=x.unsqueeze(0)
        
        x=x.reshape(x.shape[0],-1,3)
        n_particles=x.shape[1]
        
        ij=torch.triu_indices(n_particles,
                              n_particles,
                              offset=1,
                              device=x.device)
        i, j=ij[0], ij[1]
        
        dr=x[:, i, :]- x[:, j,:]
        dr=dr-self.length_box_pbc*torch.round(dr/self.length_box_pbc)
        
        r2=torch.sum(dr**2, dim=-1, keepdim=True)
        x=(r2.sqrt()).clamp_min(1e-12)
        # one_over_r=1.0/r
        # x=1.0/r
        
        
            
        out=torch.sum(self.phi(x), dim=1)
        return out.squeeze(0) if single else out
    
    
    def phi(self, x):
        for i in range(self.num_of_layers-1):
            z=self.layers[i](x)
            x=self.activations[i](z)
        return x
    