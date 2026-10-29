# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 13:19:03 2026

@author: Ale
"""

import torch
from torch.func import vmap, jvp, functional_call, jacrev

def minSR(model, particle_configurations, n_particles,
          local_energies, energy_mean, 
          optimizer, exp_function, diag_shift):
    
    particle_configurations.requires_grad_(True)
    optimizer.zero_grad()
    
    e=torch.nan_to_num(local_energies).reshape(-1)-energy_mean
    e=torch.nan_to_num(e)
    f_NN=model(particle_configurations)
    if exp_function:
        f=f_NN.reshape(-1)
    else:
        f=torch.log(torch.abs(f_NN)).reshape(-1)
     
    
    O_rows=[]
    
    Num_of_conf=particle_configurations.shape[0]
    
    params=dict(model.named_parameters())
    
    def f(params, x):
        return functional_call(model, params, (x,))
    
    jacobian_reverse_function=jacrev(f, argnums=0)
    jacobian=vmap(jacobian_reverse_function, in_dims=(None, 0))(params, particle_configurations)
    
    jac_of_all_conf_list=[]
    for k in range(Num_of_conf):
        jac_per_conf=torch.cat([jacobian[name][k].reshape(-1) for name in params]).detach()
        jac_of_all_conf_list.append(jac_per_conf)
        
    O=torch.stack(jac_of_all_conf_list, dim=0)  
    O_centered =O-O.mean(dim=0, keepdim=True)
    
    T=O_centered@O_centered.T
    scale = T.diagonal().mean().detach().clamp_min(1e-12)
    
    Id=torch.eye(
        T.shape[0],
        dtype=T.dtype,
        device=T.device,
    )
    
    T_reg = T + diag_shift * scale * Id
    

    alpha = torch.linalg.solve(T_reg, 2.0 * e.to(T.dtype))
    
    delta_theta = (O_centered.T @ alpha)
    
    lr = optimizer.param_groups[0]["lr"]

    offset = 0
    
    # with torch.no_grad():
    #     for p in params:
    #         n = p.numel()
    #         dp = delta_theta[offset:offset + n].reshape_as(p)
    
    #         p.add_(dp, alpha=lr)
    #         offset += n
    params=[p for p in model.parameters() if p.requires_grad]
    for p in params:
        n = p.numel()
    
        dp = delta_theta[offset:offset + n].reshape_as(p)
    
        p.grad = dp.detach().clone()
    
        offset += n
    
    
    
    grad_norm_before=torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=300.0
    )
    print(f"Gradient norm before clip: {grad_norm_before.item():.6f}")
    # grad_norm_after = torch.sqrt(
    #     sum(
    #         p.grad.detach().square().sum()
    #         for p in self.parameters()
    #         if p.grad is not None
    #     )
    # )
    grad_norm_after=min(grad_norm_before.item(), 300.0)
    print(f"Gradient norm after: {grad_norm_after:.6f}")
    # r_min=distance(self.particle_config).min().item()
    # r_max=distance(self.particle_config).max().item()
    # if r_min<2.555:
    #     print("ATTENTION!")
    #     print(f"minimum distance: {r_min: .04f}")
    #     print(f"maximum distance: {r_max: .04f}")
    optimizer.step()
    particle_configurations.requires_grad_(False)
    return grad_norm_before.item()