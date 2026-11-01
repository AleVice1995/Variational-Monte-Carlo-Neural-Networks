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
    
    
    finite_values_bool=torch.isfinite(local_energies)
    num_nonfinite_values=torch.sum(~finite_values_bool)
    fraction_nonfinite=num_nonfinite_values*100/local_energies.shape[0]
    

    local_energies_finite=local_energies[finite_values_bool]

    energy_median=torch.median(local_energies_finite)
    
    local_energies_cleaned=torch.where(finite_values_bool, local_energies, energy_median)

    energy_median=torch.median(local_energies_cleaned)
    
    energy_deviations=local_energies_cleaned-energy_median
    energy_dispersion=torch.mean(torch.abs(energy_deviations))
    
    factor=10

    upper_energy=energy_median+factor*energy_dispersion
    lower_energy=energy_median-factor*energy_dispersion
    
    local_energies_clipped=torch.clamp(local_energies_cleaned, min=lower_energy,  max=upper_energy)

    interval_energy_bool=(local_energies_cleaned<lower_energy) | (local_energies_cleaned>upper_energy)
    

    num_clipped_values=torch.sum(interval_energy_bool)
    fraction_clipped=num_clipped_values*100/local_energies.shape[0]
    

    energy_mean_clipped=torch.mean(local_energies_clipped)
    
    e=(local_energies_clipped-energy_mean_clipped).reshape(-1)

    energy_mean=torch.mean(local_energies)
    
    e2=torch.nan_to_num(local_energies).reshape(-1)-energy_mean
    e2=torch.nan_to_num(e2)
    
    print(f"Fraction non-finite values: {num_nonfinite_values: d}/{local_energies.shape[0]:d}={fraction_nonfinite:.02f}%")
    print(f"Fraction clipped values: {num_clipped_values: d}/{local_energies.shape[0]:d}={fraction_clipped:.02f}%")
    print(f"Median of finite values: {energy_median:.04e}")
    print(f"Median on \"cleaned\" values: {energy_median:.04e}")
    print(f"Dispersion on \"cleaned\" values: {energy_dispersion:.04e}")
    print(f"Energy mean clipped: {energy_mean_clipped:.04e}")
    print(f"Energy mean non-clipped: {energy_mean:.04e}")
    # f_NN=model(particle_configurations.reshape(particle_configurations.shape[0],-1))
    
    # if exp_function:
    #     f=f_NN.reshape(-1)
    # else:
    #     f=torch.log(torch.abs(f_NN)).reshape(-1)
     
    
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
    return grad_norm_before.item(), fraction_nonfinite.item(), fraction_clipped.item()