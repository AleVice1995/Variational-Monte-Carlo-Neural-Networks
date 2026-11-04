# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 13:19:03 2026

@author: Ale
"""

import torch
from torch.func import vmap, jvp, functional_call, jacrev

def minSR(model, particle_configurations, n_particles,
          local_energies, epoch, 
          optimizer, exp_function, diag_shift):
    
    particle_configurations.requires_grad_(True)
    optimizer.zero_grad()
    
    
    finite_values_bool=torch.isfinite(local_energies)
    num_nonfinite_values=torch.sum(~finite_values_bool)
    fraction_nonfinite=num_nonfinite_values*100/local_energies.shape[0]
    

    local_energies_finite=local_energies[finite_values_bool]

    energy_median_finite=torch.median(local_energies_finite)
    
    local_energies_cleaned=torch.where(finite_values_bool, local_energies, energy_median_finite)
    
    
    energy_median=torch.median(local_energies_cleaned)
    
    energy_deviations=local_energies_cleaned-energy_median
    energy_dispersion=torch.median(torch.abs(energy_deviations))
    energy_std=torch.std(local_energies_cleaned)
    
    k=5
    mad_threshold = 1e-4
    max_energy=torch.max(local_energies_cleaned)
    min_energy=torch.min(local_energies_cleaned)
    
    upper_energy_clipping_interval = None
    lower_energy_clipping_interval = None
    robust_scale_ratio = None
    control_value = None
    
    
    if energy_dispersion<mad_threshold:
        print("MAD ZERO CASE\n")
        upper_energy=torch.max(local_energies_cleaned)
        lower_energy=torch.min(local_energies_cleaned)
    else:
        # k_min=5
        # k_max=200
        
        #for a gaussian distribution the std=1.4826*D with D the dispersion(MAD median of aboslute deviation from median)
        
        robust_scale_ratio=energy_std/((1.4826*energy_dispersion))
        robust_scale_ratio_min_value=1.5
        control_norm_value=control_function(robust_scale_ratio_min_value)
        control_value=control_function(robust_scale_ratio)/control_norm_value
        # control_value=((robust_scale_ratio_min_value/robust_scale_ratio)**2)*(1-torch.exp(-1/robust_scale_ratio))/(1-torch.exp(torch.tensor(-1/robust_scale_ratio_min_value)))
        control_value=torch.clamp(control_value, max=1.0)
        # r_min=1
        # r_max=30
        
        # if robust_scale_ratio>r_max:
        #     control_value=0
        # elif robust_scale_ratio<r_min:
        #     control_value=1
        # else:
        #     control_value=1-(torch.log(robust_scale_ratio)/torch.log(r_max))
        # k=k_min+(k_max-k_min)*s_exp
        
        upper_energy_clipping_interval=energy_median+k*energy_dispersion
        lower_energy_clipping_interval=energy_median-k*energy_dispersion
        
        
        if max_energy>upper_energy_clipping_interval:
            upper_energy=upper_energy_clipping_interval+(max_energy-upper_energy_clipping_interval)*control_value
        else:
            upper_energy=max_energy
            
        if min_energy<lower_energy_clipping_interval:
            lower_energy=lower_energy_clipping_interval+(min_energy-lower_energy_clipping_interval)*control_value
        else:
            lower_energy=min_energy
        
    local_energies_clipped=torch.clamp(local_energies_cleaned, min=lower_energy,  max=upper_energy)

    interval_energy_bool=(local_energies_cleaned<lower_energy) | (local_energies_cleaned>upper_energy)
    

    num_clipped_values=torch.sum(interval_energy_bool)
    fraction_clipped=num_clipped_values*100/local_energies.shape[0]
    

    energy_mean_clipped=torch.mean(local_energies_clipped)
    
    e=(local_energies_clipped-energy_mean_clipped).reshape(-1)

    energy_mean=torch.mean(local_energies)
    
    # e2=torch.nan_to_num(local_energies).reshape(-1)-energy_mean
    # e2=torch.nan_to_num(e2)
    
    
    print(f"Fraction non-finite values: {num_nonfinite_values: d}/{local_energies.shape[0]:d}={fraction_nonfinite:.02f}%")
    print(f"Fraction clipped values: {num_clipped_values: d}/{local_energies.shape[0]:d}={fraction_clipped:.02f}%\n")
    print(f"Median of finite values: {energy_median_finite:.04e}")
    print(f"Median on \"cleaned\" values: {energy_median:.04e}")
    print(f"MAD Dispersion on \"cleaned\" values: {energy_dispersion:.04e}")
    print(f"Std dev on \"cleaned\" values: {energy_std:.04e}\n")
    if robust_scale_ratio is not None:
        print(f"Std to MAD ratio: {robust_scale_ratio: .04f}")
        print(f"control function: {control_value: .4f}\n")
        
        print(f"upper energy clipping interval: {upper_energy_clipping_interval: .04e}")
        print(f"max energy: {max_energy: .04e}")
        print(f"upper energy choosen: {upper_energy: .04e}")
        print(f"lower energy clipping interval: {lower_energy_clipping_interval: .04e}")
        print(f"min energy: {min_energy: .04e}")
        print(f"lower energy choosen: {lower_energy: .04e}\n")
    else:
        print(f"max energy: {max_energy: .04e}")
        print(f"min energy: {min_energy: .04e}\n")
        
    print(f"Energy mean clipped: {energy_mean_clipped:.04e}")
    print(f"Energy mean non-clipped: {energy_mean:.04e}\n")
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




def control_function(x):
    if not isinstance(x, torch.Tensor):
        x=torch.tensor(x)
        
    return (1-torch.exp(-1/x))/(x)