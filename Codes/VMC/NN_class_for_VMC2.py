# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 11:11:12 2026

@author: Ale
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 07:35:03 2026

@author: Ale
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 16:10:47 2026

@author: Ale
"""

# import numpy as np
import torch
import torch.nn as nn
import numpy as np
# from torch.utils.data import DataLoader
# from torch.utils.data import TensorDataset


class Neural_Network_VMC(nn.Module):
    
    def __init__(self, layers, activations):
        super().__init__()
        
        self.num_of_layers=len(layers)
        self.layers=nn.ModuleList()
        self.activations=nn.ModuleList(activations)
        self.particle_config=[]
        self.acceptance_ratio=0
        self.energy=0
        self.en_std_dev=0
        self.energy_list=[]
        self.en_std_dev_list=[]
        self.en_err_list=[]
        self.grad_list=[]
        self.MC_samples=0
        self.loss_list=[]
        self.acc_ratio_list=[]
        self.Delta_list=[]
        self.particles=0
        self.density=0
        self.use_Jastrow=False
        self.kinetic_factor_multiplier=0.5
        self.HO_pot_factor=0.5
        
        for i in range(self.num_of_layers-1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
            nn.init.xavier_uniform_(self.layers[i].weight)
            # nn.init.zeros_(self.layers[i].bias)
    
        nn.init.normal_(self.layers[-1].weight, mean=0.0, std=1e-3)
        # nn.init.zeros_(self.layers[-1].weight)
        # nn.init.zeros_(self.layers[-1].bias)
    def forward(self, x):
        if self.use_Jastrow:
            J=Jastrow(x, self.length_box_pbc)
        
        if self.pbc_cond:
            arg=2*torch.pi*x/self.length_box_pbc
            sin_tens=arg.sin()
            cos_tens=arg.cos()
            x=torch.stack((sin_tens, cos_tens), dim=x.ndim).reshape(*x.shape[:-1],-1)
        
        for i in range(self.num_of_layers-1):
            z=self.layers[i](x)
            x=self.activations[i](z)
        
        if self.use_Jastrow:
            return x+J
        
        return x
           
         
    def Training_NN_VMC(self, potential, starting_config, Delta, MC_samples,
                        epochs, learning_rate, reg_param, exp_function, 
                        burn_data, optimizer, scheduler=None,
                        pbc_cond=False, length_box_pbc=0, HO_pot_factor=0.5,
                        single_particle_move=False, start_delta_finder=5,
                        use_minSR=False, diag_shift=1e-12, is_pot_hard_core=False, 
                        use_continuation=False, alpha=0.0):
        
        self.MC_samples=MC_samples
        self.particle_config.append(starting_config)
        self.starting_config=starting_config
        self.min_acc_ratio=0.40
        self.max_acc_ratio=0.60
        self.pbc_cond=pbc_cond
        self.length_box_pbc=length_box_pbc
        # optimizer=torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=reg_param)
        cont=0
            
        
        
        for epoch in range(epochs):
            print(f"Epoch: {epoch+1:d}")
            current_lr=optimizer.param_groups[0]["lr"]
            print(f"Learning rate: {current_lr:.04e}\n")
            cond_delta_finder=(start_delta_finder is not None) and (epoch>=start_delta_finder)
            if (cond_delta_finder and (self.acceptance_ratio<self.min_acc_ratio or self.acceptance_ratio>self.max_acc_ratio)):
                Delta=self.delta_finder(Delta,  exp_function, samples=1000, single_particle_move=single_particle_move)
            
            self.Delta_list.append(Delta)
            self.MC_new_config(Delta, exp_function, burn_data, single_particle_move=single_particle_move, is_pot_hard_core=is_pot_hard_core)
            # values=count_small_distances(x=self.particle_config, pbc_cond=pbc_cond, length_box=length_box_pbc, alpha=2.43)
            # print(f" {values}")
            # print("Prima dell'ottimizzazione:")
            # print("Pesi finiti:", all(
            #     torch.isfinite(p).all().item()
            #     for p in self.parameters()
            # ))
            # print("Parametri massimi:", max(
            #     p.detach().abs().max().item()
            #     for p in self.parameters()
            # ))
            
            if use_continuation and epoch%100==0:
                cont+=1
                alpha=2.43*(1-(cont/99))
            
            if use_minSR:
                self.perform_one_step_minSR(potential, optimizer, exp_function, diag_shift=diag_shift, alpha=alpha)
            else:
                loss=self.perform_one_step(potential, optimizer, exp_function)
            
            # print("Dopo l'ottimizzazione:")
            # print("Pesi finiti:", all(
            #     torch.isfinite(p).all().item()
            #     for p in self.parameters()
            # ))
            # print("Parametri massimi:", max(
            #     p.detach().abs().max().item()
            #     for p in self.parameters()
            # ))
            
                        
            if scheduler is not None:
                # scheduler.step(self.energy.item())
                scheduler.step()
            print("")
            
               
    def MC_new_config(self, Delta, exp_function, burn_data, single_particle_move, is_pot_hard_core):
        
        samples=self.MC_samples
        curr_config=self.particle_config[-1]
        # curr_config=self.starting_config
        self.particle_config=[]
         
        config_list, acceptance_ratio=self.new_config(curr_config, Delta, samples,
                                                      exp_function, save_data=True, burn_data=burn_data,
                                                      single_particle_move=single_particle_move,
                                                      is_pot_hard_core=is_pot_hard_core)    
        self.particle_config=torch.stack(config_list)
        self.acceptance_ratio=acceptance_ratio
        self.acc_ratio_list.append(acceptance_ratio)
        
        print(f"Delta: {Delta:.04f}")
        print(f"Acceptance ratio: {acceptance_ratio:.04f}\n")
        
    
            
    def new_config(self, starting_config, Delta, samples,
                   exp_function, save_data, burn_data=0, single_particle_move=False,
                   is_pot_hard_core=False):
        
        curr_config=starting_config
        accepted=0
        config_list=[]
        samples_tot=samples+burn_data
        cont_burn=0
        cont_tot=0
        if self.pbc_cond:
            L=self.length_box_pbc
        with torch.no_grad():
            i=0
            while i<samples_tot:
                cont_tot+=1
            # for i in range(samples_tot):
                if single_particle_move:

                    # Seleziona un atomo casuale
                    n_particles = curr_config.numel() // 3
                    particle_idx=i%n_particles
                    # particle_idx = torch.randint(
                    #     low=0,
                    #     high=n_particles,
                    #     size=(1,)
                    # ).item()

                    # Copia la configurazione corrente
                    new_config = curr_config.clone()

                    # Indici delle tre coordinate dell'atomo selezionato
                    idx_start = 3 * particle_idx
                    idx_end = idx_start + 3

                    # Proponi uno spostamento solo per questo atomo
                    displacement = Delta * (
                        torch.rand_like(curr_config[idx_start:idx_end]) - 0.5
                    )

                    new_config[idx_start:idx_end] += displacement
                else:
                    displacement=Delta*(torch.rand_like(curr_config)-0.5)
                    new_config=curr_config+displacement
                if self.pbc_cond:
                    new_config=new_config-L*(new_config/L).round()
                # if is_pot_hard_core:
                #     cond_to_skip=is_distance_too_small(new_config, self.pbc_cond, self.length_box_pbc, alpha=2.0)
                #     if cond_to_skip:
                #         cont_burn+=1
                #         continue
                    
                    
                # print(model.layers[0].weight.dtype)
                y_new=self(new_config)
                y_curr=self(curr_config)
                
                if exp_function:
                    log_ratio=torch.clamp(2*(y_new-y_curr),max=0.0)
                    p=torch.exp(log_ratio)
                else:
                    p=(y_new/y_curr).square()
                    p=torch.clamp(p,max=1.0)
                    
                
                
                if torch.rand((), device=p.device)<=p:
                    curr_config=new_config
                    if i>=burn_data:
                        accepted+=1
                
                if save_data and i>=burn_data:
                    # print("Burn-in performed")
                    config_list.append(curr_config.clone())
                i+=1
            
            
            
            acceptance_ratio=accepted/samples
            # print(f"samples_bruciati: {cont_burn: d}")
            # print(f"totale contati: {cont_tot: d}")
            cont_tot
            if save_data:
                return config_list, acceptance_ratio
            
            
            return acceptance_ratio
    
        
    def delta_finder(self, Delta,  exp_function, samples=1000, single_particle_move=False):
        
        print("Finding best Delta parameter...")
        curr_config=self.particle_config[-1]
        delta_value_list=[]
        f_temp_list=[]
        min_acc_ratio=self.min_acc_ratio
        max_acc_ratio=self.max_acc_ratio
        f_temp=0
        
        for i in range(2):
            Delta=self.new_delta(Delta, f_temp)
            acceptance_ratio=self.new_config(curr_config, Delta, samples, exp_function,
                                             save_data=False, single_particle_move=single_particle_move)
            if acceptance_ratio>min_acc_ratio and acceptance_ratio<max_acc_ratio:
                break
            
            delta_value_list.append(Delta)
            f_temp=acceptance_ratio-0.5
            f_temp_list.append(f_temp)
            
            
                    
        while (acceptance_ratio<min_acc_ratio or acceptance_ratio>max_acc_ratio):
            if (np.sign(f_temp_list[0])==-np.sign(f_temp_list[1])):
                Delta=(delta_value_list[0]+delta_value_list[1])/2
                print(f"Testing Delta: {Delta:.04f}...")
                acceptance_ratio=self.new_config(curr_config, Delta, samples, exp_function,
                                                 save_data=False, single_particle_move=single_particle_move)
                print(f"Acc ratio: {acceptance_ratio: .04f}")
                f_temp=acceptance_ratio-0.5
                if (np.sign(f_temp)==np.sign(f_temp_list[0])):
                    delta_value_list[0]=Delta
                    f_temp_list[0]=f_temp
                else:
                    delta_value_list[1]=Delta
                    f_temp_list[1]=f_temp
            else:
                # m=(delta_value_list[1]-delta_value_list[0])/(f_temp_list[1]-f_temp_list[0])
                # Delta=delta_value_list[1]-m*f_temp_list[1]
                Delta=self.new_delta(delta_value_list[-1], f_temp)
                print(f"Testing Delta: {Delta:.04f}...")
                acceptance_ratio=self.new_config(curr_config, Delta, samples, exp_function,
                                                 save_data=False, single_particle_move=single_particle_move)
                print(f"Acc ratio: {acceptance_ratio: .04f}")
                f_temp=acceptance_ratio-0.5
                delta_value_list[0]=delta_value_list[1]
                delta_value_list[1]=Delta
                f_temp_list[0]=f_temp_list[1]
                f_temp_list[1]=f_temp
        
        
        # acceptance_ratio=self.new_config(curr_config, Delta, 10000, exp_function, save_data=False)
        # print(f"Delta: {Delta:.04f}")
        # print(f"Acceptance ratio: {acceptance_ratio:.04f}\n")
        return Delta

    def new_delta(self, Delta, f):
        if f>0:
            increment=1.5
        else:
            increment=1.9
        return Delta*(1+f*increment)

    def Energy_local(self, potential, exp_function, alpha=0.0):
        self.energy=0
        self.en_std_dev=0
        self.local_energies_list=0
        
        configurations=self.particle_config
        T=0
        T= self.Kinetic_energy_local(exp_function, self.kinetic_factor_multiplier)
        # V = torch.zeros_like(T)
        V=0
        for pot in potential:
            V=V+ pot(x=configurations, length_box_pbc=self.length_box_pbc,
                     pbc_cond=self.pbc_cond,
                      n_particles=self.particles, density=self.density,
                      sigma=2.556, eps=10.4, HO_factor_molt=self.HO_pot_factor, r_tilde=alpha)
        
        # V=potential(x=configurations)    
        local_energies=T+V
        self.local_energies_list=local_energies
        self.energy=local_energies.mean()
        self.energy_list.append(self.energy.item())
        energy_var=((local_energies**2).mean()-(self.energy**2))
        self.en_std_dev=energy_var.sqrt()
        self.en_std_dev_list.append(self.en_std_dev.item())
        if len(self.particle_config)>1:
            self.en_stoch_err=(energy_var/(len(self.particle_config)-1)).sqrt()
        else:
            self.en_stoch_err=self.en_std_dev
        self.en_err_list.append(self.en_stoch_err.item())
        N=self.particles
        if N==0:
           N=1 
        print(f"T_mean: {T.mean()/N: .04f}")
        print(f"V_mean: {V.mean()/N: .04f}")
        print(f"Energy_mean: {self.energy/N: .04f}")
        print(f"Energy std_dev: {self.en_std_dev/N: .04f}")
        print(f"Energy err: {self.en_stoch_err/N: .04f}")
        
        return local_energies
        
    def Kinetic_energy_local(self, exp_function, kinetic_factor_multiplier):
        
        configurations=self.particle_config
        if not configurations.requires_grad:
            configurations.requires_grad_(True)
        f_NN_x=self(configurations)
        df_NN_dx=torch.autograd.grad(outputs=f_NN_x, inputs=configurations, grad_outputs=torch.ones_like(f_NN_x), create_graph=True)[0]
        
        hessian_diag_terms=[]
        in_features=configurations.shape[1]
        for i in range(in_features):
            
            d2f_NN_dx2=torch.autograd.grad(outputs=df_NN_dx[:,i], inputs=configurations, grad_outputs=torch.ones_like(df_NN_dx[:,i]),
                                           create_graph=False, 
                                           retain_graph=(i<in_features-1))[0]
            hessian_diag_terms.append(d2f_NN_dx2[:,i])
        
        hessian_pure_der= torch.stack(hessian_diag_terms, dim=1)
        laplacian=torch.sum(hessian_pure_der, dim=1, keepdim=True)
        if exp_function:
            df_NN_dx_squared_summed=torch.sum(df_NN_dx**2, dim=1, keepdim=True)
            T=-kinetic_factor_multiplier*(df_NN_dx_squared_summed+laplacian)
        else:
            T=-kinetic_factor_multiplier*(laplacian/f_NN_x)
        # print(T)
        return T
        
    def perform_one_step(self, potential, optimizer, exp_function, alpha=0.0): 
        configurations=self.particle_config
        optimizer.zero_grad()
        
        local_energies=self.Energy_local(potential, exp_function, alpha=0.0)
        energy_deviations=torch.nan_to_num(local_energies)-torch.nan_to_num(self.energy)
        energy_deviations=torch.nan_to_num(energy_deviations)
        
        f_NN=self(configurations)
        if exp_function:
            f_NN_to_derive=f_NN
        else:
            f_NN_to_derive=torch.log(torch.abs(f_NN))
        # loss=self.energy
        loss=2*((energy_deviations.detach()*torch.nan_to_num(f_NN_to_derive)).mean())
        self.loss_list.append(loss.item())
        print(f"Loss: {loss: .04f}")
        loss.backward()
        
        
        # Norma globale del gradiente
                
        grad_norm_before=torch.nn.utils.clip_grad_norm_(
            self.parameters(),
            max_norm=300.0
        )
        self.grad_list.append(grad_norm_before.item())
        print(f"Gradient norm before clip: {grad_norm_before.item():.6f}")
        grad_norm_after = torch.sqrt(
            sum(
                p.grad.detach().square().sum()
                for p in self.parameters()
                if p.grad is not None
            )
        )
        
        print(f"Gradient norm after clip: {grad_norm_after.item():.6f}")
        # r_min=distance(self.particle_config).min().item()
        # r_max=distance(self.particle_config).max().item()
        # if r_min<2.555:
        #     print("ATTENTION!")
        #     print(f"minimum distance: {r_min: .04f}")
        #     print(f"maximum distance: {r_max: .04f}")
        optimizer.step()
        return loss
    
    
    def perform_one_step_minSR(self, potential, optimizer, exp_function, diag_shift=1e-4, alpha=0.0): 
        configurations=self.particle_config
        optimizer.zero_grad()
        
        # 1. Energie locali
        local_energies = self.Energy_local(potential, exp_function, alpha=alpha)
    
        # Forma [Ns]
        e = (
            torch.nan_to_num(local_energies).reshape(-1)
            - self.energy.detach()
        )
        e = torch.nan_to_num(e)
        
        f_NN=self(configurations)
        if exp_function:
            f=f_NN.reshape(-1)
        else:
            f=torch.log(torch.abs(f_NN)).reshape(-1)
         
        
        O_rows=[]
        
        Ns=f.numel()
        params=[p for p in self.parameters() if p.requires_grad]
        for k in range(Ns):
            grads=torch.autograd.grad(outputs=f[k], inputs=params,
                                      retain_graph=(k<(Ns-1)),
                                      create_graph=False, allow_unused=True)
            
            row = torch.cat([
                (torch.zeros_like(p) if g is None else g)
                .reshape(-1)
                for p, g in zip(params, grads)
            ])
            O_rows.append(row)

        O = torch.stack(O_rows, dim=0)
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
        
        for p in params:
            n = p.numel()
        
            dp = delta_theta[offset:offset + n].reshape_as(p)
        
            p.grad = dp.detach().clone()
        
            offset += n
        
        
        
        grad_norm_before=torch.nn.utils.clip_grad_norm_(
            self.parameters(),
            max_norm=1000.0
        )
        self.grad_list.append(grad_norm_before.item())
        print(f"Gradient norm before clip: {grad_norm_before.item():.6f}")
        # grad_norm_after = torch.sqrt(
        #     sum(
        #         p.grad.detach().square().sum()
        #         for p in self.parameters()
        #         if p.grad is not None
        #     )
        # )
        grad_norm_after=min(grad_norm_before.item(), 1000.0)
        print(f"Gradient norm after: {grad_norm_after:.6f}")
        # r_min=distance(self.particle_config).min().item()
        # r_max=distance(self.particle_config).max().item()
        # if r_min<2.555:
        #     print("ATTENTION!")
        #     print(f"minimum distance: {r_min: .04f}")
        #     print(f"maximum distance: {r_max: .04f}")
        optimizer.step()
        
        
    def perform_one_step_SR(self, potential, optimizer, exp_function, diag_shift=1e-4): 
        configurations=self.particle_config
        optimizer.zero_grad()
        
        # 1. Energie locali
        local_energies = self.Energy_local(potential, exp_function)
    
        # Forma [Ns]
        e = (
            torch.nan_to_num(local_energies).reshape(-1)
            - self.energy.detach()
        )
        e = torch.nan_to_num(e)
        
        f_NN=self(configurations)
        if exp_function:
            f=f_NN.reshape(-1)
        else:
            f=torch.log(torch.abs(f_NN)).reshape(-1)
         
        
        O_rows=[]
        
        Ns=f.numel()
        params=[p for p in self.parameters() if p.requires_grad]
        for k in range(Ns):
            grads=torch.autograd.grad(outputs=f[k], inputs=params,
                                      retain_graph=(k<(Ns-1)),
                                      create_graph=False, allow_unused=True)
            
            row = torch.cat([
                (torch.zeros_like(p) if g is None else g)
                .reshape(-1)
                for p, g in zip(params, grads)
            ])
            O_rows.append(row)

        O = torch.stack(O_rows, dim=0)
        O_centered =O-O.mean(dim=0, keepdim=True)
        
        S=O_centered.T@O_centered
        scale = S.diagonal().mean().detach().clamp_min(1e-12)
        
        Id=torch.eye(
            S.shape[0],
            dtype=S.dtype,
            device=S.device,
        )
        S_reg = S + diag_shift * scale * Id
        
        matr_O_e=O_centered.T@(2*e)
        delta_theta = -torch.linalg.solve(S_reg, matr_O_e)
        
        
        
        lr = optimizer.param_groups[0]["lr"]

        offset = 0
        
        for p in params:
            n = p.numel()
        
            dp = delta_theta[offset:offset + n].reshape_as(p)
        
            p.grad = dp.detach().clone()
        
            offset += n
        
        
        
        # grad_norm_before=torch.nn.utils.clip_grad_norm_(
        #     self.parameters(),
        #     max_norm=300.0
        # )
        # self.grad_list.append(grad_norm_before.item())
        # print(f"Gradient norm before clip: {grad_norm_before.item():.6f}")
        grad_norm_after = torch.sqrt(
            sum(
                p.grad.detach().square().sum()
                for p in self.parameters()
                if p.grad is not None
            )
        )
        print(f"Gradient norm: {grad_norm_after.item():.6f}")
        
        optimizer.step()  
            
    def autocorrelation_length(self, potential, exp_function, autocorr_terms=100):
        
        c_autocorr=torch.zeros(autocorr_terms, 
                               device=self.local_energies_list.device, dtype=self.local_energies_list.dtype)
        samples=self.MC_samples
        local_energies=self.local_energies_list
        cont=0
        if autocorr_terms<samples:
            with torch.no_grad():
                for i in range(autocorr_terms, samples):
                    cont+=1
                    for j in range(autocorr_terms):
                        c_autocorr[j]+=local_energies[i,0]*local_energies[i-j,0]
                
                print(f"cont: {cont:d}")
                c_autocorr=c_autocorr/cont
                c_autocorr=(c_autocorr-self.energy**2)/(self.en_std_dev**2)
            return c_autocorr
        else:
            print("Number of autocorrelations terms less than MC samples")



    
def HO_potential(x, HO_factor_molt=0.5, *args, **kwargs):
    return HO_factor_molt*torch.sum(x**2, dim=1, keepdim=True)
    
# def Lennard_jones_potential(x, sigma, eps, pbc_cond=False, length_box_pbc=0,r_tilde=0.0, *args, **kwargs):
    
#     L=length_box_pbc
#     r_cut=length_box_pbc/2
#     x=x.reshape(x.shape[0],-1,3)
#     n_particles=x.shape[1]
    
#     V = torch.zeros(
#         (x.shape[0], 1),
#         dtype=x.dtype,
#         device=x.device
#     )
#     for i in range(n_particles):
#         for j in range(i+1,n_particles):
#             coords=x[:,i,:]-x[:,j,:]
#             if pbc_cond:
#                 coords=coords-L*(coords/L).round()
#             r_squared=torch.sum(coords**2, dim=1, keepdim=True) #this is the modulus square of distance
#             r_squared=torch.where(r_squared<r_tilde**2, r_tilde**2, r_squared)
#             if pbc_cond:
#                 particles_counted=r_squared.sqrt()<=r_cut
#             else:
#                 particles_counted=r_squared.bool()
#             attr_term=(sigma**6)/(r_squared**3)
#             rep_term=attr_term**2
#             v_pair=4*eps*(rep_term-attr_term)
#             # torch.where(condizione, valore_se_vera, valore_se_falsa)
#             V=V+torch.where(particles_counted, v_pair, torch.zeros_like(v_pair))
            
    
    
#     return V
            
def V_tail(n_particles, density, length_box_pbc, sigma, eps, *args, **kwargs):
    
    r_cut=length_box_pbc/2
    pot=(n_particles*8*torch.pi*eps*density*(sigma**6)*(sigma**6-3*(r_cut**6)))/(9*(r_cut**9))
    return pot
            
def distance(x):
    x=x.reshape(x.shape[0],-1,3)
    dr=x[:,0,:]-x[:,1,:]
    r_squared=torch.sum(dr**2, dim=1, keepdim=True)
    
    return r_squared.sqrt()
    
    
# def Jastrow(x, length_box_pbc):
    
#     b=2.95
#     L=length_box_pbc
#     r_cut=L/2.0
#     dim_conf=x.ndim
#     if dim_conf==1:
        
#         x=x.reshape(-1,3)
#         n_particles=x.shape[0]
#         J = torch.zeros(
#             (1),
#             dtype=x.dtype,
#             device=x.device
#         )
#     else:
#         x=x.reshape(x.shape[0],-1,3)
#         n_particles=x.shape[1]
#         J = torch.zeros(
#             (x.shape[0], 1),
#             dtype=x.dtype,
#             device=x.device
#         )
#     for i in range(n_particles):
#         for j in range(i+1,n_particles):
#             if dim_conf==1:
#                 coords=x[i,:]-x[j,:]
                
#             else:
#                 coords=x[:,i,:]-x[:,j,:]
#             coords=coords-L*(coords/L).round()
#             r_squared=torch.sum(coords**2, dim=dim_conf-1, keepdim=True) #this is the modulus square of distance
#             r = torch.sqrt(torch.clamp(r_squared, min=1e-12))
            
#             particles_inside_cutoff=r_squared.sqrt()<=r_cut
#             factor=(b/r)**5
#             # torch.where(condizione, valore_se_vera, valore_se_falsa)
#             J=J-0.5*torch.where(particles_inside_cutoff, factor, torch.zeros_like(factor))
    
#     return J




def Jastrow(x, length_box_pbc, b=2.95):
    L = length_box_pbc
    # Supporta sia una configurazione singola [3*N]
    # sia un batch di configurazioni [batch, 3*N]
    single_config = (x.ndim == 1)

    if single_config:
        x = x.unsqueeze(0)

    x = x.reshape(x.shape[0], -1, 3)
    n_particles = x.shape[1]

    # Indici di tutte le coppie i < j
    ij = torch.triu_indices(
        n_particles,
        n_particles,
        offset=1,
        device=x.device
    )
    i, j = ij[0], ij[1]

    # Differenze di posizione per tutte le coppie
    dr = x[:, i, :] - x[:, j, :]

    # Minimum-image convention
    dr = dr - L * torch.round(dr / L)

    # Distanze
    r_squared = torch.sum(dr**2, dim=-1)
    r = torch.sqrt(r_squared.clamp_min(1e-12))

    # Cutoff
    inside = r <= L / 2
    
    r_cut = L / 2.0
    # Switch regolare: valore, prima e seconda derivata
    # si annullano al cutoff.
    switch = (1.0 - (r / r_cut)**2).clamp_min(0.0)**3
    # pair_j = switch*(b / r)**5
    pair_j = (b / r)**5
    # Contributo di Jastrow per ogni coppia
    pair_terms = torch.where(
        inside,
        pair_j,
        torch.zeros_like(r)
    )

    # Somma su tutte le coppie
    J = -0.5 * pair_terms.sum(dim=1, keepdim=True)

    if single_config:
        return J.squeeze(0)

    return J


def Lennard_jones_potential(x, sigma, eps, pbc_cond=False, length_box_pbc=0,r_tilde=0.0, *args, **kwargs):
    
    L=length_box_pbc
    r_cut=length_box_pbc/2
    x=x.reshape(x.shape[0],-1,3)
    n_particles=x.shape[1]
    
    V = torch.zeros(
        (x.shape[0], 1),
        dtype=x.dtype,
        device=x.device
    )
    
    ij = torch.triu_indices(
        n_particles,
        n_particles,
        offset=1,
        device=x.device
    )
    i, j = ij[0], ij[1]

    # Differenze di posizione per tutte le coppie
    dr = x[:, i, :] - x[:, j, :]

    # Minimum-image convention
    if pbc_cond:
        dr = dr - L * torch.round(dr / L)

    # Distanze
    r_squared = torch.sum(dr**2, dim=-1)
    r = torch.sqrt(r_squared.clamp_min(1e-12))
    r=torch.where(r<r_tilde, r_tilde, r)
    if pbc_cond:
        particles_counted=r<=r_cut
    else:
        particles_counted=r.bool()
    
    attr_term=(sigma**6)/(r**6)
    rep_term=attr_term**2
    v_pair=4*eps*(rep_term-attr_term)
    # torch.where(condizione, valore_se_vera, valore_se_falsa)
    V=torch.where(particles_counted, v_pair, torch.zeros_like(v_pair)).sum(dim=1, keepdim=True)
    return V
            
            
def tot_para_NN(layers):
    tot_weights=0
    tot_biases=0
    
    for i in range(len(layers)):
        if i<len(layers)-1:
            tot_weights+=layers[i]*layers[i+1]
        if i>0:
            tot_biases+=layers[i]
            
    return tot_biases+tot_weights
    
def is_distance_too_small(x, pbc_cond, length_box, alpha=0.0):
    
    x=x.reshape(-1,3)
    n_part=x.shape[0]
    for i in range(n_part):
        for j in range(i+1, n_part):
            dr=x[i,:]-x[j,:]
            if pbc_cond:
                dr=dr-length_box*(dr/length_box).round()
            r_squared=torch.sum(dr**2, dim=0, keepdim=True) #this is the modulus square of distance
            r = torch.sqrt(torch.clamp(r_squared, min=1e-12))
            # print(f"r: {r.item():.04f}")
            if r<=alpha:
                # print(f"r {i}-{j}: {r.item():.04f}")
                return True
                break
    return False
    

def count_small_distances(x, pbc_cond, length_box, alpha=0.0):
    
    small_distances=[]
    x=x.reshape(x.shape[0],-1,3)
    n_part=x.shape[1]
    r_conf=[]
    ij = torch.triu_indices(
        n_part,
        n_part,
        offset=1,
        device=x.device
    )
    i, j = ij[0], ij[1]

    # Differenze di posizione per tutte le coppie
    dr = x[:, i, :] - x[:, j, :]

    # Minimum-image convention
    dr = dr - length_box * torch.round(dr / length_box)

    # Distanze
    r_squared = torch.sum(dr**2, dim=-1)
    r_conf = torch.sqrt(r_squared.clamp_min(1e-12))
        
            # print(f"r: {r.item():.04f}")
            
    
    
    mask=r_conf<=alpha
    n_true=mask.sum(dim=1)
    n_false=(~mask).sum(dim=1)
    
    values=torch.stack([n_true, n_false], dim=1)
    return values
    

