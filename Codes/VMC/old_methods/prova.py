# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 19:50:40 2026

@author: Ale
"""


import numpy as np
import torch




def MC_new_config(self, Delta, exp_function):
    
    samples=self.MC_samples
    curr_config=self.particle_config[-1]
    # old_config=self.starting_config
    self.particle_config=[]
    # self.eval()
    
    
    
    # accepted=0
    # with torch.no_grad():
    #     for i in range(samples):
    #         x=Delta*(torch.rand_like(curr_config)-0.5)
    #         new_config=curr_config+x
    #         # print(model.layers[0].weight.dtype)
    #         y_new=self(new_config)
    #         y_curr=self(curr_config)
            
    #         if exp_function:
    #             log_ratio=torch.clamp(2*(y_new-y_curr),max=0.0)
    #             p=torch.exp(log_ratio)
    #         else:
    #             p=(y_new/y_curr).square()
    #             p=torch.clamp(p,max=1.0)
                
            
            
    #         if torch.rand((), device=p.device)<=p:
    #             curr_config=new_config
    #             accepted+=1
                
    #         self.particle_config.append(curr_config.clone())
    #     self.acceptance_ratio=accepted/samples
        
        
    config_list, acceptance_ratio=self.new_config(curr_config, Delta, samples, exp_function, save_data=True)    
    self.particle_config=torch.stack(config_list)
    self.acceptance_ratio=acceptance_ratio
    self.acc_ratio_list.append(acceptance_ratio)
    # batch=batch.squeeze(-1)
    # print(batch)
    # self.train()
    print(f"Acceptance ratio: {acceptance_ratio:.04f}\n")  




        
        
def new_config(self, starting_config, Delta, samples, exp_function, save_data):
    
    curr_config=starting_config
    accepted=0
    config_list=[]
    with torch.no_grad():
        for i in range(samples):
            x=Delta*(torch.rand_like(curr_config)-0.5)
            new_config=curr_config+x
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
                accepted+=1
            
            if save_data:
                config_list.append(curr_config.clone())
        acceptance_ratio=accepted/samples
        if save_data:
            return config_list, acceptance_ratio
        
        return acceptance_ratio
    
    
    
    
    
    
    
    
    def MC_new_config(self, Delta, exp_function):
        
        samples=self.MC_samples
        curr_config=self.particle_config[-1]
        # old_config=self.starting_config
        self.particle_config=[]
        # self.eval()
        accepted=0
        with torch.no_grad():
            for i in range(samples):
                x=Delta*(torch.rand_like(curr_config)-0.5)
                new_config=curr_config+x
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
                    accepted+=1
                    
                self.particle_config.append(curr_config.clone())
            self.acceptance_ratio=accepted/samples
            self.particle_config=torch.stack(self.particle_config)
            self.acc_ratio_list.append(self.acceptance_ratio)
        # batch=batch.squeeze(-1)
        # print(batch)
        # self.train()
        print(f"Acceptance ratio: {self.acceptance_ratio:.04f}\n")  
        
        
        
        
        
        
        
#%%#DA TESTARE
def my_sum(x1, x2):
    return x1+x2

def my_diff(x1,x2):
    return x1-x2


fun1=my_sum+my_diff

gigi=fun1(2,3)

print(gigi)


#%%
#introdurre i potenziali come una lista di potenziali da applicare ad ogni configurazione

def hamiltoniana(q, p, masse, T, potenziali):
    energia_cinetica_totale = T(p, masse)

    energia_potenziale_totale = sum(
        V(q) for V in potenziali
    )
    
    
#%%



import torch
from torch.func import functional_call, jvp

def make_sr_operator(model, configurations, damping=1e-3):
    names, shapes, sizes = [], [], []
    params_dict = {}

    for name, p in model.named_parameters():
        if p.requires_grad:
            names.append(name)
            shapes.append(p.shape)
            sizes.append(p.numel())
            params_dict[name] = p

    params = tuple(params_dict.values())
    theta = torch.cat([p.detach().reshape(-1) for p in params])
    buffers = dict(model.named_buffers())

    def f_flat(t):
        new_params = {}
        offset = 0

        for name, shape, size in zip(names, shapes, sizes):
            new_params[name] = t[offset:offset + size].view(shape)
            offset += size

        out = functional_call(
            model, (new_params, buffers), (configurations,)
        )
        return out.reshape(-1)

    # Valutazione della funzione sui parametri correnti
    theta = theta.requires_grad_(True)
    f = f_flat(theta)
    N = f.numel()

    def matvec(v):
        # J @ v, senza costruire J
        _, Jv = jvp(f_flat, (theta,), (v,))

        # Centratura delle derivate logaritmiche
        Jv = Jv - Jv.mean()

        # J.T @ Jv / N, senza costruire S
        STv = torch.autograd.grad(
            f, theta,
            grad_outputs=Jv,
            retain_graph=True
        )[0] / N

        return STv + damping * v

    return theta, matvec

























#%%

gigi=5.1
L=10
gigi=gigi-L*round(gigi/L)
print(gigi)


#%%
L=5
curr_config=torch.tensor(([1.0,2.0,3.0,4.0]))
# curr_config=torch.cat()
Delta=0.5
x=Delta*(torch.rand_like(curr_config)-0.5)
new_config=curr_config+x
print(new_config)
new_config=new_config-L*(new_config/L).round()
print(new_config)




#%%

for i in range(5):
    for j in range(i+1, 5):
        print(f"{i}{j}")





#%%

h=6.626e-34
m_He=3.73e9
PM_He=4.002
Na=6.022e23
m_He=(PM_He/Na)/1000
k_B=1.38e-23
# e=1.602e-19
from_met_to_Angs=1.0e10
# c=299792458.0




# h_square_2m_He=((h*from_met_to_Angs*c)**2)  /(((2*np.pi)**2)*2*m_He*e*k_B)

factor=((h*from_met_to_Angs)**2)/(((2*np.pi)**2)*2*m_He*k_B)
#%%

He_mw=4.003 
He_mass=(He_mw*10**(-3))/(6.022*10**(23)) 
h_bar=1.054571817*10**(-14) 
kB=1.380649*10**(-3) 


gigi=((h_bar**2)/(2*He_mass*kB))

#%%

N=5
i=0
while (i <N):
    if i%3==0 and i!=0:
        i-=1
    
    print(i)
    i+=1


#%%
n=100
for i in range(n):
    print(f"{2.43*(1-(i/(n-1)))}")
    
    
    #%%
    
from torch.func import jacrev, vmap
import torch

data=torch.rand([10,1])
x=3

f=(x*data)**2

grad=jacrev(f)

gigi=vmap(grad)(data)
#%%

import torch
from torch.func import vjp

def f(x):
    return torch.stack([
        x[0]**2,
        x[0] * x[1],
        x[1]**2
    ])

x = torch.tensor([2.0, 3.0])

y, vjp_fn = vjp(f, x)

v = torch.tensor([1.0, 2.0, 3.0])
result = vjp_fn(v)

print(y)       # tensor([4., 6., 9.])
print(result)  # (tensor([14., 22.]),)