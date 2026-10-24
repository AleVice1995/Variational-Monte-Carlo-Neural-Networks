# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 01:33:47 2026

@author: Ale
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from Pytorch_codes.NN_classes_with_pytorch import Neural_Network as NN
from Pytorch_codes.NN_classes_with_pytorch import training_model_sequential
import time
import copy

#%%
#Simulation of a function in 2 variables
start=time.time()

points=int(np.sqrt(10000))
x1_arr = np.linspace(-5, 5, points, dtype=np.float32)
x2_arr = np.linspace(-5, 5, points, dtype=np.float32)
data_x1_x2_arr=[]
for i in range(points):
    for j in range(points):
        data_x1_x2_arr.append((x1_arr[i], x2_arr[j]))
        
data_x1_x2_arr=np.array(data_x1_x2_arr)
# X1, X2=np.meshgrid(x1, x2)

# Z= X1**2+X2**2
z_arr=data_x1_x2_arr[:,0]**2+data_x1_x2_arr[:,1]**2

data_x1_x2=torch.from_numpy(data_x1_x2_arr)
z=torch.from_numpy(z_arr).reshape(-1, 1)
dataset = TensorDataset(data_x1_x2, z)





#Construction of the NN
layers=[2,30, 30, 30, 1]
activations=[nn.ReLU(), nn.ReLU(), nn.ReLU(), nn.Identity()]

modules=[]
for i in range(len(layers)-1):
    modules.append(nn.Linear(layers[i],layers[i+1]))
    modules.append(activations[i])

# the * is used to pass the elements of the lists as separate items to nn.Sequential()
model=nn.Sequential(*modules)
model_initial=copy.deepcopy(model)



epochs=100
batch_size=100
learning_rate=1*10**-3
#reg_parameter is related to the L2 regularization
reg_parameter=0 #10/len(dataset)
loss_function=nn.MSELoss()#BCELoss()
data=training_model_sequential(model, dataset,
                               epochs=epochs, batch_size=batch_size, 
                               learning_rate=learning_rate, loss_function=loss_function,
                               reg_param= reg_parameter)



end=time.time()
elapsed=end-start
hours, remainder=divmod(elapsed, 3600)
minutes, seconds=divmod(remainder, 60)


print(f"\nElapsed time: {hours:.0f} hh {minutes:.0f} mm {seconds:.2f} ss\n")
training_Cost=data[1]

#%%
plt.close('all')
iterations=[i for i in range(epochs)]
plt.figure(num=1, dpi=120)

plt.plot(iterations, training_Cost, linestyle='-', marker='.', label="Training")
plt.xlabel("Epochs")
plt.ylabel("Cost")
plt.grid(True)
plt.legend()
plt.show()

chosen_points=2
# fig=plt.figure(num=2, dpi=120)

# # plt.contourf(X1, X2, Z, levels=50)
# # plt.colorbar(label="f(x1, x2)")
# # plt.xlabel("x1")
# # plt.ylabel("x2")
# # plt.show()

# ax = fig.add_subplot(111, projection='3d')
# # ax.plot_surface(X1, X2, Z)
# ax.scatter(data_x1_x2[::chosen_points,0], data_x1_x2[::chosen_points,1], z[::chosen_points])

# ax.set_xlabel("x1")
# ax.set_ylabel("x2")
# ax.set_zlabel("f(x1, x2)")

# plt.show()

with torch.no_grad():
    y_guess = model(data_x1_x2)  
    y_guess_initial=model_initial(data_x1_x2)
y_guess_final = y_guess.detach().numpy()
y_guess_initial=y_guess_initial.detach().numpy()
# print(f"NN ouput: {y_guess_final}")

fig=plt.figure(num=3, dpi=120)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data_x1_x2[::chosen_points,0], data_x1_x2[::chosen_points,1], z[::chosen_points], label='model')
ax.scatter(data_x1_x2[::chosen_points,0], data_x1_x2[::chosen_points,1], y_guess_final[::chosen_points], label='trained NN')
ax.scatter(data_x1_x2[::chosen_points,0], data_x1_x2[::chosen_points,1], y_guess_initial[::chosen_points], label='initial NN')
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("f(x1, x2)")
plt.legend()
plt.show()
