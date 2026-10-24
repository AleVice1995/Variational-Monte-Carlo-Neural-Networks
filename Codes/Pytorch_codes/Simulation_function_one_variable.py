# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 01:32:26 2026

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
#SIMULATION OF A FUNCTION
#Using nn.Sequential to generate a similar approach
start=time.time()
#DATA
x_arr=np.linspace(-10, 10, 50000, dtype=np.float32)
y_arr=np.exp(-x_arr**2)
x=torch.from_numpy(x_arr).reshape(-1, 1)
y=torch.from_numpy(y_arr).reshape(-1, 1)
dataset = TensorDataset(x, y)



#Construction of the NN
layers=[1,30, 30, 30, 1]
activations=[nn.ReLU(), nn.ReLU(), nn.ReLU(), nn.Identity()]

modules=[]
for i in range(len(layers)-1):
    modules.append(nn.Linear(layers[i],layers[i+1]))
    modules.append(activations[i])

# the * is used to pass the elements of the lists as separate items to nn.Sequential()
model=nn.Sequential(*modules)
model_initial=copy.deepcopy(model)



epochs=20
batch_size=50
learning_rate=1*10**-2
#reg_parameter is related to the L2 regularization
reg_parameter=0 #10/len(dataset)
loss_function=nn.MSELoss()#BCELoss()
data=training_model_sequential(model, dataset,
                               epochs=epochs, batch_size=batch_size, 
                               learning_rate=learning_rate, loss_function=loss_function,
                               reg_param= reg_parameter, schedule_lr=True)



end=time.time()
elapsed=end-start
hours, remainder=divmod(elapsed, 3600)
minutes, seconds=divmod(remainder, 60)


print(f"\nElapsed time: {hours:.0f} hh {minutes:.0f} mm {seconds:.2f} ss\n")
training_Cost=data[1]
#%%
plt.close('all')
iterations=[i for i in range(epochs)]
plt.figure(num=2, dpi=120)

plt.plot(iterations, training_Cost, linestyle='-', marker='.', label="Training")
plt.xlabel("Epochs")
plt.ylabel("Cost")
plt.grid(True)
plt.legend()
plt.show()


# x_arr=np.linspace(-10, 10, 1000, dtype=np.float32)
# y_arr=np.exp(x_arr)#x_arr**3

x=torch.from_numpy(x_arr).reshape(-1, 1)
with torch.no_grad():
    y_guess = model(x)  
    y_guess_initial=model_initial(x)
y_guess_final = y_guess.detach().numpy()
y_guess_initial=y_guess_initial.detach().numpy()
# print(f"NN ouput: {y_guess_final}")



plt.figure(num=3, dpi=120)

plt.plot(x_arr, y_arr, linestyle='-', marker='None', label='correct')
plt.plot(x_arr, y_guess_initial, linestyle='-', marker='None', label='initial NN')
plt.plot(x_arr, y_guess_final, linestyle='-', marker='None', label='trained NN')
plt.grid('True')
plt.legend()
plt.show()
