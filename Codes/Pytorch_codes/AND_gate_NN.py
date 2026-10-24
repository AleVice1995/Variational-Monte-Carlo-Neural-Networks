# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 14:53:17 2026

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
#SIMULATION OF "AND" GATE
#Implementazione NN tramite nn.Module of pytorch


x=torch.tensor([
    [0.0,0.0],
    [0.0,1.0],
    [1.0,0.0],
    [1.0,1.0],])

y=torch.tensor([
    [0.0],
    [0.0],
    [0.0],
    [1.0],])

dataset = TensorDataset(x, y)
# print(dataset.tensors)


layers=[2,16,16,1]
activations=[nn.Sigmoid(), nn.Sigmoid(), nn.Sigmoid()]
model=NN(layers, activations)

y_guess_initial=model(x).detach().numpy()
model.training_model(dataset, 10000, 2, 0.5, nn.MSELoss())
y_guess_final=model(x).detach().numpy()  


print(f"NN ouput: {y_guess_initial}")
print(f"NN ouput: {y_guess_final}")





