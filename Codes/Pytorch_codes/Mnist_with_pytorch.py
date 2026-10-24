# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 18:39:29 2026

@author: Ale
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from Handwriting_NN_code.neural_networks_and_deep_learning.src import mnist_loader
from Pytorch_codes.NN_classes_with_pytorch import training_model_sequential
from Pytorch_codes.NN_classes_with_pytorch import from_list_to_Tensors
import time
import matplotlib.pyplot as plt
#%%
#SOLVING MNIST PROBLEM
start=time.time()

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)
validation_data = list(validation_data)
test_data = list(test_data)

training_dataset=from_list_to_Tensors(training_data)
validation_dataset=from_list_to_Tensors(validation_data)
test_dataset=from_list_to_Tensors(test_data)

#%%
layers=[784,100,10]
activations=[nn.Sigmoid(), nn.Sigmoid()]

modules=[]
for i in range(len(layers)-1):
    modules.append(nn.Linear(layers[i],layers[i+1]))
    modules.append(activations[i])

# the * is used to pass the elements of the lists as separate items to nn.Sequential()
model=nn.Sequential(*modules)

epochs=30
batch_size=50
learnin_rate=3
loss_function=nn.MSELoss()
data=training_model_sequential(model, training_dataset,
                          epochs=epochs, batch_size=batch_size, 
                          learning_rate=learnin_rate, loss_function=loss_function,
                          test_data=test_dataset, 
                          monitor_training_accuracy=True, monitor_training_cost=True)
end=time.time()
elapsed=end-start
hours, remainder=divmod(elapsed, 3600)
minutes, seconds=divmod(remainder, 60)

print(f"\nElapsed time: {hours:.0f} hh {minutes:.0f} mm {seconds:.2f} ss\n")


training_accuracy=data[0]
training_Cost=data[1]
test_accuracy=data[2]
test_Cost=data[3]


max_acc_training=training_accuracy[np.argmax(training_accuracy)]
print(f"Max Accuracy on evaluation {max_acc_training:.4f}")

max_acc_test=test_accuracy[np.argmax(test_accuracy)]
print(f"Max Accuracy on evaluation {max_acc_test:.4f}")
#%%
plt.close('all')
iterations=[i for i in range(epochs)]
plt.figure(num=1, dpi=120)

plt.plot(iterations, test_accuracy, linestyle='-', marker='.', label="Test")
plt.plot(iterations, training_accuracy, linestyle='-', marker='.', label="Training")
plt.xlabel("Epochs")
plt.ylabel("Accuracy [%]")
plt.ylim((0.8,1))
plt.grid(True)
plt.legend()
plt.show()


plt.figure(num=2, dpi=120)

plt.plot(iterations, test_Cost, linestyle='-', marker='.', label="Test")
plt.plot(iterations, training_Cost, linestyle='-', marker='.', label="Training")
plt.xlabel("Epochs")
plt.ylabel("Cost")
plt.grid(True)
plt.legend()
plt.show()

#%%
#Simulation MNIST problemwith weight inizialization, L2 regularization,
#Cross-entropy lss function
start=time.time()

training_dataset=from_list_to_Tensors(training_data)
validation_dataset=from_list_to_Tensors(validation_data)
test_dataset=from_list_to_Tensors(test_data)


layers=[784,30,10]
activations=[nn.Sigmoid(), nn.Sigmoid()]

modules=[]
for i in range(len(layers)-1):
    modules.append(nn.Linear(layers[i],layers[i+1]))
    modules.append(activations[i])

# the * is used to pass the elements of the lists as separate items to nn.Sequential()
model=nn.Sequential(*modules)

#new weight initialization
def init_weights_biases(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 1/np.sqrt(m.in_features))
        nn.init.normal_(m.bias, 0, 1)        
model.apply(init_weights_biases)

epochs=30
batch_size=10
learnin_rate=3
#reg_parameter is related to the L2 regularization
reg_parameter=0#5/len(training_dataset)
loss_function=nn.MSELoss()#BCELoss()
data=training_model_sequential(model, training_dataset,
                               epochs=epochs, batch_size=batch_size, 
                               learning_rate=learnin_rate, loss_function=loss_function,
                               reg_param= reg_parameter, test_data=test_dataset, 
                               monitor_training_accuracy=True, monitor_training_cost=True)


end=time.time()
elapsed=end-start
hours, remainder=divmod(elapsed, 3600)
minutes, seconds=divmod(remainder, 60)


print(f"\nElapsed time: {hours:.0f} hh {minutes:.0f} mm {seconds:.2f} ss\n")


training_accuracy=data[0]
training_Cost=data[1]
test_accuracy=data[2]
test_Cost=data[3]


max_acc_training=training_accuracy[np.argmax(training_accuracy)]
print(f"Max Accuracy on training {max_acc_training:.4f}")

max_acc_test=test_accuracy[np.argmax(test_accuracy)]
print(f"Max Accuracy on test {max_acc_test:.4f}")


#%%
plt.close('all')
iterations=[i for i in range(epochs)]
plt.figure(num=3, dpi=120)

plt.plot(iterations, test_accuracy, linestyle='-', marker='.', label="Test")
plt.plot(iterations, training_accuracy, linestyle='-', marker='.', label="Training")
plt.xlabel("Epochs")
plt.ylabel("Accuracy [%]")
plt.ylim((0.8,1))
plt.grid(True)
plt.legend()
plt.show()


plt.figure(num=4, dpi=120)

plt.plot(iterations, test_Cost, linestyle='-', marker='.', label="Test")
plt.plot(iterations, training_Cost, linestyle='-', marker='.', label="Training")
plt.xlabel("Epochs")
plt.ylabel("Cost")
plt.grid(True)
plt.legend()
plt.show()









