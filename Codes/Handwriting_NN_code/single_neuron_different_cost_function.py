# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 04:46:24 2026

@author: Ale
"""

import numpy as np
import Handwriting_NN_code.neural_network as nn
import Handwriting_NN_code.utils as utils


net1_neuron=nn.Network(([1,1]))

net1_neuron.biases[0]=np.array(([0.9],))
net1_neuron.weights[0]=np.array(([0.6],))

training_data=[(np.array(([1.0,],)), np.array(([0.0,],)))]

net1_neuron.SGD(training_data,
                300, 1, 0.15, 
                act_func=utils.sigmoid, cost_fun=utils.Cost_quadratic,
                print_values=True, num_training_file_name=1)

print(net1_neuron.biases[0])
print(net1_neuron.weights[0])

#%%

net2_neuron=nn.Network(([1,1]))

net2_neuron.biases[0]=np.array(([2.0],))
net2_neuron.weights[0]=np.array(([2.0],))

training_data=[(np.array(([1.0,],)), np.array(([0.0,],)))]

net2_neuron.SGD(training_data,
                300, 1, 0.15, 
                act_func=utils.sigmoid, cost_fun=utils.Cost_quadratic,
                print_values=True, num_training_file_name=2)

print(net2_neuron.biases[0])
print(net2_neuron.weights[0])

#%%
net3_neuron=nn.Network(([1,1]))

net3_neuron.biases[0]=np.array(([0.9],))
net3_neuron.weights[0]=np.array(([0.6],))

training_data=[(np.array(([1.0,],)), np.array(([0.0,],)))]

net3_neuron.SGD(training_data,
                300, 1, 0.05, 
                act_func=utils.sigmoid, cost_fun=utils.Cost_cross_entropy,
                print_values=True, num_training_file_name=3)

print(net3_neuron.biases[0])
print(net3_neuron.weights[0])

#%%
net4_neuron=nn.Network(([1,1]))

net4_neuron.biases[0]=np.array(([2.0],))
net4_neuron.weights[0]=np.array(([2.0],))

training_data=[(np.array(([1.0,],)), np.array(([0.0,],)))]

net4_neuron.SGD(training_data,
                300, 1, 0.05, 
                act_func=utils.sigmoid, cost_fun=utils.Cost_cross_entropy,
                print_values=True, num_training_file_name=4)

print(net4_neuron.biases[0])
print(net4_neuron.weights[0])


