# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 10:51:46 2026

@author: Ale
"""

import numpy as np
import Handwriting_NN_code.utils as utils
from Handwriting_NN_code.neural_networks_and_deep_learning.src import mnist_loader
import time
import Handwriting_NN_code.neural_network2 as nn2
import matplotlib.pyplot as plt



training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)
validation_data = list(validation_data)
test_data = list(test_data)

start=time.time()
net_new=nn2.Network([784,30, 10], cost=nn2.CrossEntropyCost)

epochs=30
batch_size=10
learning_parameter_eta=0.5
lmbda=5
data=net_new.SGD(training_data, epochs, batch_size, learning_parameter_eta, lmbda ,
                 evaluation_data=validation_data, 
                 monitor_training=True,
                 monitor_evaluation=True)






end=time.time()
t_sec=end-start
t_min=t_sec/60
t_hour=(np.floor(t_min))/60

ss=(t_min-np.floor(t_min))*60
mm=(t_hour-np.floor(t_hour))*60
hh=np.floor(t_hour)

print(f"\nElapsed time: {hh:.0f} hh {mm:.0f} mm {ss:.2f} ss")

#%%
evaluation_cost=data[0]
evaluation_acc=data[1]
training_cost=data[2]
training_acc=data[3]
epochs_vec=np.arange(0, epochs, 1)

plt.close('all')
plt.figure(num=1, dpi=120)
plt.plot(epochs_vec, evaluation_cost, marker='None', label="Cost on validation data")
plt.plot(epochs_vec, training_cost, marker='None', label="Cost on training data")
plt.xlabel("Epochs")
plt.ylabel("Cost")
plt.grid(True)
plt.legend()

plt.figure(num=2, dpi=120)
plt.plot(epochs_vec, evaluation_acc, marker='None', label="Accuracy on validation data")
plt.plot(epochs_vec, training_acc, marker='None', label="Accuracy on training data")
plt.xlabel("Epochs")
plt.ylabel("%")
plt.ylim((0.9,1))
plt.grid(True)
plt.legend()

#%%
max_acc_training=training_acc[np.argmax(training_acc)]
max_acc_evaluation=evaluation_acc[np.argmax(evaluation_acc)]
print(f"Max Accuracy on training {max_acc_training:.4f}")
print(f"Max Accuracy on evaluation {max_acc_evaluation:.4f}")







