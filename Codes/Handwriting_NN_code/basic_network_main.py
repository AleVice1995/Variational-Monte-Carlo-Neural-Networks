# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 02:54:57 2026

@author: Ale
"""

#main.py

import numpy as np
import Handwriting_NN_code.neural_network as nn
import Handwriting_NN_code.utils as utils
from Handwriting_NN_code.neural_networks_and_deep_learning.src import mnist_loader
import time
# %%
start=time.time()

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)
validation_data = list(validation_data)
test_data = list(test_data)

net=nn.Network([784, 30, 10])
#%%
batch_size=10

net.SGD(training_data, 30, 10, 3.09,
        act_func=utils.sigmoid, cost_fun=utils.Cost_quadratic,
        test_data=test_data, print_values=False, num_training_file_name=None)

end=time.time()
t_sec=end-start
t_min=t_sec/60
t_hour=(np.floor(t_min))/60

ss=(t_min-np.floor(t_min))*60
mm=(t_hour-np.floor(t_hour))*60
hh=np.floor(t_hour)

print(f"\nElapsed time: {hh:.0f} hh {mm:.0f} mm {ss:.2f} ss")

