# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 06:09:30 2026

@author: Ale
"""

import numpy as np
import matplotlib.pyplot as plt
import Handwriting_NN_code.utils as utils



mean_value=2
standard_deviation=1
x_start=mean_value-10
x_stop=mean_value+10
x_num_points=200
x=np.linspace(x_start,x_stop, x_num_points)

mean_value=2
standard_deviation=1
y=(1/(standard_deviation*np.sqrt(2*np.pi)))*np.exp(-((x-mean_value)**2)/(2*(standard_deviation**2)))

plt.close("all")
plt.figure(num=1, dpi=120)
plt.plot(x, y, linestyle="None", marker='.', label=f"mean: {mean_value:.2f}, dev: {standard_deviation:.2f}")
plt.xlabel("x")
plt.ylabel("Gaussian")
plt.legend()
plt.grid("True")
plt.show


z=utils.sigmoid(x)
plt.figure(num=2, dpi=120)
plt.plot(x, z, marker=".", linestyle="None", label="Sigmoid")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid("True")
plt.legend()
#%%

gigi=None

if gigi:
    print("si")
else:
    print("no")