# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 16:58:46 2026

@author: Ale
"""

import Handwriting_NN_code.utils as utils
from scipy.special import softmax
import numpy as np

gigi=np.array(([2,4,5,6]))

piero=softmax(gigi)

file=open("../Networks and Deep Learning/Documents/data_softmax.csv", "w")
file.write("z1,z2,z3,z4,a1,a2,a3,a4\n")

z1=2.5
z2=-1
z3=3.2

z4_min=-5
z4_max=5
incr=0.1
iterations=int((z4_max-z4_min)/incr)

for i in range(iterations):
    z4=z4_min+i*incr
    z_vect=np.array(([z1,z2,z3,z4]))
    a_vect=softmax(z_vect)
    file.write(f"{z_vect[0]:.1f},"
               f"{z_vect[1]:.1f},"
               f"{z_vect[2]:.1f},"
               f"{z_vect[3]:.1f},"
               f"{a_vect[0]:.3f},"
               f"{a_vect[1]:.3f},"
               f"{a_vect[2]:.3f},"
               f"{a_vect[3]:.3f}\n"
               )
    
z4=5.0
while z4>-1.9:
    z4=round((z4-incr),2)
    z_vect=np.array(([z1,z2,z3,z4]))
    a_vect=softmax(z_vect)
    file.write(f"{z_vect[0]:.1f},"
               f"{z_vect[1]:.1f},"
               f"{z_vect[2]:.1f},"
               f"{z_vect[3]:.1f},"
               f"{a_vect[0]:.3f},"
               f"{a_vect[1]:.3f},"
               f"{a_vect[2]:.3f},"
               f"{a_vect[3]:.3f}\n"
               )
    
while z3>-2.5:
    z3=round((z3-incr),2)
    z_vect=np.array(([z1,z2,z3,z4]))
    a_vect=softmax(z_vect)
    file.write(f"{z_vect[0]:.1f},"
               f"{z_vect[1]:.1f},"
               f"{z_vect[2]:.1f},"
               f"{z_vect[3]:.1f},"
               f"{a_vect[0]:.3f},"
               f"{a_vect[1]:.3f},"
               f"{a_vect[2]:.3f},"
               f"{a_vect[3]:.3f}\n"
               )    


while z2<5:
    z2=round((z2+incr),2)
    z_vect=np.array(([z1,z2,z3,z4]))
    a_vect=softmax(z_vect)
    file.write(f"{z_vect[0]:.1f},"
               f"{z_vect[1]:.1f},"
               f"{z_vect[2]:.1f},"
               f"{z_vect[3]:.1f},"
               f"{a_vect[0]:.3f},"
               f"{a_vect[1]:.3f},"
               f"{a_vect[2]:.3f},"
               f"{a_vect[3]:.3f}\n"
               ) 


while z1<5:
    z1=round((z1+incr),2)
    z_vect=np.array(([z1,z2,z3,z4]))
    a_vect=softmax(z_vect)
    file.write(f"{z_vect[0]:.1f},"
               f"{z_vect[1]:.1f},"
               f"{z_vect[2]:.1f},"
               f"{z_vect[3]:.1f},"
               f"{a_vect[0]:.3f},"
               f"{a_vect[1]:.3f},"
               f"{a_vect[2]:.3f},"
               f"{a_vect[3]:.3f}\n"
               ) 
file.close()