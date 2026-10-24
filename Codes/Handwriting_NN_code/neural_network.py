# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 05:10:07 2026

@author: Ale
"""

#neural_network.py

import numpy as np
import Handwriting_NN_code.utils as utils
import random

class Network():
    def __init__(self, sizes):
        self.num_layers=len(sizes)
        self.sizes=sizes
        
        #Uses list comprehension (a little bit faster and easy to read)
        self.biases=[np.random.randn(neur,1) for neur in sizes[1:]]
        #loop
        # self.biases=[]
        # for i in sizes[1:]:
        #     self.biases.append(np.random.randn(i, 1))
        
        
        #Uses list comprehension (a little bit faster and easy to read)
        self.weights=[
            np.random.randn(neur,inp)
            for inp,neur in zip(sizes[:-1], sizes[1:])]
        #loop
        # self.weights=[]
        # for inp,neur in zip(sizes[:-1], sizes[1:]):
        #     self.weights.append(np.random.randn(neur,inp))
        
    def feedforward(self, a, act_func):
        # x=np.array([a,]).T
        x=a
        for i in range(self.num_layers-1):
            w=self.weights[i]
            b=self.biases[i]
            # a=utils.sigmoid((w@x)+b)
            a=act_func((w@x)+b)
            x=a
        return a
    
    def evaluate(self, test_data, act_func):
        n_test=len(test_data)
        cont=0
        for i in range(n_test):
            x, y=test_data[i]
            y_guess=np.argmax(self.feedforward(x, act_func))
            if y_guess==y:
                cont+=1
        return cont
    
    def SGD(self,training_data, epochs, mini_batch_size, eta, act_func, cost_fun,
            test_data=None, print_values=False, num_training_file_name=None):
        if print_values:
            file=open("../Networks and Deep Learning/Documents/training"+ str(num_training_file_name) +".csv", "w") 
            file.write("w,b,a,C\n")
            i=0
        num_trainin=len(training_data)
        for i in range(epochs):
            random.shuffle(training_data)
            mini_batches=[training_data[k:k+mini_batch_size]
                          for k in range(0, num_trainin, mini_batch_size)]
            for mini_batch in mini_batches:
                if print_values:
                    i+=1
                    if (i%1)==0:
                        x,y=mini_batch[0]
                        guess=self.feedforward(x, act_func)
                        C=cost_fun(y,guess)
                        file.write(
                            f"{self.weights[0].item():+.2f},"
                            f"{self.biases[0].item():+.2f},"
                            f"{guess.item():.2f},"
                            f" {C:f}\n")
                self.update_mini_batch2(mini_batch, eta, act_func, cost_fun)
            if test_data:
                n_test=len(test_data)
                print(f"Epoch {i:02d}: {self.evaluate(test_data, act_func)}/ {n_test}")
            else:
                print(f"Epoch {i:02d} complete")
        if print_values:
            file.close()
            
    # 'Single example approach'  
    def update_mini_batch(self, mini_batch, eta, act_func):
        var_biases=[np.zeros(b.shape) for b in self.biases]
        var_weights=[np.zeros(w.shape) for w in self.weights]
        for x,y in mini_batch:
            der_C_b, der_C_w=self.backprop(x,y, act_func)
            var_biases=[vb+db for vb,db in zip(var_biases, der_C_b)]
            var_weights=[vw+dw for vw,dw in zip(var_weights, der_C_w)]
        self.biases=[(b-(eta*vb/len(mini_batch))) for b, vb in zip(self.biases, var_biases)]
        self.weights=[(w-(eta*vw/len(mini_batch))) for w, vw in zip(self.weights, var_weights)]
        
        
    def backprop(self, x, y, act_func):
        if act_func==utils.sigmoid:
            act_prime=utils.sigmoid_prime
        else:
            act_prime=utils.ReLU_prime
            
        der_C_b=[np.zeros(b.shape) for b in self.biases]
        der_C_w=[np.zeros(w.shape) for w in self.weights]
        z_input=[]
        a_act=[x]
        # y=np.array([y]).T
        for i in range(self.num_layers-1):
            w=self.weights[i]
            b=self.biases[i]
            z_input.append(w@a_act[i]+b)
            # a_act.append(utils.sigmoid(z_input[i]))
            a_act.append(act_func(z_input[i]))
            
        for j in range(self.num_layers-1):
            i=(self.num_layers-2)-j
            if i==self.num_layers-2:
                # der_C_b[i]=utils.sigmoid_prime(z_input[i])*(a_act[-1]-y)
                der_C_b[i]=act_prime(z_input[i])*(a_act[-1]-y)
            else:
                
                # der_C_b[i]=(self.weights[i+1].T@der_C_b[i+1])*utils.sigmoid_prime(z_input[i])
                der_C_b[i]=(self.weights[i+1].T@der_C_b[i+1])*act_prime(z_input[i])
            der_C_w[i]=der_C_b[i]@a_act[i].T
        
        return der_C_b, der_C_w
    
    # 'Fully matrix-based approach'
    def update_mini_batch2(self, mini_batch, eta, act_func, cost_fun):
        # var_biases=[np.zeros(b.shape) for b in self.biases]
        # var_weights=[np.zeros(w.shape) for w in self.weights]
        X_mat=np.hstack([x for x,_ in mini_batch])
        Y_mat=np.hstack([y for _,y in mini_batch])
    
        var_biases, var_weights=self.backprop2(X_mat,Y_mat, act_func, cost_fun)
        self.biases=[(b-(eta*vb/len(mini_batch))) for b, vb in zip(self.biases, var_biases)]
        self.weights=[(w-(eta*vw/len(mini_batch))) for w, vw in zip(self.weights, var_weights)]
                
        
    def backprop2(self, x, y, act_func, cost_fun):
        if act_func==utils.sigmoid:
            act_prime=utils.sigmoid_prime
        else:
            act_prime=utils.ReLU_prime
        
        
        der_C_b=[np.zeros(b.shape) for b in self.biases]
        der_C_w=[np.zeros(w.shape) for w in self.weights]
        z_input=[]
        a_act=[x]
        # y=np.array([y]).T
        'FORWARD PROPAGATION'
        for i in range(self.num_layers-1):
            w=self.weights[i]
            b=self.biases[i]
            z_input.append(w@a_act[i]+b)
            # a_act.append(utils.sigmoid(z_input[i]))
            a_act.append(act_func(z_input[i]))
        'BACKWARD PROPAGATION'
        for j in range(self.num_layers-1):
            i=(self.num_layers-2)-j
            if i==self.num_layers-2:
                if cost_fun is utils.Cost_quadratic:
                    # temp_mat=utils.sigmoid_prime(z_input[i])*(a_act[-1]-y)
                    temp_mat=act_prime(z_input[i])*(a_act[-1]-y)
                elif cost_fun is utils.Cost_cross_entropy:
                    temp_mat=(a_act[-1]-y)
            else:
                # temp_mat=(self.weights[i+1].T@temp_mat)*utils.sigmoid_prime(z_input[i])
                temp_mat=(self.weights[i+1].T@temp_mat)*act_prime(z_input[i])
            der_C_b[i]=np.sum(temp_mat, axis=1, keepdims=True)
            der_C_w[i]=temp_mat@a_act[i].T
        
        return der_C_b, der_C_w
        
        