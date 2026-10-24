# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 10:43:04 2026

@author: Ale
"""

import numpy as np
import Handwriting_NN_code.utils as utils
import random


class Sigmoid():
    
    @staticmethod
    def activation(z):
        return 1.0/(1.0+np.exp(-z))
    @staticmethod
    def derivative(z):
        a=Sigmoid.activation(z)
        return a*(1-a)

class ReLU():
    
    @staticmethod
    def activation(z):
        return np.maximum(0,z)
    @staticmethod
    def derivative(z):
        bool_vect=z>0
        return bool_vect.astype(int)  
    
    
class CrossEntropyCost():
    
    @staticmethod
    def fn(a,y):
        term1=y*np.log(a)
        term2=(1-y)*np.log((1-a))
        return np.sum(np.nan_to_num(-(term1+term2)))
    
    @staticmethod
    def delta(z,a,y):
        return (a-y)
    


class QuadraticCost():
    
    @staticmethod
    def fn(a,y):
        diff=(y-a)**2
        return 0.5*(np.sum(diff))
    
    @staticmethod
    def delta(z,a,y):
        return ((a-y)*Sigmoid.derivative(z))
    
    
class Network():
    def __init__(self, sizes, cost=CrossEntropyCost, act_and_der=Sigmoid):
        self.num_layers=len(sizes)
        self.sizes=sizes
        self.default_weight_initializer()
        self.cost=cost
        self.act_and_der=act_and_der
        
    
    def default_weight_initializer(self):
        self.biases=[np.random.randn(neur,1) for neur in self.sizes[1:]]
        self.weights=[
            np.random.normal(0, 1/np.sqrt(inp), (neur, inp))
            for inp,neur in zip(self.sizes[:-1], self.sizes[1:])]
        
    def large_weight_initializer(self):
        self.biases=[np.random.randn(neur,1) for neur in self.sizes[1:]]
        self.weights=[
            np.random.randn(neur,inp)
            for inp,neur in zip(self.sizes[:-1], self.sizes[1:])]
        
    def feedforward(self, a):
         for w, b in zip(self.weights, self.biases):
             a=self.act_and_der.activation((w@a)+b)
         return a
    
    def evaluate(self, test_data, lmbda):
        n_test=len(test_data)
        cont=0
        C_value=0.0
        for i in range(n_test):
            x_vect, output_y=test_data[i]
            if np.size(output_y)==1:
                y=output_y
                y_vect=np.zeros((self.sizes[-1],1))
                y_vect[y]=1
            else:
                y_vect=output_y
                y=np.argmax(y_vect)
                
            a=self.feedforward(x_vect)
            C_value+=self.cost.fn(a, y_vect)
            y_guess=np.argmax(a)
            if y_guess==y:
                cont+=1
        acc=cont/n_test
        C_value=C_value/n_test
        weight_total=sum(np.sum(array**2) for array in self.weights)
        C_value+=(lmbda*weight_total*0.5/n_test)
        return acc, C_value
        

    
    def SGD(self, training_data, epochs, mini_batch_size, eta, lmbda = 0.0,
            evaluation_data=None,
            monitor_evaluation=False,
            monitor_training=False):
        
        num_training=len(training_data)
        evaluation_cost_vec=[]
        evaluation_accuracy_vec=[]
        training_cost_vec=[]
        training_accuracy_vec=[]
        for i in range(epochs):
            random.shuffle(training_data)
            mini_batches=[training_data[k:k+mini_batch_size]
                  for k in range(0, num_training, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda, num_training)
            print(f"Epoch {i:d} completed:")
            if monitor_training:
                acc, cost=self.evaluate(training_data, lmbda)
                training_accuracy_vec.append(acc)
                training_cost_vec.append(cost)
                print(f"Training Accuracy: {acc:.3f}")
                print(f"Training Cost: {cost:.3f}")
            if monitor_evaluation:                 
                acc, cost=self.evaluate(evaluation_data, lmbda)
                evaluation_accuracy_vec.append(acc)
                evaluation_cost_vec.append(cost)
                print(f"Evaluation Accuracy: {acc:.3f}")
                print(f"Evaluation Cost: {cost:.3f}")
            print("")
        return evaluation_cost_vec, evaluation_accuracy_vec,\
                    training_cost_vec,  training_accuracy_vec
        
    
    def update_mini_batch(self, mini_batch, eta, lmbda, num_training):
        # var_biases=[np.zeros(b.shape) for b in self.biases]
        # var_weights=[np.zeros(w.shape) for w in self.weights]
        X_mat=np.hstack([x for x,_ in mini_batch])
        Y_mat=np.hstack([y for _,y in mini_batch])
    
        var_biases, var_weights=self.backprop(X_mat,Y_mat)
        self.biases=[(b-(eta*vb/len(mini_batch))) for b, vb in zip(self.biases, var_biases)]
        self.weights=[(w*(1-eta*lmbda/num_training)-(eta*vw/len(mini_batch))) for w, vw in zip(self.weights, var_weights)]
        
        
    def backprop(self, x, y):
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
            a_act.append(self.act_and_der.activation(z_input[i]))
        'BACKWARD PROPAGATION'
        for j in range(self.num_layers-1):
            i=(self.num_layers-2)-j
            if i==self.num_layers-2:
                temp_mat=self.cost.delta(z_input[i], a_act[-1], y)
            else:
                # temp_mat=(self.weights[i+1].T@temp_mat)*utils.sigmoid_prime(z_input[i])
                temp_mat=(self.weights[i+1].T@temp_mat)*self.act_and_der.derivative(z_input[i])
            der_C_b[i]=np.sum(temp_mat, axis=1, keepdims=True)
            der_C_w[i]=temp_mat@a_act[i].T
        
        return der_C_b, der_C_w 
     

                   
        
        
