# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 14:44:24 2026

@author: Ale
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset


class Neural_Network(nn.Module):
    
    def __init__(self, layers, activations):
        super().__init__()
        
        self.num_layers=len(layers)
        self.layers=nn.ModuleList()
        self.activations=nn.ModuleList(activations)
        # self.activations=activations
        for layer in range(self.num_layers-1):
            self.layers.append(nn.Linear(layers[layer],layers[layer+1]))
        
    def forward(self, x):
        for layer in range(len(self.layers)):
            x=self.activations[layer](self.layers[layer](x))
        
        return x
    
    def training_model(self, training_data, epochs, batch_size, learning_rate, loss_function):
        
        # loss_function=nn.MSELoss()
        optimizer=torch.optim.SGD(self.parameters(),lr=learning_rate)
        dataloader=DataLoader(training_data, batch_size=batch_size, shuffle=True)
        for epoch in range(epochs):
            for batch_x, batch_y in dataloader:
                
                optimizer.zero_grad()
                y_guess=self(batch_x)
                loss=loss_function(y_guess,batch_y)
                loss.backward()
                optimizer.step()
            
            if epoch%100==0:
                print(f"epoch:{epoch}, {loss.item():.04f}")
                # print(batch_x, batch_y)
                
        


def training_model_sequential(model, training_data, epochs, batch_size, learning_rate, loss_function, reg_param=0, test_data=None,
                              monitor_training_accuracy=False, monitor_training_cost=True, schedule_lr=False):
        
        dataloader=DataLoader(training_data, batch_size=batch_size, shuffle=True)
        optimizer=torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=reg_param)
        if schedule_lr:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                   mode='min',
                                                                   factor=0.5,
                                                                   patience=5,
                                                                   threshold=0.0001,
                                                                   eps=10**(-10))
            
        
        accuracy_evaluation=[]
        Cost_evaluation=[]
        accuracy_training=[]
        Cost_training=[]
        for epoch in range(epochs):
            train_one_epoch(model, dataloader, optimizer, loss_function)
            
            print(f"Epoch {epoch: 02d}")
            
            # if epoch%100==0:
            #     print(f"Epoch: {epoch}, Cost: {loss.item():.4f}")
            #     # print(batch_x, batch_y)
            if monitor_training_accuracy or monitor_training_cost:
                acc, cost=evaluate(model, training_data, loss_function, batch_size)
                accuracy_training.append(acc)
                Cost_training.append(cost)
                learning_rate = optimizer.param_groups[0]['lr']
                print(f"Learning rate: {learning_rate: .3e}")
                if schedule_lr:
                    scheduler.step(cost)
                    
                    
                # print(
                #     f"cost={cost:.10f}, "
                #     f"lr={optimizer.param_groups[0]['lr']:.3e}, "
                #     f"best={scheduler.best:.10f}, "
                #     f"num_bad_epochs={scheduler.num_bad_epochs}"
                # )
                
                if monitor_training_accuracy and monitor_training_cost:
                    print(f"Training Accuracy: {acc:.04f}")
                    print(f"Training Cost: {cost:.04f}")
                elif monitor_training_accuracy:
                    print(f"Training Accuracy: {acc:.04f}")
                else:
                    print(f"Training Cost: {cost:.04f}")
                
            if test_data:
                n_test=len(test_data)
                acc, cost=evaluate(model, test_data, loss_function, batch_size)
                accuracy_evaluation.append(acc)
                Cost_evaluation.append(cost)
                print(f"Test Accuracy: {acc:.04f}")
                print(f"Test Cost: {cost:.04f}\n")
            print("")
        
        return accuracy_training, Cost_training, accuracy_evaluation, Cost_evaluation

def train_one_epoch(model, dataloader, optimizer, loss_function):
    for batch_x, batch_y in dataloader:
        
        optimizer.zero_grad()
        y_guess=model(batch_x)
        loss=loss_function(y_guess,batch_y)
        loss.backward()
        optimizer.step()

def evaluate(model, dataset, loss_function, batch_size=100):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    correct = 0
    C_value = 0.0
    total=len(dataset)
    model.eval()

    with torch.no_grad():
        for x, y in dataloader:
            a = model(x)
            # Se y è one-hot (training)
            if y.ndim > 1:
                y_class = torch.argmax(y, dim=1)

            # Se y contiene direttamente la classe (test)
            else:
                y_class = y.long()
                # Trasformiamo temporaneamente y in one-hot
                # perché BCELoss vuole un vettore di 10 valori
                y_one_hot = torch.zeros(y.size(0),a.size(1))
                y_one_hot.scatter_(1, y.unsqueeze(1).long(), 1)
                y = y_one_hot
            
            
            C_value += loss_function(a, y).item() * x.size(0)
            y_guess = torch.argmax(a, dim=1)
            correct += (y_guess == y_class).sum().item()

    model.train()

    accuracy = correct / total
    C_value = C_value / total
    return accuracy, C_value
    



def from_list_to_Tensors(dataset, num_data=None):
    x_data, y_data= zip(*dataset[:num_data])
    x_data=torch.tensor(np.array(x_data), dtype=torch.float32)
    x_data=x_data.squeeze(-1)
    y_data=torch.tensor(np.array(y_data), dtype=torch.float32)
    y_data=y_data.squeeze(-1)
    new_dataset = TensorDataset(x_data, y_data)
    
    return new_dataset
