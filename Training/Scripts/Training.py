#!/usr/bin/env python
# coding: utf-8

import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import gc 
import os

model_save_dir = "Models/"

def get_accuracy(model, data_loader, use_cuda):

    cor = 0
    total = 0
    n = 0
    for imgs, labels in data_loader:

        #To Enable GPU Usage
        if use_cuda and torch.cuda.is_available():
            imgs = imgs.cuda()
            labels = labels.cuda()
        #############################################
        
        output = model(imgs)
        pred = output.max(1, keepdim=True)[1]
        cor = cor + pred.eq(labels.view_as(pred)).sum().item()
        total = total + imgs.shape[0]
        n = n+1
    return cor / total

def train(model, train_dataset, batch_size = 64, learning_rate=0.01, num_epochs=20, use_cuda = False):
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    iters, losses, train_acc, val_acc = [], [], [], []
    
    if not os.path.exists(model_save_dir + model.name):
        os.mkdir(model_save_dir + model.name)    
    
    training_loader = torch.utils.data.DataLoader(train_dataset, batch_size= batch_size)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size= 32)
    start_time = time.time()
    for epoch in range(num_epochs):
        gc.collect()
        count = 0
        for imgs, labels in iter(training_loader):

            #To Enable GPU Usage
            if use_cuda and torch.cuda.is_available():
                imgs = imgs.cuda()
                labels = labels.cuda()
            #############################################

            outputs = model(imgs)             
            loss = criterion(outputs, labels) 
            loss.backward()               
            optimizer.step()              
            optimizer.zero_grad()         

            # save the current training information
            iters.append(n)
            losses.append(float(loss)/batch_size)            
            count = count+1
            # print(epoch, count)
            
        print("Epoch", epoch, "Loss", loss)
        
        # Save the current model (checkpoint) to a file
        model_path = model_save_dir + model.name + "/{0}_bs_{1}_lr_{2}_epoch.pt".format(
                                                   batch_size,
                                                   learning_rate,
                                                   epoch)
        torch.save(model, model_path)
          

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Time Elapsed", elapsed_time)
    
    # Write the train/test loss/err into CSV file for plotting later
    np.savetxt("{0}/Train_loss_timeElapsed_{1}.csv".format(model_save_dir + model.name, elapsed_time), losses)