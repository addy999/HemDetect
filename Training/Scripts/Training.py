import torch.nn as nn
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
import sys
import os
import pandas as pd

model_save_dir = "../Models/"

def getAccuracyMultiClass(model, data_loader):

    #cor = 0
    #total = 0
    #n = 0
    #for imgs, labels in data_loader:

        #To Enable GPU Usage
        #imgs = imgs.cuda()
        #labels = labels.cuda()
        #############################################
        
        #output = model(imgs)
        #pred = output.max(1, keepdim=True)[1]
        #cor = cor + pred.eq(labels.view_as(pred)).sum().item()
        #total = total + imgs.shape[0]
        #n = n+1
    #return cor / total

    correct = 0
    total = 0
    
    for imgs, labels in data_loader:
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            labels = labels.cuda()
        output = model(imgs)
        #select index with maximum prediction score
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += imgs.shape[0]
    return correct / total

def getAccuracyBinaryClass(model, data_loader):
    
    total_train_err = 0
    n = 0
    for imgs, labels in data_loader:
        #To Enable GPU Usage
        imgs = imgs.cuda()
        labels = labels.cuda()
        #############################################
        output = model(imgs)
        corr = (output > 0.0).squeeze().long() != labels
        total_train_err += int(corr.sum()) / len(imgs)
        n += 1
    
    return 1 - total_train_err/n 

def train(model, train_dataset, val_dataset, batch_size = 64, learning_rate=0.01, num_epochs=20, optim_param="sgd"):
    # Figure out if binary    
    if len(train_dataset.label_dict) > 2:
        criterion = nn.CrossEntropyLoss()
        binary = False
    else:
        criterion = nn.BCEWithLogitsLoss()
        binary = True

    if optim_param == "sgd":    
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0)
    elif optim_param == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
    iters, losses, train_acc, val_acc = [], [], [], []
    
    if not os.path.exists(model_save_dir + model.name):
        os.mkdir(model_save_dir + model.name)    
    
    training_loader = torch.utils.data.DataLoader(train_dataset, batch_size= batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size= batch_size, shuffle=True) 

    start_time = time.time()
    for epoch in range(num_epochs):
        gc.collect()
        count = 0
        for imgs, labels in iter(training_loader):
            #torch.cuda.empty_cache() 
            #To Enable GPU Usage
            imgs = imgs.cuda()
            labels = labels.cuda()
            #############################################
            
            # outputs = model(imgs.reshape(-1)) #for baseline
            outputs = model(imgs)
            if binary:
                loss = criterion(outputs, labels.float()) 
            else:
                loss = criterion(outputs, labels) 
            loss.backward()               
            optimizer.step()              
            optimizer.zero_grad()         
            #print(loss)
            losses.append(float(loss)/batch_size)            
            count = count+1
            #if count%100==0:
                #print(float(loss)/batch_size)
            
        print("Epoch", epoch, "Loss", losses[-1])
        
        # Save the current model (checkpoint) to a file
        model_path = model_save_dir + model.name + "/{}_epoch.pt".format(epoch)
        torch.save(model, model_path)
          
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Time Elapsed", elapsed_time)
    
    # Write the train/test loss/err into CSV file for plotting later
    if binary:
        string = "Binary"
    else:
        string = "Multi"
        
    np.savetxt("{0}/Train_loss_timeElapsed_{1}_s_{2}.csv".format(model_save_dir + model.name, elapsed_time, string), losses)
    print("Saved losses on disk.")
    
    #################### Final acc #############################
    print("Compute final accuracies.")
    if binary:
        acc_func = getAccuracyBinaryClass
    else:
        acc_func = getAccuracyMultiClass
    
    print("> Train acc")
    train_acc = acc_func(model, training_loader)
    print("> Val acc")
    val_acc = acc_func(model, val_loader)
    
    print("Train acc: {0}, Val acc: {1}".format(train_acc, val_acc))
    
    return train_acc, val_acc
