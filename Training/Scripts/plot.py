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
from PIL import Image
import sys
import os
import pandas as pd

sys.path.append(r'../../Data/')
from dataloader import Data
from Training import *

def plotLosses(model_save_path):
    
    # Find csv
    files = os.listdir(model_save_path)
    csv_file = [file for file in files if ".csv" in file][0] 
    loss = pd.read_csv(os.path.join(model_save_path, csv_file))
    #loss = loss[::int(len(loss)/20)]
    
    plt.title("Training Loss Curve")
    plt.plot(loss, label="Train")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()
    return loss
    
def plotAccuracy(model_save_path, val_dataset, train_dataset):
    
    all_files = [file for file in os.listdir(model_save_path)]
    models = [file for file in all_files if ".csv" not in file]
    
    binaries = [file for file in all_files if "Binary" in file]
    if binaries:
        accFunc = getAccuracyBinaryClass
    else:
        accFunc = getAccuracyMultiClass

    t_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64)
    v_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64)
    epochs = dict(zip([i for i in range(len(models))], [[] for i in range(len(models))]))
    i = 0
    
    for model_params,epoch in zip(models, epochs):
#         print("Epoch", i+1)
        # load model
        model = torch.load(os.path.join(model_save_path, model_params)).cuda()
        # get acc for model and append
        epochs[epoch].append(accFunc(model, t_data_loader))
        epochs[epoch].append(accFunc(model, v_data_loader))
        i+=1
    
    # Plot    
    
    sorted_epochs = {i:epochs[i] for i in sorted(epochs)}
    
    plt.title("Accuracy Curves")
    plt.plot([i for i in sorted_epochs], [sorted_epochs[i][0] for i in sorted_epochs], label="Train")
    plt.plot([i for i in sorted_epochs], [sorted_epochs[i][1] for i in sorted_epochs], label="Validation")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.show() 
    
    return epochs