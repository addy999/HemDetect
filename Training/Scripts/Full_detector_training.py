#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
import os
import sys
sys.path.append(r'../Data/')
from dataloader import Data

# ## **Hemorrhage Classifier:**

# ### Alexnet classifier

# In[3]:


import torch.nn as nn
import torchvision.models

alexnet_model = torchvision.models.alexnet(pretrained=True)
alexnet_model.features[0] = nn.Conv2d(1, 64, kernel_size= 7, stride= 2, padding= 3)

class HemorrhageClassifier(nn.Module):
    def __init__(self):
        super(HemorrhageClassifier, self).__init__()
        self.name = "Classifier"

        for param in alexnet_model.parameters():
          param.requires_grad = False

        self.fc1 = nn.Linear(512 * 512 * 1, 100)
        self.fc2 = nn.Linear(100, 5)

    def forward(self, x):
        x = alexnet_model.features(x)
        x = x.view(-1, 512 * 512 * 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


# ### Resnet classifer

# In[4]:


import torch.nn as nn
import torchvision.models

resnet152 = torchvision.models.resnet152(pretrained=True)
modules = list(resnet152.children())
modules[0].in_channels = 1

class HemorrhageClassifier2(nn.Module):
    def __init__(self):
        super(HemorrhageClassifier2, self).__init__()
        self.name = "Classifier 2"

        for param in resnet152.parameters():
          param.requires_grad = False

        self.fc1 = nn.Linear(512 * 512 * 1, 100)
        self.fc2 = nn.Linear(100, 5)

    def forward(self, x):
        x = resnet152(x)
        x = x.view(-1, 512 * 512 * 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# # **Hemorrhage Detector**

# **Two convulational layers:**
#   - each have a 5x5 receptive field, 2x2 stride, and 2x2 padding
# 
# **The number of feature maps:**
#   - for the 1st convulational layer is 10
#   - for the 2nd convolutional layer is 20
# 
# **Two Pooling Layers:**
#   - Receptive field: 2x2, stride: 2x2
# 
# **Fully Connected layer:**
#   - 100 nodes
#   - 2 output nodes (binary classification)
# 
# **Training Parameters:**
# 
# lr = 0.02
# momentum =0.01
# 4000 iterations (60 epochs)
# Batch size of 20
# Stochastic gradient descent

# ### Alexnet detector

# In[5]:


import torch.nn as nn
import torchvision.models

alexnet_model = torchvision.models.alexnet(pretrained=True)
alexnet_model.features[0] = nn.Conv2d(1, 64, kernel_size= 7, stride= 2, padding= 3)

class HemorrhageDetector(nn.Module):
    def __init__(self):
        super(HemorrhageDetector, self).__init__()
        self.name = "Detector"

        for param in alexnet_model.parameters():
            param.requires_grad = False
        
        self.alex_output_size = 256*7*7
        self.fc1 = nn.Linear(self.alex_output_size, 100)
        self.fc2 = nn.Linear(100, 2)

    def forward(self, x):
        x = alexnet_model.features(x)
        #print("Alex output", x.shape)
        #full size: x = x.view(-1, 256*31*31)
        x = x.view(-1, self.alex_output_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

class HemorrhageDetector3(nn.Module):
    def __init__(self):
        super(HemorrhageDetector3, self).__init__()
        self.name = "Detector3"

        for param in alexnet_model.parameters():
            param.requires_grad = False

        self.alex_output_size = 256*15*15
        self.fc1 = nn.Linear(self.alex_output_size, 1000)
        self.fc2 = nn.Linear(1000, 100)
        self.fc3 = nn.Linear(100, 2)

    def forward(self, x):
        x = alexnet_model.features(x)
       # print("Alex output", x.shape)
        #full size: x = x.view(-1, 256*31*31)
        x = x.view(-1, self.alex_output_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


# ### Resnet detector

# In[6]:


import torch.nn as nn
import torchvision.models

resnet152 = torchvision.models.resnet152(pretrained=True)
modules = list(resnet152.children())
modules[0].in_channels = 1

class HemorrhageDetector2(nn.Module):
    def __init__(self):
        super(HemorrhageDetector2, self).__init__()
        self.name = "Detector 2"

        for param in resnet152.parameters():
          param.requires_grad = False

        self.fc1 = nn.Linear(512 * 512 * 1, 100)
        self.fc2 = nn.Linear(100, 2)

    def forward(self, x):
        x = resnet152(x)
        x = x.view(-1, 512 * 512 * 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ---

# # Training

# In[7]:


import time

def get_accuracy(model, data_loader, use_cuda):

    cor = 0
    total = 0
    n = 0
    for imgs, labels in data_loader:
#         imgs = features = torch.load(f'{name}_features_batchcount{n}.tensor')
        imgs = torch.from_numpy(imgs.detach().numpy())
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

import gc 

def train(model, train_dataset, batch_size = 64, learning_rate=0.01, num_epochs=20, use_cuda = False):
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    iters, losses, train_acc, val_acc = [], [], [], []
    
    if not os.path.exists("Model_" + model.name):
        os.mkdir("./Model_" + model.name)    
    
    training_loader = torch.utils.data.DataLoader(train_dataset, batch_size= batch_size)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size= 32)
    start_time = time.time()
    n = 0 
    for epoch in range(num_epochs):
        gc.collect()
        count = 0
        for imgs, labels in iter(training_loader):
#             imgs = torch.load(f'training_features_batchcount{count}.tensor')
            #imgs = torch.from_numpy(imgs.detach().numpy())

            #To Enable GPU Usage
            if use_cuda and torch.cuda.is_available():
                imgs = imgs.cuda()
                labels = labels.cuda()
            #############################################
            
            #print("imgs.shape, labels.shape") 
            #print(imgs.shape, labels.shape)

            outputs = model(imgs)             
            loss = criterion(outputs, labels) 
            loss.backward()               
            optimizer.step()              
            optimizer.zero_grad()         

            # save the current training information
            iters.append(n)
            losses.append(float(loss)/batch_size)             # finding the average loss
#             train_acc.append(get_accuracy(model, training_loader, use_cuda)) # finding the training accuracy 
#             val_acc.append(get_accuracy(model, val_loader, use_cuda))  # finding the validation accuracy
            n = n+1
            count = count+1
            # print(epoch, count)
            
        print("Epoch", epoch, "Loss", loss)
        
        # Save the current model (checkpoint) to a file
        model_path = "Model_" + model.name + "/{0}_bs_{1}_lr_{2}_epoch".format(
                                                   batch_size,
                                                   learning_rate,
                                                   epoch)
        torch.save(model.state_dict(), model_path)
          

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Time", elapsed_time)
    
    # Write the train/test loss/err into CSV file for plotting later
    np.savetxt("{0}_train_loss_{1}.csv".format(model_path, elapsed_time), losses)

#     plt.title("Training Loss Curve")
#     plt.plot(iters, losses, label="Train")
#     plt.xlabel("Iterations")
#     plt.ylabel("Loss")
#     plt.show()

#     plt.title("Training vs. Validation Accuracy Curves")
#     plt.plot(iters, train_acc, label="Train")
#     plt.plot(iters, val_acc, label="Validation")
#     plt.xlabel("Iterations")
#     plt.ylabel("Accuracy")
#     plt.legend(loc='best')
#     plt.show()


print("Train...")

training_folders = [
    "../Data/Processed/train/epidural",
    "../Data/Processed/train/intraparenchymal",
    "../Data/Processed/train/subarachnoid",
    "../Data/Processed/train/intraventricular",
    "../Data/Processed/train/subdural",
    "../Data/Processed/train/nohem",
]

train_data = Data(training_folders, {
    "epidural":"any", 
    "intraparenchymal":"any", 
    "subarachnoid":"any", 
    "intraventricular":"any", 
    "subdural":"any", 
}, 100, False, 256)

# print("Val....")

# val_folders = [
#     "../Data/Processed/val/epidural",
#     "../Data/Processed/val/intraparenchymal",
#     "../Data/Processed/val/subarachnoid",
#     "../Data/Processed/val/intraventricular",
#     "../Data/Processed/val/subdural",
#     "../Data/Processed/val/nohem",
# ]

# val_data = Data(val_folders, {
#     "epidural":"any", 
#     "intraparenchymal":"any", 
#     "subarachnoid":"any", 
#     "intraventricular":"any", 
#     "subdural":"any", 
# }, 64)

print("Amound of train data being used:", len(train_data))
#print("Type", type(train_data[0][0]))

model = HemorrhageDetector3()
model.name = "resizing_test_complex_bs=32"
#print(torch.max(train_data[0][0]))
print("Starting training", train_data[0][0].shape)
train(model, train_data, batch_size=32, use_cuda=False)


# ## Classfier run

# training_folders = [
#     "Processed/train/epidural",
#     "Processed/train/intraparenchymal",
#     "Processed/train/subarachnoid",
#     "Processed/train/intraventricular",
#     "Processed/train/subdural",
# ]

# train_data = Data(training_folders)

