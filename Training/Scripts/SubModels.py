# Alexnet classifier
import torch.nn as nn
import torchvision.models
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

class AlexNetSubdural(nn.Module):
    
    ''' 3 channel alexnet '''
    
    def __init__(self,img_size):
        super(AlexNetSubdural, self).__init__()
        self.name = "AlexNetSubdural"
        
        if img_size == 256:
            self.alex_output_size = 256*7*7
            
        self.fc1 = nn.Linear(self.alex_output_size, 1000)
        self.fc2 = nn.Linear(1000, 100)
        self.fc3 = nn.Linear(100, 1)
        self.d1 = nn.Dropout(0.0)

    def forward(self, x):
        x = x.view(-1, self.alex_output_size)
        x = F.relu(self.fc1(self.d1(x)))
        x = F.relu(self.fc2(self.d1(x)))
        x = self.fc3(self.d1(x)).squeeze(1)
        return x

class AlexNetIntrav(nn.Module):
    
    ''' 3 channel alexnet '''
    
    def __init__(self,img_size):
        super(AlexNetIntrav, self).__init__()
        self.name = "AlexNetIntrav"
        
        if img_size == 256:
            self.alex_output_size = 256*7*7
            
        self.fc1 = nn.Linear(self.alex_output_size, 1000)
        self.fc2 = nn.Linear(1000, 100)
        self.fc3 = nn.Linear(100, 1)
        #self.fc4 = nn.Linear(500, 1)
        self.d1 = nn.Dropout(0.4)

    def forward(self, x):
        x = x.view(-1, self.alex_output_size)
        x = F.relu(self.fc1(self.d1(x)))
        x = F.relu(self.fc2(self.d1(x)))
        #x = F.relu(self.fc3(self.d1(x)))
        x = F.sigmoid(self.fc3(self.d1(x)))
        x = x.squeeze(1)
        return x
    
class AlexNetSubara(nn.Module):
    
    ''' 3 channel alexnet '''
    
    def __init__(self,img_size):
        super(AlexNetSubara, self).__init__()
        self.name = "AlexNetSubara"
        
        if img_size == 256:
            self.alex_output_size = 256*7*7
            
        self.fc1 = nn.Linear(self.alex_output_size, 1000)
        self.fc2 = nn.Linear(1000, 100)
        self.fc3 = nn.Linear(100, 1)
        self.d1 = nn.Dropout(0.0)

    def forward(self, x):
        x = x.view(-1, self.alex_output_size)
        x = F.relu(self.fc1(self.d1(x)))
        x = F.relu(self.fc2(self.d1(x)))
        x = self.fc3(self.d1(x)).squeeze(1)
        return x

class AlexNetIntrap(nn.Module):
    
    ''' 3 channel alexnet '''
    
    def __init__(self,img_size):
        super(AlexNetIntrap, self).__init__()
        self.name = "AlexNetIntrap"
        
        if img_size == 256:
            self.alex_output_size = 256*7*7
            
        self.fc1 = nn.Linear(self.alex_output_size, 1000)
        self.fc2 = nn.Linear(1000, 100)
        self.fc3 = nn.Linear(100, 1)
        self.d1 = nn.Dropout(0.0)

    def forward(self, x):
        x = x.view(-1, self.alex_output_size)
        x = F.relu(self.fc1(self.d1(x)))
        x = F.relu(self.fc2(self.d1(x)))
        x = self.fc3(self.d1(x)).squeeze(1)
        return x
    
