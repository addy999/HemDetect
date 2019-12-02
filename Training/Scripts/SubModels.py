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
        self.name = "AlexNetEpidural"
        
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
    
