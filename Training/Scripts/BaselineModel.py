import torch.nn as nn
import torchvision.models
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import numpy as np
import torch

class BaselineModel(nn.Module):
    def __init__(self):
        super(BaselineModel, self).__init__()
        self.name = "BaselineModel"
        dim_x = 512 * 512 * 1
        dim_h = 32
        dim_out = 5   
        self.fc1 = nn.Linear(dim_x, dim_h)
        self.fc2 = nn.Linear(dim_h, dim_out)

    def forward(self, x):
        x = x.view(-1, 512 * 512 * 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.squeeze(1)
        return x
    
class ResnetDetector1(nn.Module):
    def __init__(self, size):
        super(ResnetDetector1, self).__init__()
        self.name = "ResnetDetector1"
        
        if size == 256:
            self.resnet_output_size = 2048
            
        self.fc1 = nn.Linear(self.resnet_output_size, 100)
        self.fc2 = nn.Linear(100, 5)

    def forward(self, x):
        x = x.view(-1, self.resnet_output_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x).squeeze(1)
        return x
    
class ResnetDetector2(nn.Module):
    def __init__(self, size):
        super(ResnetDetector2, self).__init__()
        self.name = "ResnetDetector2"
        
        if size == 256:
            self.resnet_output_size = 2048
            
        self.fc1 = nn.Linear(self.resnet_output_size, 1000)
        self.fc2 = nn.Linear(1000, 100)
        self.fc3 = nn.Linear(100, 5)

    def forward(self, x):
        x = x.view(-1, self.resnet_output_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x).squeeze(1)
        return x
