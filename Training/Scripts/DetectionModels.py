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

# alexnet_model
alexnet_model = torchvision.models.alexnet(pretrained=True)
alexnet_model.features[0] = nn.Conv2d(1, 64, kernel_size= 7, stride= 2, padding= 3)
alexnet_model.cuda()

class AlexNetDetector1(nn.Module):
    def __init__(self,img_size):
        super(AlexNetDetector1, self).__init__()
        self.name = "AlexNetDetector1"

        for param in alexnet_model.parameters():
            param.requires_grad = False
        
        self.alex_output_size = 256*15*15
        self.fc1 = nn.Linear(self.alex_output_size, 100)
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x):
        x = alexnet_model.features(x)
        #full size: x = x.view(-1, 256*31*31)
        x = x.view(-1, self.alex_output_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x).squeeze(1)

        #print("output",x.shape)

        return x
    
class AlexNetDetector2(nn.Module):
    def __init__(self,img_size):
        super(AlexNetDetector2, self).__init__()
        self.name = "AlexNetDetector2"

        for param in alexnet_model.parameters():
            param.requires_grad = False
        
        if img_size == 256:
            self.alex_output_size = 256*15*15
        elif img_size == 128:
            self.alex_output_size = 256*7*7
        elif img_size == 512:
            self.alex_output_size = 256*31*31
            
        self.fc1 = nn.Linear(self.alex_output_size, 1000)
        self.fc2 = nn.Linear(1000, 100)
        self.fc3 = nn.Linear(100, 1)

    def forward(self, x):
        x = alexnet_model.features(x)
#         print("Alex output", x.shape)
        x = x.view(-1, self.alex_output_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.squeeze(1)

        return x
    
class AlexNetDetector3(nn.Module):
    def __init__(self,img_size):
        super(AlexNetDetector3, self).__init__()
        self.name = "AlexNetDetector3"

        for param in alexnet_model.parameters():
            param.requires_grad = False

        self.alex_output_size = 256*15*15
        self.fc1 = nn.Linear(self.alex_output_size, 10000)
        self.fc2 = nn.Linear(10000, 1000)
        self.fc3 = nn.Linear(1000, 100)
        self.fc4 = nn.Linear(100, 1)

    def forward(self, x):
        x = alexnet_model.features(x)
#         print("Alex output", x.shape)
        #full size: x = x.view(-1, 256*31*31)
        x = x.view(-1, self.alex_output_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = x.squeeze(1)

        return x

# Resnet detector

resnet152 = torchvision.models.resnet152(pretrained=True).cuda()
# remove last FC layer
resnet152 = torch.nn.Sequential(*(list(resnet152.children())[:-1]))
for param in resnet152.parameters():
    param.requires_grad = False

class ResnetDetector1(nn.Module):
    def __init__(self, size):
        super(ResnetDetector1, self).__init__()
        self.name = "ResnetDetector1"
        
        if size == 256:
            self.resnet_output_size = 2048
            
        self.fc1 = nn.Linear(self.resnet_output_size, 100)
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x):
        # Convert 1 channel to 3 by duplication
        x = np.stack((x.cpu().clone().numpy(),)*3, axis=1).squeeze(2)
#         print("Stacked shape", x.shape)
        x = resnet152(torch.Tensor(x).cuda())
#         print("Resnet output shape", x.shape)
        x = x.view(-1, self.resnet_output_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x).squeeze(1)
        return x
    
resnet50 = torchvision.models.resnet50(pretrained=True).cuda()
# remove last FC layer
resnet50 = torch.nn.Sequential(*(list(resnet50.children())[:-1]))
for param in resnet50.parameters():
    param.requires_grad = False

class ResnetDetector2(nn.Module):
    def __init__(self, size):
        super(ResnetDetector1, self).__init__()
        self.name = "ResnetDetector2"
        
        if size == 256:
            self.resnet_output_size = 2048
            
        self.fc1 = nn.Linear(self.resnet_output_size, 100)
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x):
        # Convert 1 channel to 3 by duplication
        x = np.stack((x.cpu().clone().numpy(),)*3, axis=1).squeeze(2)
#         print("Stacked shape", x.shape)
        x = resnet50(torch.Tensor(x).cuda())
#         print("Resnet output shape", x.shape)
        x = x.view(-1, self.resnet_output_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x).squeeze(1)
        return x
