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

# alexnet_model
alexnet_model = torchvision.models.alexnet(pretrained=True)
alexnet_model.features[0] = nn.Conv2d(1, 64, kernel_size= 7, stride= 2, padding= 3)
alexnet_model.cuda()

# Alexnet with fewer connected CNN layers
class AlexNetClassifierCNN(nn.Module):
    def __init__(self,img_size):
        super(AlexNetClassifierCNN, self).__init__()
        self.name = "AlexNetClassifierCNN"
        self.img_size = img_size
        
        if img_size == 512:
            self.alex_output_size = 256 * 31 * 31
            # Conv, Size = 256 * 31 * 31
            self.conv1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=2, stride=1, padding=1)
            # Pool, Size = 32 *32 * 512
            self.pool = nn.MaxPool2d(2, 2) 
            # Conv, Size = 16 * 16 * 512
            self.conv2 = nn.Conv2d(512, 512, 4, stride=2, padding=1)
            # Size = 8 * 8 * 512
            # Pool = 4 * 4 * 512
            # FC Layers 
            self.fc1 = nn.Linear(4 * 4 * 512, 32)
            self.fc2 = nn.Linear(32, 5)
        elif img_size == 256:
            self.alex_output_size = 256 * 15 * 15
            # Conv, Size = 256 * 15 * 15
            self.conv1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=2, stride=1, padding=1)
            # Pool, Size = 16 *16 * 512
            self.pool = nn.MaxPool2d(2, 2) 
            # Conv, Size = 8 * 8 * 512
            self.conv2 = nn.Conv2d(512, 512, 4, stride=2, padding=1)
            # Size = 4 * 4 * 512
            # Pool = 2 * 2 * 512
            # FC Layers 
            self.fc1 = nn.Linear(2 * 2 * 512, 32)
            self.fc2 = nn.Linear(32, 5)
        elif img_size == 128:
            self.alex_output_size = 256 * 7 * 7
            # Conv, Size = 256 * 7 * 7
            self.conv1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=2, stride=1, padding=1)
            # Pool, Size = 8 * 8 * 512
            self.pool = nn.MaxPool2d(2, 2) 
            # Conv, Size = 4 * 4 * 512
            self.conv2 = nn.Conv2d(512, 512, 4, stride=2, padding=1)
            # Size = 2 * 2 * 512
            # Pool = 1 * 1 * 512
            # FC Layers 
            self.fc1 = nn.Linear(2 * 2 * 512, 32)
            self.fc2 = nn.Linear(32, 5)

    def forward(self, x):
        x = alexnet_model.features(x)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        if self.img_size == 512:
            x = x.view(-1, 4 * 4 * 512)
        elif self.img_size == 256:
            x = x.view(-1, 2 * 2 * 512)
        elif self.img_size == 128:
            x = x.view(-1, 1 * 1 * 512)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        s = nn.Softmax(dim=1).cuda()
        return s(x)

# Vanilla CNN
class HemorrhageClassifier(nn.Module):
    def __init__(self, img_size):
        super(HemorrhageClassifier, self).__init__()
        self.name = "HemorrhageClassifier"
        self.img_size = img_size
        if img_size == 256:
            # Conv, Size = 256 * 256 * 1
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=1)
            # Pool, Size = 256 * 256 * 3
            self.pool = nn.MaxPool2d(2, 2) 
            # Conv, Size = 128 * 128 * 3
            self.conv2 = nn.Conv2d(3, 6, 4, stride=2, padding=1)
            # Size = 64 * 64 * 6
            # Pool = 32 * 32 * 6
            # FC Layers 
            self.fc1 = nn.Linear(32 * 32 * 6, 32)
            self.fc2 = nn.Linear(32, 5)
        elif img_size == 512:
            # Conv, Size = 512 * 512 * 1
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=1)
            # Pool, Size = 512 *512 * 3
            self.pool = nn.MaxPool2d(2, 2) 
            # Conv, Size = 256 * 256 * 3
            self.conv2 = nn.Conv2d(3, 6, 4, stride=2, padding=1)
            # Size = 128 * 128 * 6
            # Pool = 64 * 64 * 6
            # FC Layers 
            self.fc1 = nn.Linear(64 * 64 * 6, 32)
            self.fc2 = nn.Linear(32, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        if self.img_size == 256:
            x = x.view(-1, 32 * 32 * 6)
        elif self.img_size == 512:
            x = x.view(-1, 64 * 64 * 6)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        s = nn.Softmax(dim=1).cuda()
        return s(x)
 
# Alexnet with many connected MLP layers
class AlexNetClassifier3(nn.Module):
    def __init__(self,img_size):
        super(AlexNetClassifier3, self).__init__()
        self.name = "AlexNetClassifier3"
        
        if img_size == 256:
            self.alex_output_size = 256*7*7
            
        self.fc1 = nn.Linear(self.alex_output_size, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 32)
        self.fc4 = nn.Linear(32, 5)

    def forward(self, x):
        x = x.view(-1, self.alex_output_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = x.squeeze(1)

        return x

# Alexnet with fewer connected MLP layers
class AlexNetClassifier2(nn.Module):
    
    ''' 3 channel alexnet '''
    
    def __init__(self,img_size):
        super(AlexNetClassifier2, self).__init__()
        self.name = "AlexNetClassifier2"

        for param in alexnet_model.parameters():
            param.requires_grad = False
        
        if img_size == 256:
            self.alex_output_size = 256*7*7
            
        self.fc1 = nn.Linear(self.alex_output_size, 1000)
        self.fc2 = nn.Linear(1000, 100)
        self.fc3 = nn.Linear(100, 5)

    def forward(self, x):
        x = x.view(-1, self.alex_output_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x).squeeze(1)
        return x
    
# Resnet classifer
class ResnetClass1(nn.Module):
    def __init__(self, size):
        super(ResnetClass1, self).__init__()
        self.name = "ResnetClass1"
        
        if size == 256:
            self.resnet_output_size = 2048
            
        self.fc1 = nn.Linear(self.resnet_output_size, 100)
        self.fc2 = nn.Linear(100, 5)

    def forward(self, x):
        x = x.view(-1, self.resnet_output_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x).squeeze(1)
        return x
    
class ResnetClass2(nn.Module):
    def __init__(self, size):
        super(ResnetClass2, self).__init__()
        self.name = "ResnetClass2"
        
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
