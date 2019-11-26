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
        self.fc2 = nn.Linear(100, 5)

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
        self.fc2 = nn.Linear(100, 5)

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