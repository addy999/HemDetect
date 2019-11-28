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
    
