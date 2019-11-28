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

# alexnet_model_1
alexnet_model_1 = torchvision.models.alexnet(pretrained=True)
alexnet_model_1.features[0] = nn.Conv2d(1, 64, kernel_size= 7, stride= 2, padding= 3)
alexnet_model_1.cuda()
for param in alexnet_model_1.parameters():
    param.requires_grad = False


# alexnet_model_1
alexnet_model_3 = torchvision.models.alexnet(pretrained=True).cuda()
for param in alexnet_model_3.parameters():
    param.requires_grad = False

# Resnet detector

resnet152_3 = torchvision.models.resnet152(pretrained=True).cuda()
# remove last FC layer
resnet152_3 = torch.nn.Sequential(*(list(resnet152_3.children())[:-1]))
for param in resnet152_3.parameters():
    param.requires_grad = False
