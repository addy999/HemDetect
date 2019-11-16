import torch.nn as nn
import torchvision.models

# alexnet_model
alexnet_model = torchvision.models.alexnet(pretrained=True)
alexnet_model.features[0] = nn.Conv2d(1, 64, kernel_size= 7, stride= 2, padding= 3)

class AlexNetDetector1(nn.Module):
    def __init__(self):
        super(AlexNetDetector1, self).__init__()
        self.name = "AlexNetDetector1"

        for param in alexnet_model.parameters():
            param.requires_grad = False
        
        self.alex_output_size = 256*15*15
        self.fc1 = nn.Linear(self.alex_output_size, 100)
        self.fc2 = nn.Linear(100, 2)

    def forward(self, x):
        x = alexnet_model.features(x)
        print("Alex output", x.shape)
        #full size: x = x.view(-1, 256*31*31)
        x = x.view(-1, self.alex_output_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
    
class AlexNetDetector2(nn.Module):
    def __init__(self):
        super(AlexNetDetector2, self).__init__()
        self.name = "AlexNetDetector2"

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

# Resnet detector
resnet152 = torchvision.models.resnet152(pretrained=True)
modules = list(resnet152.children())
modules[0].in_channels = 1
resnet152 = nn.Sequential(*modules)
for param in resnet152.parameters():
    param.requires_grad = False

class ResnetDetector1(nn.Module):
    def __init__(self):
        super(ResnetDetector1, self).__init__()
        self.name = "ResnetDetector1"
        self.fc1 = nn.Linear(512 * 512 * 1, 100)
        self.fc2 = nn.Linear(100, 2)

    def forward(self, x):
        x = resnet152(x)
        print("Resnet output shape", x.shape)
        x = x.view(-1, 512 * 512 * 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
