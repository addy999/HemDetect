import torch.nn as nn
import torchvision.models

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


# Resnet classifer

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