#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
from PIL import Image


# In[2]:


import os
import _pickle as pickle 
import torch
import numpy as np
from multiprocessing import Pool
import gc

class Data:
    
    def __init__(self, path_to_pickle_folders, replace_classes = {}, maximum_per_folder = None):
        
        if type(path_to_pickle_folders) != list:
            path_to_pickle_folders = [path_to_pickle_folders]
        
        self.data = []
        
        for folder in path_to_pickle_folders:
            print("Unpacking", os.path.basename(folder))
            working_label = os.path.basename(folder)
            
            if os.path.basename(folder) in replace_classes:
                working_label = replace_classes[os.path.basename(folder)]
            
            files_to_unpickle = [os.path.join(folder, img) for img in os.listdir(folder)]
            files_to_unpickle = files_to_unpickle[:maximum_per_folder]
            
#             gc.disable()
            #p = Pool()
            #results = p.map(self.parsePickle, files_to_unpickle)
            #p.close()
            #p.join()
            results = [self.parsePickle(file) for file in files_to_unpickle]
#             gc.enable()
            
            # add to data
            for file in results:
                try:
                    if file:
                        pass
                except:
                    self.data.append({
                        working_label : file
                    })

        self.convetLabels()
        self.dataToTensor()
    
    def dataToTensor(self):
        i = 0
        for data_dict in self.data:
            array = list(data_dict.values())[0]
            array = torch.Tensor(array).unsqueeze(0)
            print(array.shape)
            self.data[i] = {
                list(data_dict.keys())[0] : array
            }
            i+=1
    
    def convetLabels(self):
        all_labels = np.array([list(data.keys())[0] for data in self.data])
        unique_labels = list(np.unique(all_labels))
        self.label_dict = {label:unique_labels.index(label) for label in unique_labels}
    
    def parsePickle(self, path_to_pickle):
        try:
            f=open(path_to_pickle,'rb')
            
            gc.disable()
            img=pickle.load(f)
            gc.enable()
            
            f.close()
            return img
        except:
            pass
    
    def __getitem__(self, idx):
        ''' Return img, label'''
        data = self.data[idx]
        img = list(data.values())[0]
        word_label = list(data.keys())[0]
        label = self.label_dict[word_label]

        return img, label
    
    def __len__(self):
        return len(self.data)


# ## **Hemorrhage Classifier:**

# ### Alexnet classifier

# In[3]:


import torch.nn as nn
import torchvision.models

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


# ### Resnet classifer

# In[4]:


import torch.nn as nn
import torchvision.models

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
# 
# **Training Parameters:**
# 
# lr = 0.02
# momentum =0.01
# 4000 iterations (60 epochs)
# Batch size of 20
# Stochastic gradient descent

# ### Alexnet detector

# In[5]:


import torch.nn as nn
import torchvision.models

alexnet_model = torchvision.models.alexnet(pretrained=True)
alexnet_model.features[0] = nn.Conv2d(1, 64, kernel_size= 7, stride= 2, padding= 3)

class HemorrhageDetector(nn.Module):
    def __init__(self):
        super(HemorrhageDetector, self).__init__()
        self.name = "Detector"

        # for param in alexnet_model.parameters():
        #       param.requires_grad = False

        self.fc1 = nn.Linear(256*31*31, 100)
        self.fc2 = nn.Linear(100, 2)

    def forward(self, x):
        x = alexnet_model.features(x)
#         print(x.shape)
        x = x.view(-1, 256*31*31)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


# ### Resnet detector

# In[6]:


import torch.nn as nn
import torchvision.models

resnet152 = torchvision.models.resnet152(pretrained=True)
modules = list(resnet152.children())
modules[0].in_channels = 1

class HemorrhageDetector2(nn.Module):
    def __init__(self):
        super(HemorrhageDetector2, self).__init__()
        self.name = "Detector 2"

        for param in resnet152.parameters():
          param.requires_grad = False

        self.fc1 = nn.Linear(512 * 512 * 1, 100)
        self.fc2 = nn.Linear(100, 2)

    def forward(self, x):
        x = resnet152(x)
        x = x.view(-1, 512 * 512 * 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ---

# # Training

# In[7]:


import time

def get_accuracy(model, data_loader, use_cuda):

    cor = 0
    total = 0
    n = 0
    for imgs, labels in data_loader:
#         imgs = features = torch.load(f'{name}_features_batchcount{n}.tensor')
        imgs = torch.from_numpy(imgs.detach().numpy())
        #To Enable GPU Usage
        if use_cuda and torch.cuda.is_available():
            imgs = imgs.cuda()
            labels = labels.cuda()
        #############################################
        output = model(imgs)
        pred = output.max(1, keepdim=True)[1]
        cor = cor + pred.eq(labels.view_as(pred)).sum().item()
        total = total + imgs.shape[0]
        n = n+1
    return cor / total

def train(model, train_dataset, val_dataset, batch_size = 64, learning_rate=0.01, num_epochs=30, use_cuda = False):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    iters, losses, train_acc, val_acc = [], [], [], []
    
    training_loader = torch.utils.data.DataLoader(train_dataset, batch_size= batch_size)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size= 32)
    start_time = time.time()
    n = 0 
    for epoch in range(num_epochs):
        count = 0
        for imgs, labels in iter(training_loader):
#             imgs = torch.load(f'training_features_batchcount{count}.tensor')
            imgs = torch.from_numpy(imgs.detach().numpy())

            #To Enable GPU Usage
            if use_cuda and torch.cuda.is_available():
                imgs = imgs.cuda()
                labels = labels.cuda()
            #############################################
            
            outputs = model(imgs)             
            loss = criterion(outputs, labels) 
            loss.backward()               
            optimizer.step()              
            optimizer.zero_grad()         

            # save the current training information
            iters.append(n)
            losses.append(float(loss)/batch_size)             # finding the average loss
#             train_acc.append(get_accuracy(model, training_loader, use_cuda)) # finding the training accuracy 
#             val_acc.append(get_accuracy(model, val_loader, use_cuda))  # finding the validation accuracy
            n = n+1
            count = count+1
        
        print("Epoch", epoch, "Loss", loss)
        
        # Save the current model (checkpoint) to a file
        model_path = "Model_1000_each/model_{0}_bs{1}_lr{2}_epoch{3}".format(model.name,
                                                   batch_size,
                                                   learning_rate,
                                                   epoch)
        torch.save(model.state_dict(), model_path)
          

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Time", elapsed_time)
    
    # Write the train/test loss/err into CSV file for plotting later
    np.savetxt("{0}_train_loss_{1}.csv".format(model_path, elapsed_time), losses)

#     plt.title("Training Loss Curve")
#     plt.plot(iters, losses, label="Train")
#     plt.xlabel("Iterations")
#     plt.ylabel("Loss")
#     plt.show()

#     plt.title("Training vs. Validation Accuracy Curves")
#     plt.plot(iters, train_acc, label="Train")
#     plt.plot(iters, val_acc, label="Validation")
#     plt.xlabel("Iterations")
#     plt.ylabel("Accuracy")
#     plt.legend(loc='best')
#     plt.show()


print("Train...")

training_folders = [
    "../Data/Processed/train/epidural",
    "../Data/Processed/train/intraparenchymal",
    "../Data/Processed/train/subarachnoid",
    "../Data/Processed/train/intraventricular",
    "../Data/Processed/train/subdural",
    "../Data/Processed/train/nohem",
]

train_data = Data(training_folders, {
    "epidural":"any", 
    "intraparenchymal":"any", 
    "subarachnoid":"any", 
    "intraventricular":"any", 
    "subdural":"any", 
}, 64)

print("Val....")

val_folders = [
    "../Data/Processed/val/epidural",
    "../Data/Processed/val/intraparenchymal",
    "../Data/Processed/val/subarachnoid",
    "../Data/Processed/val/intraventricular",
    "../Data/Processed/val/subdural",
    "../Data/Processed/val/nohem",
]

val_data = Data(val_folders, {
    "epidural":"any", 
    "intraparenchymal":"any", 
    "subarachnoid":"any", 
    "intraventricular":"any", 
    "subdural":"any", 
}, 64)

print("Amound of train+val data being used:", len(train_data), len(val_data))

np.savetxt("./done_data_1000.csv", [1,2,3])

model = HemorrhageDetector().cuda()
train(model, train_data, val_data, use_cuda=True)


# ## Classfier run

# training_folders = [
#     "Processed/train/epidural",
#     "Processed/train/intraparenchymal",
#     "Processed/train/subarachnoid",
#     "Processed/train/intraventricular",
#     "Processed/train/subdural",
# ]

# train_data = Data(training_folders)

