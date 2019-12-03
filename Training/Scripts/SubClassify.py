import os
import sys
sys.path.append(r"./")
sys.path.append(r'../../Data/')
from dataloader import Data
from Training import *
from SubModels import *

print("Import Train Data...")

img_size = 256 # try 128

training_folders = [
    "../../Data/Processed/train/epidural",
    "../../Data/Processed/train/intraparenchymal",
    "../../Data/Processed/train/subarachnoid",
    "../../Data/Processed/train/intraventricular",
    "../../Data/Processed/train/subdural",
]

val_folders = [
    "../../Data/Processed/val/epidural",
    "../../Data/Processed/val/intraparenchymal",
    "../../Data/Processed/val/subarachnoid",
    "../../Data/Processed/val/intraventricular",
    "../../Data/Processed/val/subdural",
]


print("Load Alexnet data")

train_data = Data(training_folders,
            {
                "intraparenchymal":"not", 
                "subarachnoid":"not", 
#                 "intraventricular":"not", 
                "epidural":"not", 
                "subdural" : "not",
            }, 
            maximum_per_folder = 5000, #5000
            size = img_size, tl_model = "alexnet", in_channels=3,
           )

val_data = Data(val_folders,
            {
                "intraparenchymal":"not", 
                "subarachnoid":"not", 
#                 "intraventricular":"not", 
                "epidural":"not", 
                "subdural" : "not",
            }, 
            maximum_per_folder = 1500, #1500 
            size = img_size, tl_model = "alexnet", in_channels=3,
            )

print("Amound of train data being used:", len(train_data))

model = AlexNetIntrav(256).cuda()
model.name = "alexIntrav,imgs=27k,size=256,bs=256,epochs=30,lr=0.0001"
train(model, train_data, val_data, batch_size=256, num_epochs=30, learning_rate=0.0001, optim_param="sgd")
