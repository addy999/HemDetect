import os
import sys
sys.path.append(r"./")
sys.path.append(r'../../Data/')
from dataloader import Data
from Training import *
from ClassifierModels import *

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
            maximum_per_folder = 2500, #5000
            size = img_size, tl_model = "resnet", in_channels=3,
            )

val_data = Data(val_folders,
            maximum_per_folder = 750, #1500 
            size = img_size, tl_model = "resnet", in_channels=3,
            )

print("Amound of train data being used:", len(train_data))

model = ResnetClass3(256).cuda()
model.name = "classify_resnet3,imgs=12k,size=256,bs=64,epochs=20,lr=0.0001"
train(model, train_data, val_data, batch_size=64, num_epochs=20, learning_rate=0.0001, optim_param="sgd")