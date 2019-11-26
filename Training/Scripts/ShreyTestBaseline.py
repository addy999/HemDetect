import os
import sys
sys.path.append(r"./")
sys.path.append(r'../../Data/')
from dataloader import Data
from Training import *
from BaselineModel import *

print("Import Train Data...")

img_size = 256 # try 128

training_folders = [
    "../../Data/Processed/train/epidural",
    "../../Data/Processed/train/intraparenchymal",
    "../../Data/Processed/train/subarachnoid",
    "../../Data/Processed/train/intraventricular",
    "../../Data/Processed/train/subdural",
]

train_data = Data(training_folders,
            maximum_per_folder = 5000, #5000
            multi_pool = False,
            size = img_size
            )

val_folders = [
    "../../Data/Processed/val/epidural",
    "../../Data/Processed/val/intraparenchymal",
    "../../Data/Processed/val/subarachnoid",
    "../../Data/Processed/val/intraventricular",
    "../../Data/Processed/val/subdural",
]

val_data = Data(val_folders,
            maximum_per_folder = 1500, #1500
            multi_pool = False, 
            size = img_size
            )

print("Amound of train data being used:", len(train_data))

model = ResnetDetector1(256).cuda()

print("Starting training")
print(model.name)
      
model.name = "shrey_baseline_56_size_27k_0.001lr"

train(model, train_data, val_data, batch_size=32, num_epochs=100, learning_rate=0.0001, optim_param="sgd")
