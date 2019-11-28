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
            maximum_per_folder = 1000, #5000
            multi_pool = False,
            size = img_size, tl_model = "alexnet", in_channels=3,
            )

val_folders = [
    "../../Data/Processed/val/epidural",
    "../../Data/Processed/val/intraparenchymal",
    "../../Data/Processed/val/subarachnoid",
    "../../Data/Processed/val/intraventricular",
    "../../Data/Processed/val/subdural",
]

val_data = Data(val_folders,
            maximum_per_folder = 300, #1500
            multi_pool = False, 
            size = img_size, tl_model = "alexnet", in_channels=3,
            )

print("Amound of train data being used:", len(train_data))

model = AlexNetClassifer2(256).cuda()

print("Starting training")
model.name = "classify_res2,imgs=6k,size=256,bs=32,epochs=40,lr=0.001"

train(model, train_data, val_data, batch_size=32, num_epochs=40, learning_rate=0.001, optim_param="sgd")
