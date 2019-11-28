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
            maximum_per_folder = 1, #5000
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
            maximum_per_folder = 1, #1500 
            size = img_size, tl_model = "alexnet", in_channels=3,
            )


res_model = AlexNetClassifier2(256).cuda()
res_model.name = "data_save_test"
train(res_model, train_data, val_data, batch_size=32, num_epochs=40, learning_rate=0.001, optim_param="sgd")