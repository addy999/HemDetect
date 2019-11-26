import os
import sys
sys.path.append(r"./")
sys.path.append(r'../../Data/')
from dataloader import Data
from ClassifierModels import *
from Training import *

print("Import Train Data...")

img_size = 512 # try 128

training_folders = [
    "../../Data/Processed/train/epidural",
    "../../Data/Processed/train/intraparenchymal",
    "../../Data/Processed/train/subarachnoid",
    "../../Data/Processed/train/intraventricular",
    "../../Data/Processed/train/subdural",
]

train_data = Data(training_folders,
            maximum_per_folder = 500, #5000
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
            maximum_per_folder = 150, #1500
            multi_pool = False, 
            size = img_size
            )

print("Amound of train data being used:", len(train_data))

model = AlexNetClassifierCNN(img_size).cuda()
model.name = "shrey_AlexNetClassifierCNN_adam_256_size_27k_0.001lr"

print("Starting training")
# train(model, train_data, val_data, batch_size=20, num_epochs=100, learning_rate=0.02, optim_param="sgd")
