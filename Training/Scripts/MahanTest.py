import os
import sys
sys.path.append(r"./")
sys.path.append(r'../../Data/')
from dataloader import Data
from DetectionModels import *
from Training import *

print("Import Train Data...")

img_size = 256

training_folders = [
    "../../Data/Processed/train/epidural",
    "../../Data/Processed/train/intraparenchymal",
    "../../Data/Processed/train/subarachnoid",
    "../../Data/Processed/train/intraventricular",
    "../../Data/Processed/train/subdural",
    "../../Data/Processed/train/nohem",
]

train_data = Data(training_folders, 
            {
                "epidural":"any", 
                "intraparenchymal":"any", 
                "subarachnoid":"any", 
                "intraventricular":"any", 
                "subdural":"any", 
            }, 
            maximum_per_folder = 2500, #5000
            multi_pool = False, 
            size = img_size, in_channels=3,
            )

print("Import Val Data...")

val_folders = [
    "../../Data/Processed/val/epidural",
    "../../Data/Processed/val/intraparenchymal",
    "../../Data/Processed/val/subarachnoid",
    "../../Data/Processed/val/intraventricular",
    "../../Data/Processed/val/subdural",
    "../../Data/Processed/val/nohem",
]

val_data = Data(val_folders, 
            {
                "epidural":"any", 
                "intraparenchymal":"any", 
                "subarachnoid":"any", 
                "intraventricular":"any", 
                "subdural":"any", 
            }, 
            maximum_per_folder = 750, #1500
            multi_pool = False, 
            size = img_size, in_channels = 3,
            )

print("Amound of train data being used:", len(train_data))

model = AlexNetDetector3(img_size).cuda()
model.name = "alex3, imgs=13k, bs=64, epoch=30, lr=0.0001"

print("Starting training")
train(model, train_data, val_data, batch_size=64, num_epochs=30, learning_rate=0.0001)
