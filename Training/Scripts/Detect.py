import os
import sys
sys.path.append(r"./")
sys.path.append(r'../../Data/')
from dataloader import Data
from DetectionModels import *
from Training import *
from SubModels import *

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
            maximum_per_folder = 5000, #5000
            size = img_size, in_channels=3,
            )

#print(train_data._label_dict)
print("Import Val Data...")
val_folders = [
    "../../Data/Processed/train/epidural",
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
            maximum_per_folder = 1500, #1500
            size = img_size, in_channels = 3,
            )

print("Amound of train data being used:", len(train_data))

model = AlexNetIntrav(img_size).cuda()
model.name = "yeet/detect_alex3, imgs=32k, bs=32, epoch=12, lr=0.0001,d0.4"
print("Starting training")
train(model, train_data, val_data, batch_size=32, num_epochs=20, learning_rate=0.0001)
