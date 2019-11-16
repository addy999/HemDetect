import os
import sys
sys.path.append(r"./")
sys.path.append(r'../../Data/')
from dataloader import Data
from DetectionModels import *
from Training import *

print("Import Train Data...")

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
            maximum_per_folder = 100, 
            multi_pool = False, 
            size = 256
            )

print("Amound of train data being used:", len(train_data))

model = ResnetDetector1()
model.name = "Resnet_Test"
print("Starting training")
train(model, train_data, use_cuda=False)