import os
import sys
sys.path.append(r"./")
sys.path.append(r'../../Data/')
from dataloader import Data
from ClassifierModels import *
from Training import *

print("Import Train Data...")

img_size = 256

training_folders = [
    "../../Data/Processed/train/epidural",
    "../../Data/Processed/train/intraparenchymal",
    "../../Data/Processed/train/subarachnoid",
    "../../Data/Processed/train/intraventricular",
    "../../Data/Processed/train/subdural",
#     "../../Data/Processed/train/nohem",
]

train_data = Data(training_folders,
            maximum_per_folder = 50, #5000
            multi_pool = False, 
            size = img_size
            )

val_folders = [
    "../../Data/Processed/val/epidural",
    "../../Data/Processed/val/intraparenchymal",
    "../../Data/Processed/val/subarachnoid",
    "../../Data/Processed/val/intraventricular",
    "../../Data/Processed/val/subdural",
#     "../../Data/Processed/val/nohem",
]

val_data = Data(val_folders,
            maximum_per_folder = 10, #1500
            multi_pool = False, 
            size = img_size
            )

print("Amound of train data being used:", len(train_data))

model = AlexNetClassifier2(img_size).cuda()
model.name = "shrey_classifier2_adam_256_size_27k_0.01lr"

print(len(train_data.label_dict))

print("Starting training")
train(model, train_data, val_data, batch_size=1, num_epochs=20, learning_rate=0.001, optim_param="sgd")
