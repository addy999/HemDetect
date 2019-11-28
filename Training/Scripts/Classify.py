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

train_data_a = Data(training_folders,
            maximum_per_folder = 1000, #5000
            size = img_size, tl_model = "alexnet", in_channels=3,
            )

val_data_a = Data(val_folders,
            maximum_per_folder = 300, #1500 
            size = img_size, tl_model = "alexnet", in_channels=3,
            )

print("Load resnet data")
train_data_a = Data(training_folders,
            maximum_per_folder = 1000, #5000
            size = img_size, tl_model = "resnet", in_channels=3,
            )

val_data_a = Data(val_folders,
            maximum_per_folder = 300, #1500 
            size = img_size, tl_model = "resnet", in_channels=3,
            )

print("Amound of train data being used:", len(train_data_a))

print("Starting Alex training")
alex_model = AlexNetClassifier2(256).cuda()
alex_model.name = "classify_alex2,imgs=6k,size=256,bs=32,epochs=40,lr=0.001"
train(model, train_data_a, val_data_a, batch_size=32, num_epochs=40, learning_rate=0.001, optim_param="sgd")


print("Starting Resnet training")
res_model = ResnetClass2(256).cuda()
res_model.name = "classify_res2,imgs=6k,size=256,bs=32,epochs=40,lr=0.001"
train(model, train_data_r, val_data_r, batch_size=32, num_epochs=40, learning_rate=0.001, optim_param="sgd")