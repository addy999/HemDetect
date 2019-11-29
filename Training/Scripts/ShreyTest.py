import os
import sys
sys.path.append(r"./")
sys.path.append(r'../../Data/')
from dataloader import Data
from BaselineModel import *
from Training import *

print("Import Train Data...")

img_size = 256

training_folders = [
    "../../Data/Processed/train/epidural",
    "../../Data/Processed/train/intraparenchymal",
    "../../Data/Processed/train/subarachnoid",
    "../../Data/Processed/train/intraventricular",
    "../../Data/Processed/train/subdural",
#    "../../Data/Processed/train/nohem",
]

train_data = Data(training_folders, 
 #           {
  #              "epidural":"any", 
   #             "intraparenchymal":"any", 
    #            "subarachnoid":"any", 
     #           "intraventricular":"any", 
      #          "subdural":"any", 
       #     }, 
            maximum_per_folder = 1000, #5000
            size = img_size, tl_model=None,
            )

print("Import Val Data...")
val_folders = [
    "../../Data/Processed/val/epidural",
    "../../Data/Processed/val/intraparenchymal",
    "../../Data/Processed/val/subarachnoid",
    "../../Data/Processed/val/intraventricular",
    "../../Data/Processed/val/subdural",
    #"../../Data/Processed/val/nohem",
]

val_data = Data(val_folders, 
     #       {
      #          "epidural":"any", 
       #         "intraparenchymal":"any", 
        #        "subarachnoid":"any", 
         #       "intraventricular":"any", 
          #      "subdural":"any", 
            #}, 
            maximum_per_folder = 300, #1500
            size = img_size, tl_model=None,
            )

print("Amound of train data being used:", len(train_data))

model = BaselineModel().cuda()
model.name = "baseline_classify_6k"
# model.name = "detect_alex3, imgs=27k, bs=32, epoch=30, lr=0.0001" #87

print("Starting training")
train(model, train_data, val_data, batch_size=32, num_epochs=20, learning_rate=0.0001)
