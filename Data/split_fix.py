# Execute from Data folder
import os
from shutil import move
import numpy as np

folders = [
    'epidural',
    'intraparenchymal',
    'intraventricular',
    'subarachnoid', 
    'subdural',
    'nohem'
]

train_imgs = os.listdir('./Processed/train')
val_imgs = os.listdir('./Processed/val')

print("cleaning up train")

for img in train_imgs:
    for cat in folders:
        if cat in img and "pickle" in img:
            src = './Processed/train/' + img
            dest = './Processed/train/' + cat + "/" + img.replace(cat, "")
            try:
                move(src, dest)
            except:
                pass
            break

print("cleaning up val")

for img in val_imgs:
    for cat in folders:
        if cat in img and "pickle" in img:
            src = './Processed/val/' + img
            dest = './Processed/val/' + cat + "/" + img.replace(cat, "")
            try:
                move(src, dest)
            except:
                pass
            break
        
        
        
