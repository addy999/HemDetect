import os
import pydicom
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
from multiprocessing import Pool

train_data_path = '../dataset/stage_1_train_images'
train_csv_path = "../dataset/stage_1_train.csv"
img_names = os.listdir(train_data_path)

def loadTrainingData():
    train_pred_data = pd.read_csv(train_csv_path, index_col=False)
    
   # print(train_pred_data[34343:34343+40])
    
    train_pred_data.index = train_pred_data['ID'].values
    return train_pred_data


train_pred_data = loadTrainingData()


# # Split data into categories
categorized_images = {
    'epidural' : {0:[], 1:[]}, 
    'intraparenchymal' : {0:[], 1:[]}, 
    'intraventricular' : {0:[], 1:[]}, 
    'subarachnoid' : {0:[], 1:[]}, 
    'subdural' : {0:[], 1:[]},
    'any' : {0:[], 1:[]}
}

print('Categorizing...')
for img_id_cat,label in train_pred_data.values:
    img_id = img_id_cat[:12]
    cat = img_id_cat[13:]
    if label not in [0, 1]: print(img_id_cat)
    categorized_images[cat][label].append(img_id)

sums = []
for cat, splits in categorized_images.items():
    sums.append(len(splits[1]))

print(dict(zip(categorized_images.keys(), sums)))

#print({type(b) for a,b in categorized_images.items()})
