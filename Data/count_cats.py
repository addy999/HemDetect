import time
import pickle
import os
import numpy as np
import pandas as pd

#print('Create category map...')
train_csv_path = "../dataset/stage_1_train.csv"
train_pred_data = pd.read_csv(train_csv_path, index_col=False)
train_pred_data.index = train_pred_data['ID'].values

categorized_images = {
    'epidural' : [], 
    'intraparenchymal' : [], 
    'intraventricular' : [], 
    'subarachnoid' : [], 
    'subdural' : [],
    'hem' : [],
    'nohem' : []

}
for img_id_cat,label in train_pred_data.values:
    img_id = img_id_cat[:12]
    cat = img_id_cat[13:]

    if cat == "any":
        if label > 0:
            cat = "hem"
        else:
            cat = "nohem"

    if label >0:
        categorized_images[cat].append(img_id)

print({a:len(b) for a,b in categorized_images.items()})

#pickle.dump(categorized_images, open("category_map.pickle", "wb"))


