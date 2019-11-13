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
    categorized_images[cat][label].append(img_id)


def getImg(img_name):
    '''Scale pixel data to Hounsfield units with a linear transformation'''
    dc = pydicom.dcmread(os.path.join(train_data_path, img_name)+".dcm")
    intercept = dc[('0028','1052')].value
    slope = dc[('0028','1053')].value
    return dc.pixel_array * slope + intercept   

def cleanDC(img, clip_range = (13,75)):
    
    # Clamp pixel values to only HU range we want
    clipped = np.clip(img, clip_range[0], clip_range[1])
    
    # Normlaize pixels from 0-1
    # print(max(clipped.flatten()) - min(clipped.flatten()))
    try:
        norm = (clipped - min(clipped.flatten())) / (max(clipped.flatten())-min(clipped.flatten()))
    except:
        raise ValueError("no norm :(")
    # Scale values from 0-255 for CNN's
    #rgb_scaled = 255 * norm

    return norm

# Save
hem = categorized_images["any"][1]
nohem = categorized_images["any"][0]

# os.mkdir("Processed")
# os.mkdir("Processed/Binary")
# os.mkdir("Processed/Binary/Hem")
# os.mkdir("Processed/Binary/NoHem")

def saveFile(img_array, img_name, dir_path):
    pickle.dump(img_array, open(dir_path+"/"+img_name+".pickle", 'wb'))
def cleanImg(img_name):
    img = getImg(img_name)
    return cleanDC(img)

def cleanNSave(img):
    
    if os.path.exists("Processed/Binary/NoHem/" + img + ".dcm"): 
        return None
    try:
        cleaned_img = cleanImg(img)
        saveFile(cleaned_img, img, "Processed/Binary/NoHem")
    except:
#        print(img, "Unsuccessfull")
        return img

print('Starting pool...')

nohem.reverse()

p = Pool()
undone = p.map(cleanNSave, nohem)
p.close()
p.join()

while None in undone:
    undone.remove(None)
print(len(undone))
