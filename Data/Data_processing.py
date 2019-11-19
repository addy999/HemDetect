import os
import pydicom
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
from multiprocessing import Pool

train_data_path = 'dataset/stage_1_train_images'
test_data_path = 'dataset/stage_2/stage_2_train/rsna-intracranial-hemorrhage-detection/stage_2_train'
train_csv_path = "dataset/stage_1_train.csv"
test_csv_path = "dataset/stage_2/rsna-intracranial-hemorrhage-detection/stage_2_train.csv"
img_names = os.listdir(test_data_path)

def loadTrainingData():
    pred_data = pd.read_csv(test_csv_path, index_col=False)
    pred_data.index = pred_data['ID'].values
    return pred_data


pred_data = loadTrainingData()


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
for img_id_cat,label in pred_data.values:
    img_id = img_id_cat[:12]
    cat = img_id_cat[13:]
    categorized_images[cat][label].append(img_id)


def getImg(img_name):
    '''Scale pixel data to Hounsfield units with a linear transformation'''
    dc = pydicom.dcmread(os.path.join(test_data_path, img_name)+".dcm")
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
    
    if os.path.exists("Processed/test/" + img + ".dcm"): 
        return None
    try:
        cleaned_img = cleanImg(img)
        # remove file
       # os.remove(os.path.join(test_data_path, img_name))
        saveFile(cleaned_img, img, "Processed/test")
    except:
#        print(img, "Unsuccessfull")
        return img

print('Starting pool...')
print(len(hem))
to_process = hem

undone = []
#for img in to_process:
 #   undone.append(cleanNSave(img))

p = Pool()
undone = p.map(cleanNSave, to_process)
p.close()
p.join()

while None in undone:
    undone.remove(None)
print("Undone", len(undone))
