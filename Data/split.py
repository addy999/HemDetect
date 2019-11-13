# Execute from Data folder
import os
from shutil import move
from category_map_2 import categorized_images
import numpy as np

processed_imgs = [img.replace(".pickle", "") for img in os.listdir('./Processed/')]
# categorized_images.pop("any")

split_ratios = {
    'train' : 0.80,
    'val' : 0.20,
}

for cat in categorized_images:
    
    print('Splitting', cat)
    
    all_in_cat = np.array(categorized_images[cat][1])
    if cat == "any":
        cat = "nohem"
        all_in_cat = np.array(categorized_images["any"][0])
    
    processed_in_cat = np.intersect1d(all_in_cat, processed_imgs)
    
    # Make folders
    for split,ratio in split_ratios.items():
        if not os.path.exists('./Processed/' + split + '/'+cat):
            os.mkdir('./Processed/' + split + '/'+cat)
    
    # Split and move
    
    ## Train
    for i in range(0, int(len(processed_in_cat) * split_ratios['train'])):
        img = processed_in_cat[i] + ".pickle"
        src = './Processed/'+img
        dest = './Processed/train/'+cat+img
        try:
            move(src, dest)
        except:
            pass
    
    ## Val
    for i in range(int(len(processed_in_cat) * split_ratios['train']), len(processed_in_cat)):
        img = processed_in_cat[i] + ".pickle"
        src = './Processed/'+img
        dest = './Processed/val/'+cat+img
        try:
            move(src, dest)
        except:
            pass
        
        
        
        
