import os
import _pickle as pickle 
import torch
import numpy as np
from multiprocessing import Pool
import gc
import torch.nn.functional as F

import sys
sys.path.append("../Training/Scripts")
from tl import *

#  Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()
        
class Data:
    
    def __init__(self, path_to_pickle_folders, replace_classes = {}, maximum_per_folder = None, multi_pool = False, size = 256, tl_model = "alexnet", in_channels=1):
        
        if type(path_to_pickle_folders) != list:
            path_to_pickle_folders = [path_to_pickle_folders]
        
        self.data = []
        self.size = size
        self.tl_model = tl_model
        self.in_channels = in_channels
        
        for folder in path_to_pickle_folders:
            print("Unpacking", os.path.basename(folder))
            working_label = os.path.basename(folder)
            
            if os.path.basename(folder) in replace_classes:
                working_label = replace_classes[os.path.basename(folder)]
            
            files_to_unpickle = [os.path.join(folder, img) for img in os.listdir(folder)]
            files_to_unpickle = files_to_unpickle[:maximum_per_folder]

            if multi_pool:
                p = Pool()
                results = p.map(self.parsePickle, files_to_unpickle)
                p.close()
                p.join()
            else:
                #results = [self.parsePickle(file) for file in files_to_unpickle]
                results = []
                i = 1
                for file in files_to_unpickle:
                    results.append(self.parsePickle(file))
                    printProgressBar(i, len(files_to_unpickle))
                    i += 1
                            
            
            # add to data
            for file in results:
                try:
                    if file:
                        pass
                except:
                    self.data.append({
                        working_label : file
                    })

        print("> Converting labels to tensor.")
        self.convetLabels()
        print("> Converting data to tensors + resize.")
        self.dataToTensor()
        print("> Applying transfer learning.")
        self.applyTL()
    
    def applyTL(self):
        
        # Pick model
        model = None
        if self.tl_model == "alexnet":
            if self.in_channels == 1:
                model = alexnet_model_1.features
            else:
                model = alexnet_model_3.features
        
        elif self.tl_model == "resnet":
            model = resnet152_3
        
        else:
            raise ValueError("Don't be dumb.")

        # Apply forward pass
        new_data = []
        for data in self.data:
            
            label = list(data.keys())[0]
            img = list(data.values())[0].unsqueeze(0).cuda()
            
            if self.in_channels == 3:
                # Duplicate on all 3 channels
                img = np.stack((img.cpu().clone().numpy(),)*3, axis=1).squeeze(2)
                img = torch.Tensor(img).cuda()
            
            img = model(img).squeeze(0)
            
            # Resave
            new_data.append({label : img})           
        
        #print(img.shape)
        self.data = new_data

    def dataToTensor(self):
        new_data = []
        for data_dict in self.data:
            array = list(data_dict.values())[0]
            label = list(data_dict.keys())[0]

            array = torch.Tensor(array).unsqueeze(0)
            array = array * 255
            #print("before", array.shape)
            array = F.interpolate(array.unsqueeze(0), size=self.size).squeeze(0)
            #print("after", array.shape)
            if type(array) == torch.Tensor: 
                #and array.shape == torch.Size([1, self.size, self.size]):
                    new_data.append({
                     label : array
                    })
            else:
                print(type(array), array.shape)

        self.data = new_data
    
    def convetLabels(self):
        all_labels = np.array([list(data.keys())[0] for data in self.data])
        unique_labels = list(np.unique(all_labels))
        self.label_dict = {label:unique_labels.index(label) for label in unique_labels}
        
        if len(self.label_dict) > 2:
            self.binary = False
        else:
            self.binary = True
    
    def parsePickle(self, path_to_pickle):
        try:
            f=open(path_to_pickle,'rb')
            
            gc.disable()
            img=pickle.load(f)
            gc.enable()
            
            #f.close()
            return img
        except:
            pass
    
    def __getitem__(self, idx):
        ''' Return img, label'''
        data = self.data[idx]
        img = list(data.values())[0]
        word_label = list(data.keys())[0]

        label = self.label_dict[word_label]
        
        #if not self.binary:
         #   total_classes = len(self.label_dict)
          #  all_labels = list(self.label_dict.values())
           # all_labels = [test_label if test_label==label else 0.0 for test_label in all_labels]
            #label = torch.Tensor(all_labels).long()

        return img, label
    
    def __len__(self):
        return len(self.data)
