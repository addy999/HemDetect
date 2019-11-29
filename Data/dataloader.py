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
    
    def __init__(self, path_to_pickle_folders, replace_classes = {}, maximum_per_folder = None, size = 256, tl_model = "alexnet", in_channels=1):
        
        if type(path_to_pickle_folders) != list:
            path_to_pickle_folders = [path_to_pickle_folders]
        
        self.data = []
        self.size = size
        self.tl_model = tl_model
        self.in_channels = in_channels
        self.files = []
        self.directory = "/home/addy/Data/Processed2/"
        if tl_model:
            self.prefix = self.tl_model + "-" + str(self.in_channels)
        self.already_loaded_files = [img for img in os.listdir(self.directory)]
        #print("> Already loaded", self.already_loaded_files)

        for folder in path_to_pickle_folders:
            print("Unpacking", os.path.basename(folder))
            working_label = os.path.basename(folder)
            
            if os.path.basename(folder) in replace_classes:
                working_label = replace_classes[os.path.basename(folder)]
            
            file_names = [img for img in os.listdir(folder)]
            files_to_unpickle = [os.path.join(folder, img) for img in os.listdir(folder)]
            if type(maximum_per_folder) == int:
                file_names = file_names[:maximum_per_folder]
                files_to_unpickle = files_to_unpickle[:maximum_per_folder]
            elif type(maximum_per_folder) == tuple:
                file_names = file_names[maximum_per_folder[0]:maximum_per_folder[1]]
                files_to_unpickle = files_to_unpickle[maximum_per_folder[0]:maximum_per_folder[1]]

            results = []
            i = 0
            for file in files_to_unpickle:
                #print("> Analyzing", self.directory + self.prefix + "-" + file_names[i])
                if self.tl_model:
                    if self.prefix + "-" + file_names[i] in self.already_loaded_files:
                        #print("Found!")
                        file = self.directory + self.prefix + "-" + file_names[i]
                
                results.append(self.parsePickle(file))
                printProgressBar(i, len(files_to_unpickle)) 
                i += 1
                            
            
            # add to data
            i = 0
            for file in results:
                try:
                    if file:
                        pass
                except:
                    self.data.append({
                        working_label : file
                    })
                    self.files.append(file_names[i])
                
                i+=1

        print("> Converting labels to tensor.")
        self.convetLabels()
        if self.tl_model:
            print("> Applying Transfer Learning")
            self.applyTL()
            print("> Saving imgs.")
            self.saveData()
        else:
            self._dataToTensor()
            
    def _dataToTensor(self):
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
            
    
    def saveData(self):

        i = 0
        for data in self.data:
            if not os.path.exists(self.directory + self.prefix + "-" + self.files[i]):
                img = list(data.values())[0]
                pickle.dump(img, open(self.directory + self.prefix + "-" + self.files[i], 'wb'))
            i+=1
    
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
        i = 0
        for data in self.data:
            
            # If file already saved, then it has TL applied already
            if self.prefix + "-" + self.files[i] not in self.already_loaded_files:
            
                label = list(data.keys())[0]
                img = self.dataToTensor(list(data.values())[0]).unsqueeze(0).cuda()
                
                if self.in_channels == 3:
                    # Duplicate on all 3 channels
                    img = np.stack((img.cpu().clone().numpy(),)*3, axis=1).squeeze(2)
                    img = torch.Tensor(img).cuda()
                
                img = model(img).squeeze(0)
                
                # Resave
                if type(img) == torch.Tensor:
                    new_data.append({label : img})        
            
            else:
                new_data.append(data)
            
            i+=1   
        
        #print(img.shape)
        self.data = new_data

    def dataToTensor(self, array):

        array = torch.Tensor(array).unsqueeze(0)
        array = array * 255
        array = F.interpolate(array.unsqueeze(0), size=self.size).squeeze(0)
        
        return array
    
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
