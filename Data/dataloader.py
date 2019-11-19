import os
import _pickle as pickle 
import torch
import numpy as np
from multiprocessing import Pool
import gc
import torch.nn.functional as F

class Data:
    
    def __init__(self, path_to_pickle_folders, replace_classes = {}, maximum_per_folder = None, multi_pool = False, size = 512):
        
        if type(path_to_pickle_folders) != list:
            path_to_pickle_folders = [path_to_pickle_folders]
        
        self.data = []
        self.size = size
        
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
                results = [self.parsePickle(file) for file in files_to_unpickle]
            
            # add to data
            for file in results:
                try:
                    if file:
                        pass
                except:
                    self.data.append({
                        working_label : file
                    })

        self.convetLabels()
        self.dataToTensor()

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
            
            #gc.disable()
            img=pickle.load(f)
            #gc.enable()
            
            f.close()
            return img
        except:
            pass
    
    def __getitem__(self, idx):
        ''' Return img, label'''
        data = self.data[idx]
        img = list(data.values())[0]
        word_label = list(data.keys())[0]
        label = self.label_dict[word_label]
        
        if not self.binary:
            total_classes = len(self.label_dict)
            all_labels = [[label] + [0] for i in range(total_classes-1)]
            label = torch.Tensor(all_labels)

        return img, label
    
    def __len__(self):
        return len(self.data)
