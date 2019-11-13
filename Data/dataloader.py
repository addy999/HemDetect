import os
import _pickle as pickle 
import torch
import numpy as np
from multiprocessing import Pool
import gc

class Data:
    
    def __init__(self, path_to_pickle_folders, replace_classes = {}, maximum_per_folder = None):
        
        if type(path_to_pickle_folders) != list:
            path_to_pickle_folders = [path_to_pickle_folders]
        
        self.data = []
        
        for folder in path_to_pickle_folders:
            print("Unpacking", os.path.basename(folder))
            working_label = os.path.basename(folder)
            
            if os.path.basename(folder) in replace_classes:
                working_label = replace_classes[os.path.basename(folder)]
            
            files_to_unpickle = [os.path.join(folder, img) for img in os.listdir(folder)]
            files_to_unpickle = files_to_unpickle[:maximum_per_folder]
            
#             gc.disable()
            #p = Pool()
            #results = p.map(self.parsePickle, files_to_unpickle)
            #p.close()
            #p.join()
            results = [self.parsePickle(file) for file in files_to_unpickle]
#             gc.enable()
            
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
        self.cleanData()
    
    def cleanData(self):
        remove = []
        i=0
        for data in self.data:
            val = list(data.values())[0]
            if type(val) != torch.Tensor:
                remove.append(i)
            i+=1

        for i in remove:
            del self.data[i]

    def dataToTensor(self):
        i = 0
        for data_dict in self.data:
            array = list(data_dict.values())[0]
            array = torch.Tensor(array).unsqueeze(0)
            if array.shape == torch.Size([1, 512, 512]):
                self.data[i] = {
                    list(data_dict.keys())[0] : array
                }
            i+=1
    
    def convetLabels(self):
        all_labels = np.array([list(data.keys())[0] for data in self.data])
        unique_labels = list(np.unique(all_labels))
        self.label_dict = {label:unique_labels.index(label) for label in unique_labels}
    
    def parsePickle(self, path_to_pickle):
        try:
            f=open(path_to_pickle,'rb')
            
            gc.disable()
            img=pickle.load(f)
            gc.enable()
            
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

        #print(type(img), type(label))

        return img, label
    
    def __len__(self):
        return len(self.data)