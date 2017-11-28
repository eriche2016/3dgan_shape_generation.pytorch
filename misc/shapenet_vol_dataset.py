import os 
import os.path 
import numpy as np 
from glob import glob 
import torch 
import torch.utils.data as data

from IPython.core.debugger import Tracer 
debug_here = Tracer() 


def grab_files(voxel_dir):
    voxel_dir += '/' 

    return [f for f in glob(voxel_dir + '*.npy')]
    
class ShapeNet_Vol_Dataset(data.Dataset):
    def __init__(self, data_dir):
        super(ShapeNet_Vol_Dataset, self).__init__() 
        self.data_dir = data_dir 
        self.files = grab_files(self.data_dir) 
        
    def __getitem__(self, index): 
        filename = self.files[index]
        model = np.load(filename)
        model = np.array(model) 
        return model 

    def __len__(self):
        return len(self.files)
