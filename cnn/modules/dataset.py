from os.path import join
import numpy as np
import pandas as pd
from astropy.io import fits
import torch
from torch.utils.data import Dataset

import sys,time,os
import config    

### Dataset for NN calibration
class CaliDataset(Dataset):
    def __init__(self, case, inputs, true, ids):
        
        self.inputs = np.load(inputs)[case]
        self.true = np.load(true)[case]
        self.ids = np.load(ids)[case]

    def __len__(self):
        return self.ids.shape[0]

    def __getitem__(self, ID):
        
        return {'inputs': self.inputs[ID].astype(np.float32),
                'target': self.true[ID, :2].astype(np.float32),
                'id': int(self.ids[ID])}
    

### Dataset for NN calibration training
class NNDataset(Dataset):
    def __init__(self, dataset):
        self.classes_frame = dataset
        self.case_size = self.classes_frame['prediction'].shape[0]
        self.real_size = self.classes_frame['prediction'].shape[1]

    def __len__(self):
        return self.case_size*self.real_size

    def __getitem__(self, idx):
        #print(idx)
        
        idx_case = idx//self.real_size
        idx_real = idx%self.real_size
        
        measured_gal = self.classes_frame['prediction'][idx_case,idx_real,:].astype(np.float32)
        shear = self.classes_frame['true_shear'][idx_case].astype(np.float32)

        return {'input': measured_gal, 
                'label': shear, 
                'id': idx}