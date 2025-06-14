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
    def __init__(self, ids, inputs, true):
        
        self.inputs = inputs
        self.true = true
        self.ids = ids

    def __len__(self):
        return self.ids.shape[1]

    def __getitem__(self, case, ID):
        
        return {'inputs': self.inputs[case, ID],
                'target': self.true[case, ID],
                'id': self.ids[case, ID]}
    

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