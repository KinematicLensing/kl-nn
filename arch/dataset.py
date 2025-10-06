from os.path import join
import numpy as np
import pandas as pd
from astropy.io import fits
import torch
from torch.utils.data import Dataset
import pyxis as px

import sys,time,os
import config


class FiberDataset(Dataset):
    """Object for interfacing with `torch.utils.data.Dataset`.

    This object allows you to wrap a pyxis LMDB as a PyTorch
    `torch.utils.data.Dataset`. The main benefit of doing so is to utilise
    the PyTorch iterator: `torch.utils.data.DataLoader`.

    Please note that all data values are converted to `torch.Tensor` type using
    the `torch.from_numpy` function in `self.__getitem__`.

    It is not safe to use this dataset along with a dataset writer.
    Make sure you are only reading from the dataset while using the class.

    Parameter
    ---------
    dirpath : string
        Path to the directory containing the LMDB.
    """

    def __init__(self, dirpath, nGPUs, rank):
        self.dirpath = dirpath
        self.nGPUs = nGPUs
        self.rank = rank
        with px.Reader(self.dirpath) as db:
            nsamples = len(db)
            N = nsamples//nGPUs
            start = N*rank
            self.db = db[start:start+N]

    def __len__(self):
        return len(self.db)

    def __getitem__(self, key):
        data = self.db[key]
        for k in data.keys():
            data[k] = torch.from_numpy(data[k])

        return data

    def __repr__(self):
        return str(self.db)
    

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