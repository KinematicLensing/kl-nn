from os.path import join
import numpy as np
import pandas as pd
from astropy.io import fits
import torch
from torch.utils.data import Dataset

import sys,time,os
import config

class FiberDataset(Dataset):
    
    def __init__(self, size, nimgs, nspec, pars_dir, data_dir, data_stem):
        
        self.size = size
        self.pars = pd.read_csv(pars_dir)
        self.data_dir = data_dir
        self.data_stem = data_stem
        self.nimgs = nimgs
        self.nspec = nspec
        self.img_index = nspec*2+1
        self.normalize(self.pars)

    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        
        folder = index // 4000 + 1
        
        with fits.open(join(self.data_dir, f'temp_{folder}/', self.data_stem + f'{index}.fits')) as hdu:
            
            img_stack = np.full((self.nimgs, 48, 48), 0, dtype=np.float32)
            for i in range(self.nimgs):
                img_stack[i] = hdu[self.img_index + i*2].data
            img_stack /= np.max(img_stack)
                
            spec_stack = np.full((self.nspec, 64), 0, dtype=np.float32)
            for i in range(self.nspec):
                spec = hdu[2*i+1].data
                spec_stack[i, :spec.shape[0]] = spec
            spec_stack /= np.max(spec_stack)
                
            fid_pars = np.array(self.pars.iloc[index])[1:]
        
        return {'img': img_stack[None],
                'spec': spec_stack[None],
                'fid_pars': fid_pars,
                'id': index}
    
    def normalize(self, pars):
        ranges = config.par_ranges
        for par, values in self.pars.items():
            if par in ranges.keys():
                low, high = ranges[par]
                values -= low
                values /= high-low
                

class ShapeDataset(Dataset):
    
    def __init__(self, gal_pars, psf_pars):
        
        self.gal_pars = gal_pars
        self.psf_pars = psf_pars

    def __len__(self):
        return self.gal_pars["e1"].shape[0]
    
    def __set_pars(self, idx):
        
        self.param_gal = {}
        self.param_gal["e1"] = self.gal_pars["e1"][idx]
        self.param_gal["e2"] = self.gal_pars["e2"][idx]
        self.param_gal["hlr_disk"] = self.gal_pars["hlr_disk"][idx]
        self.param_gal["mag_i"] = self.gal_pars["mag_i"][idx]

        self.param_psf = {}
        # param_psfs['size'] = self.classes_frame[idx,7]
        # param_psfs['e1'] = self.classes_frame[idx,4]
        # param_psfs['e2'] = self.classes_frame[idx,5]
        self.param_psf['randint'] = self.psf_pars["randint"][idx]
        

    def __getitem__(self, idx):
        
        self.__set_pars(idx)
        
        gal_image, clean_gal, psf_image, label_ = get_sim(
                gal_param=self.param_gal,
                psf_param=self.param_psf,
                shear=None,
            )
        
        label = np.array([label_[0]/np.max(self.gal_pars["e1"]),
                          label_[1]/np.max(self.gal_pars["e2"]),
                          label_[2]/np.max(self.gal_pars["hlr_disk"]),
                          label_[3]/np.max(self.gal_pars["mag_i"])])
        
        sig_sky,_ = compute_noise(gal_image,clean_gal)
        snr = np.sqrt(np.sum(np.power(clean_gal,2))/sig_sky**2)

        return {'gal_image': gal_image[None], 
                'psf_image': psf_image[None], 
                'label': label,
                'snr': snr,
                'id': idx}
    

### Dataset for shear measurement
class ShearDataset(Dataset):
    def __init__(self, shear_set, gal_set):
        
        self.shear_set = shear_set
        self.gal_set = gal_set

    def __len__(self):
        return self.gal_set["hlr_disk"].shape[0]*self.gal_set["hlr_disk"].shape[1]
    
    def __set_pars(self, idx):
        
        case_idx = idx//self.gal_set["hlr_disk"].shape[1]
        real_idx = idx%self.gal_set["hlr_disk"].shape[1]
        
        self.param_gal = {}
        self.param_gal["hlr_disk"] = self.gal_set["hlr_disk"][case_idx,real_idx]
        self.param_gal["mag_i"] = self.gal_set["mag_i"][case_idx,real_idx]
        self.param_gal["e1"] = self.gal_set["e1"][case_idx,real_idx]
        self.param_gal["e2"] = self.gal_set["e2"][case_idx,real_idx]
        
        self.shear = self.shear_set['shear'][case_idx,:]

        self.param_psf = {}
        self.param_psf['randint'] = self.gal_set['randint'][case_idx,real_idx]

    def __getitem__(self, idx):
        
        self.__set_pars(idx)

        gal_image, clean_gal, psf_image, label_ = get_sim(
            self.param_gal,
            self.param_psf,
            shear=self.shear,
        )
        
        # note the normalization here
        label = np.array([label_[0],
                  label_[1],
                  label_[2]/1.2,
                  label_[3]/25])
        
        sig_sky,_ = compute_noise(gal_image,clean_gal)
        snr = np.sqrt(np.sum(np.power(clean_gal,2))/sig_sky**2)
        
        return {'gal_image': gal_image[None], 
                'psf_image': psf_image[None], 
                'label': label, 
                'snr': snr,
                'id': idx}
    

### Dataset for NN calibration
class CaliDataset(Dataset):
    def __init__(self, shear_set, gal_set):
        
        self.shear_set = shear_set
        self.gal_set = gal_set

    def __len__(self):
        return self.gal_set["e1"].shape[0]*self.gal_set["e2"].shape[1]
    
    def __set_pars(self, idx):
        
        case_idx = idx//self.gal_set["e1"].shape[1]
        real_idx = idx%self.gal_set["e1"].shape[1]
        
        self.param_gal = {}
        self.param_gal["hlr_disk"] = self.gal_set["hlr_disk"][case_idx][0]
        self.param_gal["mag_i"] = self.gal_set["mag_i"][case_idx][0]
        self.param_gal["e1"] = self.gal_set["e1"][case_idx,real_idx]
        self.param_gal["e2"] = self.gal_set["e2"][case_idx,real_idx]
        
        self.shear = self.shear_set['shear'][case_idx,:]

        self.param_psf = {}
        self.param_psf['randint'] = self.gal_set['randint'][case_idx]

    def __getitem__(self, idx):
        
        self.__set_pars(idx)

        gal_image, clean_gal, psf_image, label_ = get_sim(
            self.param_gal,
            self.param_psf,
            shear=self.shear,
        )
        
        # note the normalization here
        label = np.array([label_[0],
                  label_[1],
                  label_[2]/1.2,
                  label_[3]/25])
        
        sig_sky,_ = compute_noise(gal_image,clean_gal)
        snr = np.sqrt(np.sum(np.power(clean_gal,2))/sig_sky**2)
        
        return {'gal_image': gal_image[None], 
                'psf_image': psf_image[None], 
                'label': label, 
                'snr': snr,
                'id': idx}
    

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