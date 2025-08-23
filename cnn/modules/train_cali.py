import sys,time,os
from os.path import join
import logging
import numpy as np
import pandas as pd
from astropy.io import fits

import torch
from torch import optim, nn
from torch.utils.data import SubsetRandomSampler, DataLoader, Subset
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import pyxis.torch as pxt

from networks import *
from dataset import *
import config

"""
Module that manages all the trainer classes and testing functions. 
Need to create a wrapper trainer class eventually
"""

#-----------#
# Predictor #
#-----------#

class Predictor:
    '''
    Predictor uses trained CNN to make shear estimates, can be used to prepare data
    for calibration network or to test trained CNN model.
    '''
    def __init__(
        self,
        world_size: int,
        model: torch.nn.Module,
        nfeatures: int,
        cali_ds: FiberDataset,
        gpu_id: int,
        save_every: int,
        batch_size: int,
    ) -> None:
        self.world_size = world_size
        self.gpu_id = gpu_id
        self.device = torch.device(f"cuda:{gpu_id}")
        self.nfeatures = nfeatures
        self.cali_data = cali_ds
        self.optimizer = optimizer
        self.save_every = save_every
        self.model = model
        self.batch_size = batch_size
        self.ncali = len(cali_ds)//world_size
        self.nbatch_cali = self.ncali//self.batch_size
        self.logger = logging.getLogger('Trainer')
        
    def _set_tensors(self):
        '''
        Put dataset arrays on GPU for direct access in training
        '''
        # Initialize large arrays on GPU
        if self.gpu_id == 0:
            print("Initializing datasets on GPU...")
        self.img_cali = torch.empty((self.ncali, 1, 48, 48), dtype=torch.float, device=self.device)
        self.spec_vali = torch.empty((self.ncali, 1, 3, 64), dtype=torch.float, device=self.device)
        self.fid_cali = torch.empty((self.ncali, self.nfeatures), dtype=torch.float, device=self.device)
        self.SNR_cali = torch.rand((self.ncali,), device=self.device)*190+10
        
        # Fill arrays with values
        start = self.gpu_id*self.cali
        if self.gpu_id == 0:
            print("Uploading training set to GPU...")
        prev_prog = 0
        for i in range(self.ncali):
            i_db = start+i
            self.img_cali[i] = self.cali_data[i_db]['img']
            self.spec_cali[i] = self.cali_data[i_db]['spec']
            self.fid_cali[i] = self.cali_data[i_db]['fid_pars']
            if self.fid_cali[i, 2] < 0:
                self.fid_cali[i, 2] = self.fid_cali[i, 2] + 1
            prog = 100*i//self.ncali
            if prog % 10 == 0 and prog > prev_prog and self.gpu_id == 0:
                prev_prog = prog
                print(f"{prog}% complete")
        self.fid_cali[:, 2] = self.fid_cali[:, 2]*2 - 1
        
        torch.distributed.barrier()
        
    def _apply_noise(self, data, snr):
        noise = torch.randn_like(data, device=self.device)
        maxs = torch.amax(data, dim=(-1, -2, -3))
        seg = data > 0.1*maxs.view(-1, 1, 1, 1)
        npix = torch.sum(seg, dim=(-1, -2, -3))
        avg = torch.sum(data, dim=(-1, -2, -3))/npix
        factor = avg/snr
        data = data + noise*torch.sqrt(factor.view(-1, 1, 1, 1))
        #mean = data.mean()
        #std = data.std()
        #return (data-mean)/std
        return data

    def _run_batch(self, img, spec, fid):
        self.optimizer.zero_grad()
        outputs = self.model(img, spec)
        loss = self.criterion(outputs, fid)
        loss.backward()                   
        self.optimizer.step()
        return loss
        return epoch_loss

    def _predFunc(self, show_log=True):
        self.model.eval()
        results = torch.empty((self.ncali, self.nfeatures), dtype=torch.float, device=self.device)
        for i in range(self.nbatch_cali):
            start = i*self.batch_size
            end = start+self.batch_size
            batch_ids = self.valid_order[start:end]
            snr = self.SNR_cali[start:end]
            img = self._apply_noise(self.img_valid[start:end], snr)
            spec = self._apply_noise(self.spec_valid[start:end], snr)
            fid = self.fid_valid[start:end]
            
            outputs = self.model(img, spec)
            loss = self.criterion(outputs, fid)
            results[start:end] = outputs
            
            if show_log and self.gpu_id == 0 and i%100 == 0:
                #self.logger.info(f"Batch {i} complete")
                print(f"Batch {i} complete")
              
        results = results.cpu().numpy()
        np.save(join(config.cali['res_dir'], config.cali['model_name']+str(rank)), results)

        if show_log and self.gpu_id == 0:
            print(f"Loss: {loss}")
            
        return loss

    def predict(self):
        self._set_tensors()
        if self.gpu_id == 0:
            print("Running data through CNN...")
        cali_loss = self._predFunc()
        

#------------------#
# Global functions #
#------------------#

def predict(rank: int, world_size: int, Model=ForkCNN, 
            save_every=1, batch_size=100, nfeatures=2):
    '''
    Main function to train any network.
    '''
    # Set parameters based on stage
    batch_size = config.train['batch_size']
    nfeatures = config.train['feature_number']
    epoch = config.cali['epoch_number']
    
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger('Setup')
    if rank == 0:
        log.info('Initializing')
    
    ddp_setup(rank, world_size)
    log.info(f'[rank: {rank}] Successfully set up device')
    
    # Create dataset object
    cali_ds = pxt.TorchDataset(config.cali['data_dir'])
    
    # Initialize model
    model_dir = config.train['model_path'] + config.train['pretrained_name'] + '/' + config.train['pretrained_name'] + str(epoch)
    model = load_model(path=model_dir,strict=True, assign=True)
    if rank == 0:
        print(f"Loaded model {config.train['pretrained_name']} at epoch {epoch}")
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])
    
    log.info(f'[rank: {rank}] Successfully loaded training objects')
    
    predictor = Predictor(world_size, model, nfeatures, train_ds, valid_ds, optimizer, rank, save_every, batch_size)
    log.info(f'[rank: {rank}] Successfully initialized Predictor')
    torch.distributed.barrier()
    predictor.predict()
    destroy_process_group()
    
def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.synchronize()

def load_model(Model=ForkCNN, path=None, strict=True, assign=False, GPUs=1):

    model = Model(config.train['batch_size'], GPUs=GPUs)
    model.to(0)
    if GPUs > 1:
        model = DDP(model, device_ids=None)

    if path != None:
        model.load_state_dict(torch.load(path, weights_only=False), strict=strict, assign=assign)

    return model

##################################################

# Old/defective functions, not in use
    
class MSBLoss(nn.Module):
    def __init__(self):
        super(MSBLoss, self).__init__()
        
    def forward(self,x,y):
        
        if torch.std(y,axis=1).any() != 0:
            print('Warning!')
            # print(y)
        
        # print(x.shape)
        # print(y.shape)
        l = torch.mean(((torch.mean(x,axis=1)-torch.mean(y,axis=1))**2),axis=0)
        # print(torch.mean(x,axis=1)-torch.mean(y,axis=1))
        # print(l)
        return l

def cali_predict(test_dl,MODEL,criterion=MSBLoss()):

    MODEL.eval()
    losses=[]
    for i, batch in enumerate(test_dl):
        inputs, labels = batch['input'].float().to(torch.device(config.train['device'])), \
                         batch['label'].float().to(torch.device(config.train['device']))
        outputs = MODEL.forward(inputs)
        
        # remember to reshape it before feeding into loss calculation
        outputs = torch.reshape(outputs,(-1,test_dl.dataset.real_size))
        labels = torch.reshape(labels,(-1,test_dl.dataset.real_size))

        loss = criterion(outputs, labels)
        losses.append(loss.item())
        if i == 0:
            # ids = i
            res = np.mean(outputs.detach().cpu().numpy(),axis=1)
            labels_true = np.mean(labels.cpu().numpy(),axis=1)
        else:
            # ids = np.append(ids, i)
            res = np.concatenate((res, np.mean(outputs.detach().cpu().numpy(),axis=1)),axis=0)
            labels_true = np.concatenate((labels_true, np.mean(labels.cpu().numpy(),axis=1)),axis=0)  

    # combined_pred = np.column_stack((ids, res))
    # combined_true = np.column_stack((ids, labels_true))

    epoch_loss = np.sqrt(sum(losses) / len(losses))
    return res, labels_true, epoch_loss
    
