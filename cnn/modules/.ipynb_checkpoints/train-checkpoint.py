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
import config

"""
Module that manages all the trainer classes and testing functions. 
Need to create a wrapper trainer class eventually
"""

#-------------#
# CNN Trainer #
#-------------#

class CNNTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        nfeatures: int,
        train_dl: DataLoader,
        valid_dl: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
        batch_size: int,
    ) -> None:
        self.gpu_id = gpu_id
        self.nfeatures = nfeatures
        self.train_data = train_dl
        self.valid_data = valid_dl
        self.optimizer = optimizer
        self.save_every = save_every
        #self.model = DDP(model, device_ids=[gpu_id])
        self.model = model
        self.criterion = nn.MSELoss()
        self.batch_size = batch_size
        self.logger = logging.getLogger('Trainer')

    def _run_batch(self, img, spec, fid):
        self.optimizer.zero_grad()
        outputs = self.model(img, spec)
        loss = self.criterion(outputs, fid)      
        loss.backward()                   
        self.optimizer.step()
        return loss

    def _run_epoch(self, epoch, show_log=True):
        
        self.train_data.sampler.set_epoch(epoch)
        self.valid_data.sampler.set_epoch(epoch)
        train_loss = self._trainFunc(epoch)
        valid_loss = self._validFunc(epoch)
        
        return train_loss, valid_loss
    
    def _trainFunc(self,epoch,show_log=True):
        self.model.train()
        losses = []
        epoch_start = time.time()
        for i, batch in enumerate(self.train_data):
            img = batch['img'].float().to(self.gpu_id)
            spec = batch['spec'].float().to(self.gpu_id)
            fid = batch['fid_pars'].float().view(-1,self.nfeatures).to(self.gpu_id)
            loss = self._run_batch(img, spec, fid)
            losses.append(loss.item())
            if show_log and self.gpu_id == 0 and i%100 == 0:
                self.logger.info(f"Batch {i} complete")

        epoch_loss = sum(losses) / len(losses)
        epoch_loss = np.sqrt(epoch_loss) # comment out if not using MSE
        epoch_time = time.time() - epoch_start
        if show_log and self.gpu_id == 0:
            self.logger.info("[TRAIN] Epoch: {} Loss: {} Time: {:.0f}:{:.0f}".format(epoch+1, epoch_loss,
                                                                                     epoch_time // 60, 
                                                                                     epoch_time % 60))
        return epoch_loss

    def _validFunc(self,epoch,show_log=True):
        self.model.eval()
        losses = []
        epoch_start = time.time()
        for i, batch in enumerate(self.valid_data):
            img = batch['img'].float().to(self.gpu_id)
            spec = batch['spec'].float().to(self.gpu_id)
            fid = batch['fid_pars'].float().view(-1,self.nfeatures).to(self.gpu_id)
            outputs = self.model(img, spec)
            loss = self.criterion(outputs, fid)
            losses.append(loss.item())

        epoch_loss = sum(losses) / len(losses)
        epoch_loss = np.sqrt(epoch_loss) # comment out if not using MSE
        epoch_time = time.time() - epoch_start
        if show_log and self.gpu_id == 0:
            self.logger.info("[VALID] Epoch: {} Loss: {} Time: {:.0f}:{:.0f}".format(epoch+1, epoch_loss,
                                                                                     epoch_time // 60, 
                                                                                     epoch_time % 60))
        return epoch_loss
    
    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = join(config.train['model_path'], config.train['model_name'], config.train['model_name']+str(epoch))
        torch.save(ckp, PATH)

    def train(self, max_epochs: int):
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', verbose=True)
        train_losses = []
        valid_losses = []
        for epoch in range(max_epochs):
            train_loss, valid_loss = self._run_epoch(epoch)
            scheduler.step(train_loss)
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)
        losses = pd.DataFrame(np.vstack([train_losses, valid_losses]))
        model_name = config.train['model_name']
        losses.to_csv(join(config.train['model_path'], 'losses', f'losses_{model_name}.csv'), index=False)
                
#----------------#
# Deconv Trainer #
#----------------#

class DeconvTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        nfeatures: int,
        train_dl: DataLoader,
        valid_dl: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
        batch_size: int,
    ) -> None:
        self.gpu_id = gpu_id
        self.nfeatures = nfeatures
        self.train_data = train_dl
        self.valid_data = valid_dl
        self.optimizer = optimizer
        self.save_every = save_every
        self.model = model
        self.criterion = nn.MSELoss()
        self.batch_size = batch_size
        self.logger = logging.getLogger('Trainer')

    def _run_batch(self, img, spec, fid):
        self.optimizer.zero_grad()
        outputs = self.model(fid)
        loss = self.criterion(outputs, img)      
        loss.backward()                   
        self.optimizer.step()
        return loss

    def _run_epoch(self, epoch, show_log=True):
        
        self.train_data.sampler.set_epoch(epoch)
        self.valid_data.sampler.set_epoch(epoch)
        train_loss = self._trainFunc(epoch)
        valid_loss = self._validFunc(epoch)
        
        return train_loss, valid_loss
    
    def _trainFunc(self,epoch,show_log=True):
        self.model.train()
        losses = []
        epoch_start = time.time()
        for i, batch in enumerate(self.train_data):
            img = batch['img'].float().to(self.gpu_id)
            spec = batch['spec'].float().to(self.gpu_id)
            fid = batch['fid_pars'].float().view(-1,self.nfeatures).to(self.gpu_id)
            loss = self._run_batch(img, spec, fid)
            losses.append(loss.item())
            if show_log and self.gpu_id == 0 and i%100 == 0:
                self.logger.info(f"Batch {i} complete")

        epoch_loss = sum(losses) / len(losses)
        epoch_loss = np.sqrt(epoch_loss) # comment out if not using MSE
        epoch_time = time.time() - epoch_start
        if show_log and self.gpu_id == 0:
            self.logger.info("[TRAIN] Epoch: {} Loss: {} Time: {:.0f}:{:.0f}".format(epoch+1, epoch_loss,
                                                                                     epoch_time // 60, 
                                                                                     epoch_time % 60))
        return epoch_loss

    def _validFunc(self,epoch,show_log=True):
        self.model.eval()
        losses = []
        epoch_start = time.time()
        for i, batch in enumerate(self.valid_data):
            img = batch['img'].float().to(self.gpu_id)
            spec = batch['spec'].float().to(self.gpu_id)
            fid = batch['fid_pars'].float().view(-1,self.nfeatures).to(self.gpu_id)
            outputs = self.model(fid)
            loss = self.criterion(outputs, img)
            losses.append(loss.item())

        epoch_loss = sum(losses) / len(losses)
        epoch_loss = np.sqrt(epoch_loss) # comment out if not using MSE
        epoch_time = time.time() - epoch_start
        if show_log and self.gpu_id == 0:
            self.logger.info("[VALID] Epoch: {} Loss: {} Time: {:.0f}:{:.0f}".format(epoch+1, epoch_loss,
                                                                                     epoch_time // 60, 
                                                                                     epoch_time % 60))
        return epoch_loss
    
    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = join(config.train['model_path'], config.train['model_name']+str(epoch))
        torch.save(ckp, PATH)

    def train(self, max_epochs: int):
        train_losses = []
        valid_losses = []
        for epoch in range(max_epochs):
            train_loss, valid_loss = self._run_epoch(epoch)
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)
        losses = pd.DataFrame(np.vstack([train_losses, valid_losses]))
        model_name = config.train['model_name']
        losses.to_csv(join(config.train['model_path'], f'losses_{model_name}.csv'), index=False)

#-----------------------------#
# Calibration Network Trainer #
#-----------------------------#

class CaliTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        nfeatures: int,
        train_dl: DataLoader,
        valid_dl: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
        batch_size: int,
    ) -> None:
        self.gpu_id = gpu_id
        self.nfeatures = nfeatures
        self.train_data = train_dl
        self.valid_data = valid_dl
        self.optimizer = optimizer
        self.save_every = save_every
        self.model = model
        self.criterion = MSBLoss()
        self.batch_size = batch_size
        self.history = dict(train=[], valid=[])
        self.logger = logging.getLogger('Trainer')

    def _run_batch(self, img, spec, fid):
        self.optimizer.zero_grad()
        outputs = self.model(img, spec)
        loss = self.criterion(outputs, fid)      
        loss.backward()                   
        self.optimizer.step()
        return loss

    def _run_epoch(self, epoch, show_log=True):
        
        self.train_data.sampler.set_epoch(epoch)
        self.valid_data.sampler.set_epoch(epoch)
        train_loss = self._trainFunc(epoch)
        valid_loss = self._validFunc(epoch)
        
        return train_loss, valid_loss
    
    def _trainFunc(self,epoch,show_log=True):
        self.model.train()
        losses = []
        epoch_start = time.time()
        for i, batch in enumerate(self.train_data):
            img = batch['img'].float().to(self.gpu_id)
            spec = batch['spec'].float().to(self.gpu_id)
            fid = batch['fid_pars'].float().view(-1,self.nfeatures).to(self.gpu_id)
            loss = self._run_batch(img, spec, fid)
            losses.append(loss.item())
            if show_log and self.gpu_id == 0 and i%100 == 0:
                self.logger.info(f"Batch {i} complete")

        epoch_loss = sum(losses) / len(losses)
        epoch_loss = np.sqrt(epoch_loss) # comment out if not using MSE
        epoch_time = time.time() - epoch_start
        if show_log and self.gpu_id == 0:
            self.logger.info("[TRAIN] Epoch: {} Loss: {} Time: {:.0f}:{:.0f}".format(epoch+1, epoch_loss,
                                                                                     epoch_time // 60, 
                                                                                     epoch_time % 60))
        return epoch_loss

    def _validFunc(self,epoch,show_log=True):
        self.model.eval()
        losses = []
        epoch_start = time.time()
        for i, batch in enumerate(self.valid_data):
            img = batch['img'].float().to(self.gpu_id)
            spec = batch['spec'].float().to(self.gpu_id)
            fid = batch['fid_pars'].float().view(-1,self.nfeatures).to(self.gpu_id)
            outputs = self.model(img, spec)
            loss = self.criterion(outputs, fid)
            losses.append(loss.item())

        epoch_loss = sum(losses) / len(losses)
        epoch_loss = np.sqrt(epoch_loss) # comment out if not using MSE
        epoch_time = time.time() - epoch_start
        if show_log and self.gpu_id == 0:
            self.logger.info("[VALID] Epoch: {} Loss: {} Time: {:.0f}:{:.0f}".format(epoch+1, epoch_loss,
                                                                                     epoch_time // 60, 
                                                                                     epoch_time % 60))
        return epoch_loss
    
    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = join(config.train['model_path'], config.train['model_name'], config.train['model_name']+str(epoch))
        torch.save(ckp, PATH)

    def train(self, max_epochs: int):
        train_losses = []
        valid_losses = []
        for epoch in range(max_epochs):
            train_loss, valid_loss = self._run_epoch(epoch)
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)
        losses = pd.DataFrame(np.vstack([train_losses, valid_losses]))
        model_name = config.train['model_name']
        losses.to_csv(join(config.train['model_path'], 'losses', f'losses_{model_name}.csv'), index=False)
                
    # Function necessary for L-BFGS training to work per the docs
    def closure():
        lbfgs.zero_grad()
        loss = self.criterion(x_lbfgs)
        loss.backward()
        return loss


#------------------#
# Global functions #
#------------------#

def train_nn(rank: int, world_size: int, stage='train', Model=ForkCNN, Trainer=CNNTrainer, 
             save_every=1, total_epochs=50, batch_size=100, nfeatures=2):
    '''
    Main function to train any network. stage, Model, and Trainer should be specified.
    stages can be 'train', 'cali' or 'weights (WIP)'
    '''
    # Set parameters based on stage
    eval(f"total_epochs=config.{stage}['epoch_number']")
    eval(f"batch_size=config.{stage}['batch_size']")
    eval(f"nfeatures=config.{stage}['feature_number']")
    
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger('Setup')
    if rank == 0:
        log.info('Initializing')
    
    ddp_setup(rank, world_size)
    log.info(f'[rank: {rank}] Successfully set up device')
    
    eval(f"train_ds, valid_ds, model, optimizer = load_{stage}_objs(nfeatures, batch_size, world_size, Model)")
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])
    log.info(f'[rank: {rank}] Successfully loaded training objects')
    
    train_dl, valid_dl = prepare_dataloader(train_ds, valid_ds, batch_size, world_size)
    log.info(f'[rank: {rank}] Successfully prepared dataloader')
    #torch.distributed.barrier()
    
    trainer = Trainer(model, nfeatures, train_dl, valid_dl, optimizer, rank, save_every, batch_size)
    log.info(f'[rank: {rank}] Successfully initialized Trainer')
    #torch.distributed.barrier()
    trainer.train(total_epochs)
    destroy_process_group()
    
def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.synchronize()

def load_train_objs(nfeatures, batch_size, GPUs, Model):
    # Create dataset objects
    train_ds = pxt.TorchDataset(config.data['data_dir'])
    valid_ds = pxt.TorchDataset(config.test['data_dir'])
    # Initialize model and optimizer
    model = Model(batch_size, GPUs=GPUs)  # load your model
    optimizer = optim.SGD(model.parameters(), 
                          lr=config.train['initial_learning_rate'],
                          momentum=config.train['momentum'])
    return train_ds, valid_ds, model, optimizer

def load_cali_objs(nfeatures, batch_size, GPUs, Model):
    # Create dataset objects
    train_ds = pxt.TorchDataset(config.cali['train_dir'])
    valid_ds = pxt.TorchDataset(config.cali['test_dir'])
    # Initialize model and optimizer
    model = Model(batch_size, GPUs=GPUs)
    optimizer = optim.LBFGS(model.parameters(),
                            lr=config.cali['learning_rate'],
                            history_size=10,
                            max_iter=4, 
                            line_search_fn="strong_wolfe")
    return train_ds, valid_ds, model, optimizer

def prepare_dataloader(train_ds, valid_ds, batch_size, GPUs):
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        pin_memory=True,
        sampler=DistributedSampler(train_ds),
    )
    valid_dl = DataLoader(
        valid_ds,
        batch_size=batch_size,
        pin_memory=True,
        sampler=DistributedSampler(valid_ds),
    )
    return train_dl, valid_dl

def load_model(Model=ForkCNN, path=None, strict=True, assign=False, GPUs=1):

    model = Model(config.train['batch_size'], GPUs=GPUs)
    model.to(0)
    if GPUs > 1:
        model = DDP(model, device_ids=None)

    if path != None:
        model.load_state_dict(torch.load(path), strict=strict, assign=assign)

    return model

def predict(nfeatures, test_data, model, criterion=nn.MSELoss(), gpu_id=0):
    '''
    Run this function to test performance of trained models
    '''
    model.eval()
    losses=[]
    for i, batch in enumerate(test_data):
        img = batch['img'].float().to(gpu_id)
        spec = batch['spec'].float().to(gpu_id)
        fid = batch['fid_pars'].float().view(-1, nfeatures).to(gpu_id)
        outputs = model(img, spec)
        loss = criterion(outputs, fid)
        losses.append(loss.item())
        if i == 0:
            ids = batch['id'].numpy()
            preds = outputs.detach().cpu().numpy()
            fids = fid.cpu().numpy()
        else:
            ids = np.concatenate((ids, batch['id'].numpy()))
            preds = np.vstack((preds, outputs.detach().cpu().numpy()))
            fids = np.vstack((fids, fid.cpu().numpy()))

    combined_pred = np.column_stack((ids, preds))
    combined_fid = np.column_stack((ids, fids))

    epoch_loss = sum(losses) / len(losses)
    epoch_loss = np.sqrt(epoch_loss) # comment out if not using MSE

    return combined_pred, combined_fid, epoch_loss

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
    

