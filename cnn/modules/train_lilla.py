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

#-------------#
# CNN Trainer #
#-------------#

class CNNTrainer:
    def __init__(
        self,
        world_size: int,
        model: torch.nn.Module,
        nfeatures: int,
        train_ds: FiberDataset,
        valid_ds: FiberDataset,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
        batch_size: int,
    ) -> None:
        self.world_size = world_size
        self.gpu_id = gpu_id
        self.device = torch.device(f"cuda:{gpu_id}")
        self.nfeatures = nfeatures
        self.train_data = train_ds
        self.valid_data = valid_ds
        self.optimizer = optimizer
        self.save_every = save_every
        self.model = model
        self.criterion = nn.MSELoss()
        self.batch_size = batch_size
        self.ntrain = len(train_ds)//world_size
        self.nvalid = len(valid_ds)//world_size
        self.nbatch_train = self.ntrain//self.batch_size
        self.nbatch_valid = self.nvalid//self.batch_size
        self.logger = logging.getLogger('Trainer')
        
    def _set_tensors(self):
        '''
        Put dataset arrays on GPU for direct access in training
        '''
        # Initialize large arrays on GPU
        if self.gpu_id == 0:
            print("Initializing datasets on GPU...")
        self.img_train = torch.empty((self.ntrain, 1, 48, 48), dtype=torch.float, device=self.device)
        self.img_valid = torch.empty((self.nvalid, 1, 48, 48), dtype=torch.float, device=self.device)
        self.spec_train = torch.empty((self.ntrain, 1, 3, 64), dtype=torch.float, device=self.device)
        self.spec_valid = torch.empty((self.nvalid, 1, 3, 64), dtype=torch.float, device=self.device)
        self.fid_train = torch.empty((self.ntrain, self.nfeatures), dtype=torch.float, device=self.device)
        self.fid_valid = torch.empty((self.nvalid, self.nfeatures), dtype=torch.float, device=self.device)
        
        # Fill arrays with values
        start = self.gpu_id*self.ntrain
        if self.gpu_id == 0:
            print("Uploading training set to GPU...")
        prev_prog = 0
        for i in range(self.ntrain):
            i_db = start+i
            self.img_train[i] = self.train_data[i_db]['img']
            self.spec_train[i] = self.train_data[i_db]['spec']
            self.fid_train[i] = self.train_data[i_db]['fid_pars']
            if self.fid_train[i, 2] < 0:
                self.fid_train[i, 2] = self.fid_train[i, 2] + 1
            #self.fid_train[i, :3] = self.train_data[i_db]['fid_pars'][:3]
            #self.fid_train[i, 4:] = self.train_data[i_db]['fid_pars'][3:]
            prog = 100*i//self.ntrain
            if prog % 10 == 0 and prog > prev_prog and self.gpu_id == 0:
                prev_prog = prog
                print(f"{prog}% complete")
        #self.fid_train[:, 3] = torch.cos(np.pi*self.fid_train[:, 2])
        #self.fid_train[:, 2] = torch.sin(np.pi*self.fid_train[:, 2])
        self.fid_train[:, 2] = self.fid_train[:, 2]*2 - 1
        if self.gpu_id == 0:
            print(torch.max(self.fid_train[:, 2]), torch.min(self.fid_train[:, 2]))
        
        start = self.gpu_id*self.nvalid
        if self.gpu_id == 0:
            print("Uploading validation set to GPU...")
        prev_prog = 0
        for i in range(self.nvalid):
            i_db = start+i
            self.img_valid[i] = self.valid_data[i_db]['img']
            self.spec_valid[i] = self.valid_data[i_db]['spec']
            self.fid_valid[i] = self.valid_data[i_db]['fid_pars']
            if self.fid_valid[i, 2] < 0:
                self.fid_valid[i, 2] = self.fid_valid[i, 2] + 1
            #self.fid_valid[i, :3] = self.valid_data[i_db]['fid_pars'][:3]
            #self.fid_valid[i, 4:] = self.valid_data[i_db]['fid_pars'][3:]
            prog = 100*i//self.nvalid
            if prog % 10 == 0 and prog > prev_prog and self.gpu_id == 0:
                prev_prog = prog
                print(f"{prog}% complete")
        #self.fid_valid[:, 3] = torch.cos(np.pi*self.fid_valid[:, 2])
        #self.fid_valid[:, 2] = torch.sin(np.pi*self.fid_valid[:, 2])
        self.fid_valid[:, 2] = self.fid_valid[:, 2]*2 - 1
        
        torch.distributed.barrier()

    def _run_batch(self, img, spec, fid):
        self.optimizer.zero_grad()
        outputs = self.model(img, spec)
        #loss = (2*self.criterion(outputs[:, :2]**2, fid[:, :2]**2) + 3*self.criterion(outputs[:, 2:], fid[:, 2:]))
        loss = self.criterion(outputs, fid)
        loss.backward()                   
        self.optimizer.step()
        return loss

    def _run_epoch(self, epoch, show_log=True):
        
        train_loss = self._trainFunc(epoch)
        torch.distributed.barrier()
        torch.cuda.synchronize()
        valid_loss = self._validFunc(epoch)
        torch.distributed.barrier()
        torch.cuda.synchronize()
        
        return train_loss, valid_loss
    
    def _trainFunc(self, epoch, show_log=True):
        self.model.train()
        np.random.shuffle(self.train_order)
        losses = []
        epoch_start = time.time()
        
        for i in range(self.nbatch_train):
            start = i*self.batch_size
            batch_ids = self.train_order[start:start+self.batch_size]
            img = self.img_train[batch_ids]
            spec = self.spec_train[batch_ids]
            fid = self.fid_train[batch_ids]
            
            loss = self._run_batch(img, spec, fid)
            losses.append(loss.item())
            
            if show_log and self.gpu_id == 0 and i%100 == 0:
                #self.logger.info(f"Batch {i} complete")
                print(f"Batch {i} complete")

        epoch_loss = sum(losses) / len(losses)
        epoch_loss = np.sqrt(epoch_loss) # comment out if not using MSE
        epoch_time = time.time() - epoch_start
        if show_log and self.gpu_id == 0:
            #self.logger.info("[TRAIN] Epoch: {} Loss: {} Time: {:.0f}:{:.0f}".format(epoch+1, epoch_loss,
            #                                                                         epoch_time // 60, 
            #                                                                         epoch_time % 60))
            print("[TRAIN] Epoch: {} Loss: {} Time: {:.0f}:{:.0f}".format(epoch+1, epoch_loss,
                                                                          epoch_time // 60, 
                                                                          epoch_time % 60))
        return epoch_loss

    def _validFunc(self,epoch,show_log=True):
        self.model.eval()
        np.random.shuffle(self.valid_order)
        losses = []
        epoch_start = time.time()
        
        for i in range(self.nbatch_valid):
            start = i*self.batch_size
            batch_ids = self.valid_order[start:start+self.batch_size]
            img = self.img_valid[batch_ids]
            spec = self.spec_valid[batch_ids]
            fid = self.fid_valid[batch_ids]
            
            outputs = self.model(img, spec)
            loss = self.criterion(outputs, fid)
            losses.append(loss.item())
            
            if show_log and self.gpu_id == 0 and i%100 == 0:
                #self.logger.info(f"Batch {i} complete")
                print(f"Batch {i} complete")

        epoch_loss = sum(losses) / len(losses)
        epoch_loss = np.sqrt(epoch_loss) # comment out if not using MSE
        epoch_time = time.time() - epoch_start
        if show_log and self.gpu_id == 0:
            #self.logger.info("[VALID] Epoch: {} Loss: {} Time: {:.0f}:{:.0f}".format(epoch+1, epoch_loss,
            #                                                                         epoch_time // 60, 
            #                                                                         epoch_time % 60))
            print("[VALID] Epoch: {} Loss: {} Time: {:.0f}:{:.0f}".format(epoch+1, epoch_loss,
                                                                          epoch_time // 60, 
                                                                          epoch_time % 60))
        return epoch_loss
    
    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = join(config.train['model_path'], config.train['model_name'], config.train['model_name']+str(epoch))
        torch.save(ckp, PATH)

    def train(self, max_epochs: int):
        self._set_tensors()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.5, patience=5)
        train_losses = []
        valid_losses = []
        self.train_order = np.linspace(0, self.ntrain-1, self.ntrain, dtype=int)
        self.valid_order = np.linspace(0, self.nvalid-1, self.nvalid, dtype=int)
        if self.gpu_id == 0:
            print("Training start")
        for epoch in range(max_epochs):
            train_loss, valid_loss = self._run_epoch(epoch)
            scheduler.step(valid_loss)
            print(f"Current LR is {scheduler.get_last_lr()}")
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)
        losses = pd.DataFrame(np.vstack([train_losses, valid_losses]))
        model_name = config.train['model_name']
        losses.to_csv(join(config.train['model_path'], 'losses', f'losses_{model_name}.csv'), index=False)

#------------------#
# Global functions #
#------------------#

def train_nn(rank: int, world_size: int, Model=ForkCNN, Trainer=CNNTrainer, 
             save_every=1, total_epochs=50, batch_size=100, nfeatures=2):
    '''
    Main function to train any network.
    '''
    # Set parameters based on stage
    total_epochs = config.train['epoch_number']
    batch_size = config.train['batch_size']
    nfeatures = config.train['feature_number']
    
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger('Setup')
    if rank == 0:
        log.info('Initializing')
    
    ddp_setup(rank, world_size)
    log.info(f'[rank: {rank}] Successfully set up device')
    
    train_ds, valid_ds, model, optimizer = load_train_objs(nfeatures, batch_size, world_size, Model, rank)
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])
    log.info(f'[rank: {rank}] Successfully loaded training objects')
    
    #train_dl, valid_dl = prepare_dataloader(train_ds, valid_ds, batch_size, world_size)
    #log.info(f'[rank: {rank}] Successfully prepared dataloader')
    #torch.distributed.barrier()
    
    trainer = Trainer(world_size, model, nfeatures, train_ds, valid_ds, optimizer, rank, save_every, batch_size)
    log.info(f'[rank: {rank}] Successfully initialized Trainer')
    torch.distributed.barrier()
    trainer.train(total_epochs)
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

def load_train_objs(nfeatures, batch_size, nGPUs, Model, rank):
    # Create dataset objects
    train_ds = pxt.TorchDataset(config.data['data_dir'])
    valid_ds = pxt.TorchDataset(config.test['data_dir'])
    # Initialize model and optimizer
    model = Model(batch_size, GPUs=nGPUs)  # load your model
    optimizer = optim.SGD(model.parameters(), 
                          lr=config.train['initial_learning_rate'],
                          momentum=config.train['momentum'])
    return train_ds, valid_ds, model, optimizer

def prepare_dataloader(train_ds, valid_ds, batch_size, GPUs):
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
    )
    valid_dl = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=True,
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
        neg = batch['fid_pars'][:, 2] < 0
        fid = batch['fid_pars']
        fid[:, 2][neg] += 1
        fid[:, 2] = fid[:, 2]*2 - 1
        fid = fid.float().to(gpu_id)
        #cos_theta = np.cos(np.pi*fid[:, 2])
        #fid[:, 2] = np.sin(np.pi*fid[:, 2])
        #fid = np.insert(fid, 3, cos_theta, axis=1).float().view(-1, nfeatures).to(gpu_id)
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
    
