import numpy as np
from astropy.io import fits

import torch
from torch import optim, nn
from torch.utils.data import SubsetRandomSampler, DataLoader, Subset
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from .networks import ForkCNN, CaliNN
from .dataset import TrainDataset
import config

import sys,time,os
import config

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.model = DDP(model, device_ids=[gpu_id])
        self.criterion = nn.MSELoss()

    def _run_batch(self, img, spec, fid):
        self.optimizer.zero_grad()
        outputs = self.model.forward(img, spec)
        loss = self.criterion(output, fid)        
        loss = torch.sqrt(loss)           
        loss.backward()                   
        self.optimizer.step()
        return loss

    def _run_epoch(self, epoch, show_log=True):
        b_sz = len(next(iter(self.train_data))[0])
        losses = []
        epoch_start = time.time()
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        for i, batch in enumerate(self.train_data):
            img = batch['img'].float().to(self.gpu_id)
            spec = batch['spec'].float().to(self.gpu_id)
            fid = batch['fid_pars'].float().to(self.gpu_id)
            loss = self._run_batch(source, targets)
            losses.append(loss.item())
            
        epoch_loss = np.sqrt(sum(losses) / len(losses))
        epoch_time = time.time() - epoch_start
        if show_log:
            print("[TRAIN] Epoch: {} Loss: {} Time: {:.0f}:{:.0f}".format(epoch+1, epoch_loss,
                                                                          epoch_time // 60, 
                                                                          epoch_time % 60))
    
    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = join(config.train['model_path'], config.train['model_name'], str(epoch))
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)
                
    
def train_nn(rank: int, world_size: int, save_every=1, 
             total_epochs=config.train['epoch_number'], 
             batch_size=config.train['batch_size'], 
             nfeatures=config.train['feature_number'], f_valid=0.1):
    
    ddp_setup(rank, world_size)
    dataset, model, optimizer = load_train_objs(nfeatures)
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer, rank, save_every)
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

def load_train_objs(nfeatures):
    data_args = list(config.data.values())
    dataset = TrainDataset(*data_args)
    model = ForkCNN(nfeatures, self.batch_size, GPUs=1)  # load your model
    optimizer = optim.SGD(self.model.parameters(), 
                          lr=config.train['initial_learning_rate'], 
                          momentum=config.train['momentum'])
    return dataset, model, optimizer

def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )

#################################################################################

class TrainerNN:
    
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        gpu_id: int,
        save_every: int,
        f_valid: float,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.save_every = save_every
        self.model = DDP(model, device_ids=[gpu_id])
        self.f_valid = f_valid
        
    def _set_data(self, train_ds):
        '''
        Spilt the dataset into training and validation, and build dataloader.
        '''

        size = len(train_ds)
        indices = list(range(size))
        split = int(np.floor(self.f_valid * size))

        train_indices, valid_indices = indices[split:], indices[:split]
        train_sampler = DistributedSampler(train_indices)
        valid_sampler = DistributedSampler(valid_indices)

        self.train_dl = DataLoader(train_ds, 
                              batch_size=self.batch_size, 
                              num_workers=self.workers,
                              sampler=train_sampler)
        self.valid_dl = DataLoader(train_ds, 
                              batch_size=self.batch_size, 
                              num_workers=self.workers,
                              sampler=valid_sampler)
        print("Train_dl: {} Validation_dl: {}".format(len(self.train_dl), len(self.valid_dl)))
        
    def load_model(self,path=None,strict=True, assign=False):
        
        model = ForkCNN(self.features, self.batch_size, GPUs=1)
        model.to(self.device)
        if self.nGPUs > 1:
            model = DDP(model, device_ids=None)
        
        if path != None:
            model.load_state_dict(torch.load(path), strict=strict, assign=assign)
        
        return model
    
    def run(self, dataset, show_log=True):
        
        # set data loader here
        print("Setting up DDP...")
        self.ddp_setup(rank=0, world_size=1)
        
        print("Preparing Data Loader...")
        self._set_data(dataset)
        
        self.model = self.load_model()
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), 
                                   lr=config.train['initial_learning_rate'], 
                                   momentum=config.train['momentum'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')
        
        print('Begin training ...')
        
        # Loop the training and validation processes
        train_losses = []
        valid_losses = []
        for epoch in range(config.train['epoch_number']):
            self.train_dl.sampler.set_epoch(epoch)
            self.valid_dl.sampler.set_epoch(epoch)
            train_loss = self._trainFunc(epoch,show_log=show_log)
            valid_loss = self._validFunc(epoch,show_log=show_log)
            scheduler.step(train_loss)
            scheduler.get_last_lr()
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            
            if config.train['save_model']:
                if not os.path.exists(config.train['model_path']):
                    os.makedirs(config.train['model_path'])
                torch.save(self.model.state_dict(), 
                           config.train['model_path']+config.train['model_name']+str(epoch))
                
        if config.train['save_model']:
            hdu0 = fits.PrimaryHDU(train_losses)
            hdu1 = fits.ImageHDU(valid_losses)
            hdul = fits.HDUList([hdu0, hdu1])
            hdul.writeto(config.train['model_path']+'/training_loss.fits',overwrite=True)
                
        print('Finish training !')
        destroy_process_group()
        return train_losses, valid_losses
    
    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_batch(source, targets)

    def _trainFunc(self,epoch,show_log=True):
        self.model.train()
        losses = []
        epoch_start = time.time()
        for i, batch in enumerate(self.train_dl):
            inputs1, inputs2, labels = batch['gal_image'].float().to(self.device), \
                                       batch['psf_image'].float().to(self.device), \
                                       batch['label'].float().view(-1,self.features).to(self.device)

            self.optimizer.zero_grad()             
            outputs = self.model.forward(inputs1, inputs2)
            loss = self.criterion(outputs, labels) 
            losses.append(loss.item())        
            loss = torch.sqrt(loss)           
            loss.backward()                   
            self.optimizer.step()                  

        epoch_loss = np.sqrt(sum(losses) / len(losses))
        epoch_time = time.time() - epoch_start
        if show_log:
            print("[TRAIN] Epoch: {} Loss: {} Time: {:.0f}:{:.0f}".format(epoch+1, epoch_loss,
                                                                          epoch_time // 60, 
                                                                          epoch_time % 60))
        return epoch_loss

    def _validFunc(self,epoch,show_log=True):
        self.model.eval()
        losses = []
        epoch_start = time.time()
        for i, batch in enumerate(self.valid_dl):
            inputs1, inputs2, labels = batch['gal_image'].float().to(self.device), \
                                       batch['psf_image'].float().to(self.device), \
                                       batch['label'].float().view(-1,self.features).to(self.device)

            outputs = self.model.forward(inputs1, inputs2)
            loss = self.criterion(outputs, labels)
            losses.append(loss.item())

        epoch_loss = np.sqrt(sum(losses) / len(losses))
        epoch_time = time.time() - epoch_start
        if show_log:
            print("[VALID] Epoch: {} Loss: {} Time: {:.0f}:{:.0f}".format(epoch+1, epoch_loss,
                                                                        epoch_time // 60, 
                                                                        epoch_time % 60))
        return epoch_loss
    
    def _predictFunc(self,test_dl,MODEL,criterion=nn.MSELoss()):

        MODEL.eval()
        losses=[]
        for i, batch in enumerate(test_dl):
            inputs1, inputs2 = batch['gal_image'].float().to(self.device), \
                                       batch['psf_image'].float().to(self.device)
            outputs = MODEL.forward(inputs1, inputs2)
            labels_true_batch = batch['label'].float().view(-1,self.features).to(self.device)
            loss = criterion(outputs, labels_true_batch)
            losses.append(loss.item())
            if i == 0:
                ids = batch['id'].numpy()
                labels = outputs.detach().cpu().numpy()
                labels_true = labels_true_batch.cpu()
                snr = batch['snr'].numpy()
            else:
                ids = np.concatenate((ids, batch['id'].numpy()))
                labels = np.vstack((labels, outputs.detach().cpu().numpy()))
                labels_true = np.vstack((labels_true, labels_true_batch.cpu()))  
                snr = np.concatenate((snr, batch['snr'].numpy()))

        combined_pred = np.column_stack((ids, labels))
        combined_true = np.column_stack((ids, labels_true))
        combined_snr = np.column_stack((ids, snr))

        epoch_loss = np.sqrt(sum(losses) / len(losses))
        return combined_pred, combined_true, combined_snr, epoch_loss
    
    
##################################################
    
    
class MSBLoss(nn.Module):
    def __init__(self):
        super(MSBLoss, self).__init__()
        
    def forward(self,x,y):
        
        if torch.std(y,axis=1).any() != 0:
            print('Waring!')
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
    

