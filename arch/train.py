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
import normflows as nf
from normflows.nets.mlp import MLP
from nflows.flows.base import Flow
from nflows.distributions.normal import ConditionalDiagonalNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation
from nflows.nn.nets import ResidualNet

from networks import *
from dataset import *
from utils import *
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
        self.log_rank = 0
        self.device = torch.device(f"cuda:{gpu_id}")
        self.nfeatures = nfeatures
        self.train_data = train_ds
        self.valid_data = valid_ds
        self.optimizer = optimizer
        self.save_every = save_every
        self.model = model
        self.batch_size = batch_size
        self.ntrain = len(train_ds)//world_size
        self.nvalid = len(valid_ds)//world_size
        self.nbatch_train = self.ntrain//self.batch_size
        self.nbatch_valid = self.nvalid//self.batch_size
        self.transform_to_gal = config.train.get('transform_to_gal', False)
        self.logger = logging.getLogger('Trainer')
        self.sigma = config.sigma
        
    def _set_tensors(self):
        '''
        Put dataset arrays on GPU for direct access in training
        '''
        # Initialize large arrays on GPU
        if self.gpu_id == self.log_rank:
            self.logger.info("Setting up tensors on GPU")
        self.img_train = torch.empty((self.ntrain, 1, 48, 48), dtype=torch.float, device=self.device)
        self.img_valid = torch.empty((self.nvalid, 1, 48, 48), dtype=torch.float, device=self.device)
        self.spec_train = torch.empty((self.ntrain, 1, 5, 64), dtype=torch.float, device=self.device)
        self.spec_valid = torch.empty((self.nvalid, 1, 5, 64), dtype=torch.float, device=self.device)
        self.fid_train = torch.empty((self.ntrain, self.nfeatures), dtype=torch.float, device=self.device)
        self.fid_valid = torch.empty((self.nvalid, self.nfeatures), dtype=torch.float, device=self.device)
        
        # Fill arrays with values
        start = self.gpu_id*self.ntrain
        if self.gpu_id == self.log_rank:
            self.logger.info("Uploading training set to GPU...")
        prev_prog = 0
        for i in range(self.ntrain):
            i_db = start+i
            self.img_train[i] = self.train_data[i_db]['img']
            self.spec_train[i] = self.train_data[i_db]['spec']
            self.fid_train[i] = self.train_data[i_db]['fid_pars'][:self.nfeatures]
            if config.train.get('mode', 1) == 2:
                self.fid_train[i, 2] = self.train_data[i_db]['fid_pars'][5]

            prog = 100*i//self.ntrain
            if prog % 10 == 0 and prog > prev_prog and self.gpu_id == self.log_rank:
                prev_prog = prog
                self.logger.info(f"{prog}% complete")
        
        start = self.gpu_id*self.nvalid
        if self.gpu_id == self.log_rank:
            self.logger.info("Uploading validation set to GPU...")
        prev_prog = 0
        for i in range(self.nvalid):
            i_db = start+i
            self.img_valid[i] = self.valid_data[i_db]['img']
            self.spec_valid[i] = self.valid_data[i_db]['spec']
            self.fid_valid[i] = self.valid_data[i_db]['fid_pars'][:self.nfeatures]
            if config.train.get('mode', 1) == 2:
                self.fid_valid[i, 2] = self.valid_data[i_db]['fid_pars'][5]

            prog = 100*i//self.nvalid
            if prog % 10 == 0 and prog > prev_prog and self.gpu_id == self.log_rank:
                prev_prog = prog
                self.logger.info(f"{prog}% complete")
        
        torch.distributed.barrier()
        
    def _apply_noise(self, data, snr):
        return apply_noise(data, snr, device=self.device, use_iterative=True)

    def _run_batch(self, img, spec, fid):
        self.optimizer.zero_grad()
        loss = self.model(img, spec, fid)
        if ~(torch.isnan(loss) | torch.isinf(loss)):
            loss.backward()                   
            self.optimizer.step()
        return loss

    def _run_epoch(self, epoch, show_log=True):
        
        if self.gpu_id == self.log_rank:
            self.logger.info(f'Starting epoch {epoch}')
            
        self.SNR_train = self.generate_snr(size=self.ntrain, mode='rmag')
        self.SNR_valid = self.generate_snr(size=self.nvalid, mode='rmag')
        
        if self.gpu_id == self.log_rank:
            self.logger.info(f'Randomized SNR and noise for epoch {epoch}')

        train_loss, train_nans, train_infs = self._trainFunc(epoch)
        torch.distributed.barrier()
        torch.cuda.synchronize()
        valid_loss, valid_nans, valid_infs = self._validFunc(epoch)
        torch.distributed.barrier()
        torch.cuda.synchronize()

        return train_loss, train_nans, train_infs, valid_loss, valid_nans, valid_infs

    def _trainFunc(self, epoch, show_log=True):
        self.model.train()
        np.random.shuffle(self.train_order)
        losses = []
        epoch_start = time.time()
        nans = 0
        infs = 0
        
        for i in range(self.nbatch_train):
            start = i*self.batch_size
            batch_ids = self.train_order[start:start+self.batch_size]
            snr = self.SNR_train[batch_ids]
            img = self._apply_noise(self.img_train[batch_ids], snr)
            spec = self._apply_noise(self.spec_train[batch_ids], snr)
            # img = self.img_train[batch_ids]
            # spec = self.spec_train[batch_ids]
            fid = self.fid_train[batch_ids]
            
            loss = self._run_batch(img, spec, fid)
            if ~(torch.isnan(loss) | torch.isinf(loss)):
                losses.append(loss.item())
            elif torch.isnan(loss):
                nans += 1
            elif torch.isinf(loss):
                infs += 1

            if show_log and self.gpu_id == self.log_rank and i%100 == 0:
                self.logger.info(f"Batch {i} complete")

        epoch_loss = sum(losses) / len(losses)
        # epoch_loss = np.sqrt(epoch_loss) # comment out if not using MSE
        epoch_time = time.time() - epoch_start
        if show_log and self.gpu_id == self.log_rank:
            self.logger.info("[TRAIN] Epoch: {} Loss: {} Time: {:.0f}:{:.0f}".format(epoch+1, epoch_loss,
                                                                                    epoch_time // 60, 
                                                                                    epoch_time % 60))
        return epoch_loss, nans, infs

    def _validFunc(self,epoch,show_log=True):
        self.model.eval()
        np.random.shuffle(self.valid_order)
        losses = []
        epoch_start = time.time()
        nans = 0
        infs = 0
        with torch.no_grad():
            for i in range(self.nbatch_valid):
                start = i*self.batch_size
                batch_ids = self.valid_order[start:start+self.batch_size]
                snr = self.SNR_valid[batch_ids]
                img = self._apply_noise(self.img_valid[batch_ids], snr)
                spec = self._apply_noise(self.spec_valid[batch_ids], snr)
                # img = self.img_valid[batch_ids]
                # spec = self.spec_valid[batch_ids]
                fid = self.fid_valid[batch_ids]
                
                loss = self.model(img, spec, fid)
                if ~(torch.isnan(loss) | torch.isinf(loss)):
                    losses.append(loss.item())
                elif torch.isnan(loss):
                    nans += 1
                elif torch.isinf(loss):
                    infs += 1

                if show_log and self.gpu_id == self.log_rank and i%100 == 0:
                    self.logger.info(f"Batch {i} complete")

        epoch_loss = sum(losses) / len(losses)
        epoch_time = time.time() - epoch_start
        if show_log and self.gpu_id == self.log_rank:
            self.logger.info("[VALID] Epoch: {} Loss: {} Time: {:.0f}:{:.0f}".format(epoch+1, epoch_loss,
                                                                                    epoch_time // 60, 
                                                                                    epoch_time % 60))
        return epoch_loss, nans, infs
    
    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = join(config.train['model_path'], config.train['model_name'], config.train['model_name']+str(epoch))
        torch.save(ckp, PATH)

    def train(self, max_epochs: int):
        self._set_tensors()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.5, patience=10)
        train_losses = []
        valid_losses = []
        train_nans_list = []
        train_infs_list = []
        valid_nans_list = []
        valid_infs_list = []
        self.train_order = np.linspace(0, self.ntrain-1, self.ntrain, dtype=int)
        self.valid_order = np.linspace(0, self.nvalid-1, self.nvalid, dtype=int)
        if self.gpu_id == self.log_rank:
            self.logger.info("Training start")
        for epoch in range(max_epochs):
            train_loss, train_nans, train_infs, valid_loss, valid_nans, valid_infs = self._run_epoch(epoch)
            scheduler.step(valid_loss)
            if self.gpu_id == self.log_rank:
                self.logger.info(f"Current LR is {scheduler.get_last_lr()}")
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            train_nans_list.append(train_nans)
            train_infs_list.append(train_infs)
            valid_nans_list.append(valid_nans)
            valid_infs_list.append(valid_infs)
            if self.gpu_id == self.log_rank and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)
        losses = pd.DataFrame(np.vstack([train_losses, valid_losses]))
        model_name = config.train['model_name']
        losses_dir = join(config.train['model_path'], 'losses')
        os.makedirs(losses_dir, exist_ok=True)
        losses.to_csv(join(losses_dir, f'losses_{model_name}.csv'), index=False)
        nans_infs = pd.DataFrame(np.hstack([train_nans, train_infs, valid_nans, valid_infs]))
        nans_infs.to_csv(join(losses_dir, f'nans_infs_{model_name}.csv'), index=False)

    def generate_snr(self, size, mode='uniform', **kwargs):
        if mode == 'uniform':
            min_snr = kwargs.get('min', 10)
            max_snr = kwargs.get('max', 1000)
            return torch.rand((size,), device=self.device)* (max_snr - min_snr) + min_snr
        elif mode == 'rmag':
            min_rmag = kwargs.get('min', 16)
            max_rmag = kwargs.get('max', 23)
            base = np.random.power(2, size=size)
            rmag = base*(max_rmag - min_rmag) + min_rmag
            flux_factor = 10**(0.4*(23.4-rmag))
            snr = flux_factor*5
            return torch.from_numpy(snr).float().to(self.device)
        else:
            raise ValueError("Invalid SNR generation mode")

#------------------#
# Global functions #
#------------------#

def train_nn(rank: int, world_size: int, Model=ForkCNN, Trainer=CNNTrainer,
             save_every=1, total_epochs=50, batch_size=100, nfeatures=2):
    '''
    Main function to train any network.
    '''
    # Set parameters based on stage
    mode = config.train['mode']
    total_epochs = config.train['epoch_number']
    batch_size = config.train['batch_size']
    nfeatures = config.train['feature_number']
    epoch = config.train['pretrain_from'] if config.train['use_pretrain'] is True else None
    
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger('Setup')
    if rank == 0:
        log.info('Initializing')
    
    ddp_setup(rank, world_size)
    log.info(f'[rank: {rank}] Successfully set up device')

    train_ds, valid_ds, model, optimizer = load_train_objs(mode, nfeatures, batch_size, world_size, Model, rank, epoch, log=log)
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])
    log.info(f'[rank: {rank}] Successfully loaded training objects')
    
    #train_dl, valid_dl = prepare_dataloader(train_ds, valid_ds, batch_size, world_size)
    #log.info(f'[rank: {rank}] Successfully prepared dataloader')
    #torch.distributed.barrier()
    
    os.makedirs(join(config.train['model_path'], config.train['model_name']), exist_ok=True)
    trainer = Trainer(world_size, model, nfeatures, train_ds, valid_ds, optimizer, rank, save_every, batch_size)
    log.info(f'[rank: {rank}] Successfully initialized Trainer')
    torch.distributed.barrier()
    trainer.train(total_epochs)
    torch.distributed.barrier()
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

# def setup_flows():
#     # Define flows
#     K = 4

#     latent_size = config.train['feature_number']
#     hidden_units = 64
#     num_blocks = 2
#     context_size = 1024

#     flows = []
#     for i in range(K):
#         flows += [nf.flows.MaskedAffineAutoregressive(latent_size, hidden_units, 
#                                                       context_features=context_size, 
#                                                       num_blocks=num_blocks)]
#         flows += [nf.flows.LULinearPermute(latent_size)]

#     # Set base distribution
#     context_encoder = MLP([context_size, 128, 64, latent_size*2],)
#     q0 = nf.distributions.base.ConditionalDiagGaussian(latent_size, context_encoder)
#     # q0 = nf.distributions.base.Uniform(2, low=-1.5, high=1.5)

#     return q0, flows
    
def setup_flows():
    # Define flows
    num_layers = config.flow['num_layers']
    n_features = config.train['feature_number']
    hidden_units = 64
    num_blocks = 2
    context_size = 1024
    
    # Set base distribution
    base = ConditionalDiagonalNormal(shape=[n_features], 
                                     context_encoder=MLP([context_size, 128, 64, n_features*2],))

    transforms = []
    for i in range(num_layers):
        transforms.append(ReversePermutation(features=n_features))
        transforms.append(MaskedAffineAutoregressiveTransform(features=n_features, 
                                                              hidden_features=hidden_units, 
                                                              context_features=context_size))

    transform = CompositeTransform(transforms)

    return base, transform

def load_train_objs(mode, nfeatures, batch_size, nGPUs, Model, rank, epoch=None, log=None,**kwargs):
    # Create dataset objects
    train_ds = pxt.TorchDataset(config.data['data_dir'])
    valid_ds = pxt.TorchDataset(config.test['data_dir'])
    # Initialize model and optimizer
    if epoch is not None: # if epoch is specified, load pretrained model
        strict = False if mode == 2 else True
        model_dir = config.train['model_path'] + config.train['pretrained_name'] + '/' + config.train['pretrained_name'] + str(epoch)
        model = load_model(mode=mode, path=model_dir,strict=strict, assign=True)
        if rank == 0:
            if log is not None:
                log.info(f"Loaded model {config.train['pretrained_name']} at epoch {epoch}")
    else:
        model = Model(mode, **kwargs)  # initialize new model
        if rank == 0:
            if log is not None:
                log.info(f"Loaded new model {config.train['model_name']}")
    # optimizer = optim.SGD(model.parameters(), 
    #                       lr=config.train['initial_learning_rate'],
    #                       momentum=config.train['momentum'])
    optimizer = optim.AdamW(model.parameters(), 
                            lr=config.train['initial_learning_rate'], 
                            weight_decay=config.train['weight_decay'])

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

def load_model(mode=1, Model=ForkCNN, path=None, strict=True, assign=False, GPUs=1, device='cpu'):

    model = Model(mode)
    model.to(device)

    if GPUs > 1:
        model = DDP(model, device_ids=None)

    if path != None:
        model.load_state_dict(torch.load(path, weights_only=False, map_location=torch.device(device)), strict=strict, assign=assign)

    return model

def _noise_scale_from_seg(data, snr, seg, eps=1e-8):
    snr = torch.clamp(snr, min=eps)
    npix = torch.sum(seg, dim=(-1, -2, -3)).float()
    npix = torch.clamp(npix, min=1.0)
    signal = torch.sum(data*seg.float(), dim=(-1, -2, -3))
    return signal/(torch.sqrt(npix)*snr)


def _estimate_noise_rms(noise, seg, eps=1e-8):
    # Prefer background pixels to estimate RMS; fall back to full-frame RMS if needed.
    bkg = (~seg).float()
    bkg_count = torch.sum(bkg, dim=(-1, -2, -3))
    bkg_count_safe = torch.clamp(bkg_count, min=1.0)
    bkg_rms = torch.sqrt(torch.sum((noise*bkg)**2, dim=(-1, -2, -3))/bkg_count_safe)
    full_rms = torch.sqrt(torch.mean(noise**2, dim=(-1, -2, -3)))
    return torch.where(bkg_count > eps, bkg_rms, full_rms)


def apply_noise(data, snr, randgen=None, device='cpu', use_iterative=True,
                base_signal_frac=0.1, threshold_sigma=1.5, eps=1e-8):
    """
    Apply Gaussian noise to batched data using an SNR-controlled scale.

    When use_iterative=True, a second pass refines the segmentation threshold to
    threshold_sigma * estimated_noise_rms for a more realistic signal mask.
    """
    if randgen is None:
        noise = torch.randn(data.size(), device=device)
    else:
        noise = torch.randn(data.size(), device=device, generator=randgen)

    maxs = torch.amax(data, dim=(-1, -2, -3))
    seg_coarse = data > (base_signal_frac*maxs).view(-1, 1, 1, 1)
    factor_coarse = _noise_scale_from_seg(data, snr, seg_coarse, eps=eps)

    if not use_iterative:
        return data + noise*factor_coarse.view(-1, 1, 1, 1)

    coarse_noise = noise*factor_coarse.view(-1, 1, 1, 1)
    coarse_rms = _estimate_noise_rms(coarse_noise, seg_coarse, eps=eps)
    refined_threshold = threshold_sigma*coarse_rms
    seg_refined = data > refined_threshold.view(-1, 1, 1, 1)
    factor_refined = _noise_scale_from_seg(data, snr, seg_refined, eps=eps)
    return data + noise*factor_refined.view(-1, 1, 1, 1)

def predict(nfeatures, test_data, model, batch_size=100, criterion=nn.MSELoss(), device='cpu'):
    '''
    Run this function to test performance of trained models
    '''
    model.eval()
    losses=[]
    with torch.no_grad():
        for i, batch in enumerate(test_data):
            snr = torch.rand((batch_size,), device=device)*990 + 10
            # snr = torch.full((batch_size,), 900., device=0)
            img = apply_noise(batch['img'].float().to(device), snr, device=device)
            spec = apply_noise(batch['spec'].float().to(device), snr, device=device)
            img = batch['img'].float().to(device)
            spec = batch['spec'].float().to(device)
            fid = batch['fid_pars'][:, :nfeatures]
            if nfeatures > 2:
                neg = batch['fid_pars'][:, 2] < 0
                fid[:, 2][neg] += 1
                fid[:, 2] = fid[:, 2]*2 - 1
            fid = fid.float().to(device)
            outputs = model.point_estimate(img, spec)
            loss = criterion(outputs, fid)
            losses.append(loss.item())
            if i == 0:
                ids = batch['id'].numpy()
                preds = outputs.detach().cpu().numpy()
                fids = fid.cpu().numpy()
                snrs = snr.cpu().numpy()
            else:
                ids = np.concatenate((ids, batch['id'].numpy()))
                preds = np.vstack((preds, outputs.detach().cpu().numpy()))
                fids = np.vstack((fids, fid.cpu().numpy()))
                snrs = np.concatenate((snrs, snr.cpu().numpy()))

    combined_pred = np.column_stack((ids, preds))
    combined_fid = np.column_stack((ids, fids))

    epoch_loss = sum(losses) / len(losses)
    
    return combined_pred, combined_fid, epoch_loss, snrs

def estimate_density(zz, test_data, model, batch_size=100, snr=None, vcirc_mu=None,device='cpu'):
    '''
    Run this function to test performance of trained density estimation models
    '''
    if model.mode == 2:
        assert vcirc_mu is not None, "Must provide vcirc_mu for mode 2 density estimation"
    model.eval()
    true = []
    log_probs = []
    snrs = snr if snr is not None else torch.rand((batch_size*len(test_data),),device=device)*990 + 10
    with torch.no_grad():
        for i, batch in enumerate(test_data):
            snr = snrs[i*batch_size:(i+1)*batch_size]
            vcircs = vcirc_mu[i*batch_size:(i+1)*batch_size] if vcirc_mu is not None else None
            img = apply_noise(batch['img'].float().to(device), snr, device=device)
            spec = apply_noise(batch['spec'].float().to(device), snr, device=device)
            # img = batch['img'].float().to(device)
            # spec = batch['spec'].float().to(device)
            fid = batch['fid_pars'][:, :2].float().to(device)
            log_prob = model.estimate_log_prob(img, spec, zz, batch_size, vcirc_mu=vcircs)
            log_probs.append(log_prob.detach().cpu().numpy())
            true.append(fid.cpu().numpy())
    true = np.vstack(true)
    log_probs = np.vstack(log_probs)
    snrs = snrs.cpu().numpy()

    return log_probs, true, snrs

def sample_density(model, test_ds, nsamples, snr=None, vcirc_mu=None, randgen=None, return_log_prob=False,device='cpu'):
    '''
    Run this function to sample from trained density estimation models
    '''
    if snr is None and randgen is not None:
        snr = torch.rand(len(test_ds), generator=randgen, device=device)*990 + 10
    if model.mode == 2:
        assert vcirc_mu is not None, "Must provide vcirc_mu for mode 2 density estimation"
    model.eval()
    samples = []
    if return_log_prob:
        log_probs = []
    snrs = snr if snr is not None else torch.rand((len(test_ds),), device=device)*990 + 10
    with torch.no_grad():
        for i in range(len(test_ds)):
            snr = snrs[i]
            vcircs = vcirc_mu[i] if vcirc_mu is not None else None
            img = apply_noise(test_ds[i]['img'].unsqueeze(0).float().to(device), snr, randgen=randgen, device=device)
            spec = apply_noise(test_ds[i]['spec'].unsqueeze(0).float().to(device), snr, randgen=randgen, device=device)
            if return_log_prob:
                sample, log_prob = model.sample(img, spec, nsamples, return_log_prob=True, vcirc_mu=vcircs)
                log_probs.append(log_prob.detach().cpu().numpy())
            else:
                sample = model.sample(img, spec, nsamples, vcirc_mu=vcircs)
            samples.append(sample.detach().cpu().numpy())
    samples = np.vstack(samples)
    snrs = snrs.cpu().numpy()
    if return_log_prob:
        log_probs = np.vstack(log_probs)
        return samples, log_probs, snrs
    return samples, snrs
