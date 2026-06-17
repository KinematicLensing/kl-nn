import sys,time,os
from os.path import join
import logging
import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.optimize import curve_fit

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
from data import (
    _load_rmag_snr_relation,
    _resolve_handedness_flip_feature_indices,
    apply_handedness_flip,
    apply_noise,
    make_exact_half_flip_mask,
    sample_magnitudes,
)
import config
from model_registry import infer_model_name_from_checkpoint_path, load_networks_module_for_model

"""
Module that manages all the trainer classes and testing functions. 
Need to create a wrapper trainer class eventually
"""

def _resolve_amp_dtype(value: str) -> torch.dtype:
    normalized = value.strip().lower()
    if normalized in ("float16", "fp16", "half"):
        return torch.float16
    if normalized in ("bfloat16", "bf16"):
        return torch.bfloat16
    raise ValueError(f"Unsupported amp_dtype '{value}'. Use 'float16' or 'bfloat16'.")

def _patch_networkx_entry_points() -> None:
    try:
        import importlib.metadata as importlib_metadata
    except Exception:
        return
    entry_points = getattr(importlib_metadata, "entry_points", None)
    if entry_points is None or getattr(entry_points, "_kl_nn_filtered", False):
        return

    def _filtered_entry_points(*args, **kwargs):
        eps = entry_points(*args, **kwargs)
        group = kwargs.get("group")
        if group in ("networkx.backends", "networkx.backend_info"):
            try:
                return [ep for ep in eps if ep.name != "nx-loopback"]
            except TypeError:
                return eps
        return eps

    _filtered_entry_points._kl_nn_filtered = True  # type: ignore[attr-defined]
    importlib_metadata.entry_points = _filtered_entry_points


def _maybe_compile_model(
    model: torch.nn.Module,
    log: logging.Logger | None = None,
    use_compile: bool | None = None,
    compile_mode: str | None = None,
    compile_backend: str | None = None,
) -> torch.nn.Module:
    if log is not None:
        log.info(type(model))
    if use_compile is None:
        use_compile = bool(config.train.get('use_compile', False))
    if not use_compile:
        return model
    if not hasattr(torch, "compile"):
        if log is not None:
            log.warning("torch.compile is not available in this Torch build; skipping.")
        return model
    try:
        _patch_networkx_entry_points()
        mode = compile_mode if compile_mode is not None else config.train.get('compile_mode', 'default')
        backend = compile_backend if compile_backend is not None else config.train.get('compile_backend', 'inductor')
        if backend is None:
            return torch.compile(model, mode=mode)
        return torch.compile(model, mode=mode, backend=backend)
    except Exception as exc:
        if log is not None:
            log.warning("torch.compile failed; continuing without compilation: %s", exc)
        return model


def _maybe_strip_state_dict_prefix(
    state_dict: dict[str, torch.Tensor],
    model: torch.nn.Module,
    prefix: str,
) -> dict[str, torch.Tensor]:
    if not any(key.startswith(prefix) for key in state_dict.keys()):
        return state_dict
    stripped = {
        (key[len(prefix):] if key.startswith(prefix) else key): value
        for key, value in state_dict.items()
    }
    model_keys = set(model.state_dict().keys())
    raw_matches = sum(1 for key in state_dict.keys() if key in model_keys)
    stripped_matches = sum(1 for key in stripped.keys() if key in model_keys)
    return stripped if stripped_matches >= raw_matches else state_dict

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
        self.logger = logging.getLogger('Trainer')
        self.has_fib_pos = 'fib_pos' in self.train_data[0] and 'fib_pos' in self.valid_data[0]
        self.enable_handedness_flip = bool(config.train.get('enable_handedness_flip', False))
        self.use_amp = bool(config.train.get('use_amp', False)) and torch.cuda.is_available()
        self.amp_dtype = _resolve_amp_dtype(config.train.get('amp_dtype', 'float16'))
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.use_channels_last = bool(config.train.get('channels_last', False))
        self.use_noise_cache_maxs = bool(config.train.get('noise_cache_maxs', False))
        self.g2_idx = None
        self.theta_idx = None
        if self.enable_handedness_flip:
            feature_names = config.train['feature_names'][:self.nfeatures]
            self.g2_idx, self.theta_idx = _resolve_handedness_flip_feature_indices(feature_names)
        
    def _set_tensors(self):
        '''
        Put dataset arrays on GPU for direct access in training
        '''
        # Initialize large arrays on GPU
        if self.gpu_id == self.log_rank:
            self.logger.info("Setting up tensors on GPU")
        img_format = torch.channels_last if self.use_channels_last else torch.contiguous_format
        self.img_train = torch.empty(
            (self.ntrain, 1, 48, 48),
            dtype=torch.float,
            device=self.device,
            memory_format=img_format,
        )
        self.img_valid = torch.empty(
            (self.nvalid, 1, 48, 48),
            dtype=torch.float,
            device=self.device,
            memory_format=img_format,
        )
        self.spec_train = torch.empty(
            (self.ntrain, 1, 5, 64),
            dtype=torch.float,
            device=self.device,
            memory_format=img_format,
        )
        self.spec_valid = torch.empty(
            (self.nvalid, 1, 5, 64),
            dtype=torch.float,
            device=self.device,
            memory_format=img_format,
        )
        self.fid_train = torch.empty((self.ntrain, self.nfeatures), dtype=torch.float, device=self.device)
        self.fid_valid = torch.empty((self.nvalid, self.nfeatures), dtype=torch.float, device=self.device)
        if self.has_fib_pos:
            self.fibpos_train = torch.empty((self.ntrain, 5, 2), dtype=torch.float, device=self.device)
            self.fibpos_valid = torch.empty((self.nvalid, 5, 2), dtype=torch.float, device=self.device)
        
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
            if self.has_fib_pos:
                self.fibpos_train[i] = self.train_data[i_db]['fib_pos']

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
            if self.has_fib_pos:
                self.fibpos_valid[i] = self.valid_data[i_db]['fib_pos']

            prog = 100*i//self.nvalid
            if prog % 10 == 0 and prog > prev_prog and self.gpu_id == self.log_rank:
                prev_prog = prog
                self.logger.info(f"{prog}% complete")

        self.img_train_maxs = None
        self.img_valid_maxs = None
        self.spec_train_maxs = None
        self.spec_valid_maxs = None
        if self.use_noise_cache_maxs:
            if self.gpu_id == self.log_rank:
                self.logger.info("Precomputing noise max caches")
            self.img_train_maxs = torch.amax(self.img_train, dim=(-1, -2, -3))
            self.img_valid_maxs = torch.amax(self.img_valid, dim=(-1, -2, -3))
            self.spec_train_maxs = torch.amax(self.spec_train, dim=(-1, -2, -3))
            self.spec_valid_maxs = torch.amax(self.spec_valid, dim=(-1, -2, -3))

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        
    def _apply_noise(self, data, snr, maxs=None):
        output = apply_noise(
            data,
            snr,
            device=self.device,
            use_iterative=True,
            maxs=maxs,
        )
        if self.use_channels_last:
            output = output.contiguous(memory_format=torch.channels_last)
        return output

    def _run_batch(self, img, spec, fid, fp=None):
        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
            loss = self.model(img, spec, fid, fp=fp)
        if torch.isfinite(loss):
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
        return loss

    def _run_epoch(self, epoch, show_log=True):
        
        if self.gpu_id == self.log_rank:
            self.logger.info(f'Starting epoch {epoch}')
            
        self.SNR_train = self.generate_snr(size=self.ntrain, mode='uniform')
        self.SNR_valid = self.generate_snr(size=self.nvalid, mode='uniform')
        self.flip_mask_train = (
            make_exact_half_flip_mask(self.ntrain, device=self.device)
            if self.enable_handedness_flip
            else None
        )
        self.flip_mask_valid = (
            make_exact_half_flip_mask(self.nvalid, device=self.device)
            if self.enable_handedness_flip
            else None
        )
        
        if self.gpu_id == self.log_rank:
            self.logger.info(f'Randomized SNR and noise for epoch {epoch}')

        train_loss, train_nans, train_infs = self._trainFunc(epoch)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        valid_loss, valid_nans, valid_infs = self._validFunc(epoch)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        return train_loss, train_nans, train_infs, valid_loss, valid_nans, valid_infs

    def _trainFunc(self, epoch, show_log=True):
        self.model.train()
        self.train_order = torch.randperm(self.ntrain, device=self.device)
        losses = []
        epoch_start = time.time()
        nans = 0
        infs = 0
        
        for i in range(self.nbatch_train):
            start = i*self.batch_size
            batch_ids = self.train_order[start:start+self.batch_size]
            snr = self.SNR_train[batch_ids]
            img_maxs = self.img_train_maxs[batch_ids] if self.use_noise_cache_maxs else None
            spec_maxs = self.spec_train_maxs[batch_ids] if self.use_noise_cache_maxs else None
            img = self._apply_noise(self.img_train[batch_ids], snr, maxs=img_maxs)
            spec = self._apply_noise(self.spec_train[batch_ids], snr, maxs=spec_maxs)
            fid = self.fid_train[batch_ids]
            fp = self.fibpos_train[batch_ids] if self.has_fib_pos else None
            batch_flip_mask = self.flip_mask_train[batch_ids] if self.flip_mask_train is not None else None
            img, spec, fid, fp = apply_handedness_flip(
                img,
                spec,
                fid,
                fp=fp,
                flip_mask=batch_flip_mask,
                g2_idx=self.g2_idx,
                theta_idx=self.theta_idx,
            )
            if self.use_channels_last:
                img = img.contiguous(memory_format=torch.channels_last)
                spec = spec.contiguous(memory_format=torch.channels_last)
            loss = self._run_batch(img, spec, fid, fp=fp)
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
        self.valid_order = torch.randperm(self.nvalid, device=self.device)
        losses = []
        epoch_start = time.time()
        nans = 0
        infs = 0
        with torch.no_grad():
            for i in range(self.nbatch_valid):
                start = i*self.batch_size
                batch_ids = self.valid_order[start:start+self.batch_size]
                snr = self.SNR_valid[batch_ids]
                img_maxs = self.img_valid_maxs[batch_ids] if self.use_noise_cache_maxs else None
                spec_maxs = self.spec_valid_maxs[batch_ids] if self.use_noise_cache_maxs else None
                img = self._apply_noise(self.img_valid[batch_ids], snr, maxs=img_maxs)
                spec = self._apply_noise(self.spec_valid[batch_ids], snr, maxs=spec_maxs)
                fid = self.fid_valid[batch_ids]
                fp = self.fibpos_valid[batch_ids] if self.has_fib_pos else None
                batch_flip_mask = self.flip_mask_valid[batch_ids] if self.flip_mask_valid is not None else None
                img, spec, fid, fp = apply_handedness_flip(
                    img,
                    spec,
                    fid,
                    fp=fp,
                    flip_mask=batch_flip_mask,
                    g2_idx=self.g2_idx,
                    theta_idx=self.theta_idx,
                )
                if self.use_channels_last:
                    img = img.contiguous(memory_format=torch.channels_last)
                    spec = spec.contiguous(memory_format=torch.channels_last)
                loss = self.model(img, spec, fid, fp=fp)
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
        self.train_order = torch.arange(self.ntrain, device=self.device)
        self.valid_order = torch.arange(self.nvalid, device=self.device)
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
            min_snr = kwargs.get('min', 5)
            max_snr = kwargs.get('max', 1000)
            return torch.rand((size,), device=self.device)* (max_snr - min_snr) + min_snr
        elif mode == 'rmag':
            rmag = sample_magnitudes(size, m_min=15, m_max=23)
            a, b = _load_rmag_snr_relation()
            log_snr = (rmag - b) / a
            snr = 10**log_snr
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
    torch.backends.cudnn.benchmark = bool(config.train.get('cudnn_benchmark', False))

    device = torch.device(f"cuda:{rank}")
    train_ds, valid_ds, model, optimizer = load_train_objs(
        mode,
        nfeatures,
        batch_size,
        world_size,
        Model,
        rank,
        epoch,
        log=log,
        device=device,
    )
    if config.train.get('channels_last', False):
        model = model.to(memory_format=torch.channels_last)
    model = _maybe_compile_model(model, log=log)
    ddp_kwargs = {
        "device_ids": [rank],
        "find_unused_parameters": bool(config.train.get('ddp_find_unused_parameters', False)),
        "broadcast_buffers": bool(config.train.get('ddp_broadcast_buffers', False)),
        "gradient_as_bucket_view": bool(config.train.get('ddp_gradient_as_bucket_view', True)),
    }
    if config.train.get('ddp_static_graph', False):
        ddp_kwargs["static_graph"] = True
    try:
        model = DDP(model, **ddp_kwargs)
    except TypeError:
        ddp_kwargs.pop("static_graph", None)
        ddp_kwargs.pop("gradient_as_bucket_view", None)
        model = DDP(model, **ddp_kwargs)
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

def load_train_objs(
    mode,
    nfeatures,
    batch_size,
    nGPUs,
    Model,
    rank,
    epoch=None,
    log=None,
    device=None,
    **kwargs,
):
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
    if device is not None:
        model = model.to(device)
    # optimizer = optim.SGD(model.parameters(), 
    #                       lr=config.train['initial_learning_rate'],
    #                       momentum=config.train['momentum'])
    use_fused = (
        bool(config.train.get('use_fused_adamw', False))
        and torch.cuda.is_available()
        and next(model.parameters()).is_cuda
    )
    optimizer_kwargs = dict(
        lr=config.train['initial_learning_rate'],
        weight_decay=config.train['weight_decay'],
    )
    if use_fused:
        optimizer_kwargs["fused"] = True
    try:
        optimizer = optim.AdamW(model.parameters(), **optimizer_kwargs)
    except TypeError:
        if use_fused and log is not None:
            log.warning("Fused AdamW not supported in this Torch build; falling back to standard AdamW.")
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.train['initial_learning_rate'],
            weight_decay=config.train['weight_decay'],
        )

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

def load_model(
    mode=1,
    Model=ForkCNN,
    path=None,
    strict=True,
    assign=False,
    GPUs=1,
    device='cpu',
    model_name=None,
    networks_root=None,
    use_compile: bool | None = None,
    compile_mode: str | None = None,
    compile_backend: str | None = None,
    channels_last: bool | None = None,
):

    model_cls = Model
    resolved_name = None
    if path is not None:
        resolved_name = model_name or infer_model_name_from_checkpoint_path(path)
        if resolved_name:
            try:
                if networks_root is None:
                    archived_module = load_networks_module_for_model(resolved_name)
                else:
                    archived_module = load_networks_module_for_model(
                        resolved_name, networks_root=networks_root
                    )
            except FileNotFoundError:
                archived_module = None

            if archived_module is not None:
                try:
                    model_cls = getattr(archived_module, Model.__name__)
                except AttributeError as exc:
                    raise AttributeError(
                        f"Archived networks module for '{resolved_name}' "
                        f"does not define {Model.__name__}."
                    ) from exc
    model = model_cls(mode)
    model.to(device)
    if channels_last is None:
        channels_last = bool(config.train.get('channels_last', False))
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    if GPUs > 1:
        model = DDP(model, device_ids=None)

    if path != None:
        state_dict = torch.load(path, weights_only=False, map_location=torch.device(device))
        state_dict = _maybe_strip_state_dict_prefix(state_dict, model, "module.")
        state_dict = _maybe_strip_state_dict_prefix(state_dict, model, "_orig_mod.")
        model.load_state_dict(state_dict, strict=strict, assign=assign)

    if use_compile is None:
        use_compile = bool(config.train.get('use_compile', False))
    if use_compile and GPUs <= 1:
        model = _maybe_compile_model(
            model,
            log=None,
            use_compile=use_compile,
            compile_mode=compile_mode,
            compile_backend=compile_backend,
        )

    return model

def sample_density(
    model,
    test_ds,
    nsamples,
    snr=None,
    vcirc_mu=None,
    randgen=None,
    return_log_prob=False,
    device='cpu',
    flip_handedness=None,
    channels_last: bool | None = None,
    progress=None,
):
    '''
    Run this function to sample from trained density estimation models
    '''
    if channels_last is None:
        channels_last = bool(config.train.get('channels_last', False))
    if snr is None and randgen is not None:
        snr = torch.rand(len(test_ds), generator=randgen, device=device)*990 + 10
    if model.mode == 2:
        assert vcirc_mu is not None, "Must provide vcirc_mu for mode 2 density estimation"
    model.eval()
    do_flip = bool(config.train.get('enable_handedness_flip', False)) if flip_handedness is None else bool(flip_handedness)
    g2_idx = None
    theta_idx = None
    feature_names = None
    if do_flip:
        feature_names = config.train['feature_names'][: config.train['feature_number']]
        g2_idx, theta_idx = _resolve_handedness_flip_feature_indices(feature_names)
        flip_mask_all = make_exact_half_flip_mask(len(test_ds), device=device)
    else:
        flip_mask_all = None
    samples = []
    if return_log_prob:
        log_probs = []
    snrs = snr if snr is not None else torch.rand((len(test_ds),), device=device)*990 + 10
    iterator = range(len(test_ds))
    if progress is not None:
        iterator = progress(iterator, total=len(test_ds), desc="Sampling")
    with torch.no_grad():
        for i in iterator:
            snr = snrs[i]
            vcircs = vcirc_mu[i] if vcirc_mu is not None else None
            img = apply_noise(test_ds[i]['img'].unsqueeze(0).float().to(device), snr, randgen=randgen, device=device)
            spec = apply_noise(test_ds[i]['spec'].unsqueeze(0).float().to(device), snr, randgen=randgen, device=device)
            fp = test_ds[i]['fib_pos'].unsqueeze(0).float().to(device) if 'fib_pos' in test_ds[i] else None
            if do_flip:
                fid_row = torch.as_tensor(
                    test_ds[i]['fid_pars'][: len(feature_names)],
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(0)
                img, spec, _, fp = apply_handedness_flip(
                    img,
                    spec,
                    fid_row,
                    fp=fp,
                    flip_mask=flip_mask_all[i:i+1],
                    g2_idx=g2_idx,
                    theta_idx=theta_idx,
                )
            if channels_last:
                img = img.contiguous(memory_format=torch.channels_last)
                spec = spec.contiguous(memory_format=torch.channels_last)
            if return_log_prob:
                sample, log_prob = model.sample(img, spec, nsamples, fp=fp, return_log_prob=True, vcirc_mu=vcircs, sample_id=i)
                log_probs.append(log_prob.detach().cpu().numpy())
            else:
                sample = model.sample(img, spec, nsamples, fp=fp, vcirc_mu=vcircs, sample_id=i)
            samples.append(sample.detach().cpu().numpy())
    samples = np.vstack(samples)
    snrs = snrs.cpu().numpy()
    if return_log_prob:
        log_probs = np.vstack(log_probs)
        return samples, log_probs, snrs
    return samples, snrs
