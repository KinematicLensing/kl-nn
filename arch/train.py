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
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, LinearLR, SequentialLR
import pyxis.torch as pxt
from timm.optim.lars import Lars

from networks import *
from dataset import *
from utils import *
from data import (
    TFCalculator,
    _load_rmag_snr_relation,
    apply_views,
    apply_noise,
    sample_magnitudes,
    app_mag_to_snr,
    snr_to_app_mag,
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

#--------------------#
# Trainer Base Class #
#--------------------#

class Trainer:
    def __init__(
        self,
        world_size: int,
        model: torch.nn.Module,
        train_ds: FiberDataset,
        valid_ds: FiberDataset,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
        batch_size: int,
    ) -> None:
        
        self.model_name = config.train['model_name']
        self.world_size = world_size
        self.gpu_id = gpu_id
        self.log_rank = 0
        self.device = torch.device(f"cuda:{gpu_id}")
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
        self.nfeatures = config.train['feature_number']
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', factor=0.5, patience=10)
        self.logger = logging.getLogger('Trainer')
        self.use_amp = bool(config.train.get('use_amp', False)) and torch.cuda.is_available()
        self.amp_dtype = _resolve_amp_dtype(config.train.get('amp_dtype', 'float16'))
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.use_channels_last = bool(config.train.get('channels_last', False))
        self.use_noise_cache_maxs = bool(config.train.get('noise_cache_maxs', False))
    
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
        if snr is None:
            return data
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
    
    def _run_epoch(self, epoch, show_log=True):
        
        if self.gpu_id == self.log_rank:
            self.logger.info(f'Starting epoch {epoch+1}')
            
        # Generate the permutation arrays BEFORE building the corresponding SNRs
        self.train_order = torch.randperm(self.ntrain, device=self.device)
        self.valid_order = torch.randperm(self.nvalid, device=self.device)
            
        self.SNR_train = self.generate_snr(size=self.ntrain, mode='tf')
        self.SNR_valid = self.generate_snr(size=self.nvalid, mode='tf')
        
        if self.gpu_id == self.log_rank:
            self.logger.info(f'Randomized SNR and noise for epoch {epoch+1}')

        train_loss = self._trainFunc(epoch)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        valid_loss = self._validFunc(epoch)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        return train_loss, valid_loss
    
    def _trainFunc(self, epoch, show_log=True):
        raise NotImplementedError("Subclasses must implement _trainFunc")
    
    def _validFunc(self, epoch, show_log=True):
        raise NotImplementedError("Subclasses must implement _validFunc")

    def _save_checkpoint(self, epoch):
        raise NotImplementedError("Subclasses must implement _save_checkpoint")
    
    def train(self, max_epochs: int):
        self._set_tensors()
        train_losses = []
        valid_losses = []
        if self.gpu_id == self.log_rank:
            self.logger.info("Training start")
        for epoch in range(max_epochs):
            train_loss, valid_loss = self._run_epoch(epoch)
            if self.scheduler.__class__.__name__ == "ReduceLROnPlateau":
                self.scheduler.step(valid_loss)
            else:
                self.scheduler.step()
            if self.gpu_id == self.log_rank:
                self.logger.info(f"Current LR is {self.scheduler.get_last_lr()}")
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            if self.gpu_id == self.log_rank and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)

        if self.gpu_id == self.log_rank:
            losses = pd.DataFrame(np.vstack([train_losses, valid_losses]))
            losses_dir = join(config.train['model_path'], 'losses')
            os.makedirs(losses_dir, exist_ok=True)
            losses.to_csv(join(losses_dir, f'losses_{self.model_name}.csv'), index=False)
            self.logger.info("Loss history saved successfully. Training cycle complete.")
    
    def generate_snr(self, size, mode='uniform', **kwargs):
        if mode == 'none':
            return None
        if mode == 'uniform':
            min_snr = kwargs.get('min', 5)
            max_snr = kwargs.get('max', 1000)
            return torch.rand((size,), device=self.device)* (max_snr - min_snr) + min_snr
        elif mode == 'tf':
            tf_calc = TFCalculator(slope=config.tf['slope'], intercept=config.tf['intercept'], scatter=config.tf['scatter'])
            if size == self.ntrain:
                vcirc_norm = self.fid_train[self.train_order][:, 5]
            elif size == self.nvalid:
                vcirc_norm = self.fid_valid[self.valid_order][:, 5]
            vcirc_min = config.par_ranges['vcirc'][0]
            vcirc_max = config.par_ranges['vcirc'][1]
            vcirc_mu = ((vcirc_norm + 1)/2 * (vcirc_max - vcirc_min) + vcirc_min).cpu().numpy()
            mag = tf_calc.sample_mag_from_vcirc(vcirc_mu)
            snr = app_mag_to_snr(mag)
            return torch.from_numpy(snr).float().to(self.device)
        elif mode == 'rmag':
            rmag = sample_magnitudes(size, m_min=15, m_max=23)
            a, b = _load_rmag_snr_relation()
            log_snr = (rmag - b) / a
            snr = 10**log_snr
            return torch.from_numpy(snr).float().to(self.device)
        else:
            raise ValueError("Invalid SNR generation mode")

#---------------------------#
# Feature Extractor Trainer #
#---------------------------#

class FETrainer(Trainer):
    def __init__(
        self,
        world_size: int,
        model: torch.nn.Module,
        train_ds: FiberDataset,
        valid_ds: FiberDataset,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
        batch_size: int,
    ) -> None:
        
        super().__init__(world_size, model, train_ds, valid_ds, optimizer, gpu_id, save_every, batch_size)
        
        self.model_name = config.pretrain['model_name']
        self.return_components = True
        warmup_epochs = 5
        lin_scheduler = LinearLR(self.optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
        cos_scheduler = CosineAnnealingLR(self.optimizer, T_max=config.pretrain['epoch_number']-warmup_epochs, eta_min=1e-4)
        self.scheduler = SequentialLR(self.optimizer, schedulers=[lin_scheduler, cos_scheduler], milestones=[warmup_epochs])
        self.use_amp = bool(config.pretrain.get('use_amp', False)) and torch.cuda.is_available()
        self.amp_dtype = _resolve_amp_dtype(config.pretrain.get('amp_dtype', 'float16'))
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.use_channels_last = bool(config.pretrain.get('channels_last', False))
        self.use_noise_cache_maxs = bool(config.pretrain.get('noise_cache_maxs', False))
    
    def _run_batch(self, img, spec, fp, img2, spec2, fp2):
        self.optimizer.zero_grad(set_to_none=True)
        all_loss = self.model(img, spec, fp, img2, spec2, fp2, return_components=self.return_components)
        loss = all_loss[0] if self.return_components else all_loss
        if torch.isfinite(loss):
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.module.parameters(), max_norm=1.0)
            self.optimizer.step()
        return all_loss
    
    def _trainFunc(self, epoch, show_log=True):
        self.model.train()
        losses = []
        if self.return_components:
            sim_losses = []
            var_losses = []
            cov_losses = []
            eff_dim_img_list = []
            eff_dim_spec_list = []
        epoch_start = time.time()

        self.SNR_train_2 = self.generate_snr(size=self.ntrain, mode='tf')
        
        for i in range(self.nbatch_train):
            start = i*self.batch_size
            batch_ids = self.train_order[start:start+self.batch_size]
            snr = self.SNR_train[batch_ids] if self.SNR_train is not None else None
            snr2 = self.SNR_train_2[batch_ids] if self.SNR_train_2 is not None else None
            img_maxs = self.img_train_maxs[batch_ids] if self.use_noise_cache_maxs else None
            spec_maxs = self.spec_train_maxs[batch_ids] if self.use_noise_cache_maxs else None
            img = self._apply_noise(self.img_train[batch_ids], snr, maxs=img_maxs)
            spec = self._apply_noise(self.spec_train[batch_ids], snr, maxs=spec_maxs)
            img2 = self._apply_noise(self.img_train[batch_ids], snr2, maxs=img_maxs)
            spec2 = self._apply_noise(self.spec_train[batch_ids], snr2, maxs=spec_maxs)
            img, spec, img2, spec2 = apply_views(img, spec, img2, spec2)
            fp = self.fibpos_train[batch_ids]
            if self.use_channels_last:
                img = img.contiguous(memory_format=torch.channels_last)
                spec = spec.contiguous(memory_format=torch.channels_last)
                img2 = img2.contiguous(memory_format=torch.channels_last)
                spec2 = spec2.contiguous(memory_format=torch.channels_last)
            all_loss = self._run_batch(img, spec, fp, img2, spec2, fp)
            if self.return_components:
                loss, sim, var, cov, eff_dim_img, eff_dim_spec = all_loss
                sim_losses.append(sim.item())
                var_losses.append(var.item())
                cov_losses.append(cov.item())
                eff_dim_img_list.append(eff_dim_img.item())
                eff_dim_spec_list.append(eff_dim_spec.item())
            else:
                loss = all_loss
            if torch.isfinite(loss):
                losses.append(loss.item())

            if show_log and self.gpu_id == self.log_rank and i%100 == 0:
                self.logger.info(f"Batch {i} complete")

        epoch_loss = sum(losses) / len(losses)
        if self.return_components:
            epoch_sim_loss = sum(sim_losses) / len(sim_losses)
            epoch_var_loss = sum(var_losses) / len(var_losses)
            epoch_cov_loss = sum(cov_losses) / len(cov_losses)
            epoch_eff_dim_img = sum(eff_dim_img_list) / len(eff_dim_img_list)
            epoch_eff_dim_spec = sum(eff_dim_spec_list) / len(eff_dim_spec_list)
            if show_log and self.gpu_id == self.log_rank:
                self.logger.info(f"[TRAIN] Epoch: {epoch+1} Loss: {epoch_loss} Sim: {epoch_sim_loss} Var: {epoch_var_loss} Cov: {epoch_cov_loss}")
                self.logger.info(f"[TRAIN] Epoch: {epoch+1} Effective Dimensionality - Image: {epoch_eff_dim_img} Spectrum: {epoch_eff_dim_spec}")
        epoch_time = time.time() - epoch_start
        if show_log and self.gpu_id == self.log_rank:
            self.logger.info("[TRAIN] Epoch: {} Loss: {} Time: {:.0f}:{:.0f}".format(epoch+1, epoch_loss,
                                                                                    epoch_time // 60, 
                                                                                    epoch_time % 60))
        return epoch_loss

    def _validFunc(self,epoch,show_log=True):
        self.model.eval()
        losses = []
        if self.return_components:
            sim_losses = []
            var_losses = []
            cov_losses = []
            eff_dim_img_list = []
            eff_dim_spec_list = []
        epoch_start = time.time()

        self.SNR_valid_2 = self.generate_snr(size=self.nvalid, mode='tf')

        with torch.no_grad():
            for i in range(self.nbatch_valid):
                start = i*self.batch_size
                batch_ids = self.valid_order[start:start+self.batch_size]
                snr = self.SNR_valid[batch_ids] if self.SNR_valid is not None else None
                snr2 = self.SNR_valid_2[batch_ids] if self.SNR_valid_2 is not None else None
                img_maxs = self.img_valid_maxs[batch_ids] if self.use_noise_cache_maxs else None
                spec_maxs = self.spec_valid_maxs[batch_ids] if self.use_noise_cache_maxs else None
                img = self._apply_noise(self.img_valid[batch_ids], snr, maxs=img_maxs)
                spec = self._apply_noise(self.spec_valid[batch_ids], snr, maxs=spec_maxs)
                img2 = self._apply_noise(self.img_valid[batch_ids], snr2, maxs=img_maxs)
                spec2 = self._apply_noise(self.spec_valid[batch_ids], snr2, maxs=spec_maxs)
                img, spec, img2, spec2 = apply_views(img, spec, img2, spec2)
                fp = self.fibpos_valid[batch_ids]
                if self.use_channels_last:
                    img = img.contiguous(memory_format=torch.channels_last)
                    spec = spec.contiguous(memory_format=torch.channels_last)
                all_loss = self.model(img, spec, fp, img2, spec2, fp, return_components=self.return_components)
                if self.return_components:
                    loss, sim, var, cov, eff_dim_img, eff_dim_spec = all_loss
                    sim_losses.append(sim.item())
                    var_losses.append(var.item())
                    cov_losses.append(cov.item())
                    eff_dim_img_list.append(eff_dim_img.item())
                    eff_dim_spec_list.append(eff_dim_spec.item())
                else:
                    loss = all_loss
                if torch.isfinite(loss):
                    losses.append(loss.item())

                if show_log and self.gpu_id == self.log_rank and i%100 == 0:
                    self.logger.info(f"Batch {i} complete")

        epoch_loss = sum(losses) / len(losses)
        if self.return_components:
            epoch_sim_loss = sum(sim_losses) / len(sim_losses)
            epoch_var_loss = sum(var_losses) / len(var_losses)
            epoch_cov_loss = sum(cov_losses) / len(cov_losses)
            epoch_eff_dim_img = sum(eff_dim_img_list) / len(eff_dim_img_list)
            epoch_eff_dim_spec = sum(eff_dim_spec_list) / len(eff_dim_spec_list)
            if show_log and self.gpu_id == self.log_rank:
                self.logger.info(f"[VALID] Epoch: {epoch+1} Loss: {epoch_loss} Sim: {epoch_sim_loss} Var: {epoch_var_loss} Cov: {epoch_cov_loss}")
                self.logger.info(f"[VALID] Epoch: {epoch+1} Effective Dimensionality - Image: {epoch_eff_dim_img} Spectrum: {epoch_eff_dim_spec}")
        epoch_time = time.time() - epoch_start
        if show_log and self.gpu_id == self.log_rank:
            self.logger.info("[VALID] Epoch: {} Loss: {} Time: {:.0f}:{:.0f}".format(epoch+1, epoch_loss,
                                                                                    epoch_time // 60, 
                                                                                    epoch_time % 60))
        return epoch_loss
    
    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = join(config.pretrain['model_path'], config.pretrain['model_name'], config.pretrain['model_name']+str(epoch))
        torch.save(ckp, PATH)

#-------------#
# NPE Trainer #
#-------------#

class NPETrainer(Trainer):
    def __init__(
        self,
        world_size: int,
        model: torch.nn.Module,
        train_ds: FiberDataset,
        valid_ds: FiberDataset,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
        batch_size: int,
    ) -> None:
        
        super().__init__(world_size, model, train_ds, valid_ds, optimizer, gpu_id, save_every, batch_size)
        
        self.model_name = config.train['model_name']
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', factor=0.5, patience=10)

    def _run_batch(self, img, spec, fid, fp=None, snr=None):
        self.optimizer.zero_grad(set_to_none=True)
        
        if self.model.module.mode == 2:
            self.mag_train = snr_to_app_mag(snr) if snr is not None else None
            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                loss = self.model(img, spec, fid, fp=fp, mag=self.mag_train, snr=snr)
        else:
            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                loss = self.model(img, spec, fid, fp=fp)

        # Check locally if loss is valid
        valid_loss = torch.isfinite(loss).to(dtype=torch.float32).view(1)
        
        # Check globally across all ranks
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(valid_loss, op=torch.distributed.ReduceOp.MIN)

        # Only proceed if EVERY rank has a finite loss
        if valid_loss.item() == 1.0:
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
        else:
            self.invalid_loss_count += 1
                
        return loss

    def _trainFunc(self, epoch, show_log=True):
        self.model.train()
        losses = []
        epoch_start = time.time()
        self.invalid_loss_count = 0
        
        for i in range(self.nbatch_train):
            start = i*self.batch_size
            batch_ids = self.train_order[start:start+self.batch_size]
            snr = self.SNR_train[batch_ids] if self.SNR_train is not None else None
            img_maxs = self.img_train_maxs[batch_ids] if self.use_noise_cache_maxs else None
            spec_maxs = self.spec_train_maxs[batch_ids] if self.use_noise_cache_maxs else None
            img = self._apply_noise(self.img_train[batch_ids], snr, maxs=img_maxs)
            spec = self._apply_noise(self.spec_train[batch_ids], snr, maxs=spec_maxs)
            fid = self.fid_train[batch_ids]
            fp = self.fibpos_train[batch_ids]
            if self.use_channels_last:
                img = img.contiguous(memory_format=torch.channels_last)
                spec = spec.contiguous(memory_format=torch.channels_last)
            loss = self._run_batch(img, spec, fid, fp=fp, snr=snr)
            if torch.isfinite(loss):
                losses.append(loss.item())

            if show_log and self.gpu_id == self.log_rank and i%100 == 0:
                self.logger.info(f"Batch {i} complete")

        epoch_loss = sum(losses) / len(losses)
        epoch_time = time.time() - epoch_start
        if show_log and self.gpu_id == self.log_rank:
            self.logger.info("[TRAIN] Epoch: {} Loss: {} Time: {:.0f}:{:.0f}".format(epoch+1, epoch_loss,
                                                                                    epoch_time // 60, 
                                                                                    epoch_time % 60))
            self.logger.info(f"Invalid loss count for epoch {epoch+1}: {self.invalid_loss_count}")
        return epoch_loss

    def _validFunc(self,epoch,show_log=True):
        self.model.eval()
        losses = []
        epoch_start = time.time()
        self.invalid_loss_count = 0

        with torch.no_grad():
            for i in range(self.nbatch_valid):
                start = i*self.batch_size
                batch_ids = self.valid_order[start:start+self.batch_size]
                snr = self.SNR_valid[batch_ids] if self.SNR_valid is not None else None
                img_maxs = self.img_valid_maxs[batch_ids] if self.use_noise_cache_maxs else None
                spec_maxs = self.spec_valid_maxs[batch_ids] if self.use_noise_cache_maxs else None
                img = self._apply_noise(self.img_valid[batch_ids], snr, maxs=img_maxs)
                spec = self._apply_noise(self.spec_valid[batch_ids], snr, maxs=spec_maxs)
                fid = self.fid_valid[batch_ids]
                fp = self.fibpos_valid[batch_ids]
                if self.use_channels_last:
                    img = img.contiguous(memory_format=torch.channels_last)
                    spec = spec.contiguous(memory_format=torch.channels_last)
                if self.model.module.mode == 2:
                    self.mag_valid = snr_to_app_mag(snr) if snr is not None else None
                    loss = self.model(img, spec, fid, fp=fp, mag=self.mag_valid, snr=snr)
                else:
                    loss = self.model(img, spec, fid, fp=fp)
                if torch.isfinite(loss):
                    losses.append(loss.item())

                if show_log and self.gpu_id == self.log_rank and i%100 == 0:
                    self.logger.info(f"Batch {i} complete")

        epoch_loss = sum(losses) / len(losses)
        epoch_time = time.time() - epoch_start
        if show_log and self.gpu_id == self.log_rank:
            self.logger.info("[VALID] Epoch: {} Loss: {} Time: {:.0f}:{:.0f}".format(epoch+1, epoch_loss,
                                                                                    epoch_time // 60, 
                                                                                    epoch_time % 60))
            self.logger.info(f"Invalid loss count for epoch {epoch+1}: {self.invalid_loss_count}")
        return epoch_loss
    
    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = join(config.train['model_path'], config.train['model_name'], config.train['model_name']+str(epoch))
        torch.save(ckp, PATH)

#------------------#
# Global functions #
#------------------#

def train_nn(rank: int, world_size: int, Model=VICRegPretrain, Trainer=FETrainer,
             save_every=1, train_mode='pretrain'):
    '''
    Main function to train any network.
    '''
    if train_mode == 'pretrain':
        train_config = config.pretrain
    elif train_mode == 'train':
        train_config = config.train
    else:
        raise ValueError(f"Invalid train_mode '{train_mode}'. Must be 'pretrain' or 'train'.")

    # Set parameters based on stage
    total_epochs = train_config['epoch_number']
    batch_size = train_config['batch_size']
    pretrain_epoch = train_config.get('pretrain_from', None)
    
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger('Setup')
    if rank == 0:
        log.info('Initializing')
    
    ddp_setup(rank, world_size)
    log.info(f'[rank: {rank}] Successfully set up device')
    torch.backends.cudnn.benchmark = bool(train_config.get('cudnn_benchmark', False))

    device = torch.device(f"cuda:{rank}")
    train_ds, valid_ds, model, optimizer = load_train_objs(
        Model,
        rank,
        train_config,
        train_mode=train_mode,
        epoch=pretrain_epoch,
        log=log,
        device=device,
    )
    if train_config.get('channels_last', False):
        model = model.to(memory_format=torch.channels_last)
    model = _maybe_compile_model(model, log=log, use_compile=train_config.get('use_compile', False),)
    ddp_kwargs = {
        "device_ids": [rank],
        "find_unused_parameters": bool(train_config.get('ddp_find_unused_parameters', False)),
        "broadcast_buffers": bool(train_config.get('ddp_broadcast_buffers', False)),
        "gradient_as_bucket_view": bool(train_config.get('ddp_gradient_as_bucket_view', True)),
    }
    if train_config.get('ddp_static_graph', False):
        ddp_kwargs["static_graph"] = True
    try:
        model = DDP(model, **ddp_kwargs)
    except TypeError:
        ddp_kwargs.pop("static_graph", None)
        ddp_kwargs.pop("gradient_as_bucket_view", None)
        model = DDP(model, **ddp_kwargs)
    log.info(f'[rank: {rank}] Successfully loaded training objects')
    
    os.makedirs(join(train_config['model_path'], train_config['model_name']), exist_ok=True)
    trainer = Trainer(world_size, model, train_ds, valid_ds, optimizer, rank, save_every, batch_size)
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

def load_train_objs(
    Model,
    rank,
    train_config,
    train_mode='pretrain',
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
        strict = True
        model_dir = train_config['model_path'] + train_config['pretrained_name'] + '/' + train_config['pretrained_name'] + str(epoch)
        pretrained_model = load_model(train_config, Model=VICRegPretrain, path=model_dir, strict=strict, assign=True)
        model = Model(pretrained_model.backbone, **kwargs)
        # for param in model.feature_extractor.parameters():
        #     param.requires_grad = False
        if rank == 0:
            if log is not None:
                log.info(f"Loaded model {train_config['pretrained_name']} at epoch {epoch}")
    else:
        model = Model(**kwargs)  # initialize new model
        if rank == 0:
            if log is not None:
                log.info(f"Loaded new model {train_config['model_name']}")
    if device is not None:
        model = model.to(device)
    if train_mode == 'pretrain':
        optimizer = Lars(
            model.parameters(), 
            lr=train_config['initial_learning_rate'], 
            weight_decay=train_config['weight_decay'], 
            momentum=0.9,
            eps=train_config.get('eps', 1e-8),
        )
    else:
        use_fused = (
            bool(train_config.get('use_fused_adamw', False))
            and torch.cuda.is_available()
            and next(model.parameters()).is_cuda
        )
        optimizer_kwargs = dict(
            lr=train_config['initial_learning_rate'],
            weight_decay=train_config['weight_decay'],
            eps=train_config.get('eps', 1e-8)
        )
        optimizer_diff = [{'params': model.feature_extractor.parameters(), 'lr': train_config['initial_learning_rate']*0.1},
                          {'params': model.flow.parameters(), 'lr': train_config['initial_learning_rate']*10}]
        if use_fused:
            optimizer_kwargs["fused"] = True
        try:
            optimizer = optim.AdamW(optimizer_diff, **optimizer_kwargs)
        except TypeError:
            if use_fused and log is not None:
                log.warning("Fused AdamW not supported in this Torch build; falling back to standard AdamW.")
            optimizer = optim.AdamW(
                optimizer_diff,
                lr=train_config['initial_learning_rate'],
                weight_decay=train_config['weight_decay'],
            )

    return train_ds, valid_ds, model, optimizer

def load_model(
    train_config,
    Model=FeatureExtractor,
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
    else:
        model_cls = Model
    model = model_cls()
    model.to(device)
    if channels_last is None:
        channels_last = bool(train_config.get('channels_last', False))
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
        use_compile = bool(train_config.get('use_compile', False))
    if use_compile and GPUs <= 1:
        model = _maybe_compile_model(
            model,
            log=None,
            use_compile=use_compile,
            compile_mode=compile_mode,
            compile_backend=compile_backend,
        )

    return model

def point_estimate(
    model,
    test_ds,
    snr=None,
    randgen=None,
    device='cpu',
    progress=None,
):
    '''
    Run this function to get point estimates from trained density estimation models
    '''
    model.eval()
    preds = []
    trues = []
    snrs = snr if snr is not None else torch.rand((len(test_ds),), device=device)*995 + 5
    iterator = range(len(test_ds))
    if progress is not None:
        iterator = progress(iterator, total=len(test_ds), desc="Estimating")
    with torch.no_grad():
        for i in iterator:
            snr = snrs[i]
            img = apply_noise(test_ds[i]['img'].unsqueeze(0).float().to(device), snr, randgen=randgen, device=device)
            spec = apply_noise(test_ds[i]['spec'].unsqueeze(0).float().to(device), snr, randgen=randgen, device=device)
            fids = torch.as_tensor(test_ds[i]['fid_pars'][:model.nfeatures], dtype=torch.float32, device=device).unsqueeze(0)
            trues.append(fids.detach().cpu().numpy())
            fp = test_ds[i]['fib_pos'].unsqueeze(0).float().to(device) if 'fib_pos' in test_ds[i] else None
            pred = model.point_estimate(img, spec, fp)
            preds.append(pred.detach().cpu().numpy())
    
    preds = np.vstack(preds)
    trues = np.vstack(trues)
    snrs = snrs.cpu().numpy()
    return preds, trues, snrs

def sample_density(
    model,
    test_ds,
    nsamples,
    snr=None,
    vcirc_mu=None,
    randgen=None,
    return_log_prob=False,
    device='cpu',
    channels_last: bool | None = None,
    progress=None,
):
    '''
    Run this function to sample from trained density estimation models
    '''
    if channels_last is None:
        channels_last = bool(config.train.get('channels_last', False))
    if snr is None and randgen is not None:
        snr = torch.rand(len(test_ds), generator=randgen, device=device)*995 + 5
    if model.mode == 2:
        assert vcirc_mu is not None, "Must provide vcirc_mu for mode 2 density estimation"
    model.eval()
    samples = []
    if return_log_prob:
        log_probs = []
    snrs = snr if snr is not None else torch.rand((len(test_ds),), device=device)*995 + 5
    iterator = range(len(test_ds))
    if progress is not None:
        iterator = progress(iterator, total=len(test_ds), desc="Sampling")
    with torch.no_grad():
        for i in iterator:
            snr = snrs[i]
            mag = snr_to_app_mag(snr) if model.mode == 2 else None
            img = apply_noise(test_ds[i]['img'].unsqueeze(0).float().to(device), snr, randgen=randgen, device=device)
            spec = apply_noise(test_ds[i]['spec'].unsqueeze(0).float().to(device), snr, randgen=randgen, device=device)
            # img = test_ds[i]['img'].unsqueeze(0).float().to(device)
            # spec = test_ds[i]['spec'].unsqueeze(0).float().to(device)
            fp = test_ds[i]['fib_pos'].unsqueeze(0).float().to(device) if 'fib_pos' in test_ds[i] else None
            if channels_last:
                img = img.contiguous(memory_format=torch.channels_last)
                spec = spec.contiguous(memory_format=torch.channels_last)
            if return_log_prob:
                sample, log_prob = model.sample(img, spec, nsamples, fp=fp, mag=mag, snr=snr, return_log_prob=True, sample_id=i)
                log_probs.append(log_prob.detach().cpu().numpy())
            else:
                sample = model.sample(img, spec, nsamples, fp=fp, mag=mag, snr=snr, sample_id=i)
            samples.append(sample.detach().cpu().numpy())
    samples = np.vstack(samples)
    snrs = snrs.cpu().numpy()
    if return_log_prob:
        log_probs = np.vstack(log_probs)
        return samples, log_probs, snrs
    return samples, snrs


def evaluate_conditional_2d(
    model,
    test_ds,
    snr=None,
    randgen=None,
    device='cpu',
    channels_last: bool | None = None,
    progress=None,
):
    if channels_last is None:
        channels_last = bool(config.train.get('channels_last', False))
    if snr is None and randgen is not None:
        snr = torch.rand(len(test_ds), generator=randgen, device=device)*995 + 5
    model.eval()
    probs = []
    snrs = snr if snr is not None else torch.rand((len(test_ds),), device=device)*995 + 5
    # snrs = torch.full((len(test_ds),), 1000.0, device=device)
    iterator = range(len(test_ds))
    if progress is not None:
        iterator = progress(iterator, total=len(test_ds), desc="Evaluating")
    with torch.no_grad():
        for i in iterator:
            snr = snrs[i]
            img = apply_noise(test_ds[i]['img'].unsqueeze(0).float().to(device), snr, randgen=randgen, device=device)
            spec = apply_noise(test_ds[i]['spec'].unsqueeze(0).float().to(device), snr, randgen=randgen, device=device)
            fids = torch.as_tensor(test_ds[i]['fid_pars'][:model.nfeatures], dtype=torch.float32, device=device).unsqueeze(0)
            fp = test_ds[i]['fib_pos'].unsqueeze(0).float().to(device) if 'fib_pos' in test_ds[i] else None
            if channels_last:
                img = img.contiguous(memory_format=torch.channels_last)
                spec = spec.contiguous(memory_format=torch.channels_last)
            if i == 0:
                prob, g1_vals, g2_vals = model.evaluate_conditional_2d(img, spec, fids, 0, 1, fp=fp, grid_bins=200)
            else:
                prob, _, _ = model.evaluate_conditional_2d(img, spec, fids, 0, 1, fp=fp, grid_bins=200)
            probs.append(prob.unsqueeze(0).detach().cpu().numpy())
    probs = np.vstack(probs)
    g1_vals = g1_vals.cpu().numpy()
    g2_vals = g2_vals.cpu().numpy()
    snrs = snrs.cpu().numpy()
    return probs, g1_vals, g2_vals, snrs