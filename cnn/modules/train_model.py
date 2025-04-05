import os
from os.path import join
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from astropy.io import fits
import pyxis.torch as pxt

from networks import ForkCNN
from train import *
from dataset import FiberDataset
import config

if __name__ == "__main__":
    
    world_size = torch.cuda.device_count()
    save_every = 1
    nepochs = config.train['epoch_number']
    batch_size = config.train['batch_size']
    nfeatures = config.train['feature_number']

    mp.spawn(train_nn, args=(world_size, save_every, nepochs, batch_size, nfeatures), nprocs=world_size)