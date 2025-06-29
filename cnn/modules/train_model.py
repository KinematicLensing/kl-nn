import os
from os.path import join
import time
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import pyxis.torch as pxt

from networks import *
from train import *
import config

if __name__ == "__main__":

    os.system(f"mkdir {join(config.train['model_path'], config.train['model_name'])}")
    
    world_size = torch.cuda.device_count()

    mp.spawn(train_nn, args=(world_size, 'train', ForkCNN, CNNTrainer), nprocs=world_size)