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

    os.environ['NCCL_IB_DISABLE'] = '1'
    os.environ['NCCL_SOCKET_IFNAME'] = 'eno1'
    os.environ['NCCL_P2P_DISABLE'] = '1'
    os.environ['NCCL_BLOCKING_WAIT'] = '1'
    
    world_size = torch.cuda.device_count()

    mp.spawn(train_nn, args=(world_size, ForkCNN, CNNTrainer), nprocs=world_size)