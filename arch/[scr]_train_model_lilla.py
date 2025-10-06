import os
from os.path import join
import time
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import pyxis.torch as pxt

from networks import *
from train_lilla import *
import config

if __name__ == "__main__":

    os.system(f"mkdir {join(config.train['model_path'], config.train['model_name'])}")
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,3,5,6,7" #(Put the number(s) you want for the GPUs)
    
    world_size = torch.cuda.device_count() # 1

    mp.spawn(train_nn, args=(world_size, ForkCNN, CNNTrainer), nprocs=world_size)