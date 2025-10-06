import os
from os.path import join
import time
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import pyxis.torch as pxt

from networks import *
from train_cali import *
import config

if __name__ == "__main__":
    
    os.system(f"mkdir {join(config.cali['model_path'], config.cali['model_name'])}")
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,3,5,6,7" #(Put the number(s) you want for the GPUs)
    
    world_size = torch.cuda.device_count() # 1

    mp.spawn(train_cali, args=(world_size, CaliNN), nprocs=world_size)