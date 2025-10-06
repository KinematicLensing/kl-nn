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

    os.environ["CUDA_VISIBLE_DEVICES"] = "1,3,5,6,7" #(Put the number(s) you want for the GPUs)
    
    world_size = torch.cuda.device_count() # 1

    mp.spawn(predict, args=(world_size, ForkCNN), nprocs=world_size)