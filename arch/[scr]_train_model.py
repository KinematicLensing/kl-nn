import os
from os.path import join
import argparse
import time
import torch
import torch.multiprocessing as mp

from networks import *
from train import *
import config
from model_registry import save_model_artifacts

def parse_args():
    parser = argparse.ArgumentParser(description="Train a neural network model")
    parser.add_argument("--model", type=str, default="VICRegPretrain", help="Model architecture to use")
    parser.add_argument("--trainer", type=str, default="FETrainer", help="Trainer class to use")
    parser.add_argument("--train_type", type=str, default='pretrain', help="pretrain or train")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.train_type == 'pretrain':
        train_config = config.pretrain
    else:
        train_config = config.train
    artifacts = save_model_artifacts(config.MODEL_CONFIG, train_type=args.train_type, overwrite=True)
    print(f"Saved model config JSON: {artifacts['config_path']}")
    print(f"Saved networks snapshot: {artifacts['network_path']}")
    os.makedirs(join(train_config['model_path'], train_config['model_name']), exist_ok=True)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1,3,5,6,7" #(Put the number(s) you want for the GPUs)
    model_class = globals()[args.model]
    trainer_class = globals()[args.trainer]
    
    world_size = torch.cuda.device_count() # 1
    print("Training with {} GPUs".format(world_size))

    mp.spawn(train_nn, args=(world_size, model_class, trainer_class, 1, args.train_type), nprocs=world_size)
