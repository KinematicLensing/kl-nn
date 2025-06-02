import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import os

def run_ddp(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    # Minimal model; seed for consistency.
    torch.manual_seed(0)
    model = nn.Linear(10, 10).to(rank)
    print(f"[Rank {rank}] model moved to GPU {rank}")
    dist.barrier()  # Ensure all processes reach this point together
    model = DDP(model, device_ids=[rank])
    print(f"[Rank {rank}] DDP wrapping complete")
    dist.destroy_process_group()

if __name__ == '__main__':
    world_size = 2
    mp.spawn(run_ddp, args=(world_size,), nprocs=world_size, join=True)