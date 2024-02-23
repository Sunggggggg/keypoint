import os
import random
import numpy as np
from tqdm import tqdm, trange
import torch
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from config import config_parser
from set_multi_gpus import set_ddp
from dataset import load_data
from encoder import UNet
from queries import VolumeAttention

def train(rank, world_size, args):
    print(f"Local gpu id : {rank}, World Size : {world_size}")
    set_ddp(rank, world_size)

    


    

unet = UNet(3, 64)
queries = VolumeAttention(num_query=16, query_dim=64)
feat1 = unet(x1)
feat2 = unet(x2)

queries(feat1, feat2)

if __name__ == '__main__' :
    parser = config_parser()
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    mp.spawn(train, nprocs=world_size, args=(world_size, args))