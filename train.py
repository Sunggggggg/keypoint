import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp

from config import config_parser
from set_multi_gpus import set_ddp
from dataset import RealEstate10k
import models

def worker_init_fn():
    random.seed(int(torch.utils.data.get_worker_info().seed)%(2**32-1))
    np.random.seed(int(torch.utils.data.get_worker_info().seed)%(2**32-1))

def train(rank, world_size, args):
    # Multi gpus
    print(f"Local gpu id : {rank}, World Size : {world_size}")
    set_ddp(rank, world_size)

    # Dataset
    if args.dataset_name == 'realestate':
        # Train
        img_root = os.path.join(args.img_root, 'train')
        pose_root = os.path.join(args.pose_root, 'train.mat')
        train_dataset = RealEstate10k(img_root, pose_root, 
                                    args.num_ctxt_views, args.num_query_views, args.query_sparsity, 
                                    args.augment, args.lpips)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                drop_last=True, num_workers=8, pin_memory=False, worker_init_fn=worker_init_fn)

        # Val
        img_root = os.path.join(args.img_root, 'val')
        pose_root = os.path.join(args.pose_root, 'val.mat')
        val_dataset = RealEstate10k(img_root, pose_root, num_ctxt_views=args.views, num_query_views=1, augment=False)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True, drop_last=True, num_workers=4, pin_memory=False, worker_init_fn=worker_init_fn)

    elif args.dataset_name == 'adic':
        return

    # Model
    model = models.QueryAttentionRenderer()
    optimizer = torch.optim.Adam(lr=args.lrate, params=model.parameters(), betas=(0.99, 0.999))

    # Checkpoint
    if args.checkpoint_path is not None :
        print(f"Loading weights from {args.checkpoint_path}...")
        state_dict = torch.load(args.checkpoint_path, map_location=torch.device('cpu'))
        state_dict, optimizer_dict = state_dict['model'], state_dict['optimizer']
        
        if args.reconstruct:
            state_dict['latent_codes.weight'] = torch.zeros_like(state_dict['latent_codes.weight'])

        model.load_state_dict(state_dict)
        optimizer.load_state_dict(optimizer_dict)

        # cpu to gpu
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(rank)

    model = model.to(rank)

    # Output dir
    output_dir = os.path.join('outputs', args.expname)      # outputs/(exp)
    os.makedirs(output_dir, exist_ok=True)
    f = os.path.join(output_dir, 'args.txt')
    with open(f, 'w') as file:
        for arg in vars(args):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))

    # training

if __name__ == "__main__":
    args = config_parser()
    world_size = torch.cuda.device_count()
    mp.spawn(train, nprocs=world_size, args=(world_size, args))