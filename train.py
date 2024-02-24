import os
import random
import numpy as np
from tqdm import tqdm, trange
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from config import config_parser
from set_multi_gpus import set_ddp
from data import dataset_dict
from model import NeRF
from encoder import UNet
from queries import VolumeAttention
from utils import *

def train(rank, world_size, args):
    print(f"Local gpu id : {rank}, World Size : {world_size}")
    set_ddp(rank, world_size)

    # Prepare dataset
    dataset = dataset_dict[args.dataset_name]
    train_dataset = dataset(root=args.datadir, split='train', max_len=-1 , downSample=args.imgScale_train)
    val_dataset = dataset(root_dir=args.datadir, split='val', max_len=10 , downSample=args.imgScale_test)

    train_loader = DataLoader(train_dataset, shuffle=True, num_workers=8, batch_size=1, pin_memory=True)
    val_loader = DataLoader(val_dataset, shuffle=False, num_workers=1, batch_size=1, pin_memory=True)
    
    # Model(NeRF)
    nerf = NeRF(
        use_viewdirs=args.use_viewdirs,
        randomized=args.randomized,
        white_bkgd=args.white_bkgd,
        num_levels=args.num_levels,
        N_samples=args.N_samples,
        hidden=args.hidden,
        density_noise=args.density_noise,
        min_deg=args.min_deg, 
        max_deg=args.max_deg,
        viewdirs_min_deg=args.viewdirs_min_deg,
        viewdirs_max_deg=args.viewdirs_max_deg,
        device=torch.device(rank)
    )

    # Query base Cost Volume
    unet = UNet(3, args.query_dim)
    queries = VolumeAttention(num_query=args.num_query, query_dim=args.query_dim)
    
    grad_vars = []
    grad_vars += list(nerf.parameters())
    grad_vars += list(unet.parameters())
    grad_vars += list(queries.parameters())

    # Optimizer
    optimizer = optim.Adam(grad_vars, lr=args.lrate, betas=(0.9, 0.999))
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-7)
    
    # 
    for epoch in range(args.num_epochs) :
        for batch in train_loader :
            # 
            nerf.train()
            unet.train()
            queries.train()

            # training_step
            if 'scan' in batch.keys():
                batch.pop('scan')
            loss = 0

            data_mvs, pose_ref = decode_batch(batch, torch.device(rank))
            imgs = data_mvs['images']           # [B, N, 3, H, W] B=1
            proj_mats = data_mvs['proj_mats']   # [B, N, 3, 4] 
            near_fars = data_mvs['near_fars']
            depths_h = data_mvs['depths_h']

            # Feature extract
            feat1 = unet(imgs[0, :1])           
            feat2 = unet(imgs[0, 1:])           

            feat1, feat2 = queries(feat1, feat2)  # [1, 3, H, W]

            #
            imgs = unpreprocess(imgs)
            N_rays, N_samples = args.batch_size, args.N_samples
            c2ws, w2cs, intrinsics = pose_ref['c2ws'], pose_ref['w2cs'], pose_ref['intrinsics']

            rays_pts, rays_dir, target_s, rays_NDC, depth_candidates, rays_o, rays_depth, ndc_parameters = \
                build_rays(imgs, depths_h, pose_ref, w2cs, c2ws, intrinsics, near_fars, N_rays, N_samples, pad=args.pad)



if __name__ == '__main__' :
    parser = config_parser()
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    mp.spawn(train, nprocs=world_size, args=(world_size, args))