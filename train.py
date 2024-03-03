import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from tqdm import tqdm
from utils import util
from config import config_parser
from set_multi_gpus import set_ddp, average_gradients
from dataset import RealEstate10k
import loss_functions
import models
from collections import defaultdict
import imageio

to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

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
        print("Train dataset load : ", img_root, pose_root)
        train_dataset = RealEstate10k(img_root=img_root, pose_root=pose_root, num_ctxt_views=args.views, num_query_views=args.num_query_views, 
                                      query_sparsity=args.query_sparsity, augment=args.augment, lpips=args.lpips)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=False)
        
        # Val
        img_root = os.path.join(args.img_root, 'test')
        pose_root = os.path.join(args.pose_root, 'test.mat')
        print("Test dataset load : ", img_root, pose_root)
        val_dataset = RealEstate10k(img_root=img_root, pose_root=pose_root, num_ctxt_views=args.views, num_query_views=args.num_query_views, augment=False)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=False)

    elif args.dataset_name == 'adic':
        return

    # Model
    model = models.CrossAttentionRenderer(no_sample=args.no_sample, no_latent_concat=args.no_latent_concat,
                                    no_multiview=args.no_multiview, no_high_freq= args.no_high_freq, 
                                    n_view=args.views, num_queries=args.num_queries, feature_dim=args.backbone_feature_dim)
    optimizer = torch.optim.Adam(lr=args.lrate, params=model.parameters(), betas=(0.99, 0.999))
    print("Build Model...!")

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
    fig_dir = os.path.join(output_dir, 'render')
    os.makedirs(fig_dir, exist_ok=True)
    f = os.path.join(output_dir, 'args.txt')
    with open(f, 'w') as file:
        for arg in vars(args):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))

    checkpoints_dir = os.path.join(os.path.join('outputs', args.expname), 'checkpoints')
    
    # Loss
    loss_fn = val_loss_fn = loss_functions.LFLoss(args.l2_coeff, args.lpips, args.depth)
    # training
    epochs = args.epochs
    total_steps = 0
    
    print("Train start...!")
    
    for epoch in range(epochs):
        print(f"[Epoch] {epoch} Load dataset Train dataset : {len(train_dataset)} Test dataset : {len(val_dataset)}")
        for step, (model_input, gt) in enumerate(train_dataloader):
            model_input = util.dict_to_gpu(model_input)
            gt = util.dict_to_gpu(gt)

            model_output = model(model_input)
            losses, loss_summaries = loss_fn(model_output, gt, model=model)

            train_loss = 0.
            for loss_name, loss in losses.items():
                single_loss = loss.mean()
                train_loss += single_loss
            
            optimizer.zero_grad()
            train_loss.backward()

            if world_size > 1:
                average_gradients(model)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=False)

            optimizer.step()
            
            if rank == 0:
                total_steps += 1
                tqdm.write(f"[Epoch: {epoch}] [Iter: {total_steps}] Loss: {train_loss.item():.3f}\t ")
            del train_loss

        if val_dataloader is not None and total_steps % 100 == 0:
            print("Running validation set...")
            with torch.no_grad():
                model.eval()
                val_losses = defaultdict(list)
                for val_i, (model_input, gt) in enumerate(val_dataloader):
                    print("processing valid")
                    model_input = util.dict_to_gpu(model_input)
                    gt = util.dict_to_gpu(gt)

                    model_input_full = model_input
                    rgb_full = model_input['query']['rgb']
                    uv_full = model_input['query']['uv']
                    nrays = uv_full.size(2)         # 
                    # chunks = nrays // 512 + 1
                    chunks = nrays // 512 + 1
                    # chunks = nrays // 384 + 1

                    z = model.get_z(model_input)

                    rgb_chunks = torch.chunk(rgb_full, chunks, dim=2)
                    uv_chunks = torch.chunk(uv_full, chunks, dim=2)

                    model_outputs = []

                    for rgb_chunk, uv_chunk in zip(rgb_chunks, uv_chunks):
                        model_input['query']['rgb'] = rgb_chunk
                        model_input['query']['uv'] = uv_chunk
                        model_output = model(model_input, z=z, val=True)
                        del model_output['z']
                        del model_output['coords']
                        del model_output['at_wts']

                        model_output['pixel_val'] = model_output['pixel_val'].cpu()

                        model_outputs.append(model_output)

                    model_output_full = {}

                    for k in model_outputs[0].keys():
                        outputs = [model_output[k] for model_output in model_outputs]

                        if k == "pixel_val":
                            val = torch.cat(outputs, dim=-3)
                        else:
                            val = torch.cat(outputs, dim=-2)
                        model_output_full[k] = val

                    model_output = model_output_full
                    model_input['query']['rgb'] = rgb_full

                    val_loss, val_loss_smry = val_loss_fn(model_output, gt, val=True, model=model)

                    for name, value in val_loss.items():
                        val_losses[name].append(value)

                    # Render a video

                    # if val_i == batches_per_validation:
                    break

                for loss_name, loss in val_losses.items():
                    single_loss = np.mean(np.concatenate([l.reshape(-1).cpu().numpy() for l in loss], axis=0))

                if rank == 0:
                    rgbs = model_output_full['rgb'].reshape(args.batch_size, 256, 256, 3) # [B, H, W, 3]
                    rgb8 = to8b(rgbs[-1].cpu().numpy())
                    filename = os.path.join(fig_dir, f'{epoch}_{total_steps}.png')
                    imageio.imwrite(filename, rgb8)
                    print("Save Rendered images ", rgb8.shape)
            model.train()
    
        if rank == 0:
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()},
                    os.path.join(checkpoints_dir, 'model_final.pth'))
            
if __name__ == "__main__":
    args = config_parser()
    world_size = torch.cuda.device_count()
    mp.spawn(train, nprocs=world_size, args=(world_size, args))