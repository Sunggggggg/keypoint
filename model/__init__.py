import torch
from .NeRF import NeRF
from .queries import VolumeAttention
from .encoder import UNet

def create_model(args, rank):
    grad_vars = []
    # NeRF
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
    grad_vars += list(nerf.parameters())

    # Query base Cost Volume
    unet = UNet(3, args.query_dim).to(rank)
    queries_attention = VolumeAttention(num_query=args.num_query, query_dim=args.query_dim).to(rank)
    
    grad_vars += list(unet.parameters())
    grad_vars += list(queries_attention.parameters())

    return nerf, unet, queries_attention, grad_vars