import configargparse

def config_parser():
    parser = configargparse.ArgumentParser()

    # Basic
    parser.add_argument("--expname", type=str, help='experiment name')
    parser.add_argument("--basedir", type=str, default='logs')
    parser.add_argument("--eval", action='store_true', help='Eval mode')
    parser.add_argument('--nerf_config', is_config_file=True, help='config file path')
    parser.add_argument("--nerf_weight", type=str, default=None)
    # Dataset
    parser.add_argument("--datadir", type=str, default='/mnt2/SKY/dataset/nerf_synthetic/lego')
    parser.add_argument("--dataset_type", type=str, default='blender')
    parser.add_argument('--scale', default=4, type=int)
    parser.add_argument("--testskip", type=int, default=8)
    parser.add_argument("--llff_train_views", type=int, nargs="+", default=[])
    # Optimizer and scheduler
    parser.add_argument("--lr_init", type=float, default=1e-3)      # 0.001    
    parser.add_argument("--weight_decay", type=float, default=1e-5) 
    parser.add_argument("--lr_final", type=float, default=5e-5)     # 0.00005
    parser.add_argument("--lr_delay_steps", type=int, default=2500)
    parser.add_argument("--lr_delay_mult", type=float, default=0.1)
    # Loss
    parser.add_argument("--coarse_weight_decay", type=float, default=0.1)
    # Training hyperparams
    parser.add_argument("--max_iters", type=int, default=200000)
    parser.add_argument("--chunk", type=int, default=1024*8)
    parser.add_argument("--netchunk", type=int, default=1024*64)
    parser.add_argument("--N_rand", type=int, default=1024*4, help='#of Sampling rays')
    parser.add_argument("--no_ndc", action='store_true')
    parser.add_argument("--precrop_iters", type=int, default=0)
    parser.add_argument("--precrop_frac", type=float, default=0.5) 
    # Model 
    parser.add_argument("--use_viewdirs", action="store_true", help='use full 5D input instead of 3D')
    parser.add_argument("--randomized", action="store_true")
    parser.add_argument("--white_bkgd", action="store_true")           # should be False if using llff
    parser.add_argument("--num_levels", type=int, default=2)
    parser.add_argument("--N_samples", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--density_noise", type=float, default=0.0)
    parser.add_argument("--density_bias", type=float, default=-1.0)
    parser.add_argument("--min_deg", type=int, default=0)
    parser.add_argument("--max_deg", type=int, default=16)
    parser.add_argument("--viewdirs_min_deg", type=int, default=0)
    parser.add_argument("--viewdirs_max_deg", type=int, default=4)
    # logging/saving options
    parser.add_argument("--i_print", type=int, default=10)
    parser.add_argument("--i_weights", type=int, default=100)
    parser.add_argument("--i_testset", type=int, default=100)
    parser.add_argument("--i_video",   type=int, default=10000)
    
    return parser