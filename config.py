import configargparse

def config_parser():
    parser = configargparse.ArgumentParser()

    # Basic
    parser.add_argument("--expname", type=str)
    parser.add_argument("--config", is_config_file=True)
    # Dataset
    parser.add_argument("--dataset_name", type=str, default='dtu')
    parser.add_argument("--img_root", type=str, default='/home/dev4/data/SKY/datasets/data_download/realestate')
    parser.add_argument("--pose_root", type=str, default='/home/dev4/data/SKY/datasets/poses/realestate')
    parser.add_argument("--imgScale_train", type=int, default=1)
    parser.add_argument("--imgScale_test", type=int, default=1)
    # Trainer
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lrate", type=float, default=5e-4)
    # Checkpoint
    parser.add_argument("--checkpoint_path", type=str, default=None)
    # Loss
    parser.add_argument('--l2_coeff', type=float, default=0.05)
    parser.add_argument('--lpips', action='store_true', default=False)
    parser.add_argument('--depth', action='store_true', default=False)
    # Model
    parser.add_argument('--no_sample', action='store_true', default=False)
    parser.add_argument('--no_latent_concat', action='store_true', default=False)
    parser.add_argument('--no_multiview', action='store_true', default=False)
    parser.add_argument('--no_high_freq', action='store_true', default=False)
    parser.add_argument("--views", type=int, default=2)
    parser.add_argument("--num_queries", type=int, default=100)
    parser.add_argument("--backbone_feature_dim", type=int, default=256)
    
    return parser.parse_args()