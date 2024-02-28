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
    parser.add_argument("--num_epochs", type=int, default=8)
    parser.add_argument("--lrate", type=float, default=5e-4)
    # Checkpoint
    parser.add_argument("--checkpoint_path", type=str, default=None)
    # Model
    parser.add_argument("--num_ctxt_views", type=int, default=2)
    parser.add_argument("--num_query_views", type=int, default=1)
    parser.add_argument("--query_sparsity", type=int, default=192)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--lpips", action="store_true")

    parser.add_argument("--num_query_views", type=int, default=1)
    parser.add_argument("--num_query_views", type=int, default=1)
    parser.add_argument("--num_query_views", type=int, default=1)

    return parser.parse_args()