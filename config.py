import configargparse

def config_parser():
    parser = configargparse.ArgumentParser()

    # Basic
    parser.add_argument("--expname", type=str, help='experiment name')
    # Dataset
    parser.add_argument("--dataset_name", type=str, default='dtu')
    parser.add_argument("--datadir", type=str, default='/home/dev4/data/SKY/datasets/mvs_trainig/dtu')
    parser.add_argument("--imgScale_train", type=int, default=1)
    parser.add_argument("--imgScale_test", type=int, default=1)
    
    # Trainer
    parser.add_argument("--num_epochs", type=int, default=8)
    parser.add_argument("--lrate", type=float, default=5e-4)
    
    
    return parser