from dataset import RealEstate10k

img_root = '/home/dev4/data/SKY/datasets/data_download/realestate/train'
pose_root = '/home/dev4/data/SKY/datasets/poses/realestate/train.mat'
num_ctxt_views = 2
num_query_views = 1


train_dataset = RealEstate10k(img_root, pose_root, num_ctxt_views, num_query_views, query_sparsity=192, augment=True, lpips=True)