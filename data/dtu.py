import os
import numpy as np
import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from utils import read_pfm

class MVSDatasetDTU(Dataset) :
    def __init__(self, root_dir, split, n_views=2, levels=1, img_wh=None, downSample=1.0, max_len=-1):
        self.root_dir = root_dir
        self.split = split
        self.n_views = n_views
        self.img_wh = img_wh
        self.downSample = downSample
        self.max_len = max_len
        #
        self.scale_factor = 1.0 / 200

        # 
        self.build_metas()
        self.build_proj_mats()
        self.define_transforms()

    def build_metas(self):
        '''
        dtu_{self.split}_all.txt    : train/val/test scan is fixed
        dtu_pairs.txt               : ref, src image idx is fixed
        '''
        self.metas = []
        with open(f'configs/lists/dtu_{self.split}_all.txt') as f:
            self.scans = [line.rstrip() for line in f.readlines()]

        light_idxs = [3] if 'train' != self.split else range(7)
        self.id_list = []

        for scan in self.scans:
            with open(f'configs/dtu_pairs.txt') as f:
                num_viewpoint = int(f.readline()) # 49
                for _ in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())   # 0
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]] # [10, 1, 9, 12, 11, 13, 2, 8, 14, 27]
                    for light_idx in light_idxs:
                        self.metas += [(scan, light_idx, ref_view, src_views)]
                        self.id_list.append([ref_view] + src_views)

        self.id_list = np.unique(self.id_list)
        self.build_remap()
    
    def build_remap(self):
        self.remap = np.zeros(np.max(self.id_list) + 1).astype('int')
        for i, item in enumerate(self.id_list):
            self.remap[item] = i
    
    def build_proj_mats(self):
        proj_mats, near_far, intrinsics, world2cams, cam2worlds = [], [], [], [], []
        for vid in self.id_list:
            proj_mat_filename = os.path.join(self.root_dir,
                                             f'Cameras/train/{vid:08d}_cam.txt')
            intrinsic, extrinsic, near_far_l = self.read_cam_file(proj_mat_filename)
            intrinsic[:2] *= 4
            extrinsic[:3, 3] *= self.scale_factor

            intrinsic[:2] = intrinsic[:2] * self.downSample
            intrinsics += [intrinsic.copy()]

            # multiply intrinsics and extrinsics to get projection matrix
            proj_mat_l = np.eye(4)
            intrinsic[:2] = intrinsic[:2] / 4
            proj_mat_l[:3, :4] = intrinsic @ extrinsic[:3, :4]

            proj_mats += [proj_mat_l]
            near_far += [near_far_l]
            world2cams += [extrinsic]
            cam2worlds += [np.linalg.inv(extrinsic)]

        self.proj_mats, self.near_far, self.intrinsics = np.stack(proj_mats), np.stack(near_far),np.stack(intrinsics)
        self.world2cams, self.cam2worlds = np.stack(world2cams), np.stack(cam2worlds)
    
    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = [line.rstrip() for line in f.readlines()]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ')
        extrinsics = extrinsics.reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ')
        intrinsics = intrinsics.reshape((3, 3))
        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0]) * self.scale_factor
        depth_max = depth_min + float(lines[11].split()[1]) * 192 * self.scale_factor
        self.depth_interval = float(lines[11].split()[1])
        return intrinsics, extrinsics, [depth_min, depth_max]
    
    def define_transforms(self):
        self.transform = T.Compose([T.ToTensor(),
                                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
        
    def read_depth(self, filename):
        depth_h = np.array(read_pfm(filename)[0], dtype=np.float32)  # (800, 800)
        depth_h = cv2.resize(depth_h, None, fx=0.5, fy=0.5,
                             interpolation=cv2.INTER_NEAREST)  # (600, 800)
        depth_h = depth_h[44:556, 80:720]  # (512, 640)
        depth_h = cv2.resize(depth_h, None, fx=self.downSample, fy=self.downSample,
                             interpolation=cv2.INTER_NEAREST)  # !!!!!!!!!!!!!!!!!!!!!!!!!
        depth = cv2.resize(depth_h, None, fx=1.0 / 4, fy=1.0 / 4,
                           interpolation=cv2.INTER_NEAREST)  # !!!!!!!!!!!!!!!!!!!!!!!!!
        mask = depth > 0

        return depth, mask, depth_h
    
    def __len__(self):
        return len(self.metas) if self.max_len <= 0 else self.max_len

    def __getitem__(self, idx):
        sample = {}
        scan, light_idx, target_view, src_views = self.metas[idx]
        if self.split=='train':
            ids = torch.randperm(5)[:self.n_views-1]
            view_ids = [src_views[i] for i in ids] + [target_view]
        else:
            view_ids = [src_views[i] for i in range(self.n_views-1)] + [target_view]
        
        affine_mat, affine_mat_inv = [], []
        imgs, depths_h = [], []
        proj_mats, intrinsics, w2cs, c2ws, near_fars = [], [], [], [], []  # record proj mats between views
        for i, vid in enumerate(view_ids):
            img_filename = os.path.join(self.root_dir,
                                        f'Rectified/{scan}_train/rect_{vid + 1:03d}_{light_idx}_r5000.png')
            depth_filename = os.path.join(self.root_dir,
                                          f'Depths/{scan}/depth_map_{vid:04d}.pfm')

            # Read image
            img = Image.open(img_filename)
            img_wh = np.round(np.array(img.size) * self.downSample).astype('int')
            img = img.resize(img_wh, Image.BILINEAR)
            img = self.transform(img)
            imgs += [img]

            # Read parameters
            index_mat = self.remap[vid]
            proj_mat_ls = self.proj_mats[index_mat]
            near_far = self.near_far[index_mat]
            intrinsics.append(self.intrinsics[index_mat])
            w2cs.append(self.world2cams[index_mat])
            c2ws.append(self.cam2worlds[index_mat])

            affine_mat.append(proj_mat_ls)
            affine_mat_inv.append(np.linalg.inv(proj_mat_ls))
            if i == 0:  # reference view
                ref_proj_inv = np.linalg.inv(proj_mat_ls)
                proj_mats += [np.eye(4)]
            else:
                proj_mats += [proj_mat_ls @ ref_proj_inv]

            if os.path.exists(depth_filename):
                depth, mask, depth_h = self.read_depth(depth_filename)
                depth_h *= self.scale_factor
                depths_h.append(depth_h)
            else:
                depths_h.append(np.zeros((1, 1)))

            near_fars.append(near_far)

        imgs = torch.stack(imgs).float()
        depths_h = np.stack(depths_h)
        proj_mats = np.stack(proj_mats)[:, :3]
        affine_mat, affine_mat_inv = np.stack(affine_mat), np.stack(affine_mat_inv)
        intrinsics, w2cs, c2ws, near_fars = np.stack(intrinsics), np.stack(w2cs), np.stack(c2ws), np.stack(near_fars)
        view_ids_all = [target_view] + list(src_views) if type(src_views[0]) is not list else [j for sub in src_views for j in sub]
        c2ws_all = self.cam2worlds[self.remap[view_ids_all]]

        sample['images'] = imgs  # (V, H, W, 3)
        sample['depths_h'] = depths_h.astype(np.float32)  # (V, H, W)
        sample['w2cs'] = w2cs.astype(np.float32)  # (V, 4, 4)
        sample['c2ws'] = c2ws.astype(np.float32)  # (V, 4, 4)
        sample['near_fars'] = near_fars.astype(np.float32)
        sample['proj_mats'] = proj_mats.astype(np.float32)
        sample['intrinsics'] = intrinsics.astype(np.float32)  # (V, 3, 3)
        sample['view_ids'] = np.array(view_ids)
        sample['light_id'] = np.array(light_idx)
        sample['affine_mat'] = affine_mat
        sample['affine_mat_inv'] = affine_mat_inv
        sample['scan'] = scan
        sample['c2ws_all'] = c2ws_all.astype(np.float32)

        return sample