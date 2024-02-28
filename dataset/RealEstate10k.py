import random
import os
import os.path as osp
import torch
import numpy as np
from glob import glob
import sys

current_dir = osp.dirname(os.path.abspath(__file__))
sys.path.append(current_dir + '/../')

from utils import data_util, util
import json
from collections import defaultdict
import os.path as osp
from imageio import imread
from torch.utils.data import Dataset
from pathlib import Path
import cv2
from tqdm import tqdm
from scipy.io import loadmat

def augment(rgb, intrinsics, c2w_mat):
    """
    """

    # Vertical Flip with 50% probability
    # if np.random.uniform(0, 1) < 0.2:
    #     rgb = rgb[::-1, :, :]
    #     tf_flip = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    #     c2w_mat = c2w_mat @ tf_flip

    # Horizontal Flip with 50% Probability
    # Filp along x axis
    if np.random.uniform(0, 1) < 0.5:
        rgb = rgb[:, ::-1, :]
        tf_flip = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        c2w_mat = c2w_mat @ tf_flip

    # Crop by aspect ratio
    # rgm : [H, W ,3]
    if np.random.uniform(0, 1) < 0.5:
        py = np.random.randint(1, 32)
        rgb = rgb[py:-py, :, :]
    else:
        py = 0

    if np.random.uniform(0, 1) < 0.5:
        px = np.random.randint(1, 32)
        rgb = rgb[:, px:-px, :]
    else:
        px = 0

    H, W, _ = rgb.shape
    rgb = cv2.resize(rgb, (256, 256))
    xscale = 256 / W
    yscale = 256 / H

    intrinsics[0, 0] = intrinsics[0, 0] * xscale
    intrinsics[1, 1] = intrinsics[1, 1] * yscale

    return rgb, intrinsics, c2w_mat

class Camera(object):
    """ For RealExtate dataset format
    """
    def __init__(self, entry):
        fx, fy, cx, cy = entry[1:5]
        self.intrinsics = np.array([[fx, 0, cx, 0],
                                    [0, fy, cy, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]])
        w2c_mat = np.array(entry[7:]).reshape(3, 4)
        w2c_mat_4x4 = np.eye(4)
        w2c_mat_4x4[:3, :] = w2c_mat
        self.w2c_mat = w2c_mat_4x4
        self.c2w_mat = np.linalg.inv(w2c_mat_4x4)

def parse_pose(pose, timestep):
    """ 
    pose : [N, ]
    """
    timesteps = pose[:, :1]
    timesteps = np.around(timesteps)
    mask = (timesteps == timestep)[:, 0]
    pose_entry = pose[mask][0]
    camera = Camera(pose_entry)

    return camera

def unnormalize_intrinsics(intrinsics, h, w):
    intrinsics = intrinsics.copy()
    intrinsics[0] *= w
    intrinsics[1] *= h
    return intrinsics

class RealEstate10k():
    def __init__(self, img_root, pose_root,
                 num_ctxt_views, num_query_views, query_sparsity=None,
                 max_num_scenes=None, square_crop=True, augment=True, lpips=False):
        print("Loading RealEstate10k...")
        self.num_ctxt_views = num_ctxt_views
        self.num_query_views = num_query_views
        self.query_sparsity = query_sparsity

        # Load dataset (img, pose)
        all_im_dir = Path(img_root)
        self.all_pose = loadmat(pose_root)
        self.lpips = lpips
        self.eval = eval

        self.all_scenes = sorted(all_im_dir.glob('*/'))
        dummy_img_path = str(next(self.all_scenes[0].glob("*.npz")))

        if max_num_scenes:
            self.all_scenes = list(self.all_scenes)[:max_num_scenes]

        # For show
        data = np.load(dummy_img_path)
        key = list(data.keys())[0]
        im = data[key]

        print(im.shape)
        H, W = im.shape[:2]
        H, W = 256, 455
        self.H, self.W = H, W
        self.augment = augment

        self.square_crop = square_crop

        xscale = W / min(H, W)
        yscale = H / min(H, W)

        dim = min(H, W)

        self.xscale = xscale
        self.yscale = yscale

        # For now the images are already square cropped
        # Original image size : self.H, self.W
        # Square crop : dim
        self.H = 256
        self.W = 455

        print(f"Resolution is {H}, {W}.")

        if self.square_crop:
            i, j = torch.meshgrid(torch.arange(0, dim), torch.arange(0, dim))
        else:
            i, j = torch.meshgrid(torch.arange(0, W), torch.arange(0, H))

        self.uv = torch.stack([i.float(), j.float()], dim=-1).permute(1, 0, 2)  # [W, H, 2] -> [H, W, 2]

        self.uv = self.uv[None].permute(0, -1, 1, 2).permute(0, 2, 3, 1)        # [1, H, W, 2] [1, 2, H, W] [1, H, W, 2]
        self.uv = self.uv.reshape(-1, 2)                # [H, W, 2]

        self.scene_path_list = list(Path(img_root).glob("*/"))

    def __len__(self):
        return len(self.all_scenes)
        
    def __getitem__(self, idx):
        scene_path = self.all_scenes[idx]
        npz_files = sorted(scene_path.glob("*.npz"))

        name = scene_path.name          # 
        # if dont find scene, random select scene
        if name not in self.all_pose:
            return self.__getitem__(random.randint(0, len(self.all_scenes) - 1))

        if len(npz_files) == 0:
            return self.__getitem__(random.randint(0, len(self.all_scenes) - 1))
        
        pose = self.all_pose[name]
        npz_file = npz_files[0]
        try:
            data = np.load(npz_file)
        except:
            print(npz_file)
            return self.__getitem__(random.randint(0, len(self.all_scenes) - 1))

        rgb_files = list(data.keys())
        window_size = 128

        if len(rgb_files) <= 10:
            return self.__getitem__(random.randint(0, len(self.all_scenes) - 1))

        timestamps = [int(rgb_file.split('.')[0]) for rgb_file in rgb_files]
        sorted_ids = np.argsort(timestamps)

        rgb_files = np.array(rgb_files)[sorted_ids]
        timestamps = np.array(timestamps)[sorted_ids]

        assert (timestamps == sorted(timestamps)).all()
        num_frames = len(rgb_files)
        left_bound = 0
        right_bound = num_frames - 1
        candidate_ids = np.arange(left_bound, right_bound)
        nframe = 1
        nframe_view = 92

        if len(candidate_ids) < self.num_ctxt_views:
            return self.__getitem__(random.randint(0, len(self.all_scenes) - 1))

        id_feats = []

        # Select num_ctxt_views frame, (current frame nframe_view = 92)
        for i in range(self.num_ctxt_views):
            if len(candidate_ids) == 0:
                return self.__getitem__(random.randint(0, len(self.all_scenes) - 1))

            id_feat = np.random.choice(candidate_ids, size=1, replace=False)
            candidate_ids = candidate_ids[(candidate_ids < (id_feat - nframe_view)) | (candidate_ids > (id_feat + nframe_view))]

            id_feats.append(id_feat.item())

        id_feat = np.array(id_feats)

        # num_query_views : view sampling btw [nframe_view, [low, high]]
        if self.num_ctxt_views == 2:
            low = np.min(id_feat) - 64
            high = np.max(id_feat) + 64
            low = max(low, 0)
            high = min(high, num_frames - 1)

            if high <= low:
                return self.__getitem__(random.randint(0, len(self.all_scenes) - 1))

            id_render = np.random.randint(low=low, high=high, size=self.num_query_views)
        elif self.num_ctxt_views == 1:
            low = np.min(id_feat) - 64
            high = np.max(id_feat) + 64
            low = max(low, 0)
            high = min(high, num_frames - 1)

            if high <= low:
                return self.__getitem__(random.randint(0, len(self.all_scenes) - 1))

            id_render = np.random.randint(low=low, high=high, size=self.num_query_views)
        elif self.num_ctxt_views == 3:
            low = np.min(id_feat) + 64
            high = np.max(id_feat) - 64

            if high <= low:
                return self.__getitem__(random.randint(0, len(self.all_scenes) - 1))

            id_render = np.random.randint(low=low, high=high, size=self.num_query_views)
        else:
            assert False

        # make query GTs
        query_rgbs = []
        query_intrinsics = []
        query_c2w = []
        uvs = []

        for id in id_render:
            rgb_file = rgb_files[id]
            rgb = data[rgb_file]

            if rgb.shape[0] == 360:
                rgb = cv2.resize(rgb, (self.W, self.H))

            if self.square_crop:
                rgb = data_util.square_crop_img(rgb)

            cam_param = parse_pose(pose, timestamps[id])
            intrinsics = unnormalize_intrinsics(cam_param.intrinsics, self.H, self.W)

            if self.square_crop:
                intrinsics[0, 2] = intrinsics[0, 2] / self.xscale
                intrinsics[1, 2] = intrinsics[1, 2] / self.yscale

            if self.augment:
                rgb, intrinsics, cam_param.c2w_mat = augment(rgb, intrinsics, cam_param.c2w_mat)

            rgb = rgb.astype(np.float32) / 127.5 - 1    # Normalized
            rgb = rgb.reshape((-1, 3))      # [HW, 3]

            mask_lpips = 0.0

            if self.query_sparsity is not None:
                if self.lpips:
                    mask_lpips = random.randint(0, 1)
                    if mask_lpips:
                        uv = self.uv
                        uv = uv.reshape((256, 256, 2))
                        rgb = rgb.reshape((256, 256, 3))
                        offset = 32
                        x_offset, y_offset =  np.random.randint(0, 256-offset), np.random.randint(0, 256-offset)

                        uv_select = uv[y_offset:y_offset+offset, x_offset:x_offset+offset]
                        rgb_select = rgb[y_offset:y_offset+offset, x_offset:x_offset+offset]
                        uv = uv_select.reshape((-1, 2))     # [hw, 2]
                        rgb = rgb_select.reshape((-1, 3))   # [hw, 2]
                    else:
                        uv = self.uv
                        rix = np.random.permutation(uv.shape[0])
                        rix = rix[:1024]
                        uv = uv[rix]
                        rgb = rgb[rix]
                else:
                    uv = self.uv
                    rix = np.random.permutation(uv.shape[0])
                    rix = rix[:self.query_sparsity]
                    uv = uv[rix]
                    rgb = rgb[rix]
            else:
                uv = self.uv

            uvs.append(uv)
            query_rgbs.append(rgb)
            query_intrinsics.append(intrinsics)
            query_c2w.append(cam_param.c2w_mat)

        uvs = torch.Tensor(np.stack(uvs, axis=0)).float()   # [num_query_views, h, w, 2]
        ctxt_rgbs = []
        ctxt_intrinsics = []
        ctxt_c2w = []

        # make context GTs
        for id in id_feat:
            rgb_file = rgb_files[id]
            rgb = data[rgb_file]

            if rgb.shape[0] == 360:
                rgb = cv2.resize(rgb, (self.W, self.H))

            if self.square_crop:
                rgb = data_util.square_crop_img(rgb)

            cam_param = parse_pose(pose, timestamps[id])

            intrinsics = unnormalize_intrinsics(cam_param.intrinsics, self.H, self.W)

            if self.square_crop:
                intrinsics[0, 2] = intrinsics[0, 2] / self.xscale
                intrinsics[1, 2] = intrinsics[1, 2] / self.yscale

            if self.augment:
                rgb, intrinsics, cam_param.c2w_mat = augment(rgb, intrinsics, cam_param.c2w_mat)

            rgb = rgb.astype(np.float32) / 127.5 - 1

            ctxt_rgbs.append(rgb)
            ctxt_intrinsics.append(intrinsics)
            ctxt_c2w.append(cam_param.c2w_mat)

        ctxt_rgbs = np.stack(ctxt_rgbs)
        ctxt_intrinsics = np.stack(ctxt_intrinsics)
        ctxt_c2w = np.stack(ctxt_c2w)

        query_rgbs = np.stack(query_rgbs)
        query_intrinsics = np.stack(query_intrinsics)
        query_c2w = np.stack(query_c2w)

        # the 'mask' variable determines whether to enforce lpips / depth loss on a set of a points
        # (only) enforced if the points are sampled in a contiguous grid
        query = {'rgb': torch.from_numpy(query_rgbs).float(),
                 'cam2world': torch.from_numpy(query_c2w).float(),
                 'intrinsics': torch.from_numpy(query_intrinsics).float(),
                 'uv': uvs,
                 'mask': mask_lpips}

        ctxt = {'rgb': torch.from_numpy(ctxt_rgbs).float(),
                'cam2world': torch.from_numpy(ctxt_c2w).float(),
                'intrinsics': torch.from_numpy(ctxt_intrinsics).float()}

        return {'query': query, 'context': ctxt}, query