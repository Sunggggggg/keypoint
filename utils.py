import torch

def feat_mul_queries(feat, queries):
    B, E, H, W = feat.shape
    B, Q, E = queries.shape

    keypoint_map = torch.matmul(feat.view(B, E, H*W).permute(0, 2, 1), queries.permute(0, 2, 1))
    keypoint_map_norm = torch.softmax(keypoint_map, dim=1)  # [B, HW, Q]
    keypoint_map_norm = keypoint_map_norm.permute(0, 2, 1).view(B, Q, H, W)
    return keypoint_map_norm
