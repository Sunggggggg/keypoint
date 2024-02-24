import re, torch
import numpy as np

def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale

def sub_selete_data(data_batch, device, idx, filtKey=[], filtIndex=['view_ids_all','c2ws_all','scan','bbox','w2ref','ref2w','light_id','ckpt','idx']):
    data_sub_selete = {}
    for item in data_batch.keys():
        data_sub_selete[item] = data_batch[item][:,idx].float() if (item not in filtIndex and torch.is_tensor(item) and item.dim()>2) else data_batch[item].float()
        if not data_sub_selete[item].is_cuda:
            data_sub_selete[item] = data_sub_selete[item].to(device)
    return data_sub_selete

def decode_batch(batch, device, idx=list(torch.arange(4))):

    data_mvs = sub_selete_data(batch, device, idx, filtKey=[])
    pose_ref = {'w2cs': data_mvs['w2cs'].squeeze(), 'intrinsics': data_mvs['intrinsics'].squeeze(),
                'c2ws': data_mvs['c2ws'].squeeze(),'near_fars':data_mvs['near_fars'].squeeze()}

    return data_mvs, pose_ref

def unpreprocess(self, data, shape=(1,1,3,1,1)):
        # to unnormalize image for visualization
        # data N V C H W
        device = data.device
        mean = torch.tensor([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225]).view(*shape).to(device)
        std = torch.tensor([1 / 0.229, 1 / 0.224, 1 / 0.225]).view(*shape).to(device)

        return (data - mean) / std

def sample_along_rays(origins, directions, num_samples, near, far, randomized, lindisp):
    """
    origins         : [N_rays, 3]
    directions      : [N_rays, 3]
    num_samples     : N_samples
    near, far       : [N_rays, 1]

    Return
    t_vals          : [N_rays, N_samples]
    pts             : [N_rays, N_samples, 3]
    """
    batch_size = origins.shape[0]

    t_vals = torch.linspace(0., 1., num_samples,  device=origins.device)
    if lindisp:
        t_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * t_vals)
    else:
        t_vals = near * (1. - t_vals) + far * t_vals

    if randomized:
        mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
        upper = torch.cat([mids, t_vals[..., -1:]], -1)
        lower = torch.cat([t_vals[..., :1], mids], -1)
        t_rand = torch.rand(batch_size, num_samples, device=origins.device)
        t_vals = lower + (upper - lower) * t_rand
    else:
        # Broadcast t_vals to make the returned shape consistent.
        t_vals = torch.broadcast_to(t_vals, [batch_size, num_samples])
    pts = origins[..., None, :] + directions[...,None,:] * t_vals[...,:,None] 
    return t_vals, pts

def resample_along_rays(origins, directions, t_vals, weights, N_importance, randomized):
    """Resampling.
    origins         : [N_rays, 3]
    directions      : [N_rays, 3]
    t_vals          : [N_rays, N_samples]
    N_importance    : 
    weights         : [N_rays, N_samples]
    """
    t_vals_mid = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])     # [N_rays, N_samples-1]
    weights = weights[..., 1:-1]                                # [N_rays, N_samples-2], depadding
    
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)        # [N_rays, N_samples-2]
    cdf = torch.cumsum(pdf, -1)                                 # [N_rays, N_samples-2]
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)   # [N_rays, N_samples-1]

    # Take uniform samples
    if randomized :
        u = torch.linspace(0., 1., steps=N_importance)             # 
        u = u.expand(list(cdf.shape[:-1]) + [N_importance])        # [N_rays, N_importance] 
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_importance])      # [N_rays, N_importance] 

    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)                   # [N_rays, N_importance] 
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # [N_rays, N_importance, 2]

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]                   # [N_rays, N_importance, N_samples-1]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)             # [N_rays, N_importance, 2]
    bins_g = torch.gather(t_vals_mid.unsqueeze(1).expand(matched_shape), 2, inds_g)     # [N_rays, N_importance, 2]

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    new_t_vals = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])                 # [N_rays, N_importance]

    new_t_vals, _ = torch.sort(torch.cat([t_vals, new_t_vals], -1), -1)                 # [N_rays, N_importance+N_samples]
    pts = origins[..., None, :] + directions[...,None,:] * new_t_vals[...,:,None]       # [N_rays, N_importance+N_samples, 3]
    return new_t_vals, pts

def volumetric_rendering(rgb, density, t_vals, dirs, white_bkgd):
    """Volumetric Rendering Function.

    Args:
    rgb         : [N_rays, N_samples, 3]
    density     : [N_rays, N_samples, 1]    # Already activate
    t_vals      : [N_rays, N_samples]
    dirs        : [N_rays, 3]
    white_bkgd  : 

    Return:
    comp_rgb    : [N_rays, N_samples, 3]
    distance    : [N_rays]
    acc         : [N_rays]
    weights     : [N_rays, N_samples]
    alpha       : [N_rays, N_samples]
    """
    t_dists = t_vals[..., 1:] - t_vals[..., :-1]    # [N_rays, N_samples-1] 

    # Append 100000
    t_dists = torch.cat([t_dists, torch.Tensor([1e10]).expand(t_dists[...,:1].shape)], -1)  # [N_rays, N_samples]
    delta = t_dists * torch.linalg.norm(dirs[..., None, :], dim=-1)                         # [N_rays, N_samples]
    density_delta = density[..., 0] * delta                                                 # [N_rays, N_samples]
    
    alpha = 1 - torch.exp(-density_delta)               # [N_rays, N_samples]
    trans = torch.exp(-torch.cat([
        torch.zeros_like(density_delta[..., :1]),       # [N_rays, 1]
        torch.cumsum(density_delta[..., :-1], dim=-1)   # [N_rays, N_samples-1]
    ], dim=-1)) 
    weights = alpha * trans                             # [N_rays, N_samples] 

    comp_rgb = (weights[..., None] * rgb).sum(dim=-2)   # [N_rays, N_samples, 3] 
    acc = weights.sum(dim=-1)                           # [N_rays]
    distance = (weights * t_vals).sum(dim=-1) / acc     # [N_rays]
    distance = torch.clamp(torch.nan_to_num(distance), t_vals[:, 0], t_vals[:, -1])
    if white_bkgd:
        comp_rgb = comp_rgb + (1. - acc[..., None])
    return comp_rgb, distance, acc, weights, alpha