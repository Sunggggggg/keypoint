import torch
import torch.nn as nn
from utils import sample_along_rays, resample_along_rays, volumetric_rendering

class PositionalEncoding(nn.Module):
    def __init__(self, min_deg, max_deg):
        super(PositionalEncoding, self).__init__()
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.scales = nn.Parameter(torch.tensor([2 ** i for i in range(min_deg, max_deg)]), requires_grad=False)

    def forward(self, x, y=None):
        shape = list(x.shape[:-1]) + [-1]   # [B, N, -1]
        x_enc = (x[..., None, :] * self.scales[:, None]).reshape(shape) # [B, N, L, 3] -> [B, N, 3L] 
        x_enc = torch.cat((x_enc, x_enc + 0.5 * torch.pi), -1)          # [B, N, 6L]

        # PE
        x_ret = torch.sin(x_enc)
        x_ret = torch.cat([x, x_ret], -1)   # Identity
        return x_ret
    
class NeRF(nn.Module):
    def __init__(self, 
                 use_viewdirs = True,
                 randomized=False,
                 white_bkgd=False,
                 num_levels=2,
                 N_samples=64,
                 hidden=256,
                 density_noise=1, 
                 min_deg=0,
                 max_deg=10,
                 viewdirs_min_deg=0,
                 viewdirs_max_deg=4,
                 device=torch.device("cpu")
                 ) -> None:
        super(NeRF, self).__init__()
        self.use_viewdirs = use_viewdirs
        self.init_randomized = randomized
        self.randomized = randomized
        self.white_bkgd = white_bkgd
        self.num_levels = num_levels
        self.N_samples = N_samples
        self.density_input = 3 + (max_deg - min_deg) * 3 * 2
        self.rgb_input = 3 + ((viewdirs_max_deg - viewdirs_min_deg) * 3 * 2)
        self.density_noise = density_noise
        self.hidden = hidden
        self.device = device
        self.density_activation = nn.Softmax()

        # 
        self.positional_encoding = PositionalEncoding(min_deg, max_deg)
        self.density_net0 = nn.Sequential(
            nn.Linear(self.density_input, hidden),  # 0
            nn.ReLU(True),
            nn.Linear(hidden, hidden),              # 1 
            nn.ReLU(True),
            nn.Linear(hidden, hidden),              # 2
            nn.ReLU(True),
            nn.Linear(hidden, hidden),              # 3
            nn.ReLU(True),
            nn.Linear(hidden, hidden),              # 4
            nn.ReLU(True)
        )
        self.density_net1 = nn.Sequential(
            nn.Linear(self.density_input+hidden, hidden),  # 5
            nn.ReLU(True),
            nn.Linear(hidden, hidden),  # 6
            nn.ReLU(True),
            nn.Linear(hidden, hidden),  # 7
            nn.ReLU(True)
        )

        self.final_density = nn.Sequential(
            nn.Linear(hidden, 1),
        )

        input_shape = hidden
        if self.use_viewdirs:
            input_shape = hidden // 2

            self.rgb_net0 = nn.Sequential(
                nn.Linear(hidden, hidden)
            )
            self.viewdirs_encoding = PositionalEncoding(viewdirs_min_deg, viewdirs_max_deg)
            self.rgb_net1 = nn.Sequential(
                nn.Linear(hidden + self.rgb_input, input_shape),
                nn.ReLU(True),
            )
        self.final_rgb = nn.Sequential(
            nn.Linear(input_shape, 3),
            nn.Sigmoid()
        )
        _xavier_init(self)
        self.to(device)

    def forward(self, ray_batch, feat1, feat2, c1, c2):
        comp_rgbs, distances, accs = [], [], []

        rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6]                 # [N_rays, 3] each
        bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])                # [N_rays, 1, 2]
        near, far = bounds[...,0], bounds[...,1]                            # [N_rays, 1]
        view_dirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 9 else None   # [N_rays, 3]

        for l in ['coarse', 'fine']:
            if l == 'coarse' : 
                t_vals, (mean, var) = sample_along_rays(rays_o, rays_d, self.N_samples,
                                                        near, far, randomized=self.randomized, lindisp=False)   
            elif l == 'fine' :
                t_vals, (mean, var) = resample_along_rays(rays_o, rays_d, t_vals.to(rays_o.device), weights.to(rays_o.device),
                                                           self.N_samples, randomized=self.randomized)
            # do integrated positional encoding of samples
            samples_enc = self.positional_encoding(mean, var)[0]
            samples_enc = samples_enc.reshape([-1, samples_enc.shape[-1]])  # [N_rays*N_samples, 96]

            # predict density
            new_encodings = self.density_net0(samples_enc)                  # [N_rays*N_samples, 256]
            new_encodings = torch.cat((new_encodings, samples_enc), -1)     # [N_rays*N_samples, 256+96]
            new_encodings = self.density_net1(new_encodings)                # [N_rays*N_samples, 256]
            raw_density = self.final_density(new_encodings).reshape((-1, self.N_samples, 1)) # [N_rays, N_samples, 1]
            
            # predict rgb
            if self.use_viewdirs:
                #  do positional encoding of viewdirs
                viewdirs = self.viewdirs_encoding(view_dirs.to(self.device))             # [N_rays, 30]
                #viewdirs = torch.cat((viewdirs, view_dirs.to(self.device)), -1)          # [N_rays, 30]
                viewdirs = torch.tile(viewdirs[:, None, :], (1, self.N_samples, 1))      # [N_rays, N_samples, 30]
                viewdirs = viewdirs.reshape((-1, viewdirs.shape[-1]))                    # [N_rays*N_samples, 30]
                new_encodings = self.rgb_net0(new_encodings)                             # [N_rays*N_samples, 256]
                new_encodings = torch.cat((new_encodings, viewdirs), -1)                 # [N_rays*N_samples, 30+256]
                new_encodings = self.rgb_net1(new_encodings)                             # [N_rays*N_samples, 286]
            raw_rgb = self.final_rgb(new_encodings).reshape((-1, self.N_samples, 3))   # [N_rays, N_samples, 3]
            
            # Add noise to regularize the density predictions if needed.
            if self.randomized and self.density_noise:
                raw_density += self.density_noise * torch.rand(raw_density.shape, dtype=raw_density.dtype, device=raw_density.device)

            # volumetric rendering
            rgb = raw_rgb
            density = self.density_activation(raw_density)
            comp_rgb, distance, acc, weights, alpha = volumetric_rendering(rgb, density, t_vals, rays_d.to(rgb.device), self.white_bkgd)
            comp_rgbs.append(comp_rgb)
            distances.append(distance)
            accs.append(acc)
        
            return torch.stack(comp_rgbs), torch.stack(distances), torch.stack(accs)

def _xavier_init(model):
    """
    Performs the Xavier weight initialization.
    """
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)