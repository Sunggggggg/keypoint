import torch
import torch.nn as nn
import timm

from Encoder import SpatialEncoder, ImageEncoder, UNetEncoder
from midas import dpt_depth, midas_net, midas_net_custom
from resnet_block_fc import ResnetFC

class CrossAttentionRenderer(nn.Module):
    def __init__(self, no_sample=False, no_latent_concat=False, 
                 no_multiview=False, no_high_freq=False, model="midas_vit", 
                 uv=None, repeat_attention=True, n_view=1, npoints=64, num_hidden_units_phi=128):
        super().__init__()
        # 
        self.n_view = n_view
        if self.n_view == 2 or self.n_view == 1:
            self.npoints = 64
        else:
            self.npoints = 48

        if npoints:
            self.npoints = npoints

        self.repeat_attention = repeat_attention

        self.no_sample = no_sample
        self.no_latent_concat = no_latent_concat
        self.no_multiview = no_multiview
        self.no_high_freq = no_high_freq

        # Select Backbone model
        if model == "resnet":
            self.encoder = SpatialEncoder(use_first_pool=False, num_layers=4)
            self.latent_dim = 512
        elif model == 'midas':
            self.encoder = midas_net_custom.MidasNet_small(
                path=None, features=64, backbone="efficientnet_lite3", exportable=True, non_negative=True, blocks={'expand': True})
            checkpoint = ("https://github.com/AlexeyAB/MiDaS/releases/download/midas_dpt/midas_v21_small-70d6b9c8.pt")
            state_dict = torch.hub.load_state_dict_from_url(
                checkpoint, map_location=torch.device('cpu'), progress=True, check_hash=True
            )
            self.encoder.load_state_dict(state_dict)
            self.latent_dim = 512
        elif model == 'midas_vit':
            self.encoder = dpt_depth.DPTDepthModel(path=None, backbone="vitb_rn50_384", non_negative=True)
            checkpoint = ("https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-midas-501f0c75.pt")

            self.encoder.pretrained.model.patch_embed.backbone.stem.conv = timm.models.layers.std_conv.StdConv2dSame(3, 64, kernel_size=(7, 7), stride=(2, 2), bias=False)
            self.latent_dim = 512 + 64

            self.conv_map = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3)
        else:
            self.encoder = UNetEncoder()
            self.latent_dim = 32

        # 
        if self.n_view > 1 and (not self.no_latent_concat):
            self.query_encode_latent = nn.Conv2d(self.latent_dim + 3, self.latent_dim, 1)
            self.query_encode_latent_2 = nn.Conv2d(self.latent_dim, self.latent_dim // 2 , 1)
            self.latent_dim = self.latent_dim // 2
            self.update_val_merge = nn.Conv2d(self.latent_dim * 2 + 6, self.latent_dim, 1)
        elif self.no_latent_concat:
            self.feature_map = nn.Conv2d(self.latent_dim, self.latent_dim // 2 , 1)
        else:
            self.update_val_merge = nn.Conv2d(self.latent_dim + 6, self.latent_dim, 1)

        # 
        self.model = model
        self.num_hidden_units_phi = num_hidden_units_phi

        hidden_dim = 128

        if not self.no_latent_concat:
            self.latent_value = nn.Conv2d(self.latent_dim * self.n_view, self.latent_dim, 1)
            self.key_map = nn.Conv2d(self.latent_dim * self.n_view, hidden_dim, 1)
            self.key_map_2 = nn.Conv2d(hidden_dim, hidden_dim, 1)
        else:
            self.latent_value = nn.Conv2d(self.latent_dim, self.latent_dim, 1)
            self.key_map = nn.Conv2d(self.latent_dim, hidden_dim, 1)
            self.key_map_2 = nn.Conv2d(hidden_dim, hidden_dim, 1)

        self.query_embed = nn.Conv2d(16, hidden_dim, 1)
        self.query_embed_2 = nn.Conv2d(hidden_dim, hidden_dim, 1)
        self.hidden_dim = hidden_dim

        self.latent_avg_query = nn.Conv2d(9+16, hidden_dim, 1)
        self.latent_avg_query_2 = nn.Conv2d(hidden_dim, hidden_dim, 1)

        self.latent_avg_key = nn.Conv2d(self.latent_dim, hidden_dim, 1)
        self.latent_avg_key_2 = nn.Conv2d(hidden_dim, hidden_dim, 1)

        self.query_repeat_embed = nn.Conv2d(16+128, hidden_dim, 1)
        self.query_repeat_embed_2 = nn.Conv2d(hidden_dim, hidden_dim, 1)

        self.latent_avg_repeat_query = nn.Conv2d(9+16+128, hidden_dim, 1)
        self.latent_avg_repeat_query_2 = nn.Conv2d(hidden_dim, hidden_dim, 1)

        self.encode_latent = nn.Conv1d(self.latent_dim, 128, 1)

        self.phi = ResnetFC(self.n_view * 9, n_blocks=3, d_out=3,
                            d_latent=self.latent_dim * self.n_view, d_hidden=self.num_hidden_units_phi)
