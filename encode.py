import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from vit_models import vit_base_resnet50_384
import types
import math

activations = {}
def get_activation(name):
    def hook(model, input, output):
        activations[name] = output

    return hook

class ProjectReadout(nn.Module):
    def __init__(self, in_features, start_index=1):
        super(ProjectReadout, self).__init__()
        self.start_index = start_index

        self.project = nn.Sequential(nn.Linear(2 * in_features, in_features), nn.GELU())

    def forward(self, x):
        readout = x[:, 0].unsqueeze(1).expand_as(x[:, self.start_index :])
        features = torch.cat((x[:, self.start_index :], readout), -1)

        return self.project(features)

class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        x = x.transpose(self.dim0, self.dim1)
        return x
    
def _resize_pos_embed(self, posemb, gs_h, gs_w):
    posemb_tok, posemb_grid = (
        posemb[:, : self.start_index],
        posemb[0, self.start_index :],
    )

    gs_old = int(math.sqrt(len(posemb_grid)))

    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(gs_h, gs_w), mode="bilinear")
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_h * gs_w, -1)

    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)

    return posemb


def forward_flex(self, x, pose, nviews):
    b, c, h, w = x.shape

    pos_embed = self._resize_pos_embed(
        self.pos_embed, h // self.patch_size[1], w // self.patch_size[0]
    )
    pos_embed_second = self._resize_pos_embed(
        self.pos_embed_second, h // self.patch_size[1], w // self.patch_size[0]
    )

    pose_embed = self.pose_embed(pose)

    B = x.shape[0]
    if hasattr(self.patch_embed, "backbone"):
        x = self.patch_embed.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1] 

    x = self.patch_embed.proj(x).flatten(2).transpose(1, 2)
    if getattr(self, "dist_token", None) is not None:
        cls_tokens = self.cls_token.expand(B, -1, -1) 
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
    else:
        cls_tokens = self.cls_token.expand(B, -1, -1)  
        x = torch.cat((cls_tokens, x), dim=1)

    x = x + pos_embed + pose_embed[:, None, :]
    x = self.pos_drop(x)

    os = 257

    s = x.size()
    x = x.view(s[0] // nviews, nviews * s[1], *s[2:])

    for i, blk in enumerate(self.blocks):
        x = blk(x)
    x = self.norm(x)
    s = x.size()

    x = torch.flatten(x.view(s[0], nviews, os, *s[2:]), 0, 1)
    return x

class MultiviewViT(nn.Module):
    def __init__(self, features=256, vit_features=768, size=[384, 384], readout='project', pretrained=True) -> None:
        super().__init__()
        hooks = [0, 1, 8, 11]

        #model = timm.create_model("vit_base_resnet50_384", pretrained=pretrained)
        model = vit_base_resnet50_384(pretrained=False)
        pretrained = nn.Module()
        pretrained.model = model

        pretrained.model.patch_embed.backbone.stages[0].register_forward_hook(get_activation("1"))
        pretrained.model.patch_embed.backbone.stages[1].register_forward_hook(get_activation("2"))
        pretrained.model.blocks[hooks[2]].register_forward_hook(get_activation("3"))
        pretrained.model.blocks[hooks[3]].register_forward_hook(get_activation("4"))
        pretrained.activations = activations

        start_index = 1
        readout_oper = [ProjectReadout(vit_features, start_index) for out_feat in features]
        pretrained.act_postprocess1 = nn.Sequential(
            readout_oper[0], 
            Transpose(1, 2),
            nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
            nn.Conv2d(in_channels=vit_features, out_channels=features[0], kernel_size=1, stride=1,padding=0),
            nn.ConvTranspose2d(in_channels=features[0],out_channels=features[0],
                               kernel_size=4,stride=4,padding=0,bias=True,dilation=1,groups=1))

        pretrained.act_postprocess2 = nn.Sequential(
            readout_oper[1],
            Transpose(1, 2),
            nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
            nn.Conv2d(in_channels=vit_features,out_channels=features[1],kernel_size=1,stride=1,padding=0),
            nn.ConvTranspose2d(in_channels=features[1],out_channels=features[1],
                               kernel_size=2,stride=2,padding=0,bias=True,dilation=1,groups=1))

        pretrained.act_postprocess3 = nn.Sequential(
            readout_oper[2],
            Transpose(1, 2),
            nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
            nn.Conv2d(in_channels=vit_features,out_channels=features[2],kernel_size=1,stride=1,padding=0))

        pretrained.act_postprocess4 = nn.Sequential(
            readout_oper[3],
            Transpose(1, 2),
            nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
            nn.Conv2d(in_channels=vit_features,out_channels=features[3],kernel_size=1,stride=1,padding=0),
            nn.Conv2d(in_channels=features[3],out_channels=features[3],kernel_size=3,stride=2,padding=1))

        pretrained.model.start_index = start_index
        pretrained.model.patch_size = [16, 16]

        pretrained.model.forward_flex = types.MethodType(forward_flex, pretrained.model)
        pretrained.model._resize_pos_embed = types.MethodType(_resize_pos_embed, pretrained.model)

        # 
        scratch = nn.Module()
        in_shape = [256, 512, 1024, 1024]
        features
        out_shape1 = features
        out_shape2 = features
        out_shape3 = features
        out_shape4 = features

        scratch.layer1_rn = nn.Conv2d(in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=1)
        scratch.layer2_rn = nn.Conv2d(in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=1)
        scratch.layer3_rn = nn.Conv2d(in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=1)
        scratch.layer4_rn = nn.Conv2d(in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=1)