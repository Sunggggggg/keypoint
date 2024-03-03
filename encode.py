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

class Slice(nn.Module):
    def __init__(self, start_index=1):
        super(Slice, self).__init__()
        self.start_index = start_index

    def forward(self, x):
        return x[:, self.start_index :]
    
class ProjectReadout(nn.Module):
    def __init__(self, in_features, start_index=1):
        super(ProjectReadout, self).__init__()
        self.start_index = start_index

        self.project = nn.Sequential(nn.Linear(2 * in_features, in_features), nn.GELU())

    def forward(self, x):
        """
        x : [B, 1+dim, e]
        """
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

class ResidualConvUnit_custom(nn.Module):
    def __init__(self, features, activation, bn):
        super().__init__()
        self.bn = bn
        self.groups=1
        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups
        )
        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups
        )

        if self.bn==True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn==True:
            out = self.bn1(out)
       
        out = self.activation(out)
        out = self.conv2(out)
        if self.bn==True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)

class FeatureFusionBlock_custom(nn.Module):
    def __init__(self, features, activation, deconv=False, bn=False, expand=False, align_corners=True):
        super(FeatureFusionBlock_custom, self).__init__()
        self.deconv = deconv
        self.align_corners = align_corners
        self.groups=1
        self.expand = expand
        out_features = features
        if self.expand==True:
            out_features = features//2
        
        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)

        self.resConfUnit1 = ResidualConvUnit_custom(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit_custom(features, activation, bn)
        
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, *xs):
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)
        output = nn.functional.interpolate(output, scale_factor=2, mode="bilinear", align_corners=self.align_corners)
        output = self.out_conv(output)

        return output

def forward_vit(pretrained, x, rel_transform, nviews):
    b, c, h, w = x.shape

    glob = pretrained.model.forward_flex(x, rel_transform, nviews)

    layer_1 = pretrained.activations["1"]   # [B, 256, H/4, W/4]
    layer_2 = pretrained.activations["2"]   # [B, 512, H/8, W/8]
    layer_3 = pretrained.activations["3"]   # [B, 257, 768]
    layer_4 = pretrained.activations["4"]   # [B, 257, 768]

    s = layer_3.size()
    layer_3 = layer_3.view(s[0] * nviews, s[1] // nviews, s[2])

    s = layer_4.size()
    layer_4 = layer_4.view(s[0] * nviews, s[1] // nviews, s[2])
    
  
    layer_1 = pretrained.act_postprocess1[0:2](layer_1)
    layer_2 = pretrained.act_postprocess2[0:2](layer_2)
    layer_3 = pretrained.act_postprocess3[0:2](layer_3)
    layer_4 = pretrained.act_postprocess4[0:2](layer_4)

    unflatten = nn.Sequential(
        nn.Unflatten(2, torch.Size([h // pretrained.model.patch_size[1], w // pretrained.model.patch_size[0]])))

    if layer_1.ndim == 3:
        layer_1 = unflatten(layer_1)
    if layer_2.ndim == 3:
        layer_2 = unflatten(layer_2)
    if layer_3.ndim == 3:
        layer_3 = unflatten(layer_3)
    if layer_4.ndim == 3:
        layer_4 = unflatten(layer_4)

    print(layer_1.shape, layer_2.shape, layer_3.shape, layer_4.shape)
    layer_1 = pretrained.act_postprocess1[3 : len(pretrained.act_postprocess1)](layer_1)
    layer_2 = pretrained.act_postprocess2[3 : len(pretrained.act_postprocess2)](layer_2)
    layer_3 = pretrained.act_postprocess3[3 : len(pretrained.act_postprocess3)](layer_3)
    layer_4 = pretrained.act_postprocess4[3 : len(pretrained.act_postprocess4)](layer_4)

    return layer_1, layer_2, layer_3, layer_4

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
    def __init__(self, vit_features=768, size=[384, 384], readout='project', pretrained=True):
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
        features = [256, 512, 768, 768]
        readout_oper = [Slice(start_index)] * len(features)
        #readout_oper = [ProjectReadout(vit_features, start_index) for out_feat in features]

        # pretrained.act_postprocess1 = nn.Sequential(
        #     readout_oper[0], 
        #     Transpose(1, 2),
        #     nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        #     nn.Conv2d(in_channels=vit_features, out_channels=features[0], kernel_size=1, stride=1,padding=0),
        #     nn.ConvTranspose2d(in_channels=features[0],out_channels=features[0],
        #                        kernel_size=4,stride=4,padding=0,bias=True,dilation=1,groups=1))

        # pretrained.act_postprocess2 = nn.Sequential(
        #     readout_oper[1],
        #     Transpose(1, 2),
        #     nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        #     nn.Conv2d(in_channels=vit_features,out_channels=features[1],kernel_size=1,stride=1,padding=0),
        #     nn.ConvTranspose2d(in_channels=features[1],out_channels=features[1],
        #                        kernel_size=2,stride=2,padding=0,bias=True,dilation=1,groups=1))

        # pretrained.act_postprocess3 = nn.Sequential(
        #     readout_oper[2],
        #     Transpose(1, 2),
        #     nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        #     nn.Conv2d(in_channels=vit_features,out_channels=features[2],kernel_size=1,stride=1,padding=0))

        # pretrained.act_postprocess4 = nn.Sequential(
        #     readout_oper[3],
        #     Transpose(1, 2),
        #     nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        #     nn.Conv2d(in_channels=vit_features,out_channels=features[3],kernel_size=1,stride=1,padding=0),
        #     nn.Conv2d(in_channels=features[3],out_channels=features[3],kernel_size=3,stride=2,padding=1))

        pretrained.act_postprocess1 = nn.Sequential(
            nn.Identity(), nn.Identity(), nn.Identity()
        )
        pretrained.act_postprocess2 = nn.Sequential(
            nn.Identity(), nn.Identity(), nn.Identity()
        )

        pretrained.act_postprocess3 = nn.Sequential(
            readout_oper[2],
            Transpose(1, 2),
            nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
            nn.Conv2d(in_channels=vit_features,out_channels=features[2],kernel_size=1,stride=1,padding=0,
            ),
        )

        pretrained.act_postprocess4 = nn.Sequential(
            readout_oper[3],
            Transpose(1, 2),
            nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
            nn.Conv2d(in_channels=vit_features,out_channels=features[3],kernel_size=1,stride=1,padding=0,
            ),
            nn.Conv2d(in_channels=features[3],out_channels=features[3],kernel_size=3,stride=2,padding=1,
            ),
        )

        pretrained.model.start_index = start_index
        pretrained.model.patch_size = [16, 16]

        pretrained.model.forward_flex = types.MethodType(forward_flex, pretrained.model)
        pretrained.model._resize_pos_embed = types.MethodType(_resize_pos_embed, pretrained.model)
        self.pretrained = pretrained
        # 
        scratch = nn.Module()
        in_shape = [256, 512, 768, 768]
        features = 256
        out_shape1 = features
        out_shape2 = features
        out_shape3 = features
        out_shape4 = features

        scratch.layer1_rn = nn.Conv2d(in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=1)
        scratch.layer2_rn = nn.Conv2d(in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=1)
        scratch.layer3_rn = nn.Conv2d(in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=1)
        scratch.layer4_rn = nn.Conv2d(in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=1)

        scratch.refinenet1 = FeatureFusionBlock_custom(features, nn.ReLU(False), deconv=False, bn=False, expand=False, align_corners=True)
        scratch.refinenet2 = FeatureFusionBlock_custom(features, nn.ReLU(False), deconv=False, bn=False, expand=False, align_corners=True)
        scratch.refinenet3 = FeatureFusionBlock_custom(features, nn.ReLU(False), deconv=False, bn=False, expand=False, align_corners=True)
        scratch.refinenet4 = FeatureFusionBlock_custom(features, nn.ReLU(False), deconv=False, bn=False, expand=False, align_corners=True)

        self.scratch = scratch

    def forward(self, x, rel_transform, nviews):
        layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, x, rel_transform, nviews)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        return [path_2, path_1]
    
if __name__ == '__main__' :
    model = MultiviewViT()
    x = torch.rand((2, 3, 256, 256))
    y = torch.rand((2, 16))
    print(model(x, y, 2)[0].shape)