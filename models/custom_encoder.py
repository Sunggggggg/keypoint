# 0. Make Leanable Queries
# 1. Build Backbone (ResNet50-Pretraind, Multi Scale)
# 2. Positional embedding (Camera pose + Patch pose)
# 3. Accross self-attention 
# 4. Upsampling CNN(Fusion)
# 5. Skip Conect
import torch
import torch.nn as nn
from torchvision import models
from functools import partial
import math
from timm.models.layers import PatchEmbed, trunc_normal_
from timm.models.vision_transformer import Block, _init_vit_weights
from collections import OrderedDict

class VisionTransformerMultiView(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=784, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init=''):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_embed_second = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.pose_embed = nn.Linear(16, embed_dim)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        self.init_weights(weight_init)

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.pos_embed_second, std=.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=.02)
        if mode.startswith('jax'):
            # leave cls token as zeros to match jax impl
            named_apply(partial(_init_vit_weights, head_bias=head_bias, jax_impl=True), self)
        else:
            trunc_normal_(self.cls_token, std=.02)
            self.apply(_init_vit_weights)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        print(x.shape)
        x = self.blocks(x)
        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
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

class Backbone(nn.Module):
    def __init__(self, name='resnet50', pretrained=True, freeze=True) :
        super(Backbone, self).__init__()
        # 
        if name == 'resnet50' and pretrained:
            weights = models.ResNet50_Weights
            backbone = models.resnet50(weights=weights)
            
        if freeze :
            for name, param in backbone.named_parameters():
                param.requires_grad_(False)

        self.backbone = backbone

    def forward(self, x):
        #
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)

        # 
        layer1 = self.backbone.layer1(x)        # 256
        layer2 = self.backbone.layer2(layer1)   # 512
        layer3 = self.backbone.layer3(layer2)   # 1024
        layer4 = self.backbone.layer4(layer3)   # 2048
        
        return [layer1, layer2, layer3, layer4]

class MultiviewQueryEncoder(nn.Module):
    def __init__(self, num_queries, hidden_dim, layer=4,
                 img_size=256, patch_size=16, features=256, depth=12, num_heads=8
                 ) :
        super(MultiviewQueryEncoder, self).__init__()
        # 0. Make Leanable Queries
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # 1. Build Backbone (ResNet50-Pretraind, Multi Scale)
        backbone = VisionTransformerMultiView(img_size=img_size, patch_size=patch_size, in_chans=3, 
                                             embed_dim=features, depth=depth, num_heads=num_heads)
        # 1.1 branch
        feat_dim = [2**(i+8) for i in range(layer)]
        
        self.branch = nn.Module()
        self.branch.layer1 = nn.Conv2d(feat_dim[0], features, kernel_size=3, stride=1, padding=1, bias=False, groups=1)
        self.branch.layer2 = nn.Conv2d(feat_dim[1], features, kernel_size=3, stride=1, padding=1, bias=False, groups=1)
        self.branch.layer3 = nn.Conv2d(feat_dim[2], features, kernel_size=3, stride=1, padding=1, bias=False, groups=1)
        self.branch.layer4 = nn.Conv2d(feat_dim[3], features, kernel_size=3, stride=1, padding=1, bias=False, groups=1)

        # 1.2 Fusion 
        self.branch.refinenet1 = FeatureFusionBlock_custom(features, nn.ReLU(False), deconv=False, bn=False, expand=False, align_corners=True)
        self.branch.refinenet2 = FeatureFusionBlock_custom(features, nn.ReLU(False), deconv=False, bn=False, expand=False, align_corners=True)
        self.branch.refinenet3 = FeatureFusionBlock_custom(features, nn.ReLU(False), deconv=False, bn=False, expand=False, align_corners=True)
        self.branch.refinenet4 = FeatureFusionBlock_custom(features, nn.ReLU(False), deconv=False, bn=False, expand=False, align_corners=True)
        # 

    def forward(self, x):
        layer_1, layer_2, layer_3, layer_4 = self.backbone(x)


        # 
        layer_1_rn = self.branch.layer1(layer_1)
        layer_2_rn = self.branch.layer2(layer_2)
        layer_3_rn = self.branch.layer3(layer_3)
        layer_4_rn = self.branch.layer4(layer_4)

        path_4 = self.branch.refinenet4(layer_4_rn)
        path_3 = self.branch.refinenet3(path_4, layer_3_rn)
        path_2 = self.branch.refinenet2(path_3, layer_2_rn)
        path_1 = self.branch.refinenet1(path_2, layer_1_rn)

        return [path_2, path_1]

x = torch.rand((1, 3, 224, 224))
m = VisionTransformerMultiView(embed_dim=256, depth=12, num_heads=8)
print(m(x).shape)