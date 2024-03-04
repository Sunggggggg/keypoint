import torch
import torch.nn as nn
import torchvision 

class Backbone(nn.Module):
    def __init__(self, name='resnet50', num_layers=4, pretrained=True, freeze=True, hidden_dim=256) :
        super(Backbone, self).__init__()
        # 
        if name == 'resnet50' and pretrained:
            weights = torchvision.models.ResNet50_Weights
            backbone = torchvision.models.resnet50(weights=weights)
            feat_dim = [2**(i+8) for i in range(num_layers)]
            
        if freeze :
            for name, param in backbone.named_parameters():
                param.requires_grad_(False)

        self.backbone = backbone

        # Same feature dim
        self.branch = nn.Module()
        self.branch.layer1 = nn.Conv2d(feat_dim[0], hidden_dim, kernel_size=3, stride=1, padding=1, bias=False, groups=1)
        self.branch.layer2 = nn.Conv2d(feat_dim[1], hidden_dim, kernel_size=3, stride=1, padding=1, bias=False, groups=1)
        self.branch.layer3 = nn.Conv2d(feat_dim[2], hidden_dim, kernel_size=3, stride=1, padding=1, bias=False, groups=1)
        self.branch.layer4 = nn.Conv2d(feat_dim[3], hidden_dim, kernel_size=3, stride=1, padding=1, bias=False, groups=1)

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
        
        layer1 = self.branch.layer1(layer1)   # hidden_dim
        layer2 = self.branch.layer2(layer2)   # hidden_dim
        layer3 = self.branch.layer3(layer3)   # hidden_dim
        layer4 = self.branch.layer4(layer4)   # hidden_dim
        
        return [layer1, layer2, layer3, layer4]

class SelfAttention(nn.Module):
    def __init__(self, dim=256, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CrossAttention(nn.Module):
    def __init__(self, dim=256, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q.weight)
        nn.init.xavier_uniform_(self.k.weight)
        nn.init.xavier_uniform_(self.v.weight)
        nn.init.xavier_uniform_(self.proj.weight)
        if self.k.bias is not None:
            nn.init.xavier_normal_(self.k.bias)
        if self.v.bias is not None:
            nn.init.xavier_normal_(self.v.bias)
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0.)

    def forward(self, x_q, x_k, x_v):
        B, N_q, C = x_q.shape 
        _, N_kv, C = x_k.shape
        _, N_kv, C = x_v.shape
        # 
        
        # Multi-head cross attetnion
        # b, h, n, d
        q = self.q(self.norm(x_q)).reshape(B, N_q, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(self.norm(x_k)).reshape(B, N_kv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(self.norm(x_v)).reshape(B, N_kv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # [b, h, n, d] * [b, h, d, m] -> [b, h, n, m]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N_q, C)
        x = self.proj(x)
        x = self.proj_drop(x)
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

class VolumeAttention(nn.Module):
    def __init__(self, freeze=True, num_queries=100, hidden_dim=256, num_head=8, num_layers=4, depth=6):
        super(VolumeAttention, self).__init__()
        self.depth = depth
        self.num_layers = num_layers
        self.num_query = num_queries
        self.hidden_dim = hidden_dim
        
        # Backbone
        self.backbone = Backbone(pretrained=True, freeze=True, hidden_dim=hidden_dim, num_layers=num_layers)
        if freeze :
            for name, param in self.backbone.named_parameters():
                param.requires_grad_(False)
        #
        self.query_embed = nn.Parameter(torch.rand(num_queries, hidden_dim), requires_grad=True)
        self.cross_attention_blk = nn.ModuleList([
            CrossAttention(dim=hidden_dim, num_heads=num_head) for _ in range(depth)])
        self.self_attention_blk = nn.ModuleList([
            SelfAttention(dim=hidden_dim, num_heads=num_head) for _ in range(depth)])
        self.pose_embed = nn.Linear(4*4, hidden_dim)

        # 
        self.norm = nn.LayerNorm(hidden_dim)
        self.softmax = nn.Softmax(hidden_dim)
        self.keypoint_embed = nn.Sequential(
            nn.Conv2d(2*num_queries, num_queries, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True))
        
        #
        self.refinenet1 = FeatureFusionBlock_custom(num_queries, nn.ReLU(False), deconv=False, bn=False, expand=False, align_corners=True)
        self.refinenet2 = FeatureFusionBlock_custom(num_queries, nn.ReLU(False), deconv=False, bn=False, expand=False, align_corners=True)
        self.refinenet3 = FeatureFusionBlock_custom(num_queries, nn.ReLU(False), deconv=False, bn=False, expand=False, align_corners=True)
        self.refinenet4 = FeatureFusionBlock_custom(num_queries, nn.ReLU(False), deconv=False, bn=False, expand=False, align_corners=True)

    def forward(self, x, rel_transform, nviews):
        """
        x : [2B, 3, H, W]
        """
        s = x.shape
        x = x.reshape(s[0]//nviews, nviews, *s[1:])
        img1, img2 = x[:, 0], x[:, 1]       # [B, 3, H, W]

        # 
        pose_embed = self.pose_embed(rel_transform)         # [2B, hidden_dim]
        pose_embed = pose_embed.reshape(s[0]//nviews, nviews, -1)
        pose_embed1, pose_embed2 = pose_embed[:, 0], pose_embed[:, 1]   # [B, hidden_dim]

        # 
        keypoint_maps = []
        feats1, feats2 = self.backbone(img1), self.backbone(img2)   # [layer1, layer2, layer3, layer4]
        for l in range(self.num_layers):
            feat1, feat2 = feats1[l], feats2[l]     # [B, hidden_dim, H, W]
            B, _, h, w = feat1.shape 

            feat1 = feat1.reshape(B, self.hidden_dim, -1)   # [B, e, hw]
            feat2 = feat2.reshape(B, self.hidden_dim, -1)
            # feat1 += pose_embed1[:, :, None]
            # feat2 += pose_embed2[:, :, None]

            quries = self.query_embed.expand(B, self.num_query, self.hidden_dim)
            for d in range(self.depth) :
                quries = self.self_attention_blk[d](quries)

                query1, query2 = quries, quries
                query1 = self.cross_attention_blk[d](query1, feat1.permute(0, 2, 1), feat1.permute(0, 2, 1)) #[B, Q, e]
                query2 = self.cross_attention_blk[d](query2, feat2.permute(0, 2, 1), feat2.permute(0, 2, 1)) #[B, Q, e]

                # Aggregation
                matching_score = torch.matmul(query1, query2.transpose(1, 2))     # [B, Q1, Q2]
                
                refine_query1 = query1 + torch.matmul(matching_score.softmax(dim=2), query2)
                refine_query2 = query2 + torch.matmul(matching_score.softmax(dim=1).transpose(1,2), query1)

                quries = self.softmax(quries + refine_query1 + refine_query2)
                
            # 
            keypoint_map1 = torch.matmul(quries, feat1).reshape(B, self.num_query, h, w)    # [B, Q, e]*[B, e, hw] = [B, Q, h, w]
            keypoint_map2 = torch.matmul(quries, feat2).reshape(B, self.num_query, h, w)    
            keypoint_map = torch.stack([keypoint_map1, keypoint_map2], dim=1)                 # [B, 2, Q, h, w]
            keypoint_map = torch.flatten(keypoint_map, 0, 1)                # [2B, Q, H, W]
            keypoint_maps.append(keypoint_map)

        path_4 = self.refinenet4(keypoint_maps[3])
        path_3 = self.refinenet3(path_4, keypoint_maps[2])
        path_2 = self.refinenet2(path_3, keypoint_maps[1])
        path_1 = self.refinenet1(path_2, keypoint_maps[0])
        
        return [path_2, path_1]
    
if __name__ == '__main__' :
    model = VolumeAttention()
    x = torch.rand((2, 3, 256, 256))
    y = torch.rand((2, 16))
    print(model(x, y, 2)[0].shape)