import torch
import torchvision
import torch.nn as nn
from torch.nn import functional as F
from loss_functions import ContrastiveLoss

def QueryAggregation(queries1, queries2):
    """
    queries1, queries2 : [B, Q, e]
    """
    matching_score = torch.matmul(queries1, queries2.transpose(1, 2))
    matching_score1 = matching_score.softmax(dim=1)
    matching_score2 = matching_score.softmax(dim=2)

    expect_queries = \
        (torch.matmul(matching_score1.transpose(1, 2), queries1) + torch.matmul(matching_score2, queries2))/2
    
    return expect_queries

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
        #self.branch.layer4 = nn.Conv2d(feat_dim[3], hidden_dim, kernel_size=3, stride=1, padding=1, bias=False, groups=1)

    def forward(self, x):
        #
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)

        # 
        layer1 = self.backbone.layer1(x)        # 256
        layer2 = self.backbone.layer2(layer1)   # 512
        layer3 = self.backbone.layer3(layer2)   # 1024
        #layer4 = self.backbone.layer4(layer3)   # 2048
        
        layer1 = self.branch.layer1(layer1)   # hidden_dim
        layer2 = self.branch.layer2(layer2)   # hidden_dim
        layer3 = self.branch.layer3(layer3)   # hidden_dim
        #layer4 = self.branch.layer4(layer4)   # hidden_dim
        
        return [layer1, layer2, layer3]

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class SelfAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos=None):
        return tensor if pos is None else tensor + pos
        
    def forward(self, queries, query_pos=None) :
        q = k = self.with_pos_embed(queries, query_pos)
        queries2 = self.self_attn(q, k, value=queries)[0]
        queries = queries + self.dropout(queries2)
        queries = self.norm(queries)

        return queries

class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0, activation="relu"):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos=None):
        return tensor if pos is None else tensor + pos

    def forward(self, queries, feat, cam_pos=None, query_pos=None):
        queries2 = self.multihead_attn(query=self.with_pos_embed(queries, query_pos),
                                   key=self.with_pos_embed(feat, cam_pos),
                                   value=feat)[0]
        queries = queries + self.dropout(queries2)
        queries = self.norm(queries)
        
        return queries

class FFNLayer(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0, activation="relu"):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)
        self.activation = _get_activation_fn(activation)
    
        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, queries):
        queries2 = self.linear2(self.dropout(self.activation(self.linear1(queries))))
        queries = queries + self.dropout(queries2)
        queries = self.norm(queries)
        return queries

class MultiScaleQueryTransformerDecoder(nn.Module):
    def __init__(self, num_layers=3, num_queries=100, hidden_dim=256, dim_feedforward=2048, nheads=8, depth=6) :
        super().__init__()
        self.num_layers = num_layers
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.depth = depth
        # Backbone
        self.backbone = Backbone(pretrained=True, freeze=True, hidden_dim=hidden_dim, num_layers=num_layers)

        # Learnable query
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # Camera pose embedding
        self.cam_pose_embed = nn.Linear(4*4, hidden_dim)

        # Block
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers1 = nn.ModuleList()
        self.transformer_cross_attention_layers2 = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(depth) :
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0))

            self.transformer_cross_attention_layers1.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0))
            
            self.transformer_cross_attention_layers2.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0))

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0))
        
        # 
        self.encode1 = nn.Conv2d(num_queries*2, num_queries, kernel_size=1, stride=1)
        self.encode2 = nn.Conv2d(num_queries*2, num_queries, kernel_size=1, stride=1)

        self.refinenet1 = FeatureFusionBlock_custom(num_queries, nn.ReLU(False), deconv=False, bn=False, expand=False, align_corners=True)
        self.refinenet2 = FeatureFusionBlock_custom(num_queries, nn.ReLU(False), deconv=False, bn=False, expand=False, align_corners=True)
        self.refinenet3 = FeatureFusionBlock_custom(num_queries, nn.ReLU(False), deconv=False, bn=False, expand=False, align_corners=True)
        #self.refinenet4 = FeatureFusionBlock_custom(num_queries, nn.ReLU(False), deconv=False, bn=False, expand=False, align_corners=True)

        # Loss func
        self.loss_func = ContrastiveLoss(self.num_queries)

    def forward(self, x, rel_transform, nviews=2):
        """
        x               : [2B, 3, H, W]
        rel_transform   : [2B, 16]
        n_view          : 2
        """
        s = x.shape
        x = x.reshape(s[0]//nviews, nviews, *s[1:])
        img1, img2 = x[:, 0], x[:, 1]       # [B, 3, H, W]

        # Camera pose embedding
        pose_embed = self.cam_pose_embed(rel_transform)                     # [2B, hidden_dim]
        pose_embed = pose_embed.reshape(s[0]//nviews, nviews, -1)
        
        # 
        B = x.shape[0]
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)  # [B, Q, e]
        output = self.query_feat.weight.unsqueeze(0).repeat(B, 1, 1)

        # Feature map
        keypoint_maps, contra_losses =[], []
        feats1, feats2 = self.backbone(img1), self.backbone(img2)   # [layer3, layer2, layer1]
        for l in range(self.num_layers):
            feat1, feat2 = feats1[l], feats2[l]     # [B, hidden_dim, H, W]
            B, _, h, w = feat1.shape 

            feat1 = feat1.reshape(B, self.hidden_dim, -1).permute(0, 2, 1)   # [B, hw, e]
            feat2 = feat2.reshape(B, self.hidden_dim, -1).permute(0, 2, 1)   # [B, hw, e]

            pose_embed1 = pose_embed[:, 0].unsqueeze(1).repeat(1, h*w, 1)
            pose_embed2 = pose_embed[:, 1].unsqueeze(1).repeat(1, h*w, 1)

            for d in range(self.depth):
                #
                output1 = self.transformer_cross_attention_layers1[d](
                    output, feat1, cam_pos=pose_embed1, query_pos=query_embed
                )
                output2 = self.transformer_cross_attention_layers1[d](
                    output, feat2, cam_pos=pose_embed2, query_pos=query_embed
                )

                contra_loss = self.loss_func(output1, output2)
                output = QueryAggregation(output1, output2)

                #
                output = self.transformer_self_attention_layers[d](
                    output, query_pos=query_embed
                )
                output = self.transformer_ffn_layers[d](
                    output
                )
            # 
            keypoint_map11 = torch.matmul(output, feat1.transpose(1,2)).reshape(B, self.num_queries, h, w)      # [B, Q, e]*[B, e, hw] = [B, Q, h, w]
            keypoint_map21 = torch.matmul(output2, feat1.transpose(1,2)).reshape(B, self.num_queries, h, w)
            keypoint_map1 = torch.cat([keypoint_map11, keypoint_map21], dim=1)                                  # [B, 2Q, h, w]
            keypoint_map1 = self.encode1(keypoint_map1)

            keypoint_map22 = torch.matmul(output, feat2.transpose(1,2)).reshape(B, self.num_queries, h, w)
            keypoint_map12 = torch.matmul(output1, feat2.transpose(1,2)).reshape(B, self.num_queries, h, w)
            keypoint_map2 = torch.cat([keypoint_map12, keypoint_map22], dim=1)                                  # [B, 2Q, h, w]
            keypoint_map2 = self.encode2(keypoint_map2)

            keypoint_map = torch.stack([keypoint_map1, keypoint_map2], dim=1)                 
            keypoint_map = torch.flatten(keypoint_map, 0, 1)                # [2B, Q, H, W]
            
            contra_losses.append(contra_loss)
            keypoint_maps.append(keypoint_map)
        # 
        path_3 = self.refinenet3(keypoint_maps[2])
        path_2 = self.refinenet2(path_3, keypoint_maps[1])
        path_1 = self.refinenet1(path_2, keypoint_maps[0])

        return [path_2, path_1], contra_losses
    
if __name__ == '__main__' :
    x = torch.rand((4, 3, 256, 256))
    y = torch.rand((4, 16))
    model = MultiScaleQueryTransformerDecoder()
    print(model(x, y, 2)[0].shape)