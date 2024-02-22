import torch
import torch.nn as nn
from timm.models.vision_transformer import Block
from timm.models.crossvit import CrossAttentionBlock

from utils import feat_mul_queries

class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))

class PatchEmbed(nn.Module):
    def __init__(self, H=224, W=224, h=16, w=16, in_chanels=32, embed_dim=128) :
        super(PatchEmbed, self).__init__()
        self.conv = nn.Conv2d(in_chanels, embed_dim, kernel_size=(h, w), stride=(h, w))
        
        num_queries = (H//h)*(W//w) 
        self.pos_emb = nn.Parameter(torch.rand(1, num_queries, embed_dim), requires_grad=True)

    def forward(self, x):
        """
        Args
            x : [B, C, H, W]
        Return
            [B, N, D]
        """
        x = self.conv(x)                    # [B, D, h, w]
        x = x.flatten(2).transpose(1,2)     # [B, N, D]
        x = x + self.pos_emb                # 

        return x

class QueryGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=[2**3, 2**4, 2**5], 
                 H=224, W=224, h=16, w=16, 
                 embed_dim=128, num_head=4, depth=4) :
        super(QueryGenerator, self).__init__()

        # Feature Extract
        dim0, dim1, dim2 = out_channels
        self.conv0 = nn.Sequential(
                        ConvBnReLU(in_channels, dim0, kernel_size=3, stride=1, pad=1),
                        ConvBnReLU(dim0, dim0, kernel_size=3, stride=1, pad=1))

        self.conv1 = nn.Sequential(
                        ConvBnReLU(dim0, dim1, kernel_size=5, stride=2, pad=2),
                        ConvBnReLU(dim1, dim1, kernel_size=3, stride=1, pad=1),
                        ConvBnReLU(dim1, dim1, kernel_size=3, stride=1, pad=1))
        
        self.conv2 = nn.Sequential(
                        ConvBnReLU(dim1, dim2, kernel_size=5, stride=2, pad=2),
                        ConvBnReLU(dim2, dim2, kernel_size=3, stride=1, pad=1),
                        ConvBnReLU(dim2, dim2, kernel_size=3, stride=1, pad=1))

        # Patch Emd
        self.embed = PatchEmbed(H//4, W//4, h, w, dim2, embed_dim)
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_head, mlp_ratio=4.0, qkv_bias=True, norm_layer=nn.LayerNorm)
            for _ in range(depth)])
        
    def forward(self, x) :
        # Feature Extract
        x = self.conv0(x)       # (B, 8, H, W)
        x = self.conv1(x)       # (B, 16, H//2, W//2)
        x = self.conv2(x)       # (B, 32, H//4, W//4)
        
        # Patch Emd
        queries = self.embed(x)         # (B, N, d)
        for blk in self.blocks:
            queries = blk(queries)

        return x, queries

class QueryAttention(nn.Module):
    """
    Image 1과 Image2에서 Query 생성 (Self-Attention)
    생성된 Query의 Cross-Attention
    """
    def __init__(self, in_channels=3, out_channels=[2**3, 2**4, 2**5], 
                 H=224, W=224, h=16, w=16, embed_dim=128, num_head=4, depth=4) -> None:
        super().__init__()
        # 
        self.conv1 = nn.Conv2d(out_channels[-1], embed_dim, kernel_size=1)
        self.conv2 = nn.Conv2d(out_channels[-1], embed_dim, kernel_size=1)

        #
        self.query_gen = QueryGenerator(in_channels, out_channels, H, W, h, w, embed_dim, num_head)
        self.cross_block = nn.ModuleList([
            CrossAttentionBlock(embed_dim, num_head, mlp_ratio=4.0, qkv_bias=True, norm_layer=nn.LayerNorm)
            for _ in range(depth)])

    def forward(self, img1, img2):
        #
        [feat1, query1], [feat2, query2] = [self.query_gen(img) for img in [img1, img2]]
        
        for blk in self.cross_block:
            total_query = blk(query1, query2, query1)          # [B, Q, E]


        feat1, feat2 = self.conv1(feat1), self.conv2(feat2)     # [B, D, h, w] -> [B, E, h, w] h,w = (H,W)//4
        feat1 = feat_mul_queries(feat1, total_query)            # [B, Q, h, w]
        feat2 = feat_mul_queries(feat2, total_query)            # [B, Q, h, w]



if __name__ == "__main__" :
    B, C, H, W = 4, 3, 256, 256
    dim = 256
    conv0 = QueryAttention(in_channels=C, out_channels=[2**3, 2**4, 2**5], 
                 H=H, W=W, h=16, w=16, embed_dim=dim, num_head=4)

    img1 = torch.randn((B, C, H, W))
    img2 = torch.randn((B, C,H, W))
    conv0(img1, img2)