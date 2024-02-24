import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(self, dim=192, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self._reset_parameters()

    def _reset_parameters(self):
        torch.manual_seed(0)
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

        # b, h, n, d
        q = self.q(x_q).reshape(B, N_q, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x_k).reshape(B, N_kv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x_v).reshape(B, N_kv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # [b, h, n, d] * [b, h, d, m] -> [b, h, n, m]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N_q, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class VolumeAttention(nn.Module):
    def __init__(self, num_query=16, query_dim=256) -> None:
        super().__init__()
        self.num_query = num_query
        self.query_dim = query_dim

        self.quries = nn.Parameter(torch.rand(num_query, query_dim), requires_grad=True)
        self.cross_attention = CrossAttention(dim=query_dim, num_heads=4)

    def forward(self, feat1, feat2):
        B, E, H, W = feat1.shape
    
        feat1 = feat1.reshape(B, self.query_dim, -1)   # [B, e, hw]
        feat2 = feat2.reshape(B, self.query_dim, -1)

        quries = self.quries.expand(B, self.num_query, self.query_dim)
        query1 = self.cross_attention(quries, feat1.permute(0, 2, 1), feat1.permute(0, 2, 1)) #[B, Q, e]
        query2 = self.cross_attention(quries, feat2.permute(0, 2, 1), feat2.permute(0, 2, 1)) #[B, Q, e]
        query = (query1+query2)/2
        
        feat1 = torch.matmul(query, feat1)    # [B, Q, e]*[B, e, hw] = [B, Q, hw]
        feat2 = torch.matmul(query, feat2)    

        feat1 = feat1.reshape(B, self.num_query, H, W)
        feat2 = feat2.reshape(B, self.num_query, H, W)
        return feat1, feat2

############################## Patch Embedding based Queries Generation ##############################
# class PatchEmbed(nn.Module):
#     def __init__(self, H=224, W=224, h=16, w=16, in_chanels=32, embed_dim=128) :
#         super(PatchEmbed, self).__init__()
#         self.conv = nn.Conv2d(in_chanels, embed_dim, kernel_size=(h, w), stride=(h, w))
        
#         num_queries = (H//h)*(W//w) 
#         self.pos_emb = nn.Parameter(torch.rand(1, num_queries, embed_dim), requires_grad=True)

#     def forward(self, x):
#         """
#         Args
#             x : [B, C, H, W]
#         Return
#             [B, N, D]
#         """
#         x = self.conv(x)                    # [B, D, h, w]
#         x = x.flatten(2).transpose(1,2)     # [B, N, D]
#         x = x + self.pos_emb                # 

#         return x

# class QueryGenerator(nn.Module):
#     def __init__(self, in_channels=3, out_channels=[2**3, 2**4, 2**5], 
#                  H=224, W=224, h=16, w=16, 
#                  embed_dim=128, num_head=4, depth=4) :
#         super(QueryGenerator, self).__init__()

#         # Feature Extract
#         dim0, dim1, dim2 = out_channels
#         self.conv0 = nn.Sequential(
#                         ConvBnReLU(in_channels, dim0, kernel_size=3, stride=1, pad=1),
#                         ConvBnReLU(dim0, dim0, kernel_size=3, stride=1, pad=1))

#         self.conv1 = nn.Sequential(
#                         ConvBnReLU(dim0, dim1, kernel_size=5, stride=2, pad=2),
#                         ConvBnReLU(dim1, dim1, kernel_size=3, stride=1, pad=1),
#                         ConvBnReLU(dim1, dim1, kernel_size=3, stride=1, pad=1))
        
#         self.conv2 = nn.Sequential(
#                         ConvBnReLU(dim1, dim2, kernel_size=5, stride=2, pad=2),
#                         ConvBnReLU(dim2, dim2, kernel_size=3, stride=1, pad=1),
#                         ConvBnReLU(dim2, dim2, kernel_size=3, stride=1, pad=1))

#         # Patch Emd
#         self.embed = PatchEmbed(H//4, W//4, h, w, dim2, embed_dim)
#         self.blocks = nn.ModuleList([
#             Block(embed_dim, num_head, mlp_ratio=4.0, qkv_bias=True, norm_layer=nn.LayerNorm)
#             for _ in range(depth)])
        
#     def forward(self, x) :
#         # Feature Extract
#         x = self.conv0(x)       # (B, 8, H, W)
#         x = self.conv1(x)       # (B, 16, H//2, W//2)
#         x = self.conv2(x)       # (B, 32, H//4, W//4)
        
#         # Patch Emd
#         queries = self.embed(x)         # (B, N, d)
#         for blk in self.blocks:
#             queries = blk(queries)

#         return x, queries

# class QueryAttention(nn.Module):
#     """
#     Image 1과 Image2에서 Query 생성 (Self-Attention)
#     생성된 Query의 Cross-Attention
#     """
#     def __init__(self, in_channels=3, out_channels=[2**3, 2**4, 2**5], 
#                  H=224, W=224, h=16, w=16, embed_dim=128, num_head=4, depth=4) -> None:
#         super().__init__()
#         # 
#         self.conv1 = nn.Conv2d(out_channels[-1], embed_dim, kernel_size=1)
#         self.conv2 = nn.Conv2d(out_channels[-1], embed_dim, kernel_size=1)

#         #
#         self.query_gen = QueryGenerator(in_channels, out_channels, H, W, h, w, embed_dim, num_head)
        

#     def forward(self, img1, img2):
#         #
#         [feat1, query1], [feat2, query2] = [self.query_gen(img) for img in [img1, img2]]
        
#         for blk in self.cross_block:
#             total_query = blk(query1, query2, query1)          # [B, Q, E]

#         feat1, feat2 = self.conv1(feat1), self.conv2(feat2)     # [B, D, h, w] -> [B, E, h, w] h,w = (H,W)//4
#         feat1 = feat_mul_queries(feat1, total_query)            # [B, Q, h, w]
#         feat2 = feat_mul_queries(feat2, total_query)            # [B, Q, h, w]


if __name__ == '__main__' :
    B, C, H, W = 4, 3, 256, 256
    q_dim = 128

    x1, x2 = torch.rand((2, B, C, H, W))