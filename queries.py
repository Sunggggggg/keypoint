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

if __name__ == '__main__' :
    B, C, H, W = 4, 3, 256, 256
    q_dim = 128

    x1, x2 = torch.rand((2, B, C, H, W))