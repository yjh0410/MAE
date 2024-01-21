import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------ Basic Modules ------------------------
## Multi-Layer Perceptron
class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self,
                 in_dim     :int,
                 hidden_dim :int,
                 out_dim    :int,
                 drop       :float = 0.):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

## Vanilla Multi-Head Attention
class Attention(nn.Module):
    def __init__(self,
                 dim                :int,
                 qkv_bias           :bool  = False,
                 num_heads          :int   = 8,
                 num_patches        :int   = None,
                 prefix_causal_mask :bool  = False,
                 dropout            :float = 0.):
        super().__init__()
        # --------------- Basic parameters ---------------
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.prefix_causal_mask = prefix_causal_mask

        # --------------- Network parameters ---------------
        self.qkv_proj = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
        if self.prefix_causal_mask:
            assert num_patches is not None, "The num_patches should be an int type when the prefix_causal_mask is True."
            self.register_buffer(
                "attn_mask",
                torch.ones(1, num_patches, num_patches, dtype=torch.bool).tril(diagonal=0),
            )


    def forward(self, x, mask=None):
        B, N, C = x.shape
        # ----------------- Prefix mask -----------------
        if self.prefix_causal_mask:
            assert mask is not None, "A mask is required for the PrefixLM Causal Attention."
            prefix_mask = (~mask).unsqueeze(1).expand(-1, N, -1).bool()
            attn_mask = self.attn_mask.clone().expand(B, -1, -1)
            attn_mask = torch.logical_or(attn_mask, prefix_mask)
            attn_mask = attn_mask.unsqueeze(1)  # (B, 1, N, N)
        else:
            attn_mask = None

        # ----------------- Input proj -----------------
        qkv = (
            self.qkv_proj(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)

        # ----------------- Multi-head Attn -----------------
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = x.transpose(1, 2).reshape(B, N, C)

        # ----------------- Output -----------------
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

## Attention based Classifier
class AttentionPoolingClassifier(nn.Module):
    def __init__(
        self,
        in_dim      : int,
        out_dim     : int,
        num_heads   : int = 12,
        qkv_bias    : bool = False,
        num_queries : int = 1,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = in_dim // num_heads
        self.scale = head_dim**-0.5

        self.k = nn.Linear(in_dim, in_dim, bias=qkv_bias)
        self.v = nn.Linear(in_dim, in_dim, bias=qkv_bias)

        self.cls_token = nn.Parameter(torch.randn(1, num_queries, in_dim) * 0.02)
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(in_dim, affine=False, eps=1e-6)

        self.num_queries = num_queries

    def forward(self, x: torch.Tensor):
        B, N, C = x.shape

        x = self.bn(x.transpose(-2, -1)).transpose(-2, -1)
        cls_token = self.cls_token.expand(B, -1, -1)  # newly created class token

        q = cls_token.reshape(
            B, self.num_queries, self.num_heads, C // self.num_heads
        ).permute(0, 2, 1, 3)
        k = (
            self.k(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        q = q * self.scale
        v = (
            self.v(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)

        x_cls = (attn @ v).transpose(1, 2).reshape(B, self.num_queries, C)
        x_cls = x_cls.mean(dim=1)

        out = self.linear(x_cls)
        return out, x_cls


# ------------------------ Core Modules ------------------------
## ViT's Block
class ViTBlock(nn.Module):
    def __init__(
            self,
            dim                :int,
            qkv_bias           :bool  = False,
            num_heads          :int   = 8,
            num_patches        :int   = None,
            mlp_ratio          :float = 4.0,
            prefix_causal_mask :bool  = False,
            dropout            :float = 0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, qkv_bias, num_heads, num_patches, prefix_causal_mask, dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.ffn = MLP(dim, int(dim * mlp_ratio), dim, dropout)

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))

        return x

