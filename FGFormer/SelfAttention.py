import torch
import torch.nn as nn
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, DropPath, LayerNorm2d
from timm.models.vision_transformer import Attention
from einops import rearrange
from FGFormer.utils import window_partition, window_reverse
import numpy as np
import logging

class AttentionModule(nn.Module):
    def __init__(self,
                 feature_dim,

                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,

                 drop=0.,
                 attn_drop=0.,

                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,

                 window_size=8):
        super().__init__()
        """
        Args:
            feature_dim: feature size dimension.
            num_heads: number of attention head.
            mlp_ratio: MLP ratio.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            act_layer: activation function.
            norm_layer: normalization layer.
            window_size: window size.
        """
        # positional encoding for windows
        self.window_size = window_size

        self.norm1 = norm_layer(window_size ** 2 * feature_dim)
        self.attn = Attention(window_size ** 2 * feature_dim, num_heads=num_heads, qkv_bias=True)

    def forward(self, x, global_noise_info):
        """
        Args:
            x: input feature map. (Batch size, num_window, window_size^2 * dim)
            global_noise_info: Global noise information from PGA.
        """
        B, N, D = x.shape
        x = x * global_noise_info
        x = self.attn(self.norm1(x)) # (N, P, D)

        return x