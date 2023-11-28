from einops import rearrange
import torch.nn as nn
import torch

def window_partition(x, window_size):
    # This is the Patchify function that can divide the image into smaller patch
    # The output size with default value of reshape_seq is (B, H // window_size, W // window_size, window_size, window_size, C)
    # Batch size, number of window in height, number of window in width, window size, window size, channel

    # reshape_seq = True: Reshape the 2D patch into a 1D sequence
    # The output size with reshape_seq = True is (B, window_size * window_size, C) (H, W, C)
    # Batch size, Patch Sequence, Channal

    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 3, 5, 1)
    windows = rearrange(windows, 'b h w p1 p2 c -> b (h w) (p1 p2) c')
    return windows # (B N P^2 C)

def window_reverse(windows, window_size, H, W):
    # Reverse the patchified image back to the original image
    # (B, N, P^2, C)
    windows = rearrange(windows, 'b (h w) (p1 p2) c -> b h w p1 p2 c', p1=window_size, p2=window_size, h= H // window_size, w = W // window_size)
    windows = rearrange(windows, 'b h w p1 p2 c -> b c (h p1) (w p2)')
    return windows # (B C H W)

class CrossAttention(nn.Module):

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, kv) -> torch.Tensor:
        B, N, C = x.shape

        kv = self.kv(kv).reshape(B, N, 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)
        q = self.q(x).reshape(B, self.num_heads, N, self.head_dim)
        k, v = kv.unbind(0)

        q, k = self.q_norm(q), self.k_norm(k)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
