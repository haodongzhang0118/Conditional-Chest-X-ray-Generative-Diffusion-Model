import math
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from diffusion.gaussian_diffusion import _extract_into_tensor
from timm.models.vision_transformer import Mlp
from FGFormer.FourierGuidanceInfo import PGA
from FGFormer.LocalAttention import LocalWindowAttention
from FGFormer.utils import window_partition, window_reverse, CrossAttention

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core FGFOrmer Model                           #
#################################################################################
class FGFormer_block(nn.Module):
    def __init__(self, dim,
                 mlp_ratio=4.0,
                 num_heads=8,
                 bias=False,
                 window_size=8,
                 scale=2,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 ):
        super().__init__()
        self.window_size = window_size
        self.pga = PGA(dim, bias=bias, window_size=window_size, scale=scale)
        self.LW_attn = LocalWindowAttention(dim,
                                            num_heads=num_heads,
                                            mlp_ratio=mlp_ratio,
                                            qkv_bias=qkv_bias,
                                            qk_scale=qk_scale,
                                            drop=drop,
                                            attn_drop=attn_drop,
                                            act_layer=act_layer,
                                            norm_layer=norm_layer,
                                            window_size=window_size
                                            )
        
        self.norm1 = nn.LayerNorm(window_size ** 2 * dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(window_size ** 2 * dim, elementwise_affine=False, eps=1e-6)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim * window_size ** 2, bias=True)
        )

        mlp_hidden_dim = int(window_size ** 2 * dim * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp1 = Mlp(in_features=window_size ** 2 * dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

    def forward(self, x, structure_info, c):
        initial = x
        shift_msa, scale_msa, gate_msa, shift_res, scale_res, gate_res = self.adaLN_modulation(c).chunk(6, dim=1)
        x = modulate(self.norm1(x), shift_msa, scale_msa)

        structure_info = self.pga(structure_info)
        x = self.LW_attn(x, structure_info)

        x = initial + gate_msa.unsqueeze(1) * x
        x = x + gate_res.unsqueeze(1) * self.mlp1(modulate(self.norm2(x), shift_res, scale_res))

        return x, structure_info

class FinalLayer(nn.Module):
  def __init__(self,
               hidden_size=256,
               window_size=8,
               output_channel=8,
               bias=False):
    super().__init__()
    self.window_size = window_size
    self.norm_final = nn.LayerNorm(window_size ** 2 * hidden_size, elementwise_affine=False, eps=1e-6)
    self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size * window_size ** 2, bias=True)
    )
    self.linear = nn.Linear(in_features=hidden_size, out_features=output_channel, bias=True)


  def forward(self, x, c):
    _, _, D = x.shape
    shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
    x = self.norm_final(x)
    x = modulate(x, shift, scale)
    x = rearrange(x, 'b n (p c) -> b n p c', p=self.window_size ** 2, c=D // self.window_size ** 2)
    x = self.linear(x)
    return x

class FGFormer(nn.Module):
  # PGA needs the Positional Embedding
  def __init__(self,
               input_size=32,
               window_size=8,
               in_channels=4,
               hidden_size=256,
               depth=12,
               num_heads=8,
               mlp_ratio=4,
               drop=0.1,
               num_classes=1000,
               learn_sigma=True,
               bias=False,
               qkv_bias=False,
               qk_scale=None,
               scale=2,
               training=False):
    super().__init__()
    self.depth = depth
    self.learn_sigma = learn_sigma
    self.in_channels = in_channels
    self.out_channels = in_channels * 2 if learn_sigma else in_channels
    self.window_size = window_size
    self.num_heads = num_heads
    self.training = training
    self.hidden_size = hidden_size
    self.new_hidden_size = window_size ** 2 * hidden_size
    self.num_patches = (input_size // window_size) ** 2

    self.t_embedder = TimestepEmbedder(hidden_size=hidden_size)
    self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.new_hidden_size), requires_grad=False)

    self.blocks = nn.ModuleList([FGFormer_block(
                                                dim=hidden_size,
                                                mlp_ratio=mlp_ratio,
                                                num_heads=num_heads,
                                                bias=bias,
                                                window_size=window_size,
                                                scale=scale,
                                                qkv_bias=qkv_bias,
                                                qk_scale=qk_scale,
                                                drop=drop,
                                                attn_drop=drop,
                                                act_layer=nn.GELU,
                                                norm_layer=nn.LayerNorm) for _ in range(depth)])
    
    self.x_cross_attns = nn.ModuleList([CrossAttention(self.new_hidden_size, num_heads=num_heads, qkv_bias=True) for _ in range(depth - 1)])
    self.struc_cross_attns = nn.ModuleList([CrossAttention(self.new_hidden_size, num_heads=num_heads, qkv_bias=True) for _ in range(depth - 1)])

    self.final_layer_noise = FinalLayer(hidden_size=hidden_size, window_size=window_size, output_channel=in_channels * 2)
    self.cnn1 = nn.Conv2d(in_channels=in_channels, out_channels=hidden_size, kernel_size=3, padding=1, stride=1)
    self.cnn2 = nn.Conv2d(in_channels=in_channels, out_channels=hidden_size, kernel_size=3, padding=1, stride=1)
    self.initialize_weights()

  def initialize_weights(self):
    # Initialize transformer layers:

    # def _basic_init(module):
    #     if isinstance(module, nn.Linear):
    #         torch.nn.init.xavier_uniform_(module.weight)
    #         if module.bias is not None:
    #             nn.init.constant_(module.bias, 0)
    # self.apply(_basic_init)

    # Initialize (and freeze) pos_embed by sin-cos embedding:
    pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches ** 0.5))
    self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    # Initialize timestep embedding MLP:
    nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
    nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

    # Zero-out adaLN modulation layers in DiT blocks:
    for block in self.blocks:
        nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    # Zero-out output layers:
    nn.init.constant_(self.final_layer_noise.adaLN_modulation[-1].weight, 0)
    nn.init.constant_(self.final_layer_noise.adaLN_modulation[-1].bias, 0)
    nn.init.constant_(self.final_layer_noise.linear.weight, 0)
    nn.init.constant_(self.final_layer_noise.linear.bias, 0)

  def forward(self, x, t, y):
    t = self.t_embedder(t)
    _, _, H, W = x.shape
    stru = x
    x = self.cnn1(x) # (B, hidden_size, H, W)
    stru = self.cnn2(stru) # (B, hidden_size, H, W)

    x = rearrange(window_partition(x, window_size=self.window_size), 'b n p c -> b n (p c)') + self.pos_embed # (B, N, window_size ^ 2, hidden_size)
    stru = rearrange(window_partition(stru, window_size=self.window_size), 'b n p c -> b n (p c)') + self.pos_embed # (B, N, window_size ^ 2, hidden_size)

    previous = []
    for index in range(len(self.blocks)):
        x = self.x_cross_attns[index](y, x)
        stru = self.struc_cross_attns[index](y, stru)

        if index < self.depth // 2 - 1:
            x, stru = self.blocks[index](x, stru, t)
            previous.append((x, stru))
        elif index >= self.depth // 2 + 1:
            need = previous[::-1][index - (self.depth // 2 + 1)]
            x, stru = self.blocks[index](x + need[0], stru + need[1], t)
        else:
            x, stru = self.blocks[index](x, stru, t)

    x = self.final_layer_noise(x, t)

    x = window_reverse(x, self.window_size, H, W)

    return x

  def forward_with_cfg(self, x, t, y, cfg_scale):
    """
    Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
    """
    # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
    half = x[: len(x) // 2]
    combined = torch.cat([half, half], dim=0)
    model_out = self.forward(combined, t, y)
    # For exact reproducibility reasons, we apply classifier-free guidance on only
    # three channels by default. The standard approach to cfg applies it to all channels.
    # This can be done by uncommenting the following line and commenting-out the line following that.
    # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
    eps, rest = model_out[:, :3], model_out[:, 3:]
    cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
    half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
    eps = torch.cat([half_eps, half_eps], dim=0)
    return torch.cat([eps, rest], dim=1)

#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   FGFormer Configs                            #
#################################################################################
def FGFormer_XL_2(**kwargs):
   pass

def FGFormer_XL_4(**kwargs):
   pass
def FGFormer_XL_8(**kwargs):
   pass

def FGFormer_L_2(**kwargs):
   pass
def FGFormer_L_4(**kwargs):
   pass

def FGFormer_L_8(**kwargs):
   pass

def FGFormer_B_2(**kwargs):
   return FGFormer(depth=12, hidden_size=48, window_size=2, num_heads=12, **kwargs)

def FGFormer_B_4(**kwargs):
   return FGFormer(depth=12, hidden_size=48, window_size=4, num_heads=12, **kwargs)

def FGFormer_B_8(**kwargs):
   return FGFormer(depth=12, hidden_size=48, window_size=8, num_heads=12, **kwargs)

def FGFormer_S_2(**kwargs):
   pass

def FGFormer_S_4(**kwargs):
   pass

def FGFormer_S_8(**kwargs):
   pass

FGFormers = {
   'FGFormer-XL/2': FGFormer_XL_2, 'FGFormer-XL/4': FGFormer_XL_4, 'FGFormer-XL/8': FGFormer_XL_8,
   'FGFormer-L/2': FGFormer_L_2, 'FGFormer-L/4': FGFormer_L_4, 'FGFormer-L/8': FGFormer_L_8,
   'FGFormer-B/2': FGFormer_B_2, 'FGFormer-B/4': FGFormer_B_4, 'FGFormer-B/8': FGFormer_B_8,
   'FGFormer-S/2': FGFormer_S_2, 'FGFormer-S/4': FGFormer_S_4, 'FGFormer-S/8': FGFormer_S_8
}
















