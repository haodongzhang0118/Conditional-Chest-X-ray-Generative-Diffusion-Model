import torch
import torch.nn as nn
from einops import rearrange
from timm.models.vision_transformer import Mlp


def AmpAndPha(img):
    # get the Phase and Amplitude of the FFT of a given image
    fre = torch.fft.rfft2(img, norm="backward")
    amp = torch.abs(fre)
    pha = torch.angle(fre)
    return amp, pha

def inverseFFT(pha, amp):
    real = amp * torch.cos(pha)
    imag = amp * torch.sin(pha)
    fre_out = torch.complex(real, imag)

    return torch.fft.irfft2(fre_out, norm="backward").real

class PGA(nn.Module):
    def __init__(self, dim, bias=False, window_size=4, scale=2):
        super(PGA, self).__init__()
        self.window_size = window_size
        new_hidden_size = window_size ** 2 * dim
        
        self.lienar_amp = nn.Linear(in_features=new_hidden_size // 2 + 1, out_features=(new_hidden_size // 2 + 1) * 2, bias=True)
        self.linear_pha = nn.Linear(in_features=new_hidden_size // 2 + 1, out_features=new_hidden_size // 2 + 1, bias=True)
        self.norm = nn.LayerNorm(dim * window_size ** 2)

        # self.conv_q = nn.Conv2d(new_hidden_size // 2 + 1, new_hidden_size // 2 + 1, kernel_size=1, padding=0)
        # self.conv_k = nn.Conv2d(new_hidden_size // 2 + 1, new_hidden_size // 2 + 1, kernel_size=1, padding=0)
        # self.conv_v = nn.Conv2d(new_hidden_size // 2 + 1, new_hidden_size // 2 + 1, kernel_size=1, padding=0)
        self.atten_norm = nn.LayerNorm(new_hidden_size // 2 + 1)

        self.pooling = nn.AdaptiveAvgPool2d((None, None))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.norm(x) # # (B, N , P * C)
        amp, pha = AmpAndPha(x) # (B, N , P * C / 2)

        hidden_amp = self.lienar_amp(amp) # (B, N, 2 * p * c / 2)
        q = self.linear_pha(pha) # (B, N, p * c / 2)
        k, v = hidden_amp.chunk(2, dim=2) # (B, N, p * c / 2)

        attn_map = q * k  # (B, N, p * c / 2)
        structure_info = self.atten_norm(attn_map) * v  # (B, N, p * c / 2)

        structure_info = torch.fft.irfft2(structure_info) # (B, N, p * c)
        structure_info = self.softmax(self.pooling(structure_info.real))

        return structure_info
