import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os

# from models import FGFormer
from FGFormer.FGFormer import FGFormers
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL

from datetime import timedelta

model = FGFormers["FGFormer-B/4"]()

x = torch.ones(64, 4, 32, 32)
t = torch.ones(64)
y = torch.ones(64).to(torch.int64)
alphas = torch.ones(1000)
alphas_cumprod = torch.ones(1000)
alphas_cumprod_prev = torch.ones(1000)

model(x, t, y, alphas, alphas_cumprod, alphas_cumprod_prev)
