import torch
import torch.nn.functional as F
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler
from einops import rearrange, repeat
from typing import Union

def compute_downsample_rate(input_length: int, n_fft: int, downsampled_width: int) -> int:
    """计算下采样率"""
    '''stft_width = (input_length // (n_fft // 2)) + 1
    return int(round(stft_width / downsampled_width))'''
    print(round(input_length / (np.log2(n_fft) - 1) / downsampled_width), 'xxxxx')
    return round(input_length / (np.log2(n_fft) - 1) / downsampled_width) if input_length >= downsampled_width else 1



def quantize(z, vq_model, transpose_channel_length_axes=False, svq_temp:Union[float,None]=None):
    input_dim = len(z.shape) - 2
    if input_dim == 2:
        h, w = z.shape[2:]
        z = rearrange(z, 'b c h w -> b (h w) c')
        z_q, indices, vq_loss, perplexity = vq_model(z, svq_temp)
        z_q = rearrange(z_q, 'b (h w) c -> b c h w', h=h, w=w)
    elif input_dim == 1:
        if transpose_channel_length_axes:
            z = rearrange(z, 'b c l -> b (l) c')
        z_q, indices, vq_loss, perplexity = vq_model(z, svq_temp)
        if transpose_channel_length_axes:
            z_q = rearrange(z_q, 'b (l) c -> b c l')
    else:
        raise ValueError
    return z_q, indices, vq_loss, perplexity

class linear_warmup_cosine_annealingLR(_LRScheduler):
    """线性热身+余弦退火学习率调度器"""
    def __init__(self, optimizer, max_steps, warmup_ratio=0.1, min_lr=0.):
        self.max_steps = max_steps
        self.warmup_steps = int(max_steps * warmup_ratio)
        self.min_lr = min_lr
        super().__init__(optimizer)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            # 线性热身阶段
            lr_ratio = step / self.warmup_steps
        else:
            # 余弦退火阶段
            progress = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            lr_ratio = 0.5 * (1. + np.cos(np.pi * progress))
        
        return [self.min_lr + (base_lr - self.min_lr) * lr_ratio for base_lr in self.base_lrs]
    