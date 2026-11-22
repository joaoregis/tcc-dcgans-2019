"""Mede imagens/s de um passo de treino com dados sintéticos (só computação, sem disco).

    python -m dcgan.bench --resolution 64 --batches 64,128,256 --amp both --channels-last both
"""
import argparse
import time

import torch

from .config import Config
from .models import Discriminator, Generator
from .train import train_step


def bench(cfg, channels_last=True, steps=30, warmup=10):
    dev = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    fmt = torch.channels_last if channels_last else torch.contiguous_format
    G = Generator(cfg.resolution, cfg.z_dim, cfg.g_width).to(dev, memory_format=fmt)
    D = Discriminator(cfg.resolution, cfg.d_width).to(dev, memory_format=fmt)
    opt_g = torch.optim.Adam(G.parameters(), cfg.lr_g, betas=(cfg.beta1, cfg.beta2))
    opt_d = torch.optim.Adam(D.parameters(), cfg.lr_d, betas=(cfg.beta1, cfg.beta2))
    amp = torch.autocast('cuda', torch.bfloat16, enabled=cfg.amp)
    real = torch.randn(cfg.batch_size, 3, cfg.resolution, cfg.resolution, device=dev).contiguous(memory_format=fmt)
    for i in range(warmup + steps):
        if i == warmup:
            torch.cuda.synchronize(); t = time.time()
        train_step(real, G, D, opt_g, opt_d, cfg, amp)
    torch.cuda.synchronize()
    return steps * cfg.batch_size / (time.time() - t), torch.cuda.max_memory_allocated() / 2**30


def _modes(v):
    return {'on': [True], 'off': [False], 'both': [True, False]}[v]


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--resolution', type=int, default=64)
    p.add_argument('--batches', default='64,128,256')
    p.add_argument('--amp', default='both', choices=['on', 'off', 'both'])
    p.add_argument('--channels-last', default='on', choices=['on', 'off', 'both'])
    a = p.parse_args()
    print('res  batch  amp  chlast   img/s   GB')
    for b in map(int, a.batches.split(',')):
        for amp in _modes(a.amp):
            for cl in _modes(a.channels_last):
                torch.cuda.reset_peak_memory_stats()
                ips, gb = bench(Config(resolution=a.resolution, batch_size=b, amp=amp), cl)
                print(f'{a.resolution:>3}  {b:>5}  {int(amp):>3}  {int(cl):>6}  {ips:>6.0f}  {gb:>4.1f}', flush=True)
