"""Busca de hiperparâmetros com orçamento de tempo fixo: treina cada combinação por N minutos e mede o FID.

    python -m dcgan.tune --epochs 8 --sources celeba --resolution 64 --grid "batch_size=128,256 lr=1e-4,2e-4,4e-4"

Cada campo de Config pode entrar na grade. O resultado (tempo × qualidade) vai para runs/tune_<data>.csv.
"""
import argparse
import csv
import dataclasses
import itertools
from datetime import datetime
from pathlib import Path

import torch

from .config import Config
from .datasets import FaceData
from .metrics import fid
from .models import Generator
from .train import train

_TYPES = {f.name: f.type for f in dataclasses.fields(Config)}
_TYPES['lr'] = float  # atalho: lr = lr_g = lr_d


def parse_grid(text):
    """'batch_size=128,256 lr_g=1e-4,2e-4' → [{'batch_size': 128, 'lr_g': 1e-4}, ...] (produto cartesiano)."""
    axes = {}
    for item in text.split():
        key, _, values = item.partition('=')
        assert key in _TYPES, f'campo desconhecido: {key}'
        conv = (lambda v: v in ('1', 'true', 'True')) if _TYPES[key] is bool else _TYPES[key]
        axes[key] = [conv(v) for v in values.split(',')]
    return [dict(zip(axes, combo)) for combo in itertools.product(*axes.values())]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('--grid', required=True)
    p.add_argument('--minutes', type=float, default=6.0, help='orçamento de TEMPO por combinação (sensível à carga da máquina)')
    p.add_argument('--epochs', type=int, default=0, help='se >0, usa nº fixo de épocas em vez de tempo (comparação justa por passo)')
    p.add_argument('--fid-n', type=int, default=5000)
    p.add_argument('--sources', default='celeba')
    p.add_argument('--resolution', type=int, default=64)
    p.add_argument('--data-dir', default='')
    a = p.parse_args()

    base = dict(sources=a.sources, resolution=a.resolution, data_dir=a.data_dir, max_minutes=0 if a.epochs else a.minutes,
                epochs=a.epochs or 10_000, log_every=500, sample_every=10 ** 9, ckpt_every=10 ** 9)
    data = FaceData.load(a.sources, Config(**base).data_dir, a.resolution)
    out = Path('runs') / f'tune_{datetime.now():%Y-%m-%d_%H%M%S}.csv'
    out.parent.mkdir(exist_ok=True)
    rows = []
    for combo in parse_grid(a.grid):
        kw = dict(combo)
        if 'lr' in kw:
            kw['lr_g'] = kw['lr_d'] = kw.pop('lr')
        cfg = Config(**{**base, **kw})
        run = train(cfg, data)
        ck = torch.load(sorted((run / 'ckpt').glob('*.pt'))[-1], map_location='cuda')
        g = Generator(cfg.resolution, cfg.z_dim, cfg.g_width).cuda()
        g.load_state_dict(ck['ema'] or ck['G'])
        row = {**combo, 'steps': ck['step'], 'epochs_done': round(ck['step'] * cfg.batch_size / len(data), 2),
               'segundos': int((run / 'time.txt').read_text().split(' em ')[1].strip().rstrip('s')),
               'fid': round(fid(data, g, cfg.z_dim, a.fid_n), 2)}
        g.load_state_dict(ck['G'])  # mesmo treino, sem a média EMA: mostra quanto a EMA ajuda
        row['fid_sem_ema'] = round(fid(data, g, cfg.z_dim, a.fid_n), 2)
        rows.append(row)
        print('RESULTADO', row, flush=True)
        with open(out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
    print('Resumo em', out)


if __name__ == '__main__':
    main()
