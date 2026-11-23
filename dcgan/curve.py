"""Curva FID × época de uma execução: mede o FID (cópia EMA) de cada checkpoint em runs/<execução>/ckpt.

    python -m dcgan.curve runs/2026-11-22_230000 --fid-n 5000
Grava runs/<execução>/fid_curve.csv. Serve para achar o "joelho": a época a partir da qual treinar mais quase não melhora o FID.
"""
import argparse
import csv
from pathlib import Path

import torch

from .config import Config
from .datasets import FaceData
from .metrics import fid
from .models import Generator

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('run')
    p.add_argument('--fid-n', type=int, default=5000)
    a = p.parse_args()
    run = Path(a.run)
    ckpts = sorted((run / 'ckpt').glob('*.pt'))
    cfg = torch.load(ckpts[0], map_location='cpu')['config']
    data = FaceData.load(cfg['sources'], cfg['data_dir'], cfg['resolution'])
    g = Generator(cfg['resolution'], cfg['z_dim'], cfg['g_width']).cuda()
    rows = []
    for c in ckpts:
        ck = torch.load(c, map_location='cuda')
        g.load_state_dict(ck['ema'] or ck['G'])
        rows.append({'epoca': ck['epoch'], 'passo': ck['step'], 'fid': round(fid(data, g, cfg['z_dim'], a.fid_n), 2)})
        print(rows[-1], flush=True)
        with open(run / 'fid_curve.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
