"""Gera imagens com um modelo treinado (não precisa de dataset).

    python -m dcgan.generate runs/<data>/ckpt/epoch_010.pt --n 64 --seed 1
    python -m dcgan.generate runs/<data>/ckpt/epoch_010.pt --export gerador.pt     # só o gerador (~14 MB)
    python -m dcgan.generate gerador.pt --n 16 --out minhas_faces                  # usa o arquivo exportado
"""
import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from .models import Generator


def load_generator(path, device='cpu'):
    """Aceita um checkpoint de treino (usa os pesos EMA) ou um arquivo exportado. Devolve (G em modo eval, config)."""
    ck = torch.load(path, map_location=device)
    cfg = ck['config']
    g = Generator(cfg['resolution'], cfg['z_dim'], cfg['g_width']).to(device)
    g.load_state_dict(ck['ema'] or ck['G'] if 'G' in ck else ck['weights'])
    return g.eval(), cfg


def export(path, out):
    """Salva só os pesos do gerador + a config mínima para reconstruí-lo."""
    ck = torch.load(path, map_location='cpu')
    keep = {k: ck['config'][k] for k in ('resolution', 'z_dim', 'g_width')}
    torch.save({'weights': ck['ema'] or ck['G'], 'config': keep}, out)


@torch.no_grad()
def generate(g, cfg, n=64, seed=0, device='cpu'):
    """n imagens em [0, 1], forma (n, 3, R, R)."""
    gen = torch.Generator(device).manual_seed(seed) if seed else None
    z = torch.randn(n, cfg['z_dim'], 1, 1, device=device, generator=gen)
    return g(z).clamp(-1, 1) * 0.5 + 0.5


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('checkpoint')
    p.add_argument('--n', type=int, default=64)
    p.add_argument('--seed', type=int, default=0, help='mesma semente = mesmas imagens (0 = aleatório)')
    p.add_argument('--out', default='geradas', help='prefixo: grava <out>.png (grade) e <out>/NNN.png (individuais)')
    p.add_argument('--export', default='', help='em vez de gerar, salva só o gerador neste arquivo')
    a = p.parse_args()
    if a.export:
        export(a.checkpoint, a.export)
        print('Exportado:', a.export)
    else:
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        g, cfg = load_generator(a.checkpoint, dev)
        imgs = generate(g, cfg, a.n, a.seed, dev)
        save_image(imgs, f'{a.out}.png', nrow=int(a.n ** 0.5), padding=1)
        Path(a.out).mkdir(exist_ok=True)
        for i, im in enumerate(imgs):
            save_image(im, Path(a.out) / f'{i:03d}.png')
        print(f'{a.n} imagens {cfg["resolution"]}px em {a.out}.png e {a.out}/ (dispositivo: {dev})')
