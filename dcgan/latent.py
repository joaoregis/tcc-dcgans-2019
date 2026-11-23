"""Manipulação do espaço latente: interpolação entre rostos e aritmética de vetores (Radford et al., 2015, §6).

Uma seed (inteiro) só serve para sortear um vetor z; as operações são feitas nos vetores z, nunca nas seeds.

    # transição suave entre os rostos das seeds 42 e 7 (8 passos, interpolação esférica)
    python -m dcgan.latent gerador.pt interpolate --seeds 42 7 --steps 8

    # "mulher de óculos" = média(3 homens de óculos) - média(3 homens sem óculos) + média(3 mulheres sem óculos)
    python -m dcgan.latent gerador.pt arith --expr "mean(1,2,3) - mean(4,5,6) + mean(7,8,9)"
    python -m dcgan.latent gerador.pt arith --expr "0.5*10 + 0.5*20"        # mistura meio a meio das seeds 10 e 20
"""
import argparse
import math
import re

import torch
from torchvision.utils import save_image

from .generate import load_generator


def z_from_seed(seed, z_dim):
    """O vetor z (1, z_dim, 1, 1) que a seed sorteia. A mesma seed dá sempre o mesmo z."""
    return torch.randn(1, z_dim, 1, 1, generator=torch.Generator().manual_seed(seed))


def slerp(a, b, t):
    """Interpolação esférica: mantém a norma (z ~ N(0,I) vive numa casca de raio ~√z_dim); a linear a encolhe no meio."""
    fa, fb = a.flatten(), b.flatten()
    omega = torch.acos(torch.clamp(torch.dot(fa / fa.norm(), fb / fb.norm()), -1, 1))
    if omega.abs() < 1e-4:
        return (1 - t) * a + t * b
    return (torch.sin((1 - t) * omega) * a + torch.sin(t * omega) * b) / torch.sin(omega)


def interpolate(a, b, steps, spherical=True):
    """`steps` vetores indo de a até b (inclusive)."""
    ts = [i / (steps - 1) for i in range(steps)]
    return torch.cat([slerp(a, b, t) if spherical else (1 - t) * a + t * b for t in ts])


_TERM = re.compile(r'\s*([+-])?\s*(?:([0-9.]+)\s*\*\s*)?(?:mean\(([0-9,\s]+)\)|(\d+))\s*')


def eval_expr(expr, z_dim):
    """Avalia 'mean(1,2,3) - mean(4,5,6) + 0.5*7' sobre os z das seeds. Termos: [coef*](mean(seeds) | seed). Sem eval()."""
    pos, total, used = 0, torch.zeros(1, z_dim, 1, 1), []
    while pos < len(expr):
        m = _TERM.match(expr, pos)
        if not m or m.end() == pos:
            raise ValueError(f'expressão inválida perto de: {expr[pos:pos + 15]!r}')
        sign, coef, group, single = m.groups()
        seeds = [int(s) for s in (group or single).split(',')]
        used += seeds
        z = torch.stack([z_from_seed(s, z_dim) for s in seeds]).mean(0)
        total = total + (-1 if sign == '-' else 1) * float(coef or 1) * z
        pos = m.end()
    return total, used


def decode(g, z):
    """Imagens em [0, 1] para os vetores z (N, z_dim, 1, 1)."""
    with torch.no_grad():
        return g(z.to(next(g.parameters()).device)).clamp(-1, 1).mul(0.5).add(0.5).cpu()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('checkpoint')
    sub = p.add_subparsers(dest='cmd', required=True)
    i = sub.add_parser('interpolate')
    i.add_argument('--seeds', type=int, nargs=2, required=True)
    i.add_argument('--steps', type=int, default=8)
    i.add_argument('--linear', action='store_true', help='interpolação linear em vez de esférica')
    a_ = sub.add_parser('arith')
    a_.add_argument('--expr', required=True)
    for s in (i, a_):
        s.add_argument('--out', default='latent.png')
        s.add_argument('--save-z', default='', help='salva o(s) vetor(es) z resultante(s) em .pt')
    args = p.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    g, cfg = load_generator(args.checkpoint, dev)
    zd = cfg['z_dim']
    if args.cmd == 'interpolate':
        za, zb = (z_from_seed(s, zd) for s in args.seeds)
        z = interpolate(za, zb, args.steps, not args.linear)
        imgs, nrow = decode(g, z), args.steps
    else:
        result, used = eval_expr(args.expr, zd)
        ordered = list(dict.fromkeys(used))
        z = result
        # linha de cima: rostos de cada seed usada; última imagem: o resultado
        imgs = decode(g, torch.cat([torch.cat([z_from_seed(s, zd) for s in ordered]), result]))
        nrow = len(imgs)
    save_image(imgs, args.out, nrow=nrow, padding=2)
    if args.save_z:
        torch.save(z, args.save_z)
    print(f'{args.out} ({args.cmd}); a última imagem é o resultado' if args.cmd == 'arith' else f'{args.out}')
