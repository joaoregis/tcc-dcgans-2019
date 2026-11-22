"""DCGAN (Radford et al., 2015) parametrizado pela resolução: o nº de camadas cresce com log2(resolução).

Resolução 64:  G  z → 4×4×512 → 8×8×256 → 16×16×128 → 32×32×64 → 64×64×3
               D  64×64×3 → 32×32×64 → 16×16×128 → 8×8×256 → 4×4×512 → 1 logit
Resolução 128: uma camada a mais em cada rede (largura limitada a 512).
"""
import copy
import math

import torch
from torch import nn

MAX_WIDTH = 512


def widths(resolution, base):
    """Larguras dos blocos do gerador, do 4×4 (mais largo) ao penúltimo (mais estreito)."""
    n = int(math.log2(resolution)) - 2
    return [min(base * 2 ** (n - 1 - i), MAX_WIDTH) for i in range(n)]


def _init(m):
    """Inicialização do artigo DCGAN: N(0, 0.02) nos pesos; BN com escala ~N(1, 0.02)."""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


class Generator(nn.Module):
    def __init__(self, resolution=64, z_dim=100, width=64, channels=3):
        super().__init__()
        w = widths(resolution, width)
        layers = [nn.ConvTranspose2d(z_dim, w[0], 4, 1, 0, bias=False), nn.BatchNorm2d(w[0]), nn.ReLU(True)]  # 1×1 → 4×4
        for a, b in zip(w, w[1:]):                                                                          # dobra a resolução
            layers += [nn.ConvTranspose2d(a, b, 4, 2, 1, bias=False), nn.BatchNorm2d(b), nn.ReLU(True)]
        layers += [nn.ConvTranspose2d(w[-1], channels, 4, 2, 1), nn.Tanh()]                                  # saída em [-1, 1]
        self.net = nn.Sequential(*layers)
        self.apply(_init)

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, resolution=64, width=64, channels=3):
        super().__init__()
        w = widths(resolution, width)[::-1]  # do mais estreito ao mais largo
        layers = [nn.Conv2d(channels, w[0], 4, 2, 1), nn.LeakyReLU(0.2, True)]  # sem BN na 1ª camada (como no artigo)
        for a, b in zip(w, w[1:]):
            layers += [nn.Conv2d(a, b, 4, 2, 1, bias=False), nn.BatchNorm2d(b), nn.LeakyReLU(0.2, True)]
        layers += [nn.Conv2d(w[-1], 1, 4, 1, 0)]  # 4×4 → 1×1: logit
        self.net = nn.Sequential(*layers)
        self.apply(_init)

    def forward(self, x):
        return self.net(x).flatten(1).squeeze(1)  # (B,)


class EMA:
    """Média móvel exponencial dos pesos do gerador. As amostras saem desta cópia: mais estáveis e com menos ruído."""

    def __init__(self, model, decay):
        self.model = copy.deepcopy(model).eval().requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model, step):
        d = min(self.decay, (1 + step) / (10 + step))  # aquecimento: no início confia mais nos pesos novos
        torch._foreach_lerp_(list(self.model.parameters()), list(model.parameters()), 1 - d)
        for e, b in zip(self.model.buffers(), model.buffers()):
            e.copy_(b)  # estatísticas da BatchNorm são copiadas, não suavizadas
