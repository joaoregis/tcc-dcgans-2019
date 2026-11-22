"""Configuração do experimento: um único dataclass; cada campo vira uma flag de linha de comando."""
import argparse
import dataclasses
import os
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def _local_env():
    """Lê `local.env` (KEY=VALUE, ignorado no git) para configurar caminhos específicos desta máquina."""
    f = _ROOT / 'local.env'
    if not f.exists():
        return {}
    pairs = (line.split('=', 1) for line in f.read_text(encoding='utf-8').splitlines()
             if '=' in line and not line.lstrip().startswith('#'))
    return {k.strip(): v.strip() for k, v in pairs}


def default_data_dir():
    """Pasta dos datasets (raw/ e cache/). Ordem: variável de ambiente, local.env, ~/datasets/faces."""
    return os.environ.get('DCGAN_DATA_DIR') or _local_env().get('DCGAN_DATA_DIR') or str(Path.home() / 'datasets' / 'faces')


@dataclass
class Config:
    # --- Dados ---
    sources: str = 'celeba'       # 'celeba' ou 'celeba:100000,ffhq' (nome[:máx. de imagens]); ver datasets.SOURCES
    resolution: int = 64          # potência de 2 (32, 64, 128, 256)
    data_dir: str = ''
    flip: bool = True             # espelhamento horizontal aleatório (na GPU)
    # --- Modelo ---
    z_dim: int = 100
    g_width: int = 64             # nº base de filtros do gerador (a largura dobra a cada camada, até 512)
    d_width: int = 64
    # --- Otimização ---
    batch_size: int = 128
    lr_g: float = 2e-4
    lr_d: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    label_smooth: float = 0.1     # rótulo "real" = 1 - label_smooth (suavização unilateral, como em 2019)
    ema: float = 0.999            # média móvel dos pesos do gerador (usada nas amostras); 0 = desliga
    epochs: int = 10
    max_minutes: float = 0.0      # 0 = sem limite de tempo
    # --- Desempenho ---
    amp: bool = True              # bfloat16 autocast
    # --- Saídas ---
    runs_dir: str = 'runs'
    log_every: int = 100
    sample_every: int = 1000
    ckpt_every: int = 1           # a cada N épocas
    resume: str = ''              # caminho de um checkpoint .pt
    seed: int = 0                 # 0 = aleatório

    def __post_init__(self):
        self.data_dir = self.data_dir or default_data_dir()
        assert self.resolution >= 8 and self.resolution & (self.resolution - 1) == 0, 'resolution deve ser potência de 2'


def parse_args(argv=None, description=None):
    """Cria um Config a partir da linha de comando (--epochs 5, --no-amp, ...)."""
    p = argparse.ArgumentParser(description=description)
    for f in dataclasses.fields(Config):
        flag = '--' + f.name.replace('_', '-')
        if f.type is bool:
            p.add_argument(flag, default=f.default, action=argparse.BooleanOptionalAction)
        else:
            p.add_argument(flag, default=f.default, type=f.type)
    return Config(**vars(p.parse_args(argv)))
