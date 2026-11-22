import os
from dataclasses import dataclass
from pathlib import Path


def default_data_dir():
    # Fora do repo (e do OneDrive): o dataset tem ~1.5 GB.
    return os.environ.get('DCGAN_DATA_DIR') or str(Path.home() / 'datasets' / 'dcgan')


@dataclass
class Config:
    # Hiperparâmetros do experimento de 2019
    batch_size: int = 64
    z_dim: int = 100
    learning_rate: float = 0.00025
    beta1: float = 0.45
    epochs: int = 10
    alpha: float = 0.2            # Leaky ReLU
    dropout_rate: float = 0.5     # (em 2019 chamava-se keep_prob, mas era usado como rate)
    smooth_factor: float = 0.1    # label smoothing unilateral
    print_every: int = 10
    show_every: int = 100
    # Correções sobre o código de 2019 (tag v2019). False = comportamento original.
    fix_2019_bugs: bool = True    # D com dropout e batch norm em modo treino
    shuffle: bool = True          # embaralha as imagens a cada época
    # Infra
    data_dir: str = ''
    runs_dir: str = 'runs'
    limit_images: int = 0         # 0 = todas; >0 usa só as N primeiras (testes rápidos)
    max_steps: int = 0            # 0 = sem limite
    seed: int = 0                 # 0 = aleatório

    def __post_init__(self):
        self.data_dir = self.data_dir or default_data_dir()
