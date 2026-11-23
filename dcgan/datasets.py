"""Datasets de faces: registro de fontes, pré-processamento para cache `.npy` uint8 e carregamento em lotes.

Layout em disco (`data_dir`, p.ex. D:\\datasets\\faces):
    raw/<fonte>/...            imagens originais (qualquer subpasta; .jpg/.jpeg/.png)
    cache/<fonte>_<res>.npy    array uint8 (N, res, res, 3), pronto para treinar
"""
import hashlib
import os
import queue
import threading
import zipfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image
from tqdm import tqdm

IMAGE_EXT = {'.jpg', '.jpeg', '.png'}


def center_box(fraction):
    """Caixa quadrada centrada com `fraction` do lado menor (1.0 = imagem inteira)."""
    def box(w, h):
        s = int(min(w, h) * fraction)
        left, top = (w - s) // 2, (h - s) // 2
        return left, top, left + s, top + s
    return box


def _celeba_box(w, h):
    # CelebA alinhado (178×218): recorte 148×148 que inclui cabelo e queixo (usado em tutoriais DCGAN).
    return (15, 40, 163, 188) if (w, h) == (178, 218) else center_box(0.68)(w, h)


@dataclass(frozen=True)
class Source:
    box: Callable[[int, int], tuple]
    url: Optional[str] = None     # None = colocar as imagens manualmente em raw/<fonte>/
    md5: Optional[str] = None
    note: str = ''


SOURCES = {
    'celeba': Source(_celeba_box, 'https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/celeba.zip',
                     '00d2c5bc6d35e252742224ab0c1e8fcb', 'CelebA alinhado, 202 599 imagens 178×218 (~1,4 GB)'),
    'ffhq': Source(center_box(1.0), note='FFHQ thumbnails 128×128 (70 000)'),
    'celebahq': Source(center_box(1.0), note='CelebA-HQ (30 000), faces alinhadas em alta resolução'),
    'utkface': Source(center_box(1.0), note='UTKFace "aligned & cropped" (~20 000, 200×200)'),
    'lfw': Source(center_box(0.55), note='LFW (13 233, 250×250); a face ocupa só o centro'),
}


# ---------------------------------------------------------------- download / pré-processamento
def fetch_parallel(url, dest, connections=16, part=8 << 20, desc=''):
    """Download em partes com várias conexões (HTTP Range). Servidores como o do CelebA limitam cada conexão
    a ~100-300 kB/s; 16 conexões somam bem mais. Cada parte tenta até 5 vezes."""
    with urlopen(Request(url, method='HEAD')) as r:
        size = int(r.headers['Content-Length'])
    with open(dest, 'wb') as f:
        f.truncate(size)
    bar = tqdm(total=size, unit='B', unit_scale=True, desc=desc)

    def get(start):
        end = min(start + part, size) - 1
        for attempt in range(5):
            try:
                with urlopen(Request(url, headers={'Range': f'bytes={start}-{end}'}), timeout=60) as r:
                    data = r.read()
                assert len(data) == end - start + 1
                with open(dest, 'r+b') as f:   # cada parte abre o arquivo: seek independente por thread
                    f.seek(start)
                    f.write(data)
                bar.update(len(data))
                return
            except Exception:
                if attempt == 4:
                    raise
    with ThreadPoolExecutor(connections) as ex:
        list(ex.map(get, range(0, size, part)))
    bar.close()


def download(name, data_dir):
    """Baixa e extrai uma fonte com URL conhecida em raw/<nome>/. Sem URL, explica o que fazer."""
    src, raw = SOURCES[name], Path(data_dir) / 'raw' / name
    if raw.exists() and any(raw.rglob('*.*')):
        return raw
    if not src.url:
        raise FileNotFoundError(f'Coloque as imagens de {name} em {raw} ({src.note}).')
    raw.mkdir(parents=True, exist_ok=True)
    zip_path = raw / 'download.zip'
    fetch_parallel(src.url, zip_path, desc=f'Baixando {name}')
    if src.md5:
        digest = hashlib.md5()
        with open(zip_path, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b''):
                digest.update(chunk)
        assert digest.hexdigest() == src.md5, f'{zip_path} corrompido; apague e tente de novo.'
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(raw)
    zip_path.unlink()
    return raw


def load_image(path, name, resolution):
    """Abre, recorta (caixa da fonte) e reduz com Lanczos. Devolve uint8 (res, res, 3)."""
    with Image.open(path) as im:
        im = im.convert('RGB')
        im = im.crop(SOURCES[name].box(*im.size)).resize((resolution, resolution), Image.LANCZOS)
        return np.asarray(im)


def _job(args):
    return load_image(*args)


def prepare(name, data_dir, resolution, workers=None):
    """Gera cache/<nome>_<res>.npy (uma vez; reutilizado depois). Devolve o caminho."""
    out = Path(data_dir) / 'cache' / f'{name}_{resolution}.npy'
    if out.exists():
        return out
    files = sorted(p for p in download(name, data_dir).rglob('*') if p.suffix.lower() in IMAGE_EXT)
    assert files, f'nenhuma imagem em {Path(data_dir) / "raw" / name}'
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix('.tmp.npy')
    arr = np.lib.format.open_memmap(tmp, mode='w+', dtype=np.uint8, shape=(len(files), resolution, resolution, 3))
    with ProcessPoolExecutor(workers or max(1, (os.cpu_count() or 4) - 2)) as ex:
        jobs = ((str(f), name, resolution) for f in files)
        for i, img in enumerate(tqdm(ex.map(_job, jobs, chunksize=64), total=len(files), desc=f'Preparando {name}@{resolution}')):
            arr[i] = img
    arr.flush()
    del arr
    tmp.replace(out)
    return out


# ---------------------------------------------------------------- mistura de fontes e lotes
def parse_sources(spec):
    """'celeba:100000,ffhq' → [('celeba', 100000), ('ffhq', None)]."""
    out = []
    for item in spec.split(','):
        name, _, cap = item.strip().partition(':')
        assert name in SOURCES, f'fonte desconhecida: {name} (opções: {", ".join(SOURCES)})'
        out.append((name, int(cap) if cap else None))
    return out


class FaceData:
    """Várias fontes vistas como um só array (N, res, res, 3) uint8; os arquivos .npy ficam em memmap (sem cópia)."""

    def __init__(self, arrays):
        self.arrays = arrays
        self.offsets = np.cumsum([0] + [len(a) for a in arrays])

    @classmethod
    def load(cls, spec, data_dir, resolution):
        arrays = []
        for name, cap in parse_sources(spec):
            a = np.load(prepare(name, data_dir, resolution), mmap_mode='r')
            arrays.append(a[:cap] if cap else a)
        return cls(arrays)

    def __len__(self):
        return int(self.offsets[-1])

    def take(self, idx):
        """Linhas `idx` (índices globais, ordenados) como um array uint8 contíguo."""
        src = np.searchsorted(self.offsets, idx, side='right') - 1
        out = np.empty((len(idx),) + self.arrays[0].shape[1:], np.uint8)
        for s in np.unique(src):
            m = src == s
            out[m] = self.arrays[s][idx[m] - self.offsets[s]]
        return out


class Loader:
    """Itera lotes float32 em [-1, 1] já na GPU. Um thread lê/embaralha em segundo plano (sobrepõe E/S e treino);
    normalização e espelhamento acontecem na GPU. Descarta o último lote incompleto."""

    def __init__(self, data, batch_size, device, flip=True, seed=0):
        self.data, self.batch_size, self.device, self.flip = data, batch_size, device, flip
        self.rng = np.random.default_rng(seed or None)

    def __len__(self):
        return len(self.data) // self.batch_size

    def _produce(self, q):
        import torch
        perm = self.rng.permutation(len(self.data))
        for k in range(len(self)):
            idx = np.sort(perm[k * self.batch_size:(k + 1) * self.batch_size])
            q.put(torch.from_numpy(self.data.take(idx)).pin_memory())
        q.put(None)

    def __iter__(self):
        import torch  # import tardio: os processos do `prepare` não precisam carregar o PyTorch (~1 GB cada)
        q = queue.Queue(maxsize=4)
        threading.Thread(target=self._produce, args=(q,), daemon=True).start()
        while (batch := q.get()) is not None:
            x = batch.to(self.device, non_blocking=True).permute(0, 3, 1, 2).float().div_(127.5).sub_(1.0)
            if self.flip:
                x = torch.where(torch.rand(x.size(0), 1, 1, 1, device=self.device) < 0.5, x.flip(3), x)
            yield x.contiguous(memory_format=torch.channels_last)


if __name__ == '__main__':  # python -m dcgan.datasets --sources celeba --resolution 64
    from .config import parse_args
    cfg = parse_args(description='Baixa (se possível) e pré-processa as fontes em cache/*.npy')
    for source, _ in parse_sources(cfg.sources):
        print(prepare(source, cfg.data_dir, cfg.resolution))
