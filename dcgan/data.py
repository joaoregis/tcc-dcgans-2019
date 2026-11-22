"""CelebA: download, pré-processamento (recorte 108x108 + resize 28x28) e lotes."""
import hashlib
import math
import os
import zipfile
from concurrent.futures import ThreadPoolExecutor
from glob import glob
from urllib.request import urlretrieve

import numpy as np
from PIL import Image
from tqdm import tqdm

IMAGE_SIZE = 28
CELEBA_URL = 'https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/celeba.zip'
CELEBA_MD5 = '00d2c5bc6d35e252742224ab0c1e8fcb'


class _Progress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def download_extract(data_dir):
    """Baixa e extrai o CelebA em data_dir/img_align_celeba. Devolve essa pasta."""
    extract_path = os.path.join(data_dir, 'img_align_celeba')
    if os.path.exists(extract_path):
        return extract_path
    os.makedirs(data_dir, exist_ok=True)
    save_path = os.path.join(data_dir, 'celeba.zip')
    if not os.path.exists(save_path):
        with _Progress(unit='B', unit_scale=True, miniters=1, desc='Baixando celeba') as pbar:
            urlretrieve(CELEBA_URL, save_path, pbar.hook)
    md5 = hashlib.md5()
    with open(save_path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            md5.update(chunk)
    assert md5.hexdigest() == CELEBA_MD5, f'{save_path} corrompido; apague e tente de novo.'
    with zipfile.ZipFile(save_path) as zf:
        zf.extractall(data_dir)
    os.remove(save_path)
    return extract_path


def get_image(path, size=IMAGE_SIZE):
    """Recorte central 108x108 (só a face do CelebA 178x218) + resize bilinear. Devolve uint8 (size,size,3)."""
    image = Image.open(path)
    if image.size != (size, size):
        j, i = (image.size[0] - 108) // 2, (image.size[1] - 108) // 2
        image = image.crop([j, i, j + 108, i + 108]).resize([size, size], Image.BILINEAR)
    return np.array(image.convert('RGB'))


def load_celeba(data_dir, limit=0):
    """Array uint8 (N,28,28,3). Cria o cache data_dir/celeba_28.npy na primeira vez (alguns minutos)."""
    cache = os.path.join(data_dir, f'celeba_{IMAGE_SIZE}.npy')
    if not os.path.exists(cache):
        folder = download_extract(data_dir)
        files = sorted(glob(os.path.join(folder, '*.jpg')))
        with ThreadPoolExecutor() as ex:
            arr = np.stack(list(tqdm(ex.map(get_image, files), total=len(files), desc='Pré-processando')))
        np.save(cache, arr)
    arr = np.load(cache, mmap_mode='r')
    return np.asarray(arr[:limit]) if limit else np.asarray(arr)


def batches(images, batch_size, shuffle=True, rng=None):
    """Gera lotes float32 em [-1, 1] (mesma escala do tanh do gerador). Descarta o lote final incompleto."""
    idx = (rng or np.random).permutation(len(images)) if shuffle else np.arange(len(images))
    for k in range(0, len(idx) - batch_size + 1, batch_size):
        yield images[np.sort(idx[k:k + batch_size])].astype(np.float32) / 127.5 - 1.0


def images_square_grid(images):
    """Grade floor(sqrt(n))² como imagem PIL RGB; normaliza min/max do lote para 0-255."""
    n = math.floor(np.sqrt(images.shape[0]))
    images = ((images - images.min()) * 255 / (images.max() - images.min() + 1e-12)).astype(np.uint8)
    h, w = images.shape[1:3]
    grid = Image.new('RGB', (w * n, h * n))
    for k in range(n * n):
        grid.paste(Image.fromarray(images[k], 'RGB'), ((k % n) * w, (k // n) * h))
    return grid


if __name__ == '__main__':
    from .config import default_data_dir
    print(load_celeba(default_data_dir()).shape)
