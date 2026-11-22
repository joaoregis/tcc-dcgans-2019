"""Testes rápidos (sem baixar nada): .venv\Scripts\python.exe -m pytest -q"""
import numpy as np
import pytest
import torch
from PIL import Image

from dcgan import datasets
from dcgan.config import Config
from dcgan.datasets import FaceData, Loader, parse_sources
from dcgan.models import EMA, Discriminator, Generator, widths

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='precisa de GPU CUDA')


def test_widths():
    assert widths(64, 64) == [512, 256, 128, 64]
    assert widths(128, 64) == [512, 512, 256, 128, 64]


@pytest.mark.parametrize('res', [32, 64, 128, 256])
def test_shapes(res):
    g, d = Generator(res, 100, 16), Discriminator(res, 16)
    x = g(torch.randn(2, 100, 1, 1))
    assert x.shape == (2, 3, res, res) and x.abs().max() <= 1
    assert d(x).shape == (2,)


def test_ema_moves_toward_model():
    g = Generator(32, 10, 8)
    ema = EMA(g, 0.9)
    with torch.no_grad():
        for p in g.parameters():
            p.add_(1.0)
    ema.update(g, step=1000)
    diffs = [(e - p).abs().mean() for e, p in zip(ema.model.parameters(), g.parameters())]
    assert all(0 < d < 1 for d in diffs)


def test_parse_sources():
    assert parse_sources('celeba:100, ffhq') == [('celeba', 100), ('ffhq', None)]
    with pytest.raises(AssertionError):
        parse_sources('inexistente')


def test_load_image_celeba_crop(tmp_path):
    p = tmp_path / 'a.jpg'
    Image.fromarray(np.full((218, 178, 3), 7, np.uint8)).save(p)
    assert datasets.load_image(p, 'celeba', 64).shape == (64, 64, 3)


def test_prepare_and_mix(tmp_path):
    raw = tmp_path / 'raw' / 'lfw'
    raw.mkdir(parents=True)
    for i in range(5):
        Image.fromarray(np.random.randint(0, 255, (50, 50, 3), np.uint8)).save(raw / f'{i}.png')
    out = datasets.prepare('lfw', tmp_path, 16, workers=1)
    assert np.load(out).shape == (5, 16, 16, 3)
    data = FaceData.load('lfw:3,lfw', tmp_path, 16)
    assert len(data) == 8
    assert data.take(np.array([0, 2, 3, 7])).shape == (4, 16, 16, 3)


@cuda
def test_loader_range_and_drop_last():
    data = FaceData([np.random.randint(0, 256, (10, 16, 16, 3), dtype=np.uint8)])
    batches = list(Loader(data, 4, torch.device('cuda'), flip=True))
    assert len(batches) == 2 and batches[0].shape == (4, 3, 16, 16)
    assert all(-1 <= b.min() and b.max() <= 1 for b in batches)


@cuda
def test_train_smoke(tmp_path):
    from dcgan.train import train
    data = FaceData([np.random.randint(0, 256, (32, 32, 32, 3), dtype=np.uint8)])
    cfg = Config(resolution=32, g_width=8, d_width=8, batch_size=8, epochs=1, log_every=1, sample_every=2,
                 runs_dir=str(tmp_path), data_dir=str(tmp_path))
    run = train(cfg, data)
    assert (run / 'samples' / 'final.png').exists() and (run / 'ckpt' / 'epoch_001.pt').exists()
    assert (run / 'losses.csv').read_text().count('\n') == 5  # cabeçalho + 4 passos


@cuda
def test_generate_from_checkpoint_and_export(tmp_path):
    from dcgan.generate import export, generate, load_generator
    from dcgan.train import train
    data = FaceData([np.random.randint(0, 256, (16, 32, 32, 3), dtype=np.uint8)])
    cfg = Config(resolution=32, g_width=8, d_width=8, batch_size=8, epochs=1, runs_dir=str(tmp_path), data_dir=str(tmp_path))
    ck = next((train(cfg, data) / 'ckpt').glob('*.pt'))
    export(ck, tmp_path / 'g.pt')
    for path in (ck, tmp_path / 'g.pt'):
        g, c = load_generator(path)
        a, b = generate(g, c, 4, seed=1), generate(g, c, 4, seed=1)
        assert a.shape == (4, 3, 32, 32) and torch.equal(a, b) and 0 <= a.min() and a.max() <= 1
    assert (tmp_path / 'g.pt').stat().st_size < ck.stat().st_size
