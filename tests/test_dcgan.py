"""Testes rápidos (sem baixar nada): python -m pytest -q"""
import numpy as np
import pytest
from PIL import Image

from dcgan import data
from dcgan.config import Config
from dcgan.model import discriminator, generator, model_inputs, model_loss, model_opt, tf1
from dcgan.train import train


@pytest.fixture
def graph():
    with tf1.Graph().as_default() as g:
        yield g


def test_model_inputs(graph):
    real, z, lr = model_inputs(28, 28, 3, 100)
    assert real.shape.as_list() == [None, 28, 28, 3]
    assert z.shape.as_list() == [None, 100]
    assert lr.shape.as_list() == []
    assert all(t.op.type == 'Placeholder' for t in (real, z, lr))


def test_discriminator_shapes_and_reuse(graph):
    img = tf1.placeholder('float32', [None, 28, 28, 3])
    out, logits = discriminator(img)
    n = len(tf1.global_variables())
    discriminator(img, reuse=True)
    assert out.shape.as_list() == logits.shape.as_list() == [None, 1]
    assert len(tf1.global_variables()) == n  # reuse não cria variáveis


def test_generator_shape_and_reuse(graph):
    z = tf1.placeholder('float32', [None, 100])
    out = generator(z, 3)
    n = len(tf1.global_variables())
    generator(z, 3, is_train=False)
    assert out.shape.as_list() == [None, 28, 28, 3]
    assert len(tf1.global_variables()) == n


def test_loss_and_opt(graph):
    real, z, _ = model_inputs(28, 28, 3, 100)
    d_loss, g_loss = model_loss(real, z, 3)
    assert d_loss.shape.as_list() == g_loss.shape.as_list() == []
    model_opt(d_loss, g_loss, 0.001, 0.5)
    names = {v.name.split('/')[0] for v in tf1.trainable_variables()}
    assert names == {'discriminator', 'generator'}


def test_get_image_crops_celeba(tmp_path):
    p = tmp_path / 'a.jpg'
    Image.fromarray(np.zeros((218, 178, 3), np.uint8)).save(p)
    assert data.get_image(str(p)).shape == (28, 28, 3)


def test_batches_range_and_drop_last():
    imgs = np.random.randint(0, 256, (10, 28, 28, 3), dtype=np.uint8)
    out = list(data.batches(imgs, 4, shuffle=True))
    assert len(out) == 2 and out[0].shape == (4, 28, 28, 3)
    assert -1.0 <= min(b.min() for b in out) and max(b.max() for b in out) <= 1.0


def test_grid_size():
    grid = data.images_square_grid(np.random.rand(25, 28, 28, 3))
    assert grid.size == (140, 140)


def test_train_smoke(tmp_path):
    imgs = np.random.randint(0, 256, (16, 28, 28, 3), dtype=np.uint8)
    cfg = Config(batch_size=8, epochs=1, print_every=1, show_every=1, runs_dir=str(tmp_path), data_dir=str(tmp_path))
    run = train(cfg, imgs)
    assert (run / 'samples' / 'final.png').exists()
    assert (run / 'losses.csv').read_text().count('\n') == 3  # cabeçalho + 2 passos
    assert list((run / 'ckpt').glob('model-1*'))
