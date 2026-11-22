"""Treino: python -m dcgan.train [--epochs N] [--data-dir DIR] ... (ver --help)."""
import argparse
import dataclasses
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from . import data
from .config import Config
from .model import generator, model_inputs, model_loss, model_opt, tf1


def train(cfg, images):
    """Treina e devolve a pasta da execução (runs/<timestamp>/ com config.json, losses.csv, samples/, ckpt/)."""
    run = Path(cfg.runs_dir) / datetime.now().strftime('%Y-%m-%d_%H%M%S')
    (run / 'samples').mkdir(parents=True)
    (run / 'ckpt').mkdir()
    (run / 'config.json').write_text(json.dumps(dataclasses.asdict(cfg), indent=1))
    rng = np.random.RandomState(cfg.seed or None)
    sample_z = rng.uniform(-1, 1, size=(25, cfg.z_dim))  # fixo: amostras comparáveis entre passos
    channels = images.shape[-1]

    with tf1.Graph().as_default():
        if cfg.seed:
            tf1.set_random_seed(cfg.seed)
        input_real, input_z, _ = model_inputs(data.IMAGE_SIZE, data.IMAGE_SIZE, channels, cfg.z_dim)
        d_loss, g_loss = model_loss(input_real, input_z, channels, cfg.alpha, cfg.smooth_factor,
                                    cfg.dropout_rate, d_training=cfg.fix_2019_bugs)
        d_opt, g_opt = model_opt(d_loss, g_loss, cfg.learning_rate, cfg.beta1)
        sample_op = generator(input_z, channels, is_train=False, alpha=cfg.alpha, dropout_rate=cfg.dropout_rate)
        saver = tf1.train.Saver()
        steps, t0 = 0, time.time()
        with tf1.Session() as sess, open(run / 'losses.csv', 'w') as log:
            sess.run(tf1.global_variables_initializer())
            log.write('epoch,step,d_loss,g_loss\n')
            for epoch in range(1, cfg.epochs + 1):
                for batch in data.batches(images, cfg.batch_size, cfg.shuffle, rng):
                    steps += 1
                    batch_z = rng.uniform(-1, 1, size=(cfg.batch_size, cfg.z_dim))
                    sess.run(d_opt, {input_real: batch, input_z: batch_z})
                    sess.run(g_opt, {input_real: batch, input_z: batch_z})  # UPDATE_OPS da BN do D também dependem do input real
                    if steps % cfg.print_every == 0:
                        d, g = sess.run([d_loss, g_loss], {input_real: batch, input_z: batch_z})
                        log.write(f'{epoch},{steps},{d:.5f},{g:.5f}\n')
                        log.flush()
                        print(f'Epoch {epoch}/{cfg.epochs} Step {steps} D {d:.4f} G {g:.4f} ({time.time() - t0:.0f}s)', flush=True)
                    if steps % cfg.show_every == 0:
                        grid = data.images_square_grid(sess.run(sample_op, {input_z: sample_z}))
                        grid.save(run / 'samples' / f'e{epoch:02d}_s{steps:06d}.png')
                    if cfg.max_steps and steps >= cfg.max_steps:
                        break
                saver.save(sess, str(run / 'ckpt' / 'model'), global_step=epoch)
                if cfg.max_steps and steps >= cfg.max_steps:
                    break
            data.images_square_grid(sess.run(sample_op, {input_z: sample_z})).save(run / 'samples' / 'final.png')
        (run / 'time.txt').write_text(f'{steps} passos em {time.time() - t0:.0f}s\n')
    return run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for f in dataclasses.fields(Config):
        flag = '--' + f.name.replace('_', '-')
        if f.type is bool:
            p.add_argument(flag, default=f.default, action=argparse.BooleanOptionalAction)
        else:
            p.add_argument(flag, default=f.default, type=f.type)
    cfg = Config(**vars(p.parse_args()))
    images = data.load_celeba(cfg.data_dir, cfg.limit_images)
    print(f'{len(images)} imagens, dados em {cfg.data_dir}')
    print('Saída em', train(cfg, images))


if __name__ == '__main__':
    main()
