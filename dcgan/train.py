"""Treino: python -m dcgan.train [--resolution 64] [--sources celeba,ffhq] [--epochs N] ... (ver --help)."""
import dataclasses
import json
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from .config import Config, parse_args
from .datasets import FaceData, Loader
from .models import EMA, Discriminator, Generator


def make_run_dir(cfg):
    run = Path(cfg.runs_dir) / datetime.now().strftime('%Y-%m-%d_%H%M%S')
    (run / 'samples').mkdir(parents=True)
    (run / 'ckpt').mkdir()
    (run / 'config.json').write_text(json.dumps(dataclasses.asdict(cfg), indent=1))
    return run


def save_samples(g, z, path):
    with torch.no_grad():
        save_image(g(z).float().clamp(-1, 1) * 0.5 + 0.5, path, nrow=int(len(z) ** 0.5), padding=1)


def train_step(real, G, D, opt_g, opt_d, cfg, amp):
    """1 passo de D (real vs. falso) e 1 de G (enganar D). Perda não saturante com suavização unilateral dos rótulos."""
    bce = F.binary_cross_entropy_with_logits
    z = torch.randn(real.size(0), cfg.z_dim, 1, 1, device=real.device)
    with amp:
        fake = G(z)
        d_real, d_fake = D(real), D(fake.detach())
        d_loss = bce(d_real, torch.full_like(d_real, 1 - cfg.label_smooth)) + bce(d_fake, torch.zeros_like(d_fake))
    opt_d.zero_grad(set_to_none=True)
    d_loss.backward()
    opt_d.step()
    with amp:
        out = D(fake)  # D já atualizado
        g_loss = bce(out, torch.ones_like(out))
    opt_g.zero_grad(set_to_none=True)
    g_loss.backward()
    opt_g.step()
    return d_loss.detach(), g_loss.detach()


def train(cfg: Config, data=None):
    """Treina e devolve a pasta da execução. `data` (FaceData) pode ser injetado em testes."""
    assert torch.cuda.is_available(), 'GPU CUDA não encontrada'
    device = torch.device('cuda')
    if cfg.seed:
        torch.manual_seed(cfg.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True

    data = data or FaceData.load(cfg.sources, cfg.data_dir, cfg.resolution)
    loader = Loader(data, cfg.batch_size, device, cfg.flip, cfg.seed)
    G = Generator(cfg.resolution, cfg.z_dim, cfg.g_width).to(device, memory_format=torch.channels_last)
    D = Discriminator(cfg.resolution, cfg.d_width).to(device, memory_format=torch.channels_last)
    ema = EMA(G, cfg.ema) if cfg.ema else None
    opt_g = torch.optim.Adam(G.parameters(), cfg.lr_g, betas=(cfg.beta1, cfg.beta2))
    opt_d = torch.optim.Adam(D.parameters(), cfg.lr_d, betas=(cfg.beta1, cfg.beta2))
    step, epoch0 = 0, 0
    if cfg.resume:
        ck = torch.load(cfg.resume, map_location=device)
        G.load_state_dict(ck['G']); D.load_state_dict(ck['D'])
        opt_g.load_state_dict(ck['opt_g']); opt_d.load_state_dict(ck['opt_d'])
        if ema: ema.model.load_state_dict(ck['ema'])
        step, epoch0 = ck['step'], ck['epoch']

    run = make_run_dir(cfg)
    sample_z = torch.randn(64, cfg.z_dim, 1, 1, device=device)
    amp = torch.autocast('cuda', torch.bfloat16, enabled=cfg.amp)
    print(f'{len(data)} imagens | {cfg.resolution}px | G {sum(p.numel() for p in G.parameters()) / 1e6:.1f}M '
          f'D {sum(p.numel() for p in D.parameters()) / 1e6:.1f}M parâmetros | saída: {run}', flush=True)

    t0, t_log, deadline = time.time(), time.time(), cfg.max_minutes * 60
    with open(run / 'losses.csv', 'w') as log:
        log.write('epoch,step,d_loss,g_loss,img_per_s\n')
        for epoch in range(epoch0 + 1, cfg.epochs + 1):
            for real in loader:
                d_loss, g_loss = train_step(real, G, D, opt_g, opt_d, cfg, amp)
                step += 1
                if ema:
                    ema.update(G, step)
                if step % cfg.log_every == 0:
                    ips = cfg.log_every * cfg.batch_size / (time.time() - t_log)  # (sincroniza a GPU via .item())
                    d, g = d_loss.item(), g_loss.item()
                    t_log = time.time()
                    log.write(f'{epoch},{step},{d:.4f},{g:.4f},{ips:.0f}\n'); log.flush()
                    print(f'ep {epoch}/{cfg.epochs} passo {step} D {d:.3f} G {g:.3f} {ips:.0f} img/s ({(time.time() - t0) / 60:.1f} min)', flush=True)
                if step % cfg.sample_every == 0:
                    save_samples(ema.model if ema else G.eval(), sample_z, run / 'samples' / f'{step:07d}.png')
                    G.train()
                if deadline and time.time() - t0 > deadline:
                    break
            if epoch % cfg.ckpt_every == 0 or (deadline and time.time() - t0 > deadline):
                torch.save({'G': G.state_dict(), 'D': D.state_dict(), 'ema': ema.model.state_dict() if ema else None,
                            'opt_g': opt_g.state_dict(), 'opt_d': opt_d.state_dict(), 'step': step, 'epoch': epoch,
                            'config': dataclasses.asdict(cfg)}, run / 'ckpt' / f'epoch_{epoch:03d}.pt')
            if deadline and time.time() - t0 > deadline:
                print('Limite de tempo atingido.')
                break
    save_samples(ema.model if ema else G.eval(), sample_z, run / 'samples' / 'final.png')
    (run / 'time.txt').write_text(f'{step} passos em {time.time() - t0:.0f}s\n')
    return run


if __name__ == '__main__':
    print('Saída em', train(parse_args(description=__doc__)))
