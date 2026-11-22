"""FID (Fréchet Inception Distance): distância entre as estatísticas Inception de imagens reais e geradas. Menor = melhor.

Usa os pesos Inception padrão do FID (baixados uma vez pelo torch-fidelity, ~100 MB, para a pasta de cache do torch).
"""
import numpy as np
import torch


@torch.no_grad()
def fid(real, generator, z_dim, n=5000, batch=250, seed=0):
    """`real`: array uint8 (N,H,W,3) ou FaceData; `generator`: nn.Module (use a cópia EMA). Compara n reais com n falsas."""
    from torchmetrics.image.fid import FrechetInceptionDistance
    dev = next(generator.parameters()).device
    metric = FrechetInceptionDistance(feature=2048, normalize=False).to(dev)
    rng = np.random.default_rng(seed)
    n = min(n, len(real))
    idx = np.sort(rng.choice(len(real), n, replace=False))
    imgs = real.take(idx) if hasattr(real, 'take') and not isinstance(real, np.ndarray) else real[idx]
    gen = torch.Generator(device=dev).manual_seed(seed)
    generator.eval()
    for k in range(0, n, batch):
        r = torch.from_numpy(np.ascontiguousarray(imgs[k:k + batch])).to(dev).permute(0, 3, 1, 2)
        metric.update(r, real=True)
        z = torch.randn(len(r), z_dim, 1, 1, device=dev, generator=gen)
        f = ((generator(z).float().clamp(-1, 1) * 0.5 + 0.5) * 255).round().to(torch.uint8)
        metric.update(f, real=False)
    return float(metric.compute())
