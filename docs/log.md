# Log de mudanças e decisões (mais recente primeiro)

- 2026-11-22 — **Refatoração profunda para PyTorch + CUDA.** Pacote `dcgan/` reescrito (`config`, `datasets`, `models`, `train`, `metrics`, `tune`, `bench`); remove TF/`compat.v1`. Decisões do usuário: PyTorch, resolução 64 depois 128, datasets (CelebA, FFHQ, CelebA-HQ, UTKFace/LFW) em pasta dedicada no HDD (`D:\datasets\faces`, via `local.env`). Medido: 6 300 img/s a 64 px (bf16 + channels_last, batch 128), 1 300 img/s a 128 px; `torch.compile` testado e descartado (+5% só em NCHW; quebra com channels_last). Adicionados EMA, cache `.npy`, mistura de fontes, FID e busca com orçamento de tempo. Docs reescritas (`como-funciona`, `migracao`, `desempenho`, `tuning`...). Nenhum dataset baixado ainda; tuning pendente.
- 2026-11-22 — Refatoração intermediária: pacote `dcgan/` em TF 2.15 compat.v1 (CPU), pytest, `runs/`. Substituída pela versão PyTorch acima.
- 2026-11-22 — Criada `docs/` a partir da leitura do código original.
- 2026-11-22 — Repo criado; `CLAUDE.md`, `README.md`, `requirements.txt`, `GRAPHIFY.md`; `graphify-out/` ignorado; tag `v2019` marca o estado original.
