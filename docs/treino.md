# dcgan/train.py

`python -m dcgan.train [opções]` — cada campo de `Config` vira uma flag (`--epochs`, `--batch-size`, `--limit-images`, `--max-steps`, `--no-fix-2019-bugs`, `--no-shuffle`, `--seed`…). `--help` lista todas.

## `train(cfg, images)` → pasta `runs/<AAAA-MM-DD_HHMMSS>/`
- `config.json` — configuração usada; `losses.csv` — `epoch,step,d_loss,g_loss` a cada `print_every`;
  `samples/e<época>_s<passo>.png` a cada `show_every` (grade 5×5, **z fixo**) e `samples/final.png`;
  `ckpt/model-<época>` — checkpoint por época; `time.txt` — passos e segundos.
- Passo: 1× `d_opt`, 1× `g_opt` (ambos recebem `input_real` e `input_z`, ver `modelo.md`).
- Amostragem usa um `sample_op` criado uma única vez (em 2019 o grafo crescia a cada amostra).

## Desempenho (esta máquina: Windows, CPU; GPU RTX 4060 não usada)
Medido com dados sintéticos: ~0,27 s/passo (batch 64). CelebA completo: 3165 passos/época → ~14 min/época, ~2,4 h para 10 épocas. GPU exigiria WSL2 ou TF ≤ 2.10 com CUDA (não testado).

## Estado
Smoke test automático passa. **Treino real no CelebA ainda não executado.**
