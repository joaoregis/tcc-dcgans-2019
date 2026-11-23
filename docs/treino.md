# dcgan/train.py e dcgan/config.py

`python -m dcgan.train [flags]` — cada campo de `Config` é uma flag (`--help` lista tudo). Requer GPU CUDA.

## Parâmetros principais (`Config`)
| Grupo | Campo | Padrão | Nota |
|---|---|---|---|
| Dados | `sources` / `resolution` / `flip` | `celeba` / 64 / sim | |
| Modelo | `z_dim` / `g_width` / `d_width` | 100 / 64 / 64 | largura dobra por camada até 512 |
| Otimização | `batch_size` | 128 | satura a GPU (ver `desempenho.md`) |
| | `lr_g` / `lr_d` / `beta1` / `beta2` | 2e-4 / 2e-4 / 0,5 / 0,999 | valores do artigo; ajuste em `tuning.md` |
| | `label_smooth` / `ema` | 0,1 / 0,999 | |
| | `epochs` / `max_minutes` | 20 / 0 | `max_minutes` = limite de tempo (0 = sem) |
| Desempenho | `amp` | sim | bfloat16 |
| Saídas | `runs_dir`, `log_every`, `sample_every`, `ckpt_every` | `runs`, 100, 1000, 1 | |
| | `resume`, `seed` | vazio, 0 | |

## Fluxo de `train(cfg)`
1. Liga `cudnn.benchmark` e TF32; carrega `FaceData`; cria `Loader`, G, D (em `channels_last`), EMA e dois Adam (um por rede).
2. Opcional: `--resume runs/.../ckpt/epoch_003.pt` restaura redes, otimizadores, EMA, passo e época.
3. Cria `runs/<AAAA-MM-DD_HHMMSS>/` e grava `config.json`.
4. Para cada época, para cada lote: `train_step` → atualiza EMA → a cada `log_every` passos imprime perdas e **img/s** (e grava em `losses.csv`);
   a cada `sample_every` passos salva uma grade 8×8 gerada pela EMA a partir de um `z` **fixo** (assim as imagens são comparáveis ao longo do tempo).
5. Ao fim de cada época (ou ao estourar `max_minutes`) salva `ckpt/epoch_NNN.pt`. No fim, `samples/final.png` e `time.txt`.

## Como ler os resultados
- `samples/*.png`: evolução visual. `losses.csv`: `d_loss` e `g_loss` — em GANs **não** é para ambas caírem; sinais de problema são
  `d_loss → 0` (D venceu, G para de aprender) ou `g_loss` explodindo. Para qualidade objetiva use o FID (`tuning.md`).

## Estado
Treinos reais concluídos no CelebA: 64 px (40 épocas, FID 26,5) e 128 px (20 épocas, FID 31,0). Velocidades e configurações em `desempenho.md` e `tuning.md`.
