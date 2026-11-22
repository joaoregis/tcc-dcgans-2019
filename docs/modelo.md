# dcgan/model.py e dcgan/config.py

TF 2.15 em modo grafo (`tf.compat.v1`, `disable_eager_execution()` ao importar `dcgan.model`). Imagens 28×28×3 em [-1, 1].

## model.py
- **`model_inputs(w, h, c, z_dim)`** → placeholders `real_input_images`, `input_z`, `learning_rate` (este último não é usado no treino).
- **`discriminator(images, reuse, alpha, dropout_rate, training)`** → `(sigmoid(logit), logit)`; escopo `'discriminator'`.
  conv 64 (5×5, s2, glorot, com bias) → conv 128 + BN → conv 256 + BN (5×5, s2, sem bias); Leaky ReLU (`max(αx, x)`) + dropout após cada bloco; flatten 4·4·256 → dense 1.
  `training=False` = comportamento de 2019 (sem dropout, BN em inferência).
- **`generator(z, out_channels, is_train, alpha, dropout_rate)`** → imagem `tanh`; escopo `'generator'`, `reuse=(not is_train)`.
  dense 4·4·1024 → BN/LReLU/dropout → convT 512 (4×4, s1, valid → 7×7) → convT 256 (5×5, s2 → 14×14) → convT c (5×5, s2 → 28×28) → tanh.
  `is_train=False` reutiliza variáveis (amostragem) e precisa vir depois de o gerador de treino existir.
- **`model_loss(real, z, c, alpha, smooth_factor, dropout_rate, d_training)`** → `d_loss = BCE(real, 1−0.1) + BCE(fake, 0)`, `g_loss = BCE(fake, 1)`.
- **`model_opt(d_loss, g_loss, lr, beta1)`** → dois Adam, cada um com as variáveis do seu escopo, sob `control_dependencies(UPDATE_OPS)`.
  Consequência: **qualquer passo (D ou G) executa os updates da BN de D e G, que dependem do input real e do z** → o passo de G precisa receber `input_real` também (feito em `train.py`).

## config.py (`Config`, dataclass)
| Campo | Padrão | Nota |
|---|---|---|
| batch_size / z_dim | 64 / 100 | |
| learning_rate / beta1 | 0.00025 / 0.45 | Adam |
| epochs | 10 | |
| alpha / dropout_rate / smooth_factor | 0.2 / 0.5 / 0.1 | |
| print_every / show_every | 10 / 100 | passos |
| fix_2019_bugs | True | D com dropout e BN em modo treino; `False` reproduz 2019 |
| shuffle | True | embaralha por época (2019 não embaralhava) |
| data_dir | `DCGAN_DATA_DIR` ou `~/datasets/dcgan` | fora do repo/OneDrive |
| runs_dir / limit_images / max_steps / seed | `runs` / 0 / 0 / 0 | 0 = sem limite / aleatório |
