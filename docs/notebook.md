# GANS_FACE.ipynb

DCGAN em TensorFlow 1.x que gera rostos 28×28 RGB a partir do CelebA. Células de markdown dividem em seções; código na ordem abaixo.

## Fluxo das células
1. **Dados** — `data_dir = '/input'`; `helper.download_extract('celeba', data_dir)`.
2. **Imports** — `os`, `glob`, `numpy`, `matplotlib.pyplot` (`%matplotlib inline`); depois `tensorflow as tf` (sem checagem de versão/GPU, apesar de importar `LooseVersion` e `warnings`).
3. **`model_inputs(w, h, c, z_dim)`** → placeholders `real_input_images [None,w,h,c]`, `input_z [None,z_dim]`, `learning_rate []`. Testado por `tests.test_model_inputs`.
4. **`discriminator(images, reuse=False, alpha=0.2, keep_prob=0.5)`** → `(sigmoid(logits), logits)`. Escopo `'discriminator'`.
   - 28×28×n → conv 64 (5×5, s2, same, xavier, com bias, **sem BN**) → 14×14×64
   - → conv 128 + BN → 7×7×128 → conv 256 + BN → 4×4×256 (todas 5×5 s2 same, sem bias)
   - Leaky ReLU manual `max(alpha*x, x)` + `tf.layers.dropout` após cada bloco; flatten (4·4·256) → dense 1.
5. **`generator(z, out_channel_dim, is_train=True, alpha=0.2, keep_prob=0.5)`** → imagem `tanh` em [-1,1]. Escopo `'generator'`, `reuse=(not is_train)`.
   - dense 4·4·1024 (sem bias) → reshape 4×4×1024 → BN → LReLU → dropout
   - convT 512 (4×4, s1, valid) → 7×7×512 → convT 256 (5×5, s2, same) → 14×14×256 (ambos com BN+LReLU+dropout)
   - convT `out_channel_dim` (5×5, s2, same) → 28×28×n → `tanh`.
6. **`model_loss(input_real, input_z, out_channel_dim, alpha=0.2, smooth_factor=0.1)`** → `(d_loss, g_loss)`.
   - `d_loss = BCE(real, 0.9) + BCE(fake, 0)` (label smoothing unilateral: rótulo real = `1 - smooth_factor`).
   - `g_loss = BCE(fake, 1)` (não saturante).
7. **`model_opt(d_loss, g_loss, lr, beta1)`** → dois `AdamOptimizer`, cada um sobre as variáveis do seu escopo (filtro por prefixo do nome), dentro de `control_dependencies(UPDATE_OPS)` (atualiza a BN).
8. **`show_generator_output(sess, n_images, input_z, out_channel_dim, image_mode)`** → gera `n_images` com `z ~ U(-1,1)` e devolve a grade PIL via `helper.images_square_grid`.
9. **`train(epoch_count, batch_size, z_dim, learning_rate, beta1, get_batches, data_shape, data_image_mode, print_every=10, show_every=100)`**
   - Constrói inputs/loss/opt; `Saver`; `sess.run(global_variables_initializer)`.
   - Cria `output/` e `output/<época>/` (`os.mkdir`).
   - Por lote: `batch_images *= 2.0` (traz [-0.5,0.5] → [-1,1], casando com `tanh`); `z ~ U(-1,1)`; 1 passo D e 1 passo G.
   - Cada 10 passos imprime perdas D/G; cada 100 passos salva grade 5×5 em `output/<época>/<n>.png` (n = 100, 200, … global).
   - Salva checkpoint em `./model/<época>` quando `epoch_i % 10 == 0` (ver quirks).
10. **Marcadores de tempo** — escrevem `HoraInicio.txt` antes e `HoraFinal.txt` (com nº de épocas) depois do treino.
11. **Execução** — `batch_size=64, z_dim=100, learning_rate=0.00025, beta1=0.45, epochs=10`; `helper.Dataset('celeba', glob(data_dir/img_align_celeba/*.jpg))`; `train(...)` dentro de `tf.Graph().as_default()`.

## Saídas geradas (todas ignoradas ou locais)
`output/<época>/<passo>.png`, `model/` (checkpoints), `/tmp/model.ckpt`, `HoraInicio.txt`, `HoraFinal.txt`.

## Dependências entre arquivos
Usa `helper` (download, `Dataset`, `images_square_grid`) e `problem_unittests as tests` (uma chamada de teste logo após definir cada função do modelo).

## Pendente
A seção "Resultados" do notebook está vazia (sem números de treino registrados).
