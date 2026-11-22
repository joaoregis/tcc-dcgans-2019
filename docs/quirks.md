# Quirks e bugs conhecidos

Encontrados por leitura do código (não executado). Nada aqui foi corrigido: o objetivo do repo é preservar o experimento de 2019. Ao corrigir algo, mova a linha para `log.md`.

## Modelo (notebook)
- **Dropout do discriminador nunca atua**: `tf.layers.dropout(x, keep_prob)` é chamado sem `training=True` (padrão `False`). Além disso, o 2º argumento é `rate`, não `keep_prob`: `0.5` coincide, mas outro valor mudaria o significado. No gerador `training=is_train` é passado.
- **BatchNorm do discriminador sempre em modo inferência**: `tf.layers.batch_normalization(conv)` sem `training`. Usa média/variância móveis iniciais (0/1), na prática sem normalizar de verdade.
- **Gerador em `show_generator_output`** é recriado a cada chamada (`generator(input_z, ..., False)`), adicionando ops ao grafo a cada amostra.
- Comentário "Saída" do gerador com encoding quebrado (`Sa�da`) no `.ipynb`.

## Treino
- `os.mkdir('output')` falha se a pasta já existe: **reexecutar a célula de treino dá erro** sem apagar `output/`.
- `if epoch_i % 10 == 0:` está **dentro do loop de lotes**: salva checkpoint a cada passo da época 0 (e 10, 20…), não 1× por época.
- `saver.save('/tmp/model.ckpt')` seguido de `saver.restore` logo na largada é efeito nulo; `ckpt`, `coord`, `threads`, `samples`, `losses`, `cmap` e `saver` duplicado são código morto.
- Lotes não embaralhados (`Dataset.get_batches` fatia em ordem de `glob`).
- Imagens reduzidas a 28×28: baixa resolução para "rostos".

## helper.py
- `download_extract` só define variáveis para `'celeba'`; `'mnist'` levantaria `NameError`. `_ungzip`/`_read32` (MNIST) são código morto.
- Recorte 108×108 fixo "HACK" assume CelebA 178×218.
- URL de download da Udacity: disponibilidade não verificada.

## Testes
- `test_model_loss` afirma `d_loss` duas vezes (o 2º deveria ser `g_loss`).
- `_check_input` sempre usa o nome `'Real Input'` na mensagem de forma, mesmo para outros inputs.
- `test_model_opt` só confere que `tf.trainable_variables` foi chamado.
