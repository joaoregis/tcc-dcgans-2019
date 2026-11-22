# tests/test_dcgan.py

`.venv\Scripts\python.exe -m pytest -q` (~12 s, não baixa nada). Config em `pytest.ini` (silencia avisos de depreciação do `tf.layers`).

| Teste | Verifica |
|---|---|
| `test_model_inputs` | 3 placeholders com as formas certas |
| `test_discriminator_shapes_and_reuse` | saída `[None,1]`; `reuse=True` não cria variáveis |
| `test_generator_shape_and_reuse` | saída `[None,28,28,c]`; `is_train=False` não cria variáveis |
| `test_loss_and_opt` | losses escalares; escopos `discriminator`/`generator` |
| `test_get_image_crops_celeba` | JPEG 178×218 → 28×28×3 |
| `test_batches_range_and_drop_last` | lotes em [-1,1], último incompleto descartado |
| `test_grid_size` | 25 imagens → grade 140×140 |
| `test_train_smoke` | 1 época com dados sintéticos gera `final.png`, `losses.csv` e checkpoint |

Não cobre: download real do CelebA, qualidade das amostras, equivalência numérica com o TF 1.x de 2019.
