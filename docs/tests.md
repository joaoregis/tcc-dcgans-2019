# problem_unittests.py

Testes de **contrato** (formas e uso de escopos), não de qualidade do modelo. Chamados no notebook logo após cada função (`tests.test_*(função[, tf])`) e imprimem "Tests Passed". Exigem TF 1.x.

## Infra
- **`test_safe(func)`** — decorador: roda o teste dentro de um `tf.Graph()` novo (não polui o grafo do notebook) e imprime "Tests Passed".
- **`_assert_tensor_shape(tensor, shape, nome)`** — checa rank e dimensões (`None` = livre).
- **`_check_input(tensor, shape, nome, tf_name=None)`** — exige `op.type == 'Placeholder'` + forma (+ nome opcional).
- **`TmpMock(module, attr)`** — context manager que troca `module.attr` por `MagicMock` e restaura depois (usado para espionar `tf.variable_scope` e `tf.trainable_variables`).

## Testes
| Teste | Verifica |
|---|---|
| `test_model_inputs` | 3 placeholders: `[None,28,28,3]`, `[None,100]`, `[]` |
| `test_discriminator` | saída/logits `[None,1]`; `tf.variable_scope('discriminator', reuse=False)` e depois `reuse=True` quando chamado com `True` posicional |
| `test_generator` | saída `[None,28,28,out_ch]` com `is_train` True/False; escopo `'generator'` com `reuse=False` / `reuse=True` |
| `test_model_loss` | `d_loss` e `g_loss` escalares |
| `test_model_opt` | só que `tf.trainable_variables` foi chamado |

## Limites conhecidos
Ver [quirks.md](quirks.md): o teste de loss repete `d_loss` e o de opt é fraco. Nenhum teste cobre `helper.py`, `train` nem `show_generator_output`.
