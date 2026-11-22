# Quirks do código de 2019 e pendências

Código original: tag `v2019`. `--no-fix-2019-bugs` reproduz o comportamento do modelo de 2019.

## Corrigidos (em `master`, 2026-11-22)
- Dropout do discriminador nunca atuava (`training` omitido) → agora ativo (`fix_2019_bugs`).
- BN do discriminador sempre em inferência → agora em modo treino (`fix_2019_bugs`). Obriga o passo de G a receber `input_real`.
- `keep_prob` era usado como `rate` → renomeado `dropout_rate` (mesmo valor 0.5).
- `os.mkdir('output')` quebrava ao reexecutar → cada treino cria `runs/<timestamp>/`.
- Checkpoint por passo na época 0 → um por época.
- Grafo crescia a cada amostra → `sample_op` único; amostras com `z` fixo.
- Lotes sem embaralhar → `shuffle` (padrão ligado).
- Código morto (MNIST, `ckpt`, `coord`, `threads`, listas/`cmap`/`saver` duplicado) removido.
- Testes: `test_model_loss` checava `d_loss` duas vezes → reescritos em pytest.

## Pendências / riscos
- Os efeitos das correções na qualidade das amostras **não foram medidos** (treino real ainda não executado). Comparar `--fix-2019-bugs` × `--no-fix-2019-bugs`.
- Resolução 28×28 é baixa para rostos.
- O TF 2.15 pode diferir numericamente do TF 1.x (inicializadores, epsilon da BN); não há prova de equivalência.
- `model_inputs` ainda cria o placeholder `learning_rate`, que não é usado.
