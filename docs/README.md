# docs/ — documentação viva do projeto

Consulte aqui **antes** de ler o código.

| Doc | Cobre |
|---|---|
| [modelo.md](modelo.md) | `dcgan/model.py` e `dcgan/config.py`: arquitetura, loss, otimizadores, hiperparâmetros |
| [dados.md](dados.md) | `dcgan/data.py`: download, cache 28×28, lotes, grade de imagens |
| [treino.md](treino.md) | `dcgan/train.py`: loop, saídas em `runs/`, CLI, desempenho na CPU |
| [testes.md](testes.md) | `tests/test_dcgan.py` |
| [quirks.md](quirks.md) | Bugs do código de 2019 (corrigidos ou não) e pendências |
| [log.md](log.md) | Histórico de mudanças e decisões, mais recente primeiro |

## Fluxo
`data.load_celeba` baixa o CelebA e gera o cache uint8 `(N,28,28,3)` → `data.batches` entrega lotes em [-1, 1] →
`train.train` constrói o grafo (`model_inputs/model_loss/model_opt`), alterna 1 passo de D e 1 de G e grava
`runs/<data>/` (config, losses.csv, samples/, ckpt/).

## Versões
- Tag `v2019`: estado original (TF 1.x, notebook `GANS_FACE.ipynb`, `helper.py`, `problem_unittests.py`). Para ver: `git show v2019:GANS_FACE.ipynb`.
- `master`: pacote `dcgan/` em TF 2.15 (compat.v1), rodável em scripts.

## Regra de manutenção
Ao mudar código ou entender algo novo, no mesmo commit: (1) atualize o doc afetado (e `quirks.md`); (2) acrescente uma linha no topo de `log.md`; (3) se criar arquivo novo, inclua-o na tabela. Na dúvida, marque `(não verificado)`.
