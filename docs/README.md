# docs/ — documentação viva do projeto

Consulte aqui **antes** de ler o código. Cada arquivo descreve um artefato do repo.

| Doc | Cobre |
|---|---|
| [notebook.md](notebook.md) | `GANS_FACE.ipynb`: pipeline, arquitetura, loss, treino, saídas |
| [helper.md](helper.md) | `helper.py`: download, pré-processamento, `Dataset`, grade de imagens |
| [tests.md](tests.md) | `problem_unittests.py`: testes de contrato chamados no notebook |
| [quirks.md](quirks.md) | Comportamentos estranhos / bugs conhecidos (não corrigidos) |
| [log.md](log.md) | Histórico de mudanças e decisões, mais recente primeiro |

## Visão geral do fluxo
`helper.download_extract` baixa o CelebA → `helper.Dataset` entrega lotes 28×28 RGB em [-0.5, 0.5] →
o notebook define `model_inputs/discriminator/generator/model_loss/model_opt` (cada um validado por
`problem_unittests`) → `train()` treina o DCGAN e salva grades de amostras em `output/` e checkpoints em `model/`.

## Regra de manutenção (importante)
Ao mudar código ou entender algo novo, no mesmo commit:
1. Atualize o doc do artefato afetado (e `quirks.md` se corrigir/descobrir um problema).
2. Acrescente uma linha no topo de `log.md` (data, o quê, por quê).
3. Se criar um arquivo novo no repo, adicione-o à tabela acima.
Doc desatualizado é pior que doc ausente: se não tiver certeza, marque `(não verificado)`.

Estado inicial documentado em 2026-11-22, a partir da leitura do código (nada foi executado; ainda não rodamos o treino).
