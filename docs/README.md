# docs/ — documentação viva do projeto

Consulte aqui **antes** de ler o código. Se é a primeira vez: leia [como-funciona.md](como-funciona.md), depois [migracao.md](migracao.md).

| Doc | Para quê |
|---|---|
| [como-funciona.md](como-funciona.md) | **Comece aqui.** Explicação didática: o que é um DCGAN, o caminho de um dado do disco até a imagem gerada, arquivo por arquivo |
| [faq.md](faq.md) | Dúvidas que surgiram (DCGAN, FID, seeds e aritmética de vetores, usar o modelo...) |
| [migracao.md](migracao.md) | O que mudou em relação ao código de 2019 (e por quê), item por item |
| [modelo.md](modelo.md) | `dcgan/models.py`: arquitetura, resolução, EMA, perda |
| [dados.md](dados.md) | `dcgan/datasets.py`: fontes, pré-processamento, cache, mistura, lotes; como adicionar um dataset |
| [treino.md](treino.md) | `dcgan/train.py` e `dcgan/config.py`: o passo de treino, flags, saídas em `runs/`, como retomar |
| [usar-modelo.md](usar-modelo.md) | Depois do treino: gerar imagens, exportar só o gerador, usar em Python |
| [desempenho.md](desempenho.md) | Medições na RTX 4060, o que acelerou e o que não funcionou |
| [tuning.md](tuning.md) | Como ajustamos batch/lr/épocas (tempo × qualidade): método, FID, resultados |
| [testes.md](testes.md) | `tests/test_dcgan.py` |
| [quirks.md](quirks.md) | Bugs do código de 2019 e pendências |
| [log.md](log.md) | Histórico de mudanças e decisões, mais recente primeiro |

## Mapa do código
```
dcgan/config.py     Config (dataclass): todos os parâmetros; vira flags de linha de comando
dcgan/datasets.py   fontes de imagens → cache .npy → mistura → lotes na GPU
dcgan/models.py     Generator, Discriminator, EMA
dcgan/train.py      passo de treino, loop, amostras, checkpoints
dcgan/generate.py  usa o gerador treinado (gera imagens, exporta pesos)
dcgan/metrics.py    FID (qualidade das imagens)
dcgan/tune.py       busca de hiperparâmetros com orçamento de tempo
dcgan/bench.py      medição de velocidade (img/s) com dados sintéticos
tests/              pytest
```

## Regra de manutenção
Ao mudar código ou entender algo novo, no mesmo commit: (1) atualize o doc afetado (e `quirks.md`); (2) acrescente uma linha no topo de `log.md`; (3) se criar arquivo novo, inclua-o aqui. Na dúvida, marque `(não verificado)`.
