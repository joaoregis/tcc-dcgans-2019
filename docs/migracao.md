# O que mudou em relação ao código de 2019

Original preservado na tag `v2019` (`git show v2019:GANS_FACE.ipynb`). Uma etapa intermediária (TF 2.15 em CPU) existe no histórico do git.

| Tema | 2019 | Agora | Por quê |
|---|---|---|---|
| Framework | TensorFlow 1.x (`tf.placeholder`, `tf.contrib`) | **PyTorch 2.11 + CUDA 12.8** | TF 1.x não instala nas versões de Python atuais; TF no Windows nativo não usa GPU; PyTorch usa a RTX 4060 |
| Execução | Notebook Jupyter | Scripts (`python -m dcgan.train`) | Sem Jupyter; reprodutível e testável |
| Resolução | 28×28 fixo | 32–256, parametrizada (64 e 128 alvo) | O nº de camadas é calculado de `log2(res)` |
| Dados | CelebA, decodificando JPEG a cada época | Várias fontes → cache `.npy` uint8 uma vez, no HDD `D:` | Velocidade e espaço (o `C:` tem pouco livre) |
| Mistura de datasets | só CelebA | `--sources celeba:100000,ffhq` | Base maior e mais diversa |
| Precisão | float32 | **bfloat16 (autocast)** + `channels_last` | ~3,5× mais rápido (ver `desempenho.md`) |
| Lotes | fatias em ordem fixa | embaralhados por época, espelhamento aleatório na GPU, leitura em thread | Melhor generalização, sem gargalo de E/S |
| Estabilidade | — | **EMA** dos pesos de G | Amostras melhores sem custo de treino |
| Qualidade | olhar as imagens | **FID** + `tune.py` | Decidir hiperparâmetros por número, não por impressão |
| Hiperparâmetros | bs 64, lr 2,5e-4, β1 0,45 | bs 128, lr 2e-4, β1 0,5 (ajuste em `tuning.md`) | Valores do artigo DCGAN; bs ≥128 satura a GPU |
| Saídas | `output/`, `model/`, `Hora*.txt` na raiz; `mkdir` quebrava ao reexecutar | `runs/<timestamp>/` por execução, retomável (`--resume`) | Organização |
| Testes | chamados dentro do notebook | pytest (`tests/`) | Rodam em segundos, sem notebook |
| Arquitetura | G: dense 4×4×1024 + 3 convT; D com BN/dropout quebrados (ver `quirks.md`) | DCGAN canônico: G = convT+BN+ReLU, D = conv+BN+LeakyReLU, sem dropout | Era o que o artigo propõe; os dropouts de 2019 nunca atuavam no D |
| Código morto | MNIST, `ckpt`, `coord`, `threads`... | removido | Legibilidade |

## O que NÃO mudou (de propósito)
Perda não saturante, *label smoothing* de 0,1 nos rótulos reais, LeakyReLU 0,2, Adam com β1 ≈ 0,5, entrada/saída em [-1, 1] com `tanh`.

## Ressalvas honestas
- Não há prova de que a qualidade final iguala ou supera a de 2019: isso depende do treino real (pendente; ver `tuning.md`).
- A arquitetura de G/D mudou (não é a mesma rede do TCC). O comportamento de 2019 só é reproduzível pela tag `v2019` (TF 1.x).
- `torch.compile` não funciona aqui (sem Triton/MSVC); ver `desempenho.md`.
