# dcgan/models.py

## Arquitetura (resolução R, largura base w=64)
`widths(R, w)` devolve as larguras dos blocos: `[min(w·2^(n−1−i), 512) for i in range(n)]`, com `n = log2(R) − 2`.
- R=64 → `[512, 256, 128, 64]`; R=128 → `[512, 512, 256, 128, 64]`.

**Generator** (`z` de forma `(B,100,1,1)`):
`ConvT(100→512, 4×4) → BN → ReLU` (1×1 → 4×4) → para cada par de larguras `ConvT(4,2,1) → BN → ReLU` (dobra a resolução) →
`ConvT(→3) → tanh` (saída em [-1, 1]). R=64: 4→8→16→32→64.

**Discriminator**: espelho. `Conv(3→64, s2) → LeakyReLU(0.2)` (sem BN na 1ª camada) → `Conv(s2) → BN → LeakyReLU` … até 4×4 →
`Conv(512→1, 4×4)` → 1 *logit* por imagem (a sigmoide fica dentro da perda `binary_cross_entropy_with_logits`, mais estável).

Tamanho: G ≈ 3,6 M parâmetros e D ≈ 2,8 M em R=64 (valor aproximado; conferir na 1ª linha do log de treino).

**Inicialização**: pesos das convoluções N(0, 0,02), escala da BN N(1, 0,02) — receita do artigo DCGAN.

## EMA (`EMA`)
Guarda uma cópia de G cujos pesos são `ema = d·ema + (1−d)·pesos`, `d = 0,999` (com aquecimento `min(d, (1+t)/(10+t))`
para não ficar presa aos pesos iniciais). Os pesos de G oscilam passo a passo (o jogo contra D é instável); a média é mais
estável e costuma gerar imagens melhores. **Amostras e FID usam a cópia EMA.** Estatísticas da BN são copiadas, não suavizadas.
Custo: um `_foreach_lerp_` por passo (desprezível).

## Perda (em `train.train_step`)
Ver `como-funciona.md` §3. D: BCE(real, 1−0,1) + BCE(falso, 0). G: BCE(falso, 1).

## Por que sem dropout / sem camada densa
O DCGAN canônico não usa nenhum dos dois. Em 2019 o dropout do D nunca estava ativo (bug), então removê-lo não altera o que de fato rodava.
