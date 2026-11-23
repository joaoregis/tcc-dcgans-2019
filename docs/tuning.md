# Ajuste de hiperparâmetros: equilíbrio tempo × qualidade

**Estado: 64 px e 128 px concluídos. Pendente: mistura de datasets.** Dados: CelebA 64 px (202 599 imagens). Resultados em `runs/tune_*.csv` (local, não versionado).

## Métrica de qualidade: FID
*Fréchet Inception Distance* compara estatísticas de imagens reais e geradas numa rede Inception: **menor = melhor**
(0 = idênticas). `metrics.fid` usa 5 000 reais × 5 000 falsas e a cópia EMA de G. Não é perfeita, mas é objetiva e padrão na literatura.
Os pesos Inception (~100 MB) são baixados na 1ª execução pelo `torch-fidelity`. Valores só são comparáveis na **mesma resolução e mesmo conjunto real**.

## Método (`python -m dcgan.tune`)
Cada combinação treina por um **orçamento de tempo fixo** (`--minutes`) e depois mede o FID. Assim a pergunta é a do projeto:
*"dado o mesmo tempo de GPU, qual configuração gera melhores imagens?"*. Resultados em `runs/tune_<data>.csv`.
```powershell
.venv\Scripts\python.exe -m dcgan.tune --sources celeba --resolution 64 --minutes 6 --grid "batch_size=128,256 lr_g=1e-4,2e-4,4e-4"
```
Plano em fases (cada uma fixa o melhor da anterior):
1. **Batch e lr** (64 px, 6 min/combinação): `batch_size ∈ {128, 256}` × `lr ∈ {1e-4, 2e-4, 4e-4}` (lr_g = lr_d).
2. **β1 e EMA**: `beta1 ∈ {0.0, 0.5}` e `ema ∈ {0, 0.999}`.
3. **Duração**: treinar uma execução longa, medir FID por época e escolher onde a curva achata (*joelho*): mais épocas além disso custam tempo sem ganho.
4. **128 px**: repetir a fase 1 reduzida (lr, batch) e a fase 3 com o orçamento de ~2,6 min/época.
5. **Mistura de dados**: FID com `celeba` vs. `celeba+ffhq+...` (a mesma contagem total de passos).

## Valores iniciais e por quê (ainda não validados nesta máquina)
batch 128 (satura a GPU), lr 2e-4 e β1 0,5 (artigo DCGAN), EMA 0,999, label smoothing 0,1 (herdado de 2019).

## Resultados

### Fase 1 — batch e lr (64 px, CelebA, 6 min por combinação, β1 = 0,5, EMA 0,999; FID em 5 000 imagens, menor = melhor)
| batch | lr (G = D) | passos | épocas | img/s (mediana) | FID (EMA) |
|---|---|---|---|---|---|
| 128 | 1e-4 | 16 805 | 10,6 | ~6 000 | 40,2 |
| 128 | 2e-4 | 16 146 | 10,2 | ~5 700 | 32,9 |
| 128 | **4e-4** | 15 926 | 10,1 | ~5 800 | **32,3** |
| 256 | 1e-4 | 6 862 | 8,7 | ~4 900 | 65,8 |
| 256 | 2e-4 | 6 743 | 8,5 | ~4 800 | 47,4 |
| 256 | 4e-4 | 6 020 | 7,6 | ~4 300 | 38,5 |

Leitura:
- **Batch 128 vence batch 256 no mesmo tempo**: batch 256 faz metade dos passos e ainda processa *menos* imagens/s (o carregador de dados, não a GPU, vira o limite: ver `desempenho.md`). Em GANs, mais passos pesa mais que lotes maiores.
- **lr 2e-4 e 4e-4 empatam** (diferença de 0,6 de FID, provavelmente dentro do ruído de uma única execução); 1e-4 é claramente pior neste orçamento. lr maior converge mais rápido, mas é mais arriscado em treinos longos.
- Ressalva de método: duas combinações (batch 256, lr 2e-4 e 4e-4) foram descartadas na 1ª rodada porque a velocidade caiu para ~500–1 000 img/s (máquina com outros programas pesados abertos) e foram refeitas. Comparações de FID por tempo fixo só valem com a máquina ociosa.
- Cada valor é **uma execução com uma seed**; não há barras de erro.

### Fase 2 — β1 e efeito da EMA (batch 128, **8 épocas fixas** = 12 656 passos cada)
A 1ª tentativa, por tempo fixo, foi invalidada: a velocidade oscilou de ~1 600 a ~5 800 img/s por causa de outros programas, então cada combinação treinou um número diferente de passos. Passamos a comparar por **épocas fixas** (`tune --epochs 8`) e a registrar o tempo à parte.

| lr (G = D) | β1 | FID (EMA) | FID sem EMA | segundos |
|---|---|---|---|---|
| 2e-4 | 0 | 42,8 | 52,6 | 276 |
| 2e-4 | **0,5** | **34,7** | 42,6 | 272 |
| 4e-4 | 0 | 36,1 | 51,9 | 282 |
| 4e-4 | 0,5 | 36,7 | 48,4 | 555 (máquina ocupada) |

Leitura: **lr 2e-4 + β1 0,5 é o melhor** (e já eram os valores iniciais). A **EMA reduz o FID em 8 a 15 pontos** em todos os casos → mantida. lr 4e-4 não vence quando se igualam os passos.

### Fase 3 — quantas épocas? (64 px, CelebA, lr 2e-4, β1 0,5, batch 128, EMA; FID em 5 000 imagens)
Um treino de 40 épocas (63 280 passos, ~26 min; interrompido na época 25 por reinício da sessão e retomado do checkpoint da época 24 — a seed e o checkpoint garantem a continuidade, mas o embaralhamento da retomada não é idêntico). `dcgan/curve.py` mede o FID de cada checkpoint:

| época | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 | 18 | 20 | 22 | 24 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FID | 84,5 | 53,1 | 41,6 | 36,8 | 34,4 | **30,4** | 32,6 | 32,9 | 31,9 | 32,4 | 31,6 | 31,8 |

| época | 26 | 28 | 30 | 32 | 34 | 36 | 38 | 40 |
|---|---|---|---|---|---|---|---|---|
| FID | 29,6 | 29,8 | 29,3 | 27,6 | 26,5 | 26,6 | 27,7 | **26,5** |

Leitura:
- **Joelho em ~10–12 épocas**: o FID cai de 84 para ~30 e depois quase estaciona (12 → 24: 30,4 → 31,8, dentro do ruído de ±1–2).
- **Ganho lento depois**: de 12 para 40 épocas o FID melhora ~13% (30 → 26,5) com 3,3× mais tempo.
- Cada época ≈ 35–48 s (tempo varia com a carga da máquina); 12 épocas ≈ 7–9 min; 20 ≈ 12–15 min; 40 ≈ 26 min.

## Decisão (64 px, CelebA)
| Objetivo | Configuração | Tempo | FID |
|---|---|---|---|
| Rápido, boa qualidade | batch 128, lr 2e-4, β1 0,5, EMA 0,999, **12 épocas** | ~8 min | ~30 |
| **Padrão do projeto** | idem, **20 épocas** (`Config.epochs`) | ~13 min | ~31–32* |
| Melhor qualidade | idem, 40 épocas | ~26 min | ~26,5 |

\* A curva oscila entre as épocas 12 e 24; o ganho real aparece após ~26 épocas. Para o melhor resultado vale ir a 40.

Limites: uma execução por ponto, sem repetição de seeds; FID com 5 000 imagens tem ruído de ±1–2. A 128 px e com mais datasets os valores ótimos podem mudar.

## 128 px (CelebA, β1 0,5, EMA 0,999; épocas fixas)
Cache: `celeba_128.npy` (9,96 GB). Mesma GPU, ~1 460 img/s com dados reais (≈ 140–160 s por época). FID **não é comparável** com o de 64 px (resolução e conjunto de referência diferentes).

### Ajuste curto — 6 épocas por combinação
| batch | lr (G = D) | passos | segundos | FID (EMA) | FID sem EMA |
|---|---|---|---|---|---|
| 128 | 1e-4 | 9 492 | 843 | 93,8 | 100,4 |
| 128 | 2e-4 | 9 492 | 836 | 77,5 | 84,9 |
| 64 | 2e-4 | 18 990 | 885 | 64,0 | 69,4 |
| 64 | 4e-4 | 18 990 | 884 | 56,0 | 71,1 |
| **32** | **2e-4** | 37 986 | 969 | **52,7** | 66,8 |

Leitura: **a 128 px lotes menores vencem** (o oposto do que se viu a 64 px): a GPU já está saturada com batch 32, então mais passos custam quase o mesmo tempo (+10% do 128 ao 32) e pesam mais que lotes grandes. lr maior também ajuda. EMA reduz o FID em 7–15 pontos de novo. Batch 32 (52,7) e batch 64 com lr 4e-4 (56,0) estão a 3 pontos — pode ser ruído de uma execução; escolhi o de menor FID. Não testado: batch 32 com lr 4e-4, batch 16.

### Duração — treino de 20 épocas (batch 32, lr 2e-4; 126 620 passos em 3 214 s ≈ 54 min)
| época | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 | 18 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|
| FID | 83,6 | 55,8 | 48,9 | 35,4 | 35,7 | 32,7 | 34,1 | 34,1 | 33,9 | **31,0** |

Joelho em ~8–12 épocas (FID ~33–35 em 21–32 min); de 12 a 20 épocas o ganho é pequeno (~5%, dentro do ruído) por +22 min.

## Decisão a 128 px
| Objetivo | Configuração | Tempo | FID |
|---|---|---|---|
| Rápido | batch 32, lr 2e-4, 8–12 épocas | 21–32 min | 33–35 |
| **Melhor medido** | batch 32, lr 2e-4, **20 épocas** | ~54 min | 31,0 |

Comando: `-m dcgan.train --resolution 128 --batch-size 32 --lr-g 2e-4 --lr-d 2e-4 --epochs 20`.

## Próximas fases
- **Mistura de dados**: FID com `celeba` vs. `celeba+ffhq+...` no mesmo nº de passos (as outras bases ainda não foram baixadas).
