# Desempenho (RTX 4060 8 GB, Ryzen 9 7900X, 31 GB RAM, PyTorch 2.11 + CUDA 12.8)

Medido com `python -m dcgan.bench` (dados sintéticos, só computação; 30 passos após aquecimento). Medições de 2026-11-22.

## 64 px — imagens/s
| batch | bf16 + channels_last | bf16, NCHW | fp32 NCHW | fp32 channels_last |
|---|---|---|---|---|
| 64 | 5 389 | 4 426 | 2 924 | 1 638 |
| 128 | **6 294** | 5 009 | 2 860 | 1 602 |
| 256 | 6 197 | 4 984 | 2 630 | 1 493 |

Conclusões:
- **bf16 + `channels_last` é o melhor**: ~2,2× sobre fp32 (o melhor fp32 é NCHW) e ~25% sobre bf16 NCHW. (Em fp32, `channels_last` piora.)
- **Batch 128 já satura a GPU** (batch 256 não ganha) → usar ≥128; batches maiores só valem se a qualidade melhorar (ver `tuning.md`).
- Memória é irrelevante em 64 px (<0,5 GB).

## 128 px (bf16 + channels_last)
batch 32: 1 215 · 64: 1 287 · 128: 1 313 img/s; 0,4–0,9 GB. Sobra muita VRAM: 256 px é viável em memória (não medido).

## Treino completo com o `Loader` (dados sintéticos em RAM, 64 px, batch 128)
~6 200 img/s → o carregamento de dados **não** é gargalo.

## Estimativas de tempo por época (CelebA, 202 599 imagens)
- 64 px: ≈ 33 s/época; 128 px: ≈ 2,6 min/época. (TF 2.15 em CPU, antes: ≈ 14 min/época a 28 px.)

## O que não funcionou
- **`torch.compile`**: falha no Windows sem Triton e sem compilador MSVC (`cl.exe`). A opção foi removida do código. Poderia ser reavaliada com `triton-windows` + Visual Studio Build Tools; ganho esperado modesto, rede pequena.
- **Treino em CPU** (etapa TF intermediária): ~0,27 s/passo; ~35× mais lento que a GPU.

## Como remedir
`.venv\Scripts\python.exe -m dcgan.bench --resolution 128 --batches 64,128 --amp both --channels-last both`
