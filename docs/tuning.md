# Ajuste de hiperparâmetros: equilíbrio tempo × qualidade

**Estado: método pronto, resultados PENDENTES** (dependem de baixar dados reais e os pesos do Inception do FID).

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
*(a preencher: tabela de cada fase com FID, passos e épocas; decisão final e custo em minutos)*
