# dcgan/datasets.py — dados

## Pastas (`data_dir`, nesta máquina `D:\datasets\faces`, definido em `local.env`)
```
raw/<fonte>/...           imagens originais (qualquer subpasta; .jpg/.jpeg/.png)
cache/<fonte>_<res>.npy   array uint8 (N, res, res, 3); criado por `prepare`, reutilizado
```
Tudo fora do repo e fora do `C:`. Para mudar de lugar: edite `local.env` ou use `--data-dir`.

## Fontes (`SOURCES`)
Cada fonte tem um **recorte** (quais pixels viram a imagem) e, se conhecida, uma URL de download.

| Nome | Conteúdo | Recorte | Download |
|---|---|---|---|
| `celeba` | 202 599 faces alinhadas 178×218 | caixa fixa 148×148 `(15,40,163,188)` (cabelo + queixo) | automático (S3 da Udacity, MD5 conferido; ~1,4 GB) |
| `ffhq` | 70 000 faces, thumbnails 128×128 | imagem inteira | **manual** em `raw/ffhq/` |
| `celebahq` | 30 000 faces alinhadas em alta resolução | imagem inteira | **manual** |
| `utkface` | ~20 000 faces 200×200 (aligned & cropped) | imagem inteira | **manual** |
| `lfw` | 13 233 imagens 250×250, face só no centro | centro 55% | **manual** |

"Manual" = colocar os arquivos em `raw/<fonte>/`; o `prepare` encontra tudo sozinho. Fontes e licenças (todas de uso acadêmico/não comercial;
conferir cada uma) ainda serão confirmadas antes de baixar — **nenhuma foi baixada até agora**.
O recorte de CelebA mudou de 108×108 (2019, só o rosto) para 148×148 (mais contexto), melhor para 64–128 px.

## Funções
- `prepare(name, data_dir, res)` — baixa (se possível), recorta, reduz com **Lanczos** (anti-aliasing) em processos paralelos
  (núcleos − 2) e grava o `.npy` via memmap (não precisa caber na RAM). Idempotente.
- `parse_sources('celeba:100000,ffhq')` → lista `(nome, limite)`. O limite pega as N primeiras imagens (útil para balancear fontes).
- `FaceData.load(spec, data_dir, res)` — abre os `.npy` em memmap e os expõe como um array só. `take(idx)` devolve as linhas pedidas.
- `Loader(data, batch_size, device, flip, seed)` — iterável de lotes `(B,3,res,res)` float32 em [-1,1] na GPU (`channels_last`).
  Uma thread produz lotes `uint8` pinned (cópia rápida para a GPU); normalização e espelhamento são feitos na GPU. Descarta o último lote incompleto.

## Como adicionar um dataset
1. Em `SOURCES`, adicione `Source(center_box(1.0), note='...')` (ou um recorte próprio; opcionalmente `url` e `md5`).
2. Coloque as imagens em `raw/<nome>/` e rode `python -m dcgan.datasets --sources <nome> --resolution 64`.
3. Use em `--sources`. Atualize esta tabela e `log.md`.

## Espaço (uint8)
Cada imagem ocupa `res²·3` bytes: 64 px = 12 KB (CelebA inteiro ≈ 2,5 GB); 128 px = 49 KB (CelebA ≈ 10 GB; todas as fontes ≈ 16 GB).
