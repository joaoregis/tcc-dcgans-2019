# dcgan/data.py

- **`download_extract(data_dir)`** — baixa `celeba.zip` (URL S3 da Udacity, respondeu HTTP 200 em 2026-11-22), confere MD5 `00d2c5bc…`, extrai em `data_dir/img_align_celeba` (202 599 JPEGs 178×218), apaga o zip. Se a pasta existe, só devolve o caminho. **Download não executado ainda** (~1,4 GB).
- **`get_image(path, size=28)`** — recorte central 108×108 + resize bilinear → `uint8 (28,28,3)` (igual a 2019).
- **`load_celeba(data_dir, limit=0)`** — pré-processa tudo uma vez (threads) e grava o cache `data_dir/celeba_28.npy` (~476 MB); depois só lê o `.npy`. `limit>0` pega só as N primeiras.
- **`batches(images, batch_size, shuffle, rng)`** — gerador de lotes `float32` em [-1, 1] (`/127.5 − 1`); descarta o último lote incompleto; embaralha os índices por época (índices do lote ordenados para leitura eficiente).
- **`images_square_grid(images)`** — grade ⌊√n⌋² (PIL RGB), normaliza min/max do lote para 0–255.
- `python -m dcgan.data` — baixa e gera o cache.

Dependências: numpy, Pillow, tqdm. MNIST (código morto de 2019) foi removido.
