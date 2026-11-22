# helper.py

Utilitários de dados (herdados do projeto Udacity). Usado pelo notebook: `download_extract`, `Dataset`, `images_square_grid`.

## API usada
- **`download_extract(database_name, data_path)`** — só `'celeba'` funciona. Baixa `celeba.zip` (URL S3 da Udacity — disponibilidade atual não verificada), confere MD5 `00d2c5bc…`, extrai para `data_path/img_align_celeba`, apaga o zip. Se a pasta já existe, imprime "Found celeba Data" e retorna. Em erro de extração remove a pasta parcial.
- **`Dataset(dataset_name, data_files)`** — fixa imagem **28×28**; CelebA → modo `'RGB'`, 3 canais; `self.shape = (n_arquivos, 28, 28, canais)`.
  - **`get_batches(batch_size)`** — gerador; fatias consecutivas **sem embaralhar**; descarta o último lote incompleto; devolve `float` em **[-0.5, 0.5]** (`/255 - 0.5`). O notebook multiplica por 2 → [-1, 1].
- **`images_square_grid(images, mode)`** — normaliza o lote por min/max para 0–255, monta grade `floor(sqrt(n))²` e devolve imagem PIL.

## Funções internas
- **`get_image(path, w, h, mode)`** — abre a imagem; se o tamanho ≠ (w,h) (caso do CelebA, 178×218): recorte central **108×108**, depois resize bilinear para w×h. Converte para `mode` e devolve array.
- **`get_batch(files, w, h, mode)`** — empilha `get_image`, `float32`, garante 4 dimensões.
- **`_unzip`**, **`DLProgress`** (barra tqdm para `urlretrieve`).
- **`_ungzip`, `_read32`** — leitura de MNIST (idx gzip). **Não usadas** (ver quirks).

## Dependências
`numpy`, `Pillow`, `tqdm`; stdlib (`zipfile`, `gzip`, `hashlib`, `urllib`).
