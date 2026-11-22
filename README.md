# tcc-dcgans-2019

Geração de faces humanas com DCGAN (PyTorch + CUDA), treinada em CelebA e outras bases — TCC, 2019, modernizado.
O código original (TF 1.x, notebook) está preservado na tag `v2019`. Documentação: [docs/](docs/README.md).

## Ambiente (Windows + GPU NVIDIA)
```powershell
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```
Crie `local.env` com `DCGAN_DATA_DIR=D:\datasets\faces` (pasta dos datasets, fora do repo).

## Uso
```powershell
.venv\Scripts\python.exe -m pytest -q                                          # testes
.venv\Scripts\python.exe -m dcgan.bench --resolution 64                         # velocidade da GPU
.venv\Scripts\python.exe -m dcgan.datasets --sources celeba --resolution 64     # baixa e pré-processa
.venv\Scripts\python.exe -m dcgan.train --sources celeba --resolution 64        # treina
.venv\Scripts\python.exe -m dcgan.tune --minutes 6 --grid "batch_size=128,256"  # ajusta hiperparâmetros (FID)
```
Cada treino cria `runs/<data>/` com `config.json`, `losses.csv`, `samples/*.png` e `ckpt/`.
