# tcc-dcgans-2019

Geração de faces humanas com DCGAN (TensorFlow) treinada no CelebA — TCC, 2019.
O código original (TF 1.x, notebook) está preservado na tag `v2019`.

## Ambiente (Windows, CPU)
```powershell
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

## Uso
```powershell
.venv\Scripts\python.exe -m pytest -q                       # testes rápidos (sem baixar nada)
.venv\Scripts\python.exe -m dcgan.data                      # baixa o CelebA (~1,4 GB) e cria o cache 28x28
.venv\Scripts\python.exe -m dcgan.train                      # treino completo (10 épocas)
.venv\Scripts\python.exe -m dcgan.train --epochs 1 --limit-images 6400 --max-steps 100   # teste curto
.venv\Scripts\python.exe -m dcgan.train --no-fix-2019-bugs   # reproduz o comportamento de 2019
```
O dataset fica em `~/datasets/dcgan` (ou `DCGAN_DATA_DIR`), fora do repo. Cada treino cria `runs/<data>/` com `config.json`, `losses.csv`, `samples/*.png` e `ckpt/`.

## Estrutura
- `dcgan/` — `config.py`, `data.py`, `model.py`, `train.py`
- `tests/` — testes pytest
- `docs/` — documentação viva (comece por `docs/README.md`)
