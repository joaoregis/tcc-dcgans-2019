# tcc-dcgans-2019

Geração de faces humanas com DCGAN (TensorFlow 1.x) treinada no CelebA — TCC, 2019.

## Como rodar
```bash
pip install -r requirements.txt   # requer Python <= 3.7
jupyter notebook GANS_FACE.ipynb
```
O notebook baixa o CelebA automaticamente (via `helper.py`) para `/input`.

## Conteúdo
- `GANS_FACE.ipynb` — modelo e treino
- `helper.py` — download e carregamento do dataset
- `problem_unittests.py` — testes das funções do notebook
