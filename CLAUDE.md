# tcc-dcgans-2019

TCC (2019): DCGAN que gera rostos humanos a partir do CelebA. Estilo Udacity (projeto face generation).

## Arquivos
- `GANS_FACE.ipynb` — notebook principal: modelo (discriminador/gerador), loss, otimizador, treino.
- `helper.py` — download do CelebA, `Dataset`, `images_square_grid`.
- `problem_unittests.py` — testes chamados dentro do notebook (`tests.test_*`).

## Stack
Python 3.6/3.7 + **TensorFlow 1.x** (usa `tf.placeholder`, `tf.layers`, `tf.contrib`; não roda em TF2). Ver `requirements.txt`.

## Rodar
1. `pip install -r requirements.txt`
2. Abrir o notebook; a 1ª célula baixa o CelebA em `/input` (`data_dir`).
3. Executar as células em ordem. Hiperparâmetros: batch 64, z_dim 100, lr 0.00025, beta1 0.45, 10 épocas.

## Regras
- Não commitar dataset, checkpoints nem imagens geradas pesadas.
- Notebook gera `HoraInicio.txt` / `HoraFinal.txt` (tempo de treino); são artefatos locais.
- Não migrar para TF2/PyTorch sem pedido explícito: o objetivo é preservar o experimento de 2019.
