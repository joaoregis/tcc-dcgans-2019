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

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- When graphify-out/graph.json exists, run `graphify query "<question>"` before your first grep, search or source-file read for a codebase question, and put this rule in every subagent prompt that explores code. Read raw files only for lines the graph or an earlier read already located. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
