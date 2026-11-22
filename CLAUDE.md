# tcc-dcgans-2019

TCC (2019): DCGAN que gera rostos humanos 28×28 a partir do CelebA. Código original (TF 1.x, notebook) na tag `v2019`.

## Layout
- `dcgan/` — `config.py` (hiperparâmetros), `data.py` (CelebA), `model.py` (D, G, loss, otimizadores), `train.py` (CLI de treino).
- `tests/test_dcgan.py` — pytest, sem baixar dados.
- `docs/` — documentação viva.

## Ambiente
Windows, **Python 3.11 + TensorFlow 2.15 (CPU)** em `.venv/` (uv). O código usa `tf.compat.v1` em modo grafo para ser equivalente ao TF 1.x de 2019. Sem Jupyter: tudo roda por scripts. GPU nativa no Windows não é suportada por esse TF; ~0,27 s/passo na CPU (≈2,5 h para 10 épocas).

## Comandos
Use sempre `.venv\Scripts\python.exe`:
- `-m pytest -q` — testes (rápidos).
- `-m dcgan.train --help` — todas as opções (épocas, `--limit-images`, `--max-steps`, `--no-fix-2019-bugs`...).
- `-m dcgan.data` — baixa o CelebA (~1,4 GB) para `~/datasets/dcgan`.

## Docs (viva)
Consulte `docs/README.md` antes de ler o código. **Ao mudar código ou descobrir algo, atualize o doc afetado e acrescente uma linha em `docs/log.md` no mesmo commit.**

## Regras
- Não commitar dataset, `runs/`, `.venv/` nem `graphify-out/`.
- Rode `pytest` antes de commitar mudanças em `dcgan/`.
- O comportamento de 2019 é reproduzível com `--no-fix-2019-bugs`; não removê-lo sem pedido.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- When graphify-out/graph.json exists, run `graphify query "<question>"` before your first grep, search or source-file read for a codebase question, and put this rule in every subagent prompt that explores code. Read raw files only for lines the graph or an earlier read already located. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
