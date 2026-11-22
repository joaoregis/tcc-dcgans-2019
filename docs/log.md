# Log de mudanças e decisões (mais recente primeiro)

- 2026-11-22 — Refatoração: código do notebook movido para o pacote `dcgan/` (config/data/model/train); migrado para TF 2.15 em modo grafo (`compat.v1`) porque TF 1.x não roda em Python 3.13/3.14 desta máquina; sem Jupyter; testes em pytest; saídas em `runs/`; cache `.npy` do dataset fora do repo. Removidos `GANS_FACE.ipynb`, `helper.py`, `problem_unittests.py` (preservados na tag `v2019`). Ambiente: uv + Python 3.11 em `.venv/`. Medido ~0,27 s/passo na CPU. Correções dos bugs de 2019 atrás da flag `fix_2019_bugs`.
- 2026-11-22 — Criada `docs/` a partir da leitura do código original.
- 2026-11-22 — Repo criado; adicionados `CLAUDE.md`, `README.md`, `requirements.txt`, `GRAPHIFY.md`; `graphify-out/` ignorado no git; tag `v2019` marca o estado original.
