# tcc-dcgans-2019

TCC (2019): DCGAN que gera rostos humanos, agora em PyTorch + CUDA (resolução 64→128 px, várias bases de faces). Código original (TF 1.x, notebook) na tag `v2019`.

## Layout
- `dcgan/` — `config.py`, `datasets.py`, `models.py`, `train.py`, `metrics.py` (FID), `tune.py`, `bench.py`.
- `tests/test_dcgan.py` — pytest. `docs/` — documentação viva (comece por `docs/README.md` e `docs/como-funciona.md`).

## Ambiente
Windows, **Python 3.11 + PyTorch 2.11 (CUDA 12.8)** em `.venv/` (uv). GPU RTX 4060 8 GB, 31 GB RAM. Sem Jupyter: tudo por scripts.
Datasets em **`D:\datasets\faces`** (HDD; via `local.env`, ignorado pelo git) — nunca no `C:` (pouco espaço) nem no repo/OneDrive.
`torch.compile` não funciona aqui (sem Triton/MSVC).

## Comandos (sempre `.venv\Scripts\python.exe`)
`-m pytest -q` · `-m dcgan.bench` · `-m dcgan.datasets --sources celeba --resolution 64` · `-m dcgan.train --help` · `-m dcgan.tune --help`

## Docs (viva)
Consulte `docs/README.md` antes de ler o código. **Ao mudar código ou descobrir algo, atualize o doc afetado e acrescente uma linha em `docs/log.md` no mesmo commit.** O usuário quer entender o que foi feito: explique em `docs/`.

## Regras
- Não commitar dataset, `runs/`, `.venv/`, `local.env`, `graphify-out/`.
- Rode `pytest` antes de commitar mudanças em `dcgan/`.
- Pedir permissão antes de baixar datasets/pesos (tamanho e fonte) e antes de treinos longos.
- Hiperparâmetros: decidir por medição (FID × tempo, ver `docs/tuning.md`), não por palpite.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- When graphify-out/graph.json exists, run `graphify query "<question>"` before your first grep, search or source-file read for a codebase question, and put this rule in every subagent prompt that explores code. Read raw files only for lines the graph or an earlier read already located. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
