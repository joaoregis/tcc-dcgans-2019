# Quirks do código de 2019 e pendências

Código original: tag `v2019`. Estes problemas foram encontrados por leitura do código original (não executado).

## Resolvidos na reescrita (2026-11-22) — ver `migracao.md`
- Dropout do discriminador nunca atuava (`training` omitido) e o 2º argumento era `rate` chamado `keep_prob` → dropout removido (DCGAN canônico).
- BatchNorm do discriminador sempre em inferência (médias iniciais 0/1, sem normalizar de fato) → BN em modo treino.
- `os.mkdir('output')` quebrava ao reexecutar; checkpoint salvo a cada passo da época 0; grafo crescia a cada amostra; lotes sem embaralhar → `runs/<timestamp>/`, checkpoint por época, `z` fixo para amostras, embaralhamento.
- Código morto (MNIST, `ckpt`, `coord`, `threads`, listas/`cmap`/`saver` duplicado); testes que checavam `d_loss` duas vezes → removidos/reescritos.
- Resolução 28×28 → parametrizada (padrão 64).

## Pendências / riscos
- Qualidade das imagens e FID ainda **não medidos** (treino real pendente).
- Datasets além do CelebA exigem download manual; fontes/licenças a confirmar (ver `dados.md`).
- `torch.compile`: com `triton-windows` funciona, mas é mais lento que `channels_last` sem compilar e quebra junto com ele (ver `desempenho.md`).
- Equivalência com a rede de 2019 não existe: a arquitetura mudou (ver `migracao.md`).
