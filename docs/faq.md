# FAQ — dúvidas que surgiram durante o projeto

Perguntas do usuário e respostas, para consulta. Acrescente novas ao final da seção certa (e uma linha em `log.md`).

## Conceitos
**Isto é mesmo um DCGAN?** Sim. Segue Radford et al. (2015): convoluções com *stride* no lugar de *pooling*, convolução transposta no gerador,
BatchNorm nas duas redes (exceto saída do G e entrada do D), ReLU no G com `tanh` na saída, LeakyReLU no D, sem camada densa escondida
(a projeção de `z` virou uma `ConvT` 1×1→4×4, equivalente). Extras que **não** estão no artigo: *label smoothing* 0,1 (Salimans et al., 2016; já usado em 2019),
EMA dos pesos de G (prática posterior, de ProGAN/StyleGAN), bf16 e espelhamento aleatório. O código de 2019 era "estilo DCGAN" mas se afastava do canônico
(camada densa 4×4×1024, dropout, BN do D mal usada). Detalhes: `migracao.md`.

**Para que serve o FID?** *Fréchet Inception Distance*: nota objetiva de qualidade (menor = melhor; 0 = indistinguível). Compara 5 000 imagens reais e 5 000 geradas
pelas estatísticas de uma rede Inception. Precisamos dele porque as perdas do GAN **não** indicam qualidade (G e D se enfrentam, os números oscilam) e olhar
imagens não basta para decidir entre duas configurações parecidas. Só compara resultados na mesma resolução e contra o mesmo conjunto real. Pesos do Inception
(91 MB) ficam no cache do torch (`E:\Caches\torch\hub\checkpoints`). Sanidade: gerador sem treino vs. ruído deu FID ≈ 426. Ver `tuning.md`.

## Desempenho
**O que é `torch.compile` e por que não usamos?** Recurso do PyTorch 2 que analisa a rede, junta operações pequenas e gera código de GPU otimizado (via Triton).
Testamos com `triton-windows` (não precisou do MSVC): +5% em NCHW, mas o melhor modo sem compilar (`channels_last`) já é mais rápido (6 294 vs. 5 388 img/s) e compilar
com `channels_last` quebra (bug de strides no Inductor). Ganho líquido negativo; removido. Ver `desempenho.md`.

**Dá para otimizar a rede em tempo real mexendo em parâmetros?** Em parte. O Adam já adapta o passo de cada peso sozinho. Mudar lr/batch/β1 *durante* o treino é possível
(`--resume` permite parar, trocar a flag e continuar) e há técnicas (decaimento do lr, parada por platô do FID, *Population Based Training*, Optuna com poda), mas em GANs é arriscado
(muda o equilíbrio G×D e dificulta reproduzir). **Não implementado**: hoje o ajuste é *entre* execuções curtas com FID (`tune.py`). Um passo barato seria decair o lr no fim e parar quando o FID parar de melhorar.

**Por que o download do CelebA estava lento?** O servidor limita cada conexão a ~100–300 kB/s. `datasets.fetch_parallel` baixa em partes com 16 conexões (HTTP Range) e soma ~2–7 MB/s.

## Usar o modelo treinado
**Depois de treinar, dá para extrair o modelo e executá-lo?** Sim: `python -m dcgan.generate <checkpoint.pt>` (ver `usar-modelo.md`). Só o gerador é necessário.

**Os pesos são carregados por outro script?** Sim. O arquivo de pesos é só uma lista de números, não sabe qual é a rede. `generate.py` lê dele a resolução, `z_dim` e largura,
reconstrói o `Generator` do `models.py` e carrega os números (`load_state_dict`). Modelo = arquitetura (código) + pesos. Rodar sem o código Python exigiria exportar para ONNX/TorchScript (não feito).

**O discriminador pode ser aproveitado?** Sim, mas não para gerar imagens. O checkpoint de treino o guarda e ele serve para: retomar o treino; extrair características para um classificador com poucos rótulos
(o artigo DCGAN fez isso em CIFAR-10/SVHN); dar uma nota de "realismo" para filtrar imagens ruins (pouco confiável depois do treino); iniciar o treino em outro dataset. O arquivo exportado (`--export`) só tem o gerador. Nada disso está implementado.

## Seeds, vetores latentes e manipulação
**A mesma seed gera sempre o mesmo rosto?** Sim, **com o mesmo modelo treinado**. A seed só sorteia o vetor latente `z` (100 números ~N(0,1)); o rosto vem desse `z`.
Em outro modelo, ou em outra época do mesmo treino, a mesma seed dá outro rosto.

**Dá para combinar seeds com soma, subtração, multiplicação e divisão e obter rostos com as características que quero?** Quase: o certo é operar nos **vetores `z`**, não nas seeds.
- Uma *seed* é só um número inteiro (ex.: 42) que, via gerador de números aleatórios, produz um `z`. Somar duas seeds (42 + 7) **não** mistura os rostos; é só outro inteiro, que sorteia um `z` sem relação.
- O procedimento certo: (1) gere `z₁`, `z₂`… a partir das seeds que você conhece; (2) faça as operações **nesses vetores**; (3) passe o resultado pelo gerador. O resultado é um novo `z` (guarde-o em arquivo, ou a "receita": quais seeds e quais operações).
- **Soma/subtração**: funcionam. É a aritmética do artigo (média de ~3 `z` de "homem de óculos" − média de "homem sem óculos" + média de "mulher sem óculos" ≈ "mulher de óculos"). Tirar a média de vários `z` por conceito reduz o ruído; com um `z` só o resultado costuma ser instável.
- **Multiplicar/dividir por um número** (escalar): funciona como "intensidade". Dividir pela quantidade de vetores é a média. Escalas muito diferentes de 1 afastam `z` do que o gerador viu no treino e surgem artefatos.
- **Multiplicar/dividir um vetor por outro, elemento a elemento**: não tem significado útil.
- **Interpolação** `(1−t)·z₁ + t·z₂` (t de 0 a 1): transição suave entre dois rostos. Melhor com *slerp* (interpolação esférica), porque `z` ~N(0,I) fica numa "casca" de raio ≈ 10 e a média linear de dois vetores encolhe a norma.
- Limites: o modelo **não é rotulado**; você descobre quais rostos têm óculos etc. olhando as amostras, e os atributos são parcialmente "emaranhados" (mexer em um pode mudar outros). Não há garantia, só tendência. Para controle explícito de atributos seria preciso um modelo condicional (cGAN) ou um classificador para achar direções no espaço latente.
- **Resultado real (modelo de 40 épocas, 64 px)**: a receita do artigo com 3 amostras por grupo **não** deu óculos (`docs/exemplos/aritmetica_tentativa1.png`: saiu uma mulher sem óculos). Com mais amostras (4) e coeficiente 1,5 (`tentativa2`) saiu um rosto com óculos escuros. Ou seja: funciona, mas é sensível à escolha das seeds e dos coeficientes. As seeds foram escolhidas olhando um catálogo das seeds 0–63.
- **Como usar** (`dcgan/latent.py`, ver `usar-modelo.md`): `interpolate --seeds 42 7` e `arith --expr "mean(1,2,3) - mean(4,5,6) + mean(7,8,9)"`. Testado também com os modelos treinados (exemplos em `docs/exemplos/`).
