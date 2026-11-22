# Como o código funciona

## 1. A ideia (DCGAN em 1 minuto)
Duas redes neurais competem:
- **Gerador (G)** recebe um vetor de números aleatórios `z` (100 números) e desenha uma imagem de rosto.
- **Discriminador (D)** recebe uma imagem e diz se é *real* (veio do dataset) ou *falsa* (veio de G).

Em cada passo de treino: D aprende a separar reais de falsas; depois G aprende a enganar D. Com o tempo, G produz rostos
cada vez mais convincentes. "DC" = *deep convolutional*: as duas redes usam camadas convolucionais
(G usa convolução **transposta**, que aumenta a imagem; D usa convolução com *stride 2*, que a reduz).

## 2. O caminho dos dados (do disco à GPU)
```
D:\datasets\faces\raw\celeba\*.jpg        imagens originais (fora do projeto, no HDD)
        │  datasets.prepare()  — recorta a face, reduz (Lanczos) para res×res, usa todos os núcleos da CPU
        ▼
D:\datasets\faces\cache\celeba_64.npy     um array uint8 (N, 64, 64, 3): pronto, lido sem decodificar JPEG
        │  datasets.FaceData  — junta várias fontes como se fossem um array só (memmap, sem copiar)
        ▼
datasets.Loader                           thread em segundo plano: sorteia índices, copia o lote para memória "pinned"
        │  .to('cuda') + normaliza para [-1,1] + espelha aleatoriamente — tudo na GPU
        ▼
train.train_step(real, ...)               lote float32 (B,3,res,res) na GPU
```
Por que o cache? Decodificar 200 mil JPEGs a cada época desperdiçava tempo (era o gargalo em 2019). Agora isso é feito **uma vez**;
depois cada época é só ler bytes. O HDD só importa na 1ª leitura: o Windows mantém o arquivo no cache de RAM (31 GB nesta máquina).

## 3. O passo de treino (`train.train_step`)
```
z ~ N(0,1)                        # ruído
fake = G(z)                       # G desenha
d_loss = BCE(D(real), 0.9) + BCE(D(fake.detach()), 0)   # D: real→0.9 (suavizado), falso→0
atualiza D
g_loss = BCE(D(fake), 1)          # G: quer que D diga "real" (perda "não saturante")
atualiza G
```
`BCE` = entropia cruzada binária (com logits). `fake.detach()` impede que o erro de D "vaze" para G. Os rótulos reais valem 0,9
em vez de 1 (*label smoothing*), o que deixa D menos arrogante e estabiliza o treino (já era assim em 2019).

## 4. Os arquivos, por dentro
- **`config.py`** — `Config` é um dataclass com **todos** os parâmetros e seus padrões. `parse_args` gera uma flag por campo
  (`--batch-size 256`, `--no-amp`). Os caminhos desta máquina vêm de `local.env` (ignorado pelo git).
- **`models.py`** — `Generator`/`Discriminator` calculam o nº de camadas a partir da resolução (`log2(res) − 2`),
  então o mesmo código serve para 32, 64, 128, 256 px. `EMA` mantém uma média móvel dos pesos de G (ver `modelo.md`).
- **`datasets.py`** — registro `SOURCES` (um recorte e, quando conhecida, uma URL por dataset), `prepare` (gera o cache),
  `FaceData` (mistura fontes), `Loader` (lotes na GPU).
- **`train.py`** — `train_step` (acima), `train` (prepara tudo, roda épocas, grava amostras/checkpoints/log em `runs/<data>/`).
- **`metrics.py`** — `fid`: nota objetiva de qualidade (menor é melhor). **`tune.py`** usa o FID para escolher hiperparâmetros.
- **`generate.py`** — carrega o gerador treinado e cria imagens (ver `usar-modelo.md`).
- **`bench.py`** — mede imagens/segundo (ver `desempenho.md`).

## 5. Rodar
```powershell
.venv\Scripts\python.exe -m pytest -q                                   # testes
.venv\Scripts\python.exe -m dcgan.bench --resolution 64                  # velocidade da GPU (sem dados)
.venv\Scripts\python.exe -m dcgan.datasets --sources celeba --resolution 64   # baixa/pré-processa
.venv\Scripts\python.exe -m dcgan.train --sources celeba --resolution 64      # treina
```
Resultado de cada treino: `runs/<data>/` com `config.json`, `losses.csv`, `samples/*.png`, `ckpt/epoch_NNN.pt`, `time.txt`.

## 6. Perguntas frequentes
**Isto é mesmo um DCGAN?** Sim. Segue as diretrizes de Radford et al. (2015): convoluções com *stride* no lugar de *pooling*,
convolução transposta no gerador, BatchNorm nas duas redes (exceto saída do G e entrada do D), ReLU no G com `tanh` na saída,
LeakyReLU no D, nenhuma camada densa escondida (a projeção de `z` virou uma `ConvT` 1×1→4×4, equivalente). Extras que **não** estão no artigo:
*label smoothing* de 0,1 (Salimans et al., 2016; já usado em 2019), EMA dos pesos de G (prática posterior, de ProGAN/StyleGAN),
precisão bf16 e espelhamento aleatório. O código de 2019 era "estilo DCGAN" mas se afastava do canônico (camada densa 4×4×1024, dropout, BN do D mal usada).

**Dá para otimizar a rede em tempo real mexendo em parâmetros?** Em parte. Já acontece: o Adam adapta o passo de cada peso sozinho.
Ajustar hiperparâmetros (lr, batch, β1) *durante* o treino é possível (o `--resume` permite parar, mudar a flag e continuar) e há técnicas
para isso (decaimento do lr, parada por platô do FID, *Population Based Training*, Optuna com poda de tentativas), mas em GANs é arriscado:
muda o equilíbrio entre G e D e dificulta reproduzir. **Não está implementado**; hoje o ajuste é feito *entre* execuções curtas com FID (`tune.py`).
Um próximo passo barato seria decair o lr no fim do treino e parar quando o FID parar de melhorar.
