# Usar o modelo treinado (inferência)

## O que é "o modelo" depois do treino
Só o **gerador (G)** é usado para criar imagens; o discriminador serve apenas durante o treino. Os "pesos" são os números aprendidos
pelo G (~3,6 M parâmetros em 64 px ≈ 14 MB). Cada época grava `runs/<data>/ckpt/epoch_NNN.pt`, que guarda **tudo** para retomar o treino
(G, D, cópia EMA de G, otimizadores, passo, época, configuração). Para gerar imagens basta o G — de preferência a **cópia EMA**, que dá imagens melhores.

## Como gerar imagens
Gerar = sortear ruído `z` (100 números) e passar pelo G. Cada `z` diferente é um rosto diferente; **a mesma semente dá as mesmas imagens**.
```powershell
.venv\Scripts\python.exe -m dcgan.generate runs\<data>\ckpt\epoch_010.pt --n 64 --seed 1
```
Grava `geradas.png` (grade) e `geradas\000.png`, `001.png`… (individuais). Roda em GPU ou CPU, **sem dataset**.

## Exportar só o gerador (para levar a outro lugar)
```powershell
.venv\Scripts\python.exe -m dcgan.generate runs\<data>\ckpt\epoch_010.pt --export gerador.pt
.venv\Scripts\python.exe -m dcgan.generate gerador.pt --n 16 --out minhas_faces
```
`gerador.pt` tem só os pesos de G + `resolution`, `z_dim`, `g_width` (menor que o checkpoint, que carrega também D e os otimizadores).

## Em código Python
```python
from dcgan.generate import load_generator, generate
g, cfg = load_generator('gerador.pt', 'cuda')
imgs = generate(g, cfg, n=16, seed=3, device='cuda')   # tensor (16, 3, R, R) em [0, 1]
```

## Limites
- O arquivo `.pt` é um *pickle* do PyTorch: carregue apenas arquivos que você mesmo gerou.
- A resolução é fixa pelo treino (um modelo de 64 px gera 64 px). Para imagens maiores é preciso treinar com `--resolution 128`.
- Não há controle de "atributos" (idade, óculos...): o DCGAN gera rostos aleatórios. Interpolar entre dois `z` dá transições suaves (não implementado ainda).
