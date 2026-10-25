# U-Net do zero no CamVid — estudo de ablação

Segmentação semântica urbana (11 classes) com uma U-Net treinada **sem pré-treino**,
investigando o efeito de skip connections, função de perda sob desbalanceamento,
data augmentation e capacidade do modelo.

## Como rodar (Google Colab)

1. Abra `notebooks/camvid_unet.ipynb` no Colab (*File → Open notebook → GitHub*).
2. *Runtime → Change runtime type → GPU* (L4 recomendada).
3. Execute as células em ordem. Resultados e checkpoints vão para
   `MyDrive/unet-camvid/` e o treino é retomável se a sessão cair.

## Estrutura

```
src/
  data.py            download do CamVid, Dataset em RAM, pesos por classe
  model.py           U-Net parametrizável (base, skip on/off)
  losses_metrics.py  CE, CE ponderada, CE + Dice; matriz de confusão, IoU
  train.py           Config + run(): treino, seleção por validação, teste único
  experiments.py     variantes e seeds do estudo
  report.py          tabelas (média ± dp) e figuras
notebooks/
  camvid_unet.ipynb  roteiro completo
```

## Desenho experimental

| Variante  | Muda em relação ao baseline | Pergunta |
|-----------|-----------------------------|----------|
| baseline  | U-Net base 32, CE, com aug  | —        |
| no_skip   | sem skip connections        | Q1       |
| wce       | CE ponderada (median freq.) | Q2       |
| ce_dice   | CE + Dice multiclasse       | Q2       |
| no_aug    | sem flip/brilho/contraste   | Q3       |
| base16/64 | largura 16 / 64             | Q4 (opc.)|

3 seeds por variante; 120 épocas, AdamW (lr 1e-3, cosine), batch 8,
recortes aleatórios 256×256 no treino, avaliação na imagem inteira (360×480).
Métrica principal: mIoU no teste (Void ignorado), média ± desvio-padrão.

## Dados

CamVid (Brostow et al., 2009) na divisão do SegNet (367 / 101 / 233 imagens),
obtido de [alexgkendall/SegNet-Tutorial](https://github.com/alexgkendall/SegNet-Tutorial).
