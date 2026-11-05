# INFO — VarroaDataset para detecção de ácaros com YOLO

## 1. Objetivo

Preparar o **VarroaDataset** para treinar um detector de *Varroa destructor* em imagens de abelhas com YOLO (Ultralytics). Este documento diferencia o **formato original** do **formato de detecção YOLO**.

**Fontes:**
- Repositório: https://github.com/schurist/VarroaDataset
- Dados (Zenodo, versão 1.2.0, 13/10/2020): https://zenodo.org/records/4085044
- DOI da versão: https://doi.org/10.5281/zenodo.4085044

## 2. Características do conjunto de dados

- Imagens de abelhas obtidas em ambiente controlado, com dimensões **160 pixels de largura × 280 pixels de altura**.
- Imagens positivas (abelhas com ácaro visível) e negativas (sem ácaro anotado).
- Divisão pré-definida em `train` (treinamento), `val` (validação) e `test` (teste).

### Estatísticas publicadas

| Conjunto | Total de imagens | Positivas | Negativas |
|---|---:|---:|---:|
| Treino (`train`) | 8.225 | 2.554 | 5.671 |
| Validação (`val`) | 1.876 | 451 | 1.425 |
| Teste (`test`) | 3.408 | 942 | 2.466 |
| **Total** | **13.509** | **3.947** | **9.562** |

A distribuição é desbalanceada: cerca de **29,2% positivas** e **70,8% negativas**.

## 3. Estrutura original informada

```text
dataset/
├── train/
│   ├── labels/
│   │   ├── 2017-09-01_10-54-26/
│   │   │   └── *.txt
│   │   └── 2017-10-17_1-39-36/
│   │       └── *.txt
│   ├── videos/
│   │   ├── 2017-09-01_10-54-26/
│   │   │   └── *.png
│   │   └── 2017-10-17_1-39-36/
│   │       └── *.png
│   └── gt_one.csv
├── val/
│   ├── labels/
│   │   ├── 2017-09-01_10-54-26/
│   │   │   └── *.txt
│   │   └── 2017-10-17_1-39-36/
│   │       └── *.txt
│   ├── videos/
│   │   ├── 2017-09-01_10-54-26/
│   │   │   └── *.png
│   │   └── 2017-10-17_1-39-36/
│   │       └── *.png
│   └── gt_one.csv
├── test/
│   ├── labels/
│   │   ├── 2017-09-01_10-54-26/
│   │   │   └── *.txt
│   │   └── 2017-10-17_1-39-36/
│   │       └── *.txt
│   ├── videos/
│   │   ├── 2017-09-01_10-54-26/
│   │   │   └── *.png
│   │   └── 2017-10-17_1-39-36/
│   │       └── *.png
│   └── gt_one.csv
└── gt.csv
```

**Observação:** essa árvore descreve a organização relatada para os arquivos locais. O README do projeto menciona `gt.csv` dentro dos subconjuntos; os arquivos locais relatados se chamam `gt_one.csv`. A pasta `videos/` contém imagens `.png` recortadas dos vídeos.

## 4. Interpretação das anotações originais

### 4.1 CSV

Segundo a documentação do repositório, cada registro contém o caminho da imagem, um rótulo binário e, para exemplos anotados, zero ou mais caixas delimitadoras em coordenadas de cantos opostos.

Exemplos fornecidos:

```text
test/videos/2017-09-01_10-54-26/2017-09-01_10-54-26.mp4-bee_id_7784-6420-1.png 0
test/videos/2017-09-01_10-54-26/2017-09-01_10-54-26.mp4-bee_id_7785-7005-1.png 1 61 148 94 174
```

Interpretação:
- `0`: abelha classificada como negativa, sem caixa de ácaro nessa linha.
- `1`: abelha classificada como positiva; seguem coordenadas das caixas dos ácaros.
- `61 148 94 174`: caixa com canto superior esquerdo `(61, 148)` e inferior direito `(94, 174)`.
- Para múltiplos ácaros, esperar múltiplos grupos de quatro números.

**Atenção:** os delimitadores são separados por espaços, apesar da extensão `.csv`.

### 4.2 TXT original

Exemplo negativo:

```text
0
```

Exemplo positivo:

```text
1
60.724109362054634 148.32578895834905 93.93944415154019 174.15993823906007
```

A primeira linha é o rótulo **binário da imagem**. A segunda linha, quando presente, contém uma caixa em **pixels**, no formato `x_min y_min x_max y_max`, com coordenadas que podem ser decimais. Existem arquivos com mais de uma caixa.

## 5. Formato necessário para detecção YOLO

Para **detecção de ácaros** com uma única classe:

```text
0: varroa
```

Cada linha do arquivo `.txt` YOLO deve ter:

```text
<class_id> <x_center> <y_center> <width> <height>
```

As quatro medidas geométricas devem ser normalizadas para o intervalo `[0, 1]`, dividindo coordenadas horizontais pela largura `W = 160` e verticais pela altura `H = 280`.

**Fórmulas:**

```text
x_center = ((x_min + x_max) / 2) / W
y_center = ((y_min + y_max) / 2) / H
width    = (x_max - x_min) / W
height   = (y_max - y_min) / H
```

Para a caixa CSV `61 148 94 174`:

```text
0 0.484375 0.575000 0.206250 0.092857
```

Para a caixa decimal TXT `60.724109362054634 148.32578895834905 93.93944415154019 174.15993823906007`, a conversão é feita da mesma forma, **sem arredondar previamente as coordenadas**.

**Importante:**
- O `1` do dataset original significa **imagem positiva**, não classe YOLO `1`.
- A classe YOLO do ácaro será `0`, tanto na primeira quanto nas demais caixas.
- Para imagem negativa, o `.txt` YOLO deve estar **vazio** (zero linhas), e não conter `0` isolado.
- Uma imagem com dois ácaros deve ter **duas linhas**, ambas começando por `0`.
- Para treinamento YOLO, cada arquivo de imagem deve ter o arquivo de rótulo correspondente com o **mesmo nome-base**.

## 6. Estrutura YOLO recomendada (nova pasta, sem alterar o original)

```text
varroa_yolo/
├── images/
│   ├── train/
│   │   ├── 2017-09-01_10-54-26/
│   │   └── 2017-10-17_1-39-36/
│   ├── val/
│   │   ├── 2017-09-01_10-54-26/
│   │   └── 2017-10-17_1-39-36/
│   └── test/
│       ├── 2017-09-01_10-54-26/
│       └── 2017-10-17_1-39-36/
├── labels/
│   ├── train/
│   │   ├── 2017-09-01_10-54-26/
│   │   └── 2017-10-17_1-39-36/
│   ├── val/
│   │   ├── 2017-09-01_10-54-26/
│   │   └── 2017-10-17_1-39-36/
│   └── test/
│       ├── 2017-09-01_10-54-26/
│       └── 2017-10-17_1-39-36/
└── data.yaml
```

O Ultralytics reconhece a correspondência entre `images/` e `labels/` mantendo a estrutura relativa. Preservar os nomes das sessões para rastreabilidade e evitar colisões.

Exemplo de `data.yaml`, supondo execução a partir da pasta do projeto:

```yaml
path: /CAMINHO/ABSOLUTO/varroa_yolo
train: images/train
val: images/val
test: images/test
names:
  0: varroa
```

Substituir `/CAMINHO/ABSOLUTO/varroa_yolo` pelo caminho real da máquina.

## 7. Preparação e treinamento propostos

1. **Conferir a fonte de verdade**: comparar `gt.csv`, `gt_one.csv` e `.txt` originais para detectar divergências e definir qual será usado na conversão.
2. **Converter** as caixas em pixels para o formato YOLO, criando arquivos vazios para negativos.
3. **Validar** correspondência imagem/rótulo, classes e imagens ausentes.
4. **Inspecionar visualmente** amostras positivas, negativas e caixas pequenas; sobrepor os retângulos nas imagens originais.
5. **Preservar os splits originais** inicialmente; verificar vazamento por frames vizinhos ou pela mesma abelha/sessão em conjuntos distintos. Para uma avaliação de generalização mais rigorosa, considerar divisão por vídeo/sessão, sem usar o teste para ajustar o modelo.
6. **Treinar um baseline** com um modelo pequeno pré-treinado e comparar posteriormente resoluções e arquiteturas.
7. **Avaliar no teste** apenas após selecionar configurações usando a validação.

## 8. Critérios de qualidade e riscos

- **Ácaros pequenos:** comparar detecções em diferentes resoluções e examinar erros de localização.
- **Desbalanceamento:** monitorar precisão, recall, mAP50, mAP50-95 e falsos positivos por imagem negativa; não usar apenas acurácia.
- **Rótulos negativos:** uma abelha rotulada como negativa pode ter um ácaro não visível; o objetivo operacional é detectar **ácaros visíveis**, não diagnosticar infestações ocultas.
- **Sobreposição temporal:** frames próximos do mesmo vídeo podem ser muito semelhantes. Evitar interpretar bom desempenho em frames correlacionados como generalização para outras colmeias, câmeras ou condições de iluminação.
- **Integridade:** não converter rótulos binários diretamente para linhas YOLO sem coordenadas.
- **Licenciamento:** verificar os termos de uso do registro Zenodo e do material associado antes de redistribuir imagens ou um dataset convertido; o repositório GitHub apresenta uma licença MIT para seu conteúdo, o que não deve ser presumido automaticamente para todos os arquivos distribuídos separadamente.

## 9. Pendências antes da conversão definitiva

- [ ] Determinar se `gt_one.csv` é subconjunto, cópia ou transformação de `gt.csv`.
- [ ] Identificar imagens ausentes e rótulos conflitantes.
- [ ] Produzir um script de conversão reprodutível.
- [ ] Visualizar amostras anotadas após conversão.

## 10. Referência bibliográfica

Schurischuster, Stefan; Kampel, Martin (2020). *VarroaDataset* [Dataset]. Zenodo. DOI da versão consultada: https://doi.org/10.5281/zenodo.4085044. O README do GitHub também indica o DOI conceitual https://doi.org/10.5281/zenodo.4085043.
