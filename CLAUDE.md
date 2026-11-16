# CLAUDE.md — Reconhecimento de sinais de Libras em tempo real

## Objetivo
Treinar um modelo que reconheça sinais de Libras **dinâmicos** (movimento ao longo do
tempo, não só pose estática) a partir de vídeos, e rodar o reconhecimento em tempo
real pela webcam. Projeto de aprendizado: ao propor mudanças, **explique o conceito
por trás da decisão**, não só o código.

Idioma: responder e comentar código em **português**.

## Dataset: pasta unificada de classes
Todas as bases de dados são extraídas para **uma única pasta de classes**, para que o mesmo
sinal vindo de bases diferentes caia na mesma classe:
```
dados/videos/<classe>/<fonte>__<id>.mp4   única pasta com classes (o que o 1_extrair.py lê)
dados/manifestos/<fonte>.csv              manifesto organizado de cada base (+ <fonte>_original.csv)
dados/manifestos/indice.csv               índice geral de todas as bases (regenerado a cada execução)
dados/excluidos/<fonte>/                  vídeos excluídos e metadados, FORA das classes
```
- [preparacao/base_unificada.py](preparacao/base_unificada.py): núcleo comum a todas as bases.
  Normaliza rótulos, move, escreve manifestos, confere sha256 e avisa sobre **possíveis
  duplicatas** com classes de outras bases (ex.: `guarda roupa` vs `guarda-roupa`). É
  idempotente, faz dry run por padrão, só mexe nos arquivos da própria fonte, move (sem
  re-extrair) vídeos cujo rótulo mudou e detecta órfãos (`--remover-orfaos`).
- `preparacao/<fonte>_pre_tratamento.py`: um **adaptador por base**, que só sabe ler o formato
  dela e devolve uma lista de `Item`. **Base nova = adaptador novo**, copiando o da Incluir
  Tecnologia.
- O prefixo `<fonte>__` evita colisões entre bases e mantém a origem rastreável até o cache de
  landmarks (útil para validar com uma base e treinar com outra = outros sinalizantes).

### Bases incluídas
**incluir_tecnologia**: canal do YouTube **Incluir Tecnologia** (dicionário de Libras, 2011–2012),
pacote `incluir_tecnologia.tar.gz` (layout plano `videos/*.mp4` + `MANIFEST.csv`), adaptador
[preparacao/incluir_tecnologia_pre_tratamento.py](preparacao/incluir_tecnologia_pre_tratamento.py).
O id é o `video_id` do YouTube. O adaptador também aceita uma pasta já extraída.
- **1069 classes, 1187 vídeos**: 965 classes com 1 vídeo, 104 com 2–4 (91×2, 12×3, 1×4).
  Poucos vídeos por classe é a restrição que manda em todas as decisões abaixo.
- Provavelmente poucos sinalizantes (mesmo canal); resoluções de 446×360 a 1920×1080, 25/30 fps,
  sinais de 2–6 s com a pessoa parada e as mãos abaixadas (fora do quadro) no início e no fim.
- Excluídos (textos/músicas sinalizados de 100–160 s): `A sua`, `Simples desejo`, `Amor Sincero`
  (este último foi apagado à mão; no manifesto aparece como `ausente`).

**minds_libras_rgb / minds_libras_rgbd**: MINDS-Libras (UFMG, CC-BY-4.0), 20 sinais × 12
sinalizantes × 5 repetições, adaptador [preparacao/minds_libras_pre_tratamento.py](preparacao/minds_libras_pre_tratamento.py).
Lê de `/mnt/hd1tb/libras-data/raw/minds_libras_{rgb,rgbd}/extracted` (repositório bruto compartilhado:
**links simbólicos**, nunca mover; `--copiar` se precisar de cópia). `rgb` = Canon (1158 vídeos,
recortados); `rgbd` = só o RGB do Kinect (1200 vídeos de 5 s fixos; Depth/Body/Face ignorados).
- **Kinect adiado** (decisão do usuário, 2026-11-16): está em `dados/videos` e no cache, mas o
  `2_treinar.py` o deixa fora por padrão (`--excluir-fontes minds_libras_rgbd`; `--excluir-fontes`
  vazio inclui).
- As duas câmeras gravaram **a mesma execução**: o id `s<NN>_<sinal>_r<n>` é igual nas duas fontes,
  e o manifesto tem a coluna `sinalizador` = `minds_<NN>` (global entre bases, igual nas câmeras).
  A validação por pessoa do `2_treinar.py` usa essa coluna, o que mantém as duas câmeras juntas.
- Grafia corrigida no adaptador (`ROTULOS`): `Maca`→`maçã`, `America`→`américa`, `Cinco`→`5`
  (conferido visualmente contra o `5` da Incluir). 11 classes já existiam; 9 são novas
  (acontecer, aluno, banco, barulho, esquina, medo, sapo, vacina, vontade).
- `organizar(..., modo=)` em `base_unificada.py`: `mover` (padrão, Incluir), `copiar` ou `link`.

### Regras de rótulo (valem para todas as bases; decididas com o usuário)
Ficam em `normalizar_rotulo()` e `UNIR`, em `base_unificada.py`:
  - tudo em minúsculas (Windows não diferencia caixa em pastas; "Bom dia"/"Bom Dia" são o mesmo sinal);
  - sinônimos com `/` viram ` - ` ("ele - você"); `?`, ponto final e aspas removidos/trocados;
  - **variantes numeradas são sinais diferentes** (variantes regionais) e ficam em pastas
    separadas no formato `(n)`: `feliz (1)/`, `feliz (2)/`, `azul (3)/`;
  - grafias da mesma palavra unidas ("quarta feira" → `quarta-feira/`);
  - `^`, `~`, `´` (acentos do português) e `libras` **são sinais válidos**;
  - rótulo começando com `_` é proibido nas bases (reservado para classes próprias, como `_nada`).
- ⚠️ A numeração das variantes é **da base de origem**: `feliz (1)` de outra base não é
  necessariamente o mesmo sinal que `feliz (1)` da Incluir Tecnologia. Conferir ao adicionar bases.
- Excluídos e metadados ficam **fora** de `dados/videos/`: qualquer pasta lá dentro (inclusive
  `_excluidos`) seria treinada como classe.

## Estrutura do código
| Arquivo | Papel |
|---|---|
| `comum.py` | Tudo que o treino e a câmera precisam fazer **igual**: extrator MediaPipe, limpeza, features, reamostragem, modelo |
| `1_extrair.py` | Roda o MediaPipe em cada vídeo e salva os landmarks brutos em `dados/landmarks/<classe>/<video>.npz` (cache; pula o que já existe) |
| `2_treinar.py` | Carrega o cache, aplica augmentation, treina e salva `modelo_sinais.pt` (pesos + classes + n_features + duração da janela) |
| `3_tempo_real.py` | Webcam com janela deslizante + suavização das predições |
| `GUIA.md` | Explicação conceitual completa do pipeline |
| `preparacao/base_unificada.py` | Núcleo comum: leva qualquer base para a pasta unificada `dados/videos/` |
| `preparacao/<fonte>_pre_tratamento.py` | Adaptador de cada base (rodar antes do passo 1; sem `--executar` é dry run) |

```bash
pip install -r requirements.txt
python preparacao/incluir_tecnologia_pre_tratamento.py --origem incluir_tecnologia.tar.gz --executar
python preparacao/minds_libras_pre_tratamento.py --executar
python 1_extrair.py --videos dados/videos --saida dados/landmarks
python 2_treinar.py --landmarks dados/landmarks
python 3_tempo_real.py --modelo modelo_sinais.pt
```

## Decisões já tomadas (e por quê)
- **MediaPipe com a API Tasks** (`HandLandmarker` + `PoseLandmarker`, modo VIDEO).
  O `mp.solutions` (Holistic/Hands) **não existe mais** desde o mediapipe 0.10.3x (o venv usa
  1.2.3, onde a API Tasks foi testada): não usar.
  Os modelos `.task` são baixados automaticamente para `modelos_mediapipe/`.
- **Landmarks em vez de pixels**: invariância a fundo, luz e pessoa; viável com poucos dados.
- **Features guiadas pela fonologia da Libras** (por frame: 226; com velocidade: 452):
  - pose (nariz, ombros, cotovelos, punhos) normalizada pelo centro e largura dos ombros;
  - por mão: posição no referencial do corpo (ponto de articulação) + forma relativa ao
    punho escalada pelo tamanho da mão (configuração de mão) + flag de presença;
  - x multiplicado pela razão de aspecto; mãos em slots fixos (esq/dir);
  - lacunas de detecção de até 6 frames são interpoladas; o início e o fim sem mãos são recortados
    (`recortar_sem_maos`) e depois o início e o fim **parados** (`recortar_parado`: velocidade dos
    punhos da pose em larguras de ombro, piso 0,01, mantém 5–95% do movimento acumulado). O
    segundo existe porque no MINDS a mão em repouso continua visível (Kinect: 5 s → ~2,8 s).
- **Tempo**: reamostragem para 32 passos + velocidade (diferença entre frames).
- **Modelo** `ModeloSinais`: LayerNorm+Linear → Conv1D residual → GRU bidirecional →
  pooling média+máx → Linear. A GRU foi escolhida em vez de Transformer por causa
  do volume pequeno de dados.
- **Augmentation** nos landmarks brutos: espelhamento com troca de mãos, recorte temporal
  de 75–100%, rotação de ±12°, esticamento de ±10%, ruído, apagar a mão em ~5% dos frames.
- **Tempo real**: janela = mediana da duração dos sinais de treino; o sinal só é aceito
  depois de 4 predições iguais seguidas com confiança ≥ `--limiar` (0.7); o buffer é limpo
  ao aceitar. Classes com `_` no nome (ex.: `_nada`) são treinadas, mas nunca exibidas.

## ⚠️ Problema em aberto: 1069 classes, quase todas com 1 vídeo
O pipeline atual foi pensado para poucas classes com vários vídeos. Com ~1 exemplo por
classe:
1. **Validação**: `dividir()` põe os sinalizantes `--val-sinalizadores` (padrão `minds_11 minds_12`)
   só na validação (~200 vídeos, 20 classes do MINDS, pessoa nunca vista) e, nos vídeos sem
   sinalizador (Incluir), usa `--frac-val` por classe, o que dá só ~13 vídeos (`round(2 × 0.2) = 0`).
   Ainda em aberto: separar **1 vídeo de cada uma das 104 classes da Incluir com 2+ vídeos**.
2. **Classificação softmax com 1069 saídas e ~1 exemplo cada** tende a decorar. Na câmera,
   a probabilidade se espalha entre muitas classes e raramente passa de 0.7.
3. **Generalizar para outra pessoa** (quem está na câmera) é o maior risco: o modelo só viu
   um sinalizante por sinal (exceto as 20 classes do MINDS, com 10 pessoas no treino).
4. O custo de treino cresce com o nº de classes: `repeticoes` (32) amostras por classe por época
   = ~34 mil amostras/época. Na GPU, ~5–9 s por época (ver "Desempenho medido").

### Caminhos a avaliar (discutir com o usuário antes de implementar)
- **Baseline sem treino: k-NN com DTW** sobre as features. Funciona com 1 exemplo por
  classe e serve de referência.
- **Metric learning / few-shot**: treinar a rede para produzir um *embedding* (triplet
  loss, ArcFace ou redes prototípicas) e classificar pelo vizinho mais próximo. É a
  abordagem padrão para "muitas classes, pouquíssimos exemplos", e permite adicionar
  sinais novos sem re-treinar.
- **Avaliação realista**: gravar 2–3 vídeos próprios para um subconjunto (ex.: 50 sinais)
  e usar só como teste; reportar top-1 e top-5.
- **Reduzir o escopo do tempo real**: um vocabulário ativo menor (ex.: 20–100 sinais)
  facilita o reconhecimento ao vivo.
- **Pré-treino em datasets públicos de Libras** com vários sinalizantes por sinal
  (ex.: V-Librasil, MINDS-Libras). O dataset atual vem do canal Incluir Tecnologia, não deles.
- Adicionar uma classe `_nada` (pessoa parada ou gesticulando sem sinalizar).

## Estado / avisos
- **Treino com Incluir + MINDS Canon (2026-11-16)**, Kinect fora, validação = sinalizantes 11 e 12:
  early stopping na época 102 (8,4 min, RTX 5060, 4,9 s/época com `--lote 128`). Checkpoint:
  **pessoas novas top-1 86,0% / top-5 98,5%** (200 vídeos); Incluir 7/13. Todos os erros do MINDS
  caem em outra classe do MINDS: na prática é um problema de 20 classes, não de 1078.
  Erros concentrados por pessoa: `5` 0/10 (11 e 12 fazem o sinal na altura do ombro/rosto; os do
  treino, no peito), `maçã` 0/5 (12), `filho` 1/5 (11), `aproveitar` 1/5 (12). Com só 2 pessoas
  na validação o número oscila muito entre épocas (67–83%). Confiança média: acertos 0,77,
  erros 0,34; só 120 de 179 acertos passam do limiar 0,7 da câmera.
  Extração completa (3545 vídeos, Canon 1080p incluso): 50 min a ~149 frames/s nesta máquina.
- Validado com dados reais (2026-11-13):
  - `1_extrair.py`: **extração completa feita** (1187 vídeos, 0 erros, `dados/landmarks/`, 64 MB).
    Pose em 100% dos frames, mãos em ~20–80% (a parte sem mãos é a pessoa parada no início e no
    fim, removida por `recortar_sem_maos`). 13 vídeos ficaram com mão em menos de 30% dos frames.
  - `2_treinar.py`: roda de ponta a ponta na GPU com os dados reais; o treino chega a ~99% de
    acurácia de treino em 5 épocas (decora, como esperado com ~1 vídeo por classe).
  - `3_tempo_real.py`: o loop foi testado com um vídeo no lugar da webcam (prevê a classe certa);
    **webcam real ainda não testada**. ~13–14 fps em 1080p: capturar a câmera em 640×480.
- **GPU**: o venv usa `torch 2.15.0+cu132` (o `pip install torch` padrão no Windows é só-CPU;
  ver `requirements.txt`). `2_treinar.py` escolhe `cuda` sozinho (`--dispositivo`), usa
  `--workers` processos no DataLoader (padrão `min(8, núcleos-1)`) e salva os pesos na CPU,
  então o checkpoint abre em máquinas sem GPU (`3_tempo_real.py` carrega com `map_location="cpu"`).
- Corrigido em 2026-11-16 no `2_treinar.py`:
  - checkpoint e early stopping pela **perda de validação** (antes: `acc_val >` estrito, que
    salvava um modelo das primeiras épocas, com confiança ~25% na câmera);
  - relatório final com top-1/top-5 por grupo (pessoas novas / outros / total) e os pares mais
    confundidos; a matriz de confusão só é impressa com ≤ 30 classes;
  - `DatasetSinais`: exatamente `--repeticoes` (32) amostras por classe por época (~34,5 mil);
    classe com mais vídeos que isso sorteia um vídeo por amostra (antes `alvo = max(...)` subiria
    para ~48 com o MINDS em todas as 1078 classes).

### Desempenho medido (i9-14900HX + RTX 4080 Laptop 12 GB, 2026-11-13)
| Etapa | Antes | Agora |
|---|---|---|
| Extração (`1_extrair.py`, 16 processos) | ~2,5–3 h estimado (1 processo, ~13 frames/s) | **9,3 min** (~196 frames/s) |
| Época de treino, CPU, 1 processo | ~340 s (com o bug de amostras) | 85 s |
| Época de treino, GPU, `--workers 8–12` | — | **~5–9 s** (`--lote 128`: ~4,5 s) |
| 150 épocas | ~14 h | **~13–20 min** (~11 min com `--lote 128`) |
- O gargalo NÃO é a GPU: o passo leva ~11 ms com lote 32 ou 256 (custo fixo de disparar
  operações; VRAM pico ~150 MB). O que pesa é a CPU: preparar os dados (~0,8 ms/amostra em
  Python) e despachar as operações. Trocar de placa muda pouco; mais núcleos e lote maior, sim.
- A 1ª época é mais lenta (~20 s): no Windows os processos do DataLoader são criados do zero.
- Câmera: remover o sufixo `(n)` das variantes só na exibição ("feliz (2)" → "feliz").
- Em Libras, expressões faciais são gramaticais e ainda não estão nas features
  (próximo passo possível: subconjunto do `FaceLandmarker`).

## Convenções
- Qualquer mudança de pré-processamento vai em `comum.py` e vale para o treino **e** a
  câmera (evitar training-serving skew).
- Mudou as features? Atualize `N_FEATURES` e re-treine. O cache de landmarks brutos
  continua válido; só re-extraia se mudar o que o MediaPipe salva.
- Dividir dados sempre por vídeo, nunca por frame ou janela.
- Mudança na organização/rótulos do dataset vai nos scripts de `preparacao/`, nunca à mão, para
  continuar reprodutível a partir dos pacotes originais. Regra que vale para todas as bases fica
  em `base_unificada.py`; o que é específico de uma base, no adaptador dela. Mudou um rótulo?
  Re-extraia os landmarks só daquela classe (o cache em `dados/landmarks/` é indexado pelo
  nome da pasta).
