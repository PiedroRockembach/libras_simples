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
  - lacunas de detecção de até 6 frames são interpoladas; o início e o fim sem mãos são recortados.
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
1. **Validação quase inexistente hoje, mas possível.** Com `--frac-val 0.2`,
   `dividir_por_video` faz `round(2 × 0.2) = 0` para classes com 2 vídeos e sobram ~13 vídeos de
   validação. Separando **1 vídeo de cada uma das 104 classes com 2+ vídeos** há uma validação
   honesta (gravação nunca vista; não necessariamente outra pessoa). Validar com cópias
   aumentadas do mesmo vídeo é otimista demais.
2. **Classificação softmax com 1069 saídas e ~1 exemplo cada** tende a decorar. Na câmera,
   a probabilidade se espalha entre muitas classes e raramente passa de 0.7.
3. **Generalizar para outra pessoa** (quem está na câmera) é o maior risco: o modelo só viu
   um sinalizante por sinal.
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
- **Bugs conhecidos no `2_treinar.py` (ainda não corrigidos):**
  - Checkpoint só é salvo com `acc_val > melhor` (estrito). Com validação pequena a acurácia
    satura nas primeiras épocas e o arquivo salvo fica sendo esse modelo quase sem treino
    (confiança ~25% na câmera). Escolher o checkpoint pela perda de validação.
  - A matriz de confusão final seria impressa com 1069×1069: trocar por top-1/top-5 e pares mais
    confundidos.
  - Divisão de validação: hoje só 13 vídeos (ver item 1 do problema em aberto).
- Corrigido: `DatasetSinais` usava `alvo = max(vídeos por classe) × repeticoes` (uma classe com
  4 vídeos fazia TODAS terem 128 amostras: ~137 mil/época). Agora `alvo = max(repeticoes,
  max vídeos)`: ~34 mil/época.

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
