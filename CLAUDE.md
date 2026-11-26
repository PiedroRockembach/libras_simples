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

**vlibrasil**: V-LIBRASIL (Rodrigues e Zanchettin, CIn-UFPE, 2021), 1364 palavras x 3 articuladores
(4086 vídeos, chroma key, 1080p, 30/60 fps). **Fonte: cópia do Kaggle `davimedio01/v-librasil`**
(decisão do usuário, 2026-11-24; o site da UFPE ficou fora do ar; o IEEE DataPort exige assinatura).
Licença declarada no Kaggle: CC BY-NC-ND 4.0; os termos originais proíbem redistribuir e uso
comercial (dados fora do git). Adaptador [preparacao/vlibrasil_pre_tratamento.py](preparacao/vlibrasil_pre_tratamento.py):
- **Classe pelo catálogo**: o `annotations.csv` do Kaggle traz o `url_download` original da UFPE,
  o mesmo link do `catalogo_treino.csv`; o adaptador usa a classe que o catálogo decidiu. As ~203
  palavras ambíguas (várias formas conhecidas) ficam em `dados/brutos/vlibrasil`.
- **Sinalizador = articulador** (`vlibrasil_1..3`, do nome do arquivo); id = nome do vídeo na UFPE.
- Nomes de arquivo trocam `? \ "` por U+F03F/U+F05C/U+F022 (faixa privada); o adaptador desfaz.
- **Câmera lenta**: exportado a metade da velocidade (dissertação, p. 57). Medido: punhos ~2,7x
  mais lentos que na Incluir nas mesmas 12 palavras; com o `PASSO_FRAMES = {"vlibrasil": 2}` do
  `1_extrair.py` (1 de cada 2 frames, mesmo fps), cai para ~1,4x (ritmo dos intérpretes). Corrigir
  só o fator documentado (2x) foi decisão consciente: o resto varia por palavra (1,8x a 4,4x).
- Organizado em 2026-11-24: **3481 vídeos, 1161 classes** (203 somadas a classes existentes). A
  coluna `fps` do `annotations.csv` é o nº de frames, não fps. Faltam 3 vídeos (`error.csv`).
- Grafias unidas no `UNIR` ao integrar: `côco`→`coco` (classe da Incluir renomeada; cache movido),
  `melância`→`melancia`, `volêi`→`vôlei`, `comemorar (celebrar)`→`comemorar - celebrar`.
  `secretária` x `secretaria` ficam separadas (palavras diferentes; o aviso de duplicata é esperado).

### Catálogo unificado de treino (MALTA-LIBRAS + V-LIBRASIL, 2026-11-18)
[preparacao/catalogo_unificado.py](preparacao/catalogo_unificado.py) gera `dados/catalogo_treino.csv`:
uma linha por vídeo (nossos + fontes do repositório `../ISLR_LIBRAS`, do Malta Lab), só com as
classes que têm **>= 3 exemplos obteníveis** (`--min-videos`; vídeos da mesma pessoa contam,
decisão do usuário; Canon+Kinect da mesma execução = 1). Ainda **não baixa nada**: `status` =
`local` (já temos) / `baixar` (tem link) / `sem_link` (listado, não conta).
- Resultado: **1388 classes**, 7184 vídeos obteníveis (2777 já temos, 4407 a baixar).
  `confianca_classe`: **alta** = 255 classes (atingem o mínimo sem o V-LIBRASIL); **media** = 1133
  (precisam do V-LIBRASIL; 718 delas só existem nele).
- Regras: rótulo pela coluna `checked` do MALTA (a `lemma` é de lematizador e erra: "água"→"aguo");
  classe = (palavra, **variante**) pelo `group` anotado no MALTA; nossas classes ganham a variante
  pelo vídeo correspondente no "Youtube" do MALTA (= nossa Incluir; casado por título + altura;
  **não entra de novo**). Fonte sem variante (V-LIBRASIL) só une se a palavra tiver UMA forma
  conhecida. Variante nova de palavra com outras formas vira `palavra (vN)` (N = group do MALTA).
- Sem link nos CSVs: Acessibilidade v2 e UFSC Corpus (o Corpus usa os nomes de arquivo do
  SignBank, mas é outra gravação; casar por eles baixaria a duplicata errada). Spread the Sign
  com 2+ links para a palavra também fica sem link (não dá para saber qual é).
- Identidade dos 3 sinalizantes do V-LIBRASIL é desconhecida; links não testados (rede bloqueada
  no ambiente onde o catálogo foi gerado).

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
# máquina nova: baixa tudo e organiza em dados/videos/ (sem --executar, só mostra o plano)
python preparacao/baixar_videos.py --executar --organizar
# ...ou, com os pacotes já em disco:
python preparacao/incluir_tecnologia_pre_tratamento.py --origem incluir_tecnologia.tar.gz --executar
python preparacao/minds_libras_pre_tratamento.py --executar
python preparacao/vlibrasil_pre_tratamento.py --executar      # depois do catálogo (usa as classes dele)
python 1_extrair.py --videos dados/videos --saida dados/landmarks
python 2_treinar.py --landmarks dados/landmarks
python 3_tempo_real.py --modelo modelo_sinais.pt
```

### Download automático (`preparacao/baixar_videos.py`, 2026-11-18)
Reproduz os dados numa máquina nova. Brutos em `dados/brutos/<fonte>/` (fora do git), logs por
vídeo em `dados/brutos/_logs/<fonte>.csv`. Dry run por padrão; idempotente; `--limite N` para teste.
- `incluir_tecnologia`: YouTube via **yt-dlp**, pelos `video_id` de `dados/manifestos/incluir_tecnologia_original.csv`
  (só vídeo, mp4 ≤ 1080p). O sha256 pode diferir do pacote original (o YouTube reprocessa);
  se o `.tar.gz` existir, prefira-o.
- `minds_libras_rgb` / `minds_libras_rgbd`: API do **Zenodo** (2667329: 12 zips, 60 GB;
  4322984: 12 zips, 29 GB). Retoma downloads (HTTP Range), confere MD5, extrai **só** os vídeos RGB
  que o adaptador reconhece (usa os mesmos padrões `PADRAO_RGB/RGBD`) e apaga o zip
  (`--manter-zips`). Marcador `zips/<zip>.ok` evita baixar de novo.
- `vlibrasil`: **kagglehub** (`davimedio01/v-librasil`, público, sem login), dataset inteiro
  (~10,9 GB, ~7 min a ~28 MB/s em 2026-11-24) em `dados/brutos/vlibrasil/`. Manual: `curl -L -o
  v-librasil.zip https://www.kaggle.com/api/v1/datasets/download/davimedio01/v-librasil`.
- `catalogo` (ou uma fonte só, só MALTA): links `status=baixar` do `dados/catalogo_treino.csv`; nome do
  arquivo = o do link (único por fonte). **Ainda sem adaptador** para `dados/videos/`.
- `--organizar` chama os adaptadores: Incluir (move), V-LIBRASIL (move) e MINDS com `--modo mover` (opção nova do
  adaptador; no Windows, link simbólico exige administrador).
- Rede: verificação SSL pelo sistema operacional via **truststore** (o SignBank da UFSC não envia
  o certificado intermediário: o Python puro falha). No máx. 2 conexões por site; "disjuntor": 5
  falhas seguidas num site pulam o resto dele (rodar de novo tenta só o que faltou).
- Testado em 2026-11-18 (cópia isolada): YouTube, Acessibilidade v3, Spread the Sign, UFV e
  SignBank ok; retomada + MD5 ok; Zenodo aceita Range; extração seletiva ok. **Fora do ar nesse
  dia**: site do V-LIBRASIL (502 até na página inicial; por isso o Kaggle) e USP (handshake TLS
  sem resposta).

## Decisões já tomadas (e por quê)
- **MediaPipe com a API Tasks** (`HandLandmarker` + `PoseLandmarker` + `FaceLandmarker`, modo VIDEO).
  O `mp.solutions` (Holistic/Hands) **não existe mais** desde o mediapipe 0.10.3x (o venv usa
  1.2.3, onde a API Tasks foi testada): não usar.
  Os modelos `.task` são baixados automaticamente para `modelos_mediapipe/`.
- **Landmarks em vez de pixels**: invariância a fundo, luz e pessoa; viável com poucos dados.
- **Sequência = dict de partes** (`comum.py`): `pose` (T,33,3), `mao_esq`/`mao_dir` (T,21,3),
  `rosto` (T,20,3), `expressao` (T,52). Recortes no tempo (`fatiar`), aspecto e interpolação
  valem para todas as partes de uma vez; treino e câmera usam o mesmo `limpar()`.
- **Cache versionado** (`VERSAO_CACHE` em `comum.py`, campo `versao` no .npz): 1 = pose+mãos,
  **2 = + rosto** (2026-11-18). O `1_extrair.py` re-extrai sozinho os arquivos de versão menor;
  o `2_treinar.py` ainda lê os de versão 1, com o rosto "ausente" (e avisa).
- **Features guiadas pela fonologia da Libras** (por frame: 344; com velocidade: **688**):
  - pose (nariz, ombros, cotovelos, punhos) normalizada pelo centro e largura dos ombros;
  - por mão: posição no referencial do corpo (ponto de articulação) + forma relativa ao
    punho escalada pelo tamanho da mão (configuração de mão) + flag de presença
    + **punho e pontas dos dedos no referencial do rosto** (origem no nariz, unidade = largura
    do rosto) + **distância mínima mão-rosto** (sem mão/rosto = 3,0, nunca 0 = "encostando");
  - **rosto**: 20 pontos de referência (`ROSTO_IDX`: testa, olhos, sobrancelhas, nariz, boca,
    queixo, bochechas, têmporas, mandíbula) no referencial do rosto (inclinação/rotação da cabeça)
    + **51 blendshapes** (expressão: sobrancelhas, olhos, bochechas, boca; sem o `_neutral`)
    + flag de presença. Só 20 dos 478 pontos vão para o cache (os 478 o deixariam ~20x maior).
  - espelhamento troca também os pontos pareados do rosto e os blendshapes `...Left`/`...Right`;
  - limitação: a distância mão-rosto é 2D, então mão NA FRENTE do rosto conta como "encostando";
  - x multiplicado pela razão de aspecto; mãos em slots fixos (esq/dir);
  - bases em câmera lenta: `PASSO_FRAMES` no `1_extrair.py` (V-LIBRASIL: 1 de cada 2 frames;
    campo `passo` no .npz). A correção fica na extração: treino e câmera não mudam;
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

## Branch `testes`: vocabulário de 20 classes (decisão do usuário, 2026-11-26)
Nesta branch só os **20 sinais do MINDS** são usados, as únicas classes com volume real (55–65
vídeos de 11–16 pessoas somando as 3 bases; a 21ª classe tem 6). A lista fica em
[classes_ativas.txt](classes_ativas.txt), lida por `ler_classes_ativas()` (`comum.py`) no
`1_extrair.py` e no `2_treinar.py` (`--classes`, padrão = esse arquivo; `--classes ""` = todas).
O `3_tempo_real.py` não muda: usa as classes gravadas no modelo. Os vídeos do MINDS precisam estar
na máquina (`baixar_videos.py --fontes minds_libras_rgb`); sem eles sobram 13 classes com 31 vídeos.
- **Exatamente 50 vídeos por classe no treino, o resto é teste** (decisão do usuário, 2026-11-26):
  `dividir_fixo()`, `--videos-por-classe 50` (0 = divisão antiga). Sai primeiro a pessoa de teste
  inteira (`--val-sinalizadores`, agora só `minds_12`: com 2 pessoas, classes de 55 vídeos ficariam
  com 45), depois 1 repetição por vez da pessoa do MINDS com mais vídeos na classe; Incluir e
  V-LIBRASIL ficam no treino (cada vídeo é uma pessoa a mais). Simulado pelos manifestos: treino
  1000 (50 x 20), teste 189 = 100 de `minds_12` (pessoa nova) + 89 repetições de pessoas do treino.
  O teste também escolhe o checkpoint (early stopping pela perda), o que deixa o número um pouco otimista.

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
- **Rosto adicionado em 2026-11-18** (ver "Decisões"). Validado nesta máquina: regressão com as
  features antigas idêntica (diferença 0), rosto detectado em 91–100% dos frames, ~9 ms/frame a mais
  na extração, tempo real ~16–20 fps. **Pendente**: re-extrair tudo (`1_extrair.py` refaz o cache
  v1) e re-treinar; comparar com o treino sem rosto (pessoas novas top-1 86,0%). O
  `modelo_sinais.pt` atual (452 features) é recusado pelo `3_tempo_real.py` até re-treinar.
  Possível melhoria: os blendshapes têm um "neutro" diferente por pessoa (ex.: `mouthShrugLower`
  alto no sinalizante da Incluir); subtrair uma linha de base por pessoa pode ajudar.

## Convenções
- Qualquer mudança de pré-processamento vai em `comum.py` e vale para o treino **e** a
  câmera (evitar training-serving skew).
- Mudou as features? Atualize `N_FEATURES` e re-treine. O cache de landmarks brutos
  continua válido; só re-extraia se mudar o que o MediaPipe salva, e nesse caso **suba
  `VERSAO_CACHE`** (o `1_extrair.py` refaz só os arquivos desatualizados).
- Dividir dados sempre por vídeo, nunca por frame ou janela.
- Mudança na organização/rótulos do dataset vai nos scripts de `preparacao/`, nunca à mão, para
  continuar reprodutível a partir dos pacotes originais. Regra que vale para todas as bases fica
  em `base_unificada.py`; o que é específico de uma base, no adaptador dela. Mudou um rótulo?
  Re-extraia os landmarks só daquela classe (o cache em `dados/landmarks/` é indexado pelo
  nome da pasta).
