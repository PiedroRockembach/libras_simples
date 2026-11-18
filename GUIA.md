# Reconhecimento de sinais de Libras com MediaPipe: guia conceitual

## O pipeline

```
base A (ex.: incluir_tecnologia.tar.gz)    base B    base C ...   (cada uma no seu formato)
   │  preparacao/<fonte>_pre_tratamento.py  (adaptador: lê o formato da base)
   │  preparacao/base_unificada.py          (núcleo comum: rótulos limpos, move, manifestos)
   ▼
dados/videos/<classe>/<fonte>__<id>.mp4     (pasta ÚNICA de classes, todas as bases juntas)
   │  1_extrair.py   (MediaPipe: mãos + pose → landmarks por frame, cache .npz)
   ▼
dados/landmarks/<classe>/*.npz
   │  2_treinar.py   (limpeza → augmentation → features → modelo temporal)
   ▼
modelo_sinais.pt
   │  3_tempo_real.py (janela deslizante na webcam → mesmo pré-processamento → modelo)
   ▼
texto na tela
```

`comum.py` guarda tudo o que treino e câmera precisam fazer igual.

## 0. Os dados: uma pasta única de classes para todas as bases

**Por que uma pasta unificada.** O maior problema deste projeto é ter poucos
exemplos por sinal (seção 4), e a solução mais forte é juntar várias bases de
dados. Mas cada base chega num formato diferente: planilha, pastas, nomes de
arquivo. Se cada uma ficasse organizada à sua maneira, "Bom Dia" de uma e "bom dia"
de outra virariam classes diferentes, e o modelo nunca saberia que são o mesmo
sinal. Por isso todas são convertidas para **um único lugar e um único padrão**:

```
dados/videos/<classe>/<fonte>__<id>.mp4   ← o 1_extrair.py lê só daqui
dados/manifestos/<fonte>.csv              ← de onde veio cada vídeo (+ a planilha original)
dados/manifestos/indice.csv               ← todas as bases juntas
dados/excluidos/<fonte>/                  ← o que ficou de fora, fora das classes
```

A preparação é dividida em duas partes, um padrão comum de software chamado
*adaptador*:

- **Núcleo comum** (`preparacao/base_unificada.py`): as regras de rótulo, a
  movimentação dos arquivos e os manifestos. É o mesmo código para todas as bases,
  e é isso que garante que todas sigam as **mesmas** regras.
- **Um adaptador por base** (`preparacao/<fonte>_pre_tratamento.py`): só sabe ler o
  formato daquela base. Adicionar uma base nova é escrever um adaptador novo; o
  resto já está pronto.

**O prefixo `<fonte>__` no nome do arquivo** (`incluir_tecnologia__m8F3o0Y39kA.mp4`)
cumpre três papéis:
- evita colisões, porque duas bases podem ter um vídeo chamado `001.mp4`;
- deixa cada execução mexer só nos arquivos da própria base;
- mantém a origem visível até o cache de landmarks. Isso permite uma avaliação
  importante: treinar com uma base e validar com outra, ou seja, com **outros
  sinalizantes**.

**Possíveis duplicatas.** Ao adicionar uma base, o núcleo compara os rótulos dela
com as classes que já existem, ignorando acentos, espaços e pontuação, e avisa se
encontrar casos como `guarda roupa` vs `guarda-roupa`. Quem decide se são o mesmo
sinal é uma pessoa: se forem, a regra entra na lista `UNIR`. A mesma cautela vale
para as variantes: `feliz (1)` de uma base não é necessariamente o mesmo sinal que
`feliz (1)` de outra, porque a numeração é de cada base.

### Base incluída: Incluir Tecnologia

Os vídeos vêm do canal **Incluir Tecnologia** (dicionário de Libras no YouTube):
cada vídeo mostra um sinal isolado, de 2 a 6 segundos, com a pessoa parada e as
mãos abaixadas no início e no fim. Depois da preparação são **1069 classes e 1187
vídeos**: 965 classes têm 1 vídeo e 104 têm de 2 a 4.

O pacote original traz os vídeos soltos numa pasta e o rótulo numa planilha
(`MANIFEST.csv`). O pipeline espera **uma pasta por classe**, porque assim o rótulo
fica no próprio caminho do arquivo e não depende de uma planilha.

### Regras de rótulo (valem para todas as bases)

Em rótulos, um detalhe pequeno muda o que o modelo aprende, por isso cada regra
tem um motivo:

| Problema no rótulo original | Regra | Por quê |
|---|---|---|
| "Bom dia" e "Bom Dia" | tudo em minúsculas | Para o modelo seriam duas classes diferentes para o **mesmo** sinal: ele seria punido por acertar. E o Windows nem deixa criar as duas pastas. |
| "Quarta feira" e "Quarta-feira" | grafias unidas | Mesmo caso: mesma palavra, mesmo sinal. |
| "Ele/Você", "Posso?", `Cheio "comida"` | `ele - você`, `posso`, `cheio (comida)` | `/ ? "` não são permitidos em nomes de pasta. |
| "Feliz (1)" e "Feliz (2)", "Azul 1/2/3" | pastas **separadas**, formato `(n)` | São sinais diferentes para a mesma palavra (variantes regionais, com outra configuração de mão). Juntá-los criaria uma classe com formas incompatíveis. |
| Músicas e textos sinalizados de 100 a 160 s | movidos para fora do dataset | Não são sinais isolados. Reamostrados para 32 passos, virariam ruído. |

Algumas classes parecem estranhas, mas são sinais legítimos: `^`, `~` e `´` são os
sinais dos acentos do português, e `libras` é o sinal da própria língua.

Rótulos começando com `_` são proibidos nas bases: esse prefixo fica reservado
para classes criadas por você, como `_nada` (seção 5).

**Reprodutibilidade.** A organização é feita por script, e não à mão, para que
qualquer pessoa possa refazer exatamente os mesmos dados a partir dos pacotes
originais. Os scripts são *idempotentes*: rodá-los de novo não muda nada. Se uma
regra de rótulo mudar, o vídeo é movido para a pasta nova sem reprocessar o
pacote. Os scripts também conferem o sha256 de cada vídeo, o que detecta arquivos
corrompidos ou trocados. Por padrão, só mostram o plano (*dry run*): nada é movido
sem `--executar`.

**Cuidado: toda pasta dentro de `dados/videos/` vira uma classe.** Por isso os
excluídos e os metadados ficam em `dados/excluidos/`, fora dela. Uma pasta
`_excluidos` lá dentro seria treinada como classe.

## 1. Por que landmarks e não pixels

Uma CNN de vídeo aprende direto dos pixels, mas precisa de milhares de vídeos e
também aprende o que não importa: fundo, roupa, luz, tom de pele. O MediaPipe já
foi treinado em milhões de imagens para achar 21 pontos 3D por mão e 33 pontos de
pose. Usando esses pontos você:

- reduz cada frame de ~900 mil números (640×480×3) para cerca de 200;
- fica independente de fundo, iluminação e de quem está sinalizando;
- consegue treinar com poucos vídeos e rodar em tempo real numa CPU.

Isso é **transfer learning por extração de features**: o modelo pesado (MediaPipe)
fica congelado e você treina só um modelo pequeno em cima da saída dele.

**API atual:** desde a versão 0.10.3x, o `pip install mediapipe` não traz mais o
`mp.solutions` (Holistic, Hands) usado na maioria dos tutoriais. O código usa a
API Tasks (`HandLandmarker` + `PoseLandmarker`) com arquivos `.task`, que são
baixados automaticamente. Ela foi testada no mediapipe 1.2.3.

**Quanto custa.** O MediaPipe roda na CPU (no Windows ele não usa a GPU) e processa
~13 frames/s por processo: os ~124 mil frames levariam de 2,5 a 3 horas num processo
só. Mas um vídeo não depende do outro, um problema "embaraçosamente paralelo": o
`1_extrair.py` processa vários vídeos ao mesmo tempo, um por **processo**. Em Python,
threads disputam o GIL e não rodam de fato em paralelo; processos rodam. Com 16
processos num i9-14900HX, a extração inteira levou **9 minutos**. E ela roda uma vez
só: os landmarks ficam em cache, e mudanças nas features só exigem re-treinar.

**Taxa de detecção.** O script imprime em quantos frames encontrou mão. Nesses
vídeos ela varia de ~20% a ~80%, e isso é normal: no início e no fim a pessoa está
parada com as mãos abaixadas, fora do quadro. Esses trechos são recortados
(`recortar_sem_maos`). Taxa baixa só é problema se a mão sumir *durante* o sinal.

**Modo VIDEO:** em vez de detectar a mão do zero a cada frame, o MediaPipe
rastreia a partir do frame anterior. Fica mais rápido e os pontos tremem menos.

## 2. Features guiadas pela fonologia da Libras

Um sinal de Libras se define por parâmetros. Cada um vira uma parte do vetor:

| Parâmetro | Feature |
|---|---|
| Configuração de mão | pontos da mão relativos ao punho, divididos pelo tamanho da mão |
| Ponto de articulação | posição da mão relativa ao centro dos ombros (dividida pela largura dos ombros) **e relativa ao rosto** (dividida pela largura do rosto), mais a distância mínima entre mão e rosto |
| Movimento | a sequência no tempo + velocidades (diferença entre frames) |
| Orientação | implícita na coordenada 3D da mão |
| Expressões não-manuais | 51 *blendshapes* do rosto + 20 pontos de referência do rosto no referencial dele |

**Por que o rosto.** Em Libras, o rosto entra de dois jeitos:
- **Como gramática.** As expressões não-manuais marcam pergunta, negação e
  intensidade, e alguns sinais só se distinguem por elas.
- **Como lugar.** Muitos sinais diferem só pelo ponto do rosto que a mão toca:
  queixo, bochecha, testa ou boca. O referencial do corpo (ombros) é grosseiro
  demais para isso. Por isso a posição da mão também é medida no **referencial do
  rosto** (origem no nariz, unidade = largura do rosto).

Além disso, o modelo recebe pronta a **distância mínima** entre a mão e o rosto. Ele
poderia aprender isso sozinho, mas com poucos exemplos por sinal vale entregar a
informação "encostou?" explicitamente. Sem mão ou sem rosto, essa distância vale
3,0 (longe), nunca 0: zero significaria "encostando". Uma limitação: a distância é
medida na imagem 2D, então a mão *na frente* do rosto também conta como encostando.

**Blendshapes em vez de pontos para a expressão.** O `FaceLandmarker` devolve 478
pontos e 52 *blendshapes*, coeficientes de 0 a 1 com nomes como `browInnerUp`
(sobrancelhas levantadas), `jawOpen`, `cheekPuff` e `mouthPucker`. Os blendshapes
já resumem a expressão de um jeito independente do tamanho e da posição do rosto.
Dos 478 pontos guardamos só 20 de referência: eles servem para saber *onde* estão
as partes do rosto e como a cabeça está inclinada. Os 478 deixariam o cache cerca
de 20× maior.

**Espelhamento com rosto.** Ao espelhar o vídeo na augmentation, não basta inverter x:
os pontos pareados (olho esquerdo ↔ direito) e os blendshapes `...Left` ↔ `...Right`
também trocam de lugar. Senão a imagem espelhada teria o sorriso do lado errado.

**Normalização.** A mesma pessoa, a 1 m ou a 2 m da câmera, gera coordenadas
completamente diferentes. Colocar a origem no meio dos ombros e usar a largura dos
ombros como unidade faz a feature descrever o *sinal*, não a câmera.

**Razão de aspecto.** O MediaPipe devolve x como fração da largura e y como fração
da altura. Num vídeo 16:9, 0,1 em x vale 1,8× mais que 0,1 em y. O código
multiplica x pela razão largura/altura antes de tudo.

**Slot fixo por mão.** A mão esquerda sempre ocupa as mesmas posições no vetor, e
a direita também. Se a ordem mudasse entre frames, o modelo veria a mão "teletransportar".

**Ausência explícita.** Mão não detectada vira zeros + uma flag `presente=0`.
Sem a flag, o modelo não teria como distinguir "mão na origem" de "mão ausente".

**Lacunas curtas.** Em movimento rápido a mão borra e o detector perde 1–3 frames.
Buracos de até 6 frames são interpolados; buracos maiores são ausência real.

## 3. A dimensão do tempo: o que diferencia sinal dinâmico de estático

Um classificador por frame só vê formas. "Oi" e "tchau" podem ter a mesma mão e
diferir só no movimento. Por isso a entrada do modelo é uma **sequência**.

**Reamostragem para 32 passos.** Cada pessoa sinaliza num ritmo, e cada vídeo tem
um fps. Interpolar a sequência para um número fixo de passos normaliza a duração,
permite empilhar em lotes e torna o modelo tolerante a velocidade.

**Velocidade explícita.** Depois de reamostrar, concatenamos `x[t] - x[t-1]`. A
rede poderia aprender isso sozinha, mas com poucos dados vale entregar pronto.

**Arquitetura (`ModeloSinais`):**

1. `LayerNorm + Linear`: projeta as 688 features num espaço de 128.
2. `Conv1D` (kernel 5): detecta padrões locais, como "mão subindo" ou "dedo fechando",
   em ~5 frames vizinhos. Conexão residual para não atrapalhar o gradiente.
3. `GRU bidirecional`: lê a sequência nos dois sentidos e aprende a *ordem* dos
   eventos (subir e depois girar ≠ girar e depois subir). Bidirecional é válido
   porque classificamos uma janela inteira, não frame a frame.
4. Pooling média + máximo: a média resume o sinal todo; o máximo captura o
   momento mais marcante.
5. `Linear`: uma pontuação por classe.

**Por que GRU e não Transformer?** Transformers brilham com muitos dados. Com
dezenas de vídeos, um modelo recorrente pequeno (~150 mil parâmetros) com viés
indutivo de sequência generaliza melhor. GRU é uma LSTM simplificada (2 portas em
vez de 3): menos parâmetros, desempenho parecido.

**Alternativa sem rede neural:** com 1–3 vídeos por classe, *k-NN com DTW*
(Dynamic Time Warping) compara a sequência da câmera com cada exemplo, alinhando as
velocidades. Vale como baseline.

## 4. O problema central: poucos dados

Com 1 vídeo por classe, qualquer modelo decora em vez de aprender. É a situação
de 965 das 1069 classes deste dataset. Além disso, um canal só significa poucos
sinalizantes: o modelo aprende o sinal *daquela* pessoa, e quem estiver na câmera
será diferente. As defesas, em ordem de impacto:

1. **Mais dados.** Mire em 10+ vídeos por classe, de 3+ pessoas, com roupas,
   lugares e distâncias diferentes. Nenhuma técnica substitui isso.
2. **Data augmentation** (nova a cada época, sobre os landmarks):
   - *espelhamento* com troca esquerda↔direita (equivale a um sinalizante canhoto);
   - *recorte temporal* de 75–100% do sinal (na câmera a janela nunca fica perfeitamente alinhada);
   - rotação de ±12°, esticamento de ±10% e ruído gaussiano (câmera torta, corpos diferentes, tremor);
   - apagar a mão em ~5% dos frames (simula falhas do detector).
3. **Regularização:** dropout, weight decay (AdamW), label smoothing.
4. **Early stopping:** salva o melhor modelo na validação e para quando ela não melhora.
5. **Mudar a formulação do problema:** com muitas classes e pouquíssimos exemplos,
   a abordagem padrão é *metric learning*. A rede aprende um *embedding* onde vídeos
   do mesmo sinal ficam próximos, e a classificação é pelo vizinho mais próximo.
   Isso permite adicionar sinais novos sem re-treinar (veja o CLAUDE.md).

**Validação dividida por vídeo, nunca por frame.** Janelas do mesmo vídeo são
quase idênticas: se uma vai para o treino e outra para a validação, a acurácia
fica inflada. Classe com 1 vídeo não tem o que separar. Aqui, porém, 104 classes
têm 2 ou mais vídeos: separar um vídeo de cada uma dá uma validação honesta, com
gravações que o modelo nunca viu. Ela mede generalização para *outra gravação*, não
necessariamente para *outra pessoa*.

**Qual métrica escolhe o checkpoint.** Com uma validação pequena, a acurácia é uma
métrica em *degraus*: com 13 vídeos, ela só pode valer 0%, 7,7%, 15%... Ela satura
cedo, empata, e "salvar quando melhorar" passa a guardar um modelo das primeiras
épocas. A perda de validação (*cross-entropy*) é contínua: continua caindo enquanto
o modelo fica mais confiante no acerto, então é melhor critério de escolha.
**Atenção:** o código atual ainda usa a acurácia (bug listado no CLAUDE.md).

**Top-5 além de top-1.** Com mais de mil classes, errar o 1º lugar mas ter a
resposta certa entre as 5 primeiras já é útil (por exemplo, para sugerir opções na
tela). Também mostra se o modelo está "perto" ou perdido.

**Balanceamento:** cada classe aparece o mesmo número de vezes por época, senão
a classe com mais vídeos domina.

**Onde está o gargalo do treino (e por que a GPU não é tudo).** Cada passo de
treino tem duas partes: *preparar o lote* (augmentation + features, em Python na
CPU) e *calcular a rede* (na GPU). Este modelo é pequeno (490 mil parâmetros) e,
medido numa RTX 4080, o passo leva os mesmos ~11 ms com lote de 32 ou de 256: a
GPU passa quase todo o tempo esperando. Quem manda é a CPU. Por isso:
- **`--workers`**: vários processos preparam os próximos lotes enquanto a GPU treina
  o atual (um pipeline). Cada processo recebe uma cópia do dataset e precisa da
  própria semente aleatória, senão todos sorteariam as mesmas augmentations.
- **`--lote` maior** aproveita a GPU ociosa: menos passos por época com o mesmo
  custo por passo. Lotes maiores mudam um pouco a dinâmica do treino (menos
  atualizações por época), então compare a acurácia antes de adotar.
- Uma placa mais potente quase não muda o tempo; mais núcleos de CPU, sim.

Na prática (i9-14900HX + RTX 4080 Laptop): uma época leva ~85 s na CPU e **~5–9 s**
na GPU com 8–12 workers.

**Matriz de confusão:** mostra *quais* sinais se confundem. Normalmente são pares
com a mesma configuração de mão e movimento parecido, e esses pedem mais exemplos.
Com 1069 classes, a matriz inteira é ilegível: o útil é listar os pares mais
confundidos.

## 5. Tempo real: janela deslizante

No treino, cada vídeo é um sinal recortado. Na câmera chega um fluxo contínuo,
sem marcação de início e fim. A solução:

- **Buffer dos últimos N segundos**, com N = mediana da duração dos sinais de
  treino (salva no checkpoint).
- A cada 2 frames, se há mãos visíveis em mais de 60% da janela, ela passa pelo
  **mesmo `preparar_sequencia`** do treino.
- **Suavização:** o sinal só é aceito se a mesma classe vencer 4 predições
  seguidas com confiança ≥ 70%. Uma predição isolada oscila muito.
- Ao aceitar um sinal, o buffer é limpo para não contá-lo duas vezes. Abaixar as
  mãos libera repetir a mesma palavra.

**A classe `_nada` é essencial.** Softmax sempre distribui 100% entre as classes
que conhece: sem uma classe de "não é sinal", o modelo vai "ver" sinais quando
você coça o nariz. Grave vídeos parados, ajeitando o cabelo, gesticulando à toa.
Classes com `_` no nome são treinadas, mas nunca exibidas.

**Muitas classes diluem a confiança.** Com 1069 saídas, a probabilidade tende a se
espalhar, e raramente uma classe passa de 70%. Um *vocabulário ativo* menor na
câmera (20 a 100 sinais) torna o reconhecimento ao vivo bem mais confiável. Já as
variantes (`feliz (1)`, `feliz (2)`) são classes diferentes no treino, mas na tela
basta mostrar "feliz".

**Resolução.** O MediaPipe roda a ~13–14 frames/s em 1080p. Capturar a webcam em
640×480 deixa folga para o tempo real.

**Espelhamento da câmera:** o frame é processado sem espelhar (igual aos vídeos
de treino); só a imagem exibida é espelhada. Se seus vídeos foram gravados com a
câmera frontal do celular (que às vezes espelha), o espelhamento na augmentation
cobre essa diferença.

## 6. Próximos passos

- **Linha de base da expressão por pessoa:** o rosto "neutro" muda de uma pessoa
  para outra (algumas têm a boca naturalmente mais caída, por exemplo). Subtrair o
  neutro de cada pessoa dos blendshapes deixaria só a *variação* da expressão, que é
  o que carrega significado.
- **Segmentação melhor:** detectar início e fim pela velocidade das mãos, em vez
  de janela fixa.
- **Mais dados por classe:** o gargalo aqui não é o número de sinais, mas os
  exemplos por sinal. Com vários vídeos e sinalizantes por classe (milhares de
  vídeos no total), um Transformer pequeno sobre os landmarks passa a compensar.
- **Avaliação realista:** gravar 2 ou 3 vídeos próprios de um subconjunto (ex.: 50
  sinais) e usá-los só como teste, reportando top-1 e top-5.
- **Datasets públicos de Libras** para pré-treino, como MINDS-Libras e V-Librasil.

## Como rodar

```bash
pip install -r requirements.txt
# com GPU NVIDIA no Windows, troque o torch só-CPU pela versão CUDA (veja requirements.txt):
pip install torch --index-url https://download.pytorch.org/whl/cu132
# máquina nova: baixa os vídeos de todas as bases e organiza em dados/videos/
# (sem --executar, só mostra o plano e o espaço em disco; --limite 2 faz um teste rápido)
python preparacao/baixar_videos.py --executar --organizar
# ...ou, com o pacote já em disco, uma vez por base:
python preparacao/incluir_tecnologia_pre_tratamento.py --origem incluir_tecnologia.tar.gz --executar
python 1_extrair.py --videos dados/videos --saida dados/landmarks
python 2_treinar.py --landmarks dados/landmarks
python 3_tempo_real.py --modelo modelo_sinais.pt
```

Ajustes na câmera: `--limiar 0.8` (mais rigoroso) e `--estabilidade 6` (mais estável, mais lento).
