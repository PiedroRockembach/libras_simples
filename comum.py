"""
comum.py — código compartilhado entre extração, treino e tempo real.

REGRA DE OURO: o pré-processamento usado no treino e na câmera precisa ser
EXATAMENTE o mesmo. Se você normalizar de um jeito no treino e de outro na
inferência, o modelo vê "outro idioma" em tempo real e erra tudo
(isso tem nome: training-serving skew). Por isso tudo fica neste arquivo.

Uma SEQUÊNCIA é um dict de arrays com o mesmo número de frames T:
    pose      (T, 33, 3)  pontos do corpo              \
    mao_esq   (T, 21, 3)  mão no slot esquerdo           |  coordenadas de imagem
    mao_dir   (T, 21, 3)  mão no slot direito            |  (x, y em [0,1]; NaN = não detectado)
    rosto     (T, 20, 3)  pontos de referência do rosto /
    expressao (T, 52)     blendshapes do rosto (0 a 1; NaN = rosto não detectado)
Guardar tudo num dict (e não em tuplas) deixa os recortes no tempo e a augmentation valerem para
todas as partes de uma vez: uma modalidade nova entra sem mudar a assinatura das funções.
"""
import os
import urllib.request

import numpy as np
import torch
import torch.nn as nn

# ----------------------------------------------------------------------------
# Configuração
# ----------------------------------------------------------------------------
N_FRAMES = 32          # todo sinal é reamostrado para 32 passos de tempo
MAX_LACUNA = 6         # buracos de detecção de até 6 frames são interpolados
# Versão do cache de landmarks (dados/landmarks/*.npz). Suba quando mudar O QUE o MediaPipe
# salva: o 1_extrair.py re-extrai os arquivos de versão menor. 1 = pose+mãos; 2 = + rosto.
VERSAO_CACHE = 2

# Pontos da pose que usamos: nariz, ombros, cotovelos, punhos.
# Pernas e quadril não dizem nada sobre o sinal e só adicionam ruído.
POSE_IDX = [0, 11, 12, 13, 14, 15, 16]
# Pares esquerda/direita da pose (usados para espelhar o corpo na augmentation)
POSE_PARES = [(11, 12), (13, 14), (15, 16), (1, 4), (2, 5), (3, 6), (7, 8), (9, 10)]

# Rosto: dos 478 pontos do FaceLandmarker guardamos só 20 de referência. Os 478 deixariam o cache
# ~20x maior, e a EXPRESSÃO já vem resumida nos blendshapes; os pontos servem para saber ONDE estão
# as partes do rosto (para a posição da mão) e como a cabeça está inclinada/virada.
ROSTO_IDX = [10, 168, 1, 13, 14, 152,      # centro: testa, entre os olhos, ponta do nariz,
                                           #   lábio superior, lábio inferior, queixo
             33, 263, 133, 362,            # cantos externos e internos dos olhos
             105, 334,                     # sobrancelhas
             61, 291,                      # cantos da boca
             234, 454,                     # laterais do rosto (bochechas)
             127, 356,                     # têmporas
             172, 397]                     # mandíbula
ROSTO_PARES = [(6, 7), (8, 9), (10, 11), (12, 13), (14, 15), (16, 17), (18, 19)]  # índices em ROSTO_IDX
R_NARIZ, R_LARG_A, R_LARG_B = 2, 14, 15    # origem e largura do referencial do rosto
# Blendshapes do FaceLandmarker, na ordem em que o modelo devolve (conferida com o mediapipe 1.2.3).
EXPRESSOES = [
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft",
    "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen", "jawRight", "mouthClose",
    "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight", "mouthFunnel",
    "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight",
    "mouthPucker", "mouthRight", "mouthRollLower", "mouthRollUpper", "mouthShrugLower",
    "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight"]
# Ao espelhar a imagem, o "Left" de um blendshape vira o "Right" (e o _neutral não serve de feature).
EXPRESSOES_ESPELHO = [EXPRESSOES.index(n.replace("Left", "#").replace("Right", "Left").replace("#", "Right"))
                      for n in EXPRESSOES]
EXPRESSOES_USADAS = [i for i, n in enumerate(EXPRESSOES) if n != "_neutral"]
# Pontos da mão comparados com o rosto: punho e as 5 pontas dos dedos.
MAO_PONTOS_ROSTO = [0, 4, 8, 12, 16, 20]
DIST_SEM_CONTATO = 3.0   # "distância mão-rosto" quando falta a mão ou o rosto (3 larguras de rosto)

PONTOS = ("pose", "mao_esq", "mao_dir", "rosto")   # arrays (T, N, 3) em coordenadas de imagem
PARTES = PONTOS + ("expressao",)
FORMAS = {"pose": (33, 3), "mao_esq": (21, 3), "mao_dir": (21, 3),
          "rosto": (len(ROSTO_IDX), 3), "expressao": (len(EXPRESSOES),)}

MODELOS = {
    "mao": ("hand_landmarker.task",
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/latest/hand_landmarker.task"),
    "pose": ("pose_landmarker_lite.task",
             "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"),
    "rosto": ("face_landmarker.task",
              "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
              "face_landmarker/float16/latest/face_landmarker.task"),
}


def baixar_modelos(pasta="modelos_mediapipe"):
    """A API Tasks do MediaPipe não vem com os modelos no pip: baixamos os .task."""
    os.makedirs(pasta, exist_ok=True)
    caminhos = {}
    for chave, (nome, url) in MODELOS.items():
        destino = os.path.join(pasta, nome)
        if not os.path.exists(destino):
            print(f"Baixando {nome}...")
            urllib.request.urlretrieve(url, destino)
        caminhos[chave] = destino
    return caminhos


# ----------------------------------------------------------------------------
# 1) Extração de landmarks com MediaPipe (API Tasks)
# ----------------------------------------------------------------------------
class ExtratorLandmarks:
    """
    Roda HandLandmarker + PoseLandmarker + FaceLandmarker em modo VIDEO.

    Por que modo VIDEO e não IMAGE? No modo VIDEO o MediaPipe usa o frame
    anterior para RASTREAR a mão em vez de detectá-la do zero a cada frame:
    fica mais rápido e os pontos tremem menos. A exigência é que os
    timestamps sejam sempre crescentes.
    """

    def __init__(self, pasta_modelos="modelos_mediapipe"):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        self._mp = mp
        cam = baixar_modelos(pasta_modelos)
        modo = vision.RunningMode.VIDEO
        self.maos = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=cam["mao"]),
                running_mode=modo, num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5))
        self.pose = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=cam["pose"]),
                running_mode=modo, num_poses=1))
        # output_face_blendshapes: os 52 coeficientes de expressão (sobrancelha, boca, bochecha...)
        self.rosto = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=cam["rosto"]),
                running_mode=modo, num_faces=1, output_face_blendshapes=True))
        self._ultimo_ts = -1

    def processar(self, frame_bgr, timestamp_ms):
        """Retorna um frame da sequência: dict com as PARTES (sem o eixo T); NaN = não detectado."""
        import cv2
        ts = max(int(timestamp_ms), self._ultimo_ts + 1)   # garante ordem crescente
        self._ultimo_ts = ts
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)    # MediaPipe espera RGB
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)

        res_maos = self.maos.detect_for_video(img, ts)
        res_pose = self.pose.detect_for_video(img, ts)
        res_rosto = self.rosto.detect_for_video(img, ts)
        out = {k: np.full(f, np.nan, np.float32) for k, f in FORMAS.items()}

        if res_pose.pose_landmarks:
            out["pose"][:] = [[p.x, p.y, p.z] for p in res_pose.pose_landmarks[0]]

        # Cada mão vai para um "slot" fixo (esquerda/direita). Se a ordem das mãos
        # mudasse entre frames, o modelo veria a mão "pulando" de posição no vetor.
        slots = {"Left": out["mao_esq"], "Right": out["mao_dir"]}
        for lms, lado in zip(res_maos.hand_landmarks, res_maos.handedness):
            nome = lado[0].category_name
            if not np.isnan(slots[nome][0, 0]):              # slot já ocupado:
                nome = "Right" if nome == "Left" else "Left"  # usa o outro
            slots[nome][:] = [[p.x, p.y, p.z] for p in lms]

        if res_rosto.face_landmarks:
            lms = res_rosto.face_landmarks[0]
            out["rosto"][:] = [[lms[i].x, lms[i].y, lms[i].z] for i in ROSTO_IDX]
            if res_rosto.face_blendshapes:
                out["expressao"][:] = [c.score for c in res_rosto.face_blendshapes[0]]
        return out

    def fechar(self):
        self.maos.close()
        self.pose.close()
        self.rosto.close()


def empilhar(frames):
    """Lista de frames (dicts) -> sequência (dict de arrays com eixo T)."""
    if not frames:
        return {k: np.zeros((0, *f), np.float32) for k, f in FORMAS.items()}
    return {k: np.stack([f[k] for f in frames]) for k in PARTES}


def carregar_npz(caminho):
    """
    Lê um arquivo do cache. Devolve (sequência, metadados {fps, largura, altura, versao}).
    Cache antigo (versão 1, sem rosto): o rosto vem todo NaN = "não detectado", então o arquivo
    ainda serve, só sem a informação do rosto. Rode o 1_extrair.py para atualizar.
    """
    d = np.load(caminho)
    T = len(d["pose"])
    seq = {"pose": d["pose"], "mao_esq": d["mao_esq"], "mao_dir": d["mao_dir"]}
    for k in ("rosto", "expressao"):
        seq[k] = d[k] if k in d.files else np.full((T, *FORMAS[k]), np.nan, np.float32)
    meta = {"fps": float(d["fps"]), "largura": float(d["largura"]), "altura": float(d["altura"]),
            "versao": int(d["versao"]) if "versao" in d.files else 1}
    return seq, meta


# ----------------------------------------------------------------------------
# 2) Limpeza da sequência bruta
# ----------------------------------------------------------------------------
def fatiar(seq, a, b):
    """Recorte no tempo aplicado a TODAS as partes juntas (continuam alinhadas frame a frame)."""
    return {k: v[a:b] for k, v in seq.items()}


def interpolar_lacunas(seq, max_lacuna=MAX_LACUNA):
    """
    seq: (T, N, D) ou (T, N) com NaN onde não houve detecção.
    Com movimento rápido a mão borra e o MediaPipe "perde" ela por 1-3 frames.
    Buracos curtos com detecção dos dois lados são preenchidos por interpolação
    linear; buracos longos significam que a mão realmente saiu (ficam NaN).
    """
    seq = seq.copy()
    presente = ~np.isnan(seq.reshape(len(seq), -1)[:, 0])
    idx = np.where(presente)[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= max_lacuna + 1:
            for t in range(a + 1, b):
                w = (t - a) / (b - a)
                seq[t] = (1 - w) * seq[a] + w * seq[b]
    return seq


def corrigir_aspecto(seq, largura, altura):
    """
    O MediaPipe devolve x em [0,1] da LARGURA e y em [0,1] da ALTURA.
    Num vídeo 16:9, uma distância de 0.1 em x é ~1.8x maior que 0.1 em y.
    Multiplicar x pela razão de aspecto deixa as duas direções na mesma escala,
    senão rotações e distâncias ficam distorcidas. (Os blendshapes não são posição: ficam iguais.)
    """
    r = largura / altura
    out = dict(seq)
    for k in PONTOS:
        out[k] = seq[k].copy()
        out[k][..., 0] *= r
    return out


def limpar(seq, largura, altura):
    """Correção de aspecto + interpolação de lacunas em todas as partes. Treino E câmera."""
    seq = corrigir_aspecto(seq, largura, altura)
    return {k: interpolar_lacunas(v) for k, v in seq.items()}


def recortar_sem_maos(seq, margem=2):
    """Remove o início/fim do vídeo em que nenhuma mão aparece (pessoa parada)."""
    alguma = ~np.isnan(seq["mao_esq"][:, 0, 0]) | ~np.isnan(seq["mao_dir"][:, 0, 0])
    idx = np.where(alguma)[0]
    if len(idx) == 0:
        return seq
    return fatiar(seq, max(idx[0] - margem, 0), min(idx[-1] + margem + 1, len(alguma)))


def recortar_parado(seq, limiar=0.01, ini=0.05, fim=0.95, margem=3, suav=5):
    """
    Remove o início/fim em que a pessoa está PARADA, mesmo com as mãos visíveis.

    recortar_sem_maos só funciona quando as mãos saem do quadro no repouso (Incluir Tecnologia).
    No MINDS (sobretudo no Kinect, de enquadramento mais aberto) a mão abaixada continua visível e
    detectada: sem este recorte, o sinal ocupa ~40% dos 32 passos e o resto é pessoa parada.

    Como funciona: velocidade por frame dos punhos da POSE (15, 16; detectados em ~100% dos frames),
    em larguras de ombro por frame (independe de resolução e distância da câmera), suavizada por média
    móvel de `suav` frames. Tira-se um piso `limiar` (tremor do detector com a pessoa parada) e
    mantém-se o trecho entre `ini` e `fim` do movimento ACUMULADO: parado não acumula nada, então o
    repouso sai; uma pausa NO MEIO do sinal fica, porque está entre movimentos.
    Calibrado em 2026-11-16: Kinect 5,0 s -> ~2,8 s; Canon 2,9 -> 2,6 s; Incluir ~inalterado.
    """
    pose = seq["pose"]
    T = len(pose)
    if T < 8:
        return seq
    largura = np.nanmedian(np.linalg.norm(pose[:, 11, :2] - pose[:, 12, :2], axis=1))
    if not np.isfinite(largura) or largura <= 0:
        return seq
    p = pose[:, [15, 16], :2] / largura                      # (T, 2 punhos, xy)
    k = np.ones(suav) / suav
    p = np.stack([np.convolve(np.pad(p[:, i, j], suav // 2, mode="edge"), k, "valid")
                  for i in range(2) for j in range(2)], axis=1).reshape(T, 2, 2)
    v = np.linalg.norm(np.diff(p, axis=0), axis=2).max(axis=1)  # punho mais rápido em cada frame
    v = np.maximum(v - limiar, 0)
    if not np.isfinite(v).all() or v.sum() <= 0:
        return seq
    acum = np.concatenate([[0.0], np.cumsum(v)]) / v.sum()   # (T,) de 0 a 1
    a = max(int(np.searchsorted(acum, ini)) - margem, 0)
    b = min(int(np.searchsorted(acum, fim)) + 1 + margem, T)
    return fatiar(seq, a, b)


# ----------------------------------------------------------------------------
# 3) Features — pensadas a partir dos parâmetros da Libras
# ----------------------------------------------------------------------------
def _referencial(centro, escala):
    """Preenche frames sem detecção com a mediana do vídeo; devolve centro (T,2) e escala (T,1)."""
    centro, escala = centro.copy(), escala.copy()
    falta = ~np.isfinite(escala)
    if falta.all():
        return None, None
    if falta.any():
        centro[falta] = np.nanmedian(centro, axis=0)
        escala[falta] = np.nanmedian(escala)
    return centro, np.maximum(escala, 1e-3)[:, None]


def features_por_frame(seq):
    """
    Entrada: sequência já com aspecto corrigido. Saída: (T, 344)

    Os parâmetros fonológicos da Libras guiam o que extraímos:
      - CONFIGURAÇÃO DE MÃO  -> formato da mão relativo ao punho, escalado pelo
                                tamanho da mão (63 valores por mão)
      - PONTO DE ARTICULAÇÃO -> onde a mão está em relação ao CORPO (42 por mão)
                                e em relação ao ROSTO (12 por mão + distância mínima)
      - MOVIMENTO            -> vem da sequência no tempo + velocidades (depois)
      - ORIENTAÇÃO           -> fica implícita no formato 3D da mão
      - EXPRESSÕES NÃO-MANUAIS -> 51 blendshapes do rosto + forma do rosto (40)
      - Contexto do corpo    -> nariz, ombros, cotovelos, punhos (14)
    """
    pose, rosto = seq["pose"], seq["rosto"]
    T = len(pose)
    # Referencial do CORPO: origem no meio dos ombros, unidade = largura dos ombros.
    # Assim a feature não muda se a pessoa está mais perto/longe ou mais à esquerda.
    oe, od = pose[:, 11, :2], pose[:, 12, :2]
    centro, escala = _referencial((oe + od) / 2, np.linalg.norm(oe - od, axis=1))
    if centro is None:                    # sem pose nenhuma: valores padrão
        centro, escala = np.full((T, 2), 0.5, np.float32), np.full((T, 1), 0.3, np.float32)

    # Referencial do ROSTO: origem na ponta do nariz, unidade = largura do rosto. É o que permite
    # distinguir sinais que só mudam pelo lugar do rosto que a mão toca (queixo, bochecha, testa).
    tem_rosto = ~np.isnan(rosto[:, 0, 0])
    c_rosto, e_rosto = _referencial(rosto[:, R_NARIZ, :2],
                                    np.linalg.norm(rosto[:, R_LARG_A, :2] - rosto[:, R_LARG_B, :2], axis=1))
    if c_rosto is None:                   # nenhum frame com rosto: tudo vira "ausente"
        c_rosto, e_rosto = np.full((T, 2), np.nan, np.float32), np.ones((T, 1), np.float32)

    f_pose = (pose[:, POSE_IDX, :2] - centro[:, None]) / escala[:, None]
    partes = [f_pose.reshape(T, -1)]

    for mao in (seq["mao_esq"], seq["mao_dir"]):
        presente = ~np.isnan(mao[:, 0, 0])
        # ponto de articulação: posição da mão no referencial do corpo
        pos = (mao[:, :, :2] - centro[:, None]) / escala[:, None]
        # configuração de mão: forma relativa ao punho, escala = punho→base do dedo médio
        tam = np.linalg.norm(mao[:, 9, :2] - mao[:, 0, :2], axis=1)
        tam = np.maximum(np.nan_to_num(tam, nan=1.0), 1e-3)[:, None, None]
        forma = (mao - mao[:, :1]) / tam
        # mão em relação ao rosto: punho + pontas dos dedos no referencial do rosto, e a menor
        # distância entre a mão e um ponto do rosto ("encostou?"). Sem mão ou sem rosto, a
        # distância vira DIST_SEM_CONTATO e não 0: zero significaria "encostando".
        mao_rosto = (mao[:, MAO_PONTOS_ROSTO, :2] - c_rosto[:, None]) / e_rosto[:, None]
        dif = mao[:, :, None, :2] - rosto[:, None, :, :2]                 # (T, 21, 20, 2)
        dist = np.linalg.norm(dif, axis=-1).reshape(T, -1)
        dist = np.where(np.isnan(dist), np.inf, dist).min(axis=1) / e_rosto[:, 0]
        dist = np.minimum(np.where(np.isfinite(dist), dist, DIST_SEM_CONTATO), DIST_SEM_CONTATO)
        partes += [pos.reshape(T, -1), forma.reshape(T, -1),
                   presente[:, None].astype(np.float32),
                   mao_rosto.reshape(T, -1), dist[:, None]]

    # Expressões não-manuais: o rosto no próprio referencial (inclinação/rotação da cabeça, boca,
    # sobrancelhas) + os blendshapes, que já descrevem a expressão em coeficientes de 0 a 1.
    f_rosto = (rosto[:, :, :2] - c_rosto[:, None]) / e_rosto[:, None]
    partes += [f_rosto.reshape(T, -1), seq["expressao"][:, EXPRESSOES_USADAS],
               tem_rosto[:, None].astype(np.float32)]

    # Parte ausente vira zero; as flags "presente" avisam o modelo que o zero é ausência.
    return np.nan_to_num(np.concatenate(partes, axis=1), nan=0.0).astype(np.float32)


def reamostrar(x, n=N_FRAMES):
    """
    (T, F) -> (n, F) por interpolação linear no tempo.
    Pessoas fazem o mesmo sinal em velocidades diferentes; reamostrar para um
    número fixo de passos normaliza a duração e permite empilhar em lotes.
    """
    T = len(x)
    if T == 1:
        return np.repeat(x, n, axis=0)
    t_orig = np.linspace(0, 1, T)
    t_novo = np.linspace(0, 1, n)
    return np.stack([np.interp(t_novo, t_orig, x[:, j]) for j in range(x.shape[1])],
                    axis=1).astype(np.float32)


def adicionar_velocidade(x):
    """Concatena a diferença entre frames: o MOVIMENTO explícito. (n,F)->(n,2F)"""
    v = np.diff(x, axis=0, prepend=x[:1])
    return np.concatenate([x, v], axis=1)


def preparar_sequencia(seq):
    """Pipeline completo de features. Usado no treino E na câmera."""
    f = features_por_frame(seq)
    return adicionar_velocidade(reamostrar(f))


N_FEATURES_MAO = 42 + 63 + 1 + 2 * len(MAO_PONTOS_ROSTO) + 1           # 119
N_FEATURES_ROSTO = 2 * len(ROSTO_IDX) + len(EXPRESSOES_USADAS) + 1     # 92
N_FEATURES = 2 * (2 * len(POSE_IDX) + 2 * N_FEATURES_MAO + N_FEATURES_ROSTO)   # 2 x 344 = 688


# ----------------------------------------------------------------------------
# 4) Modelo temporal
# ----------------------------------------------------------------------------
class ModeloSinais(nn.Module):
    """
    Conv1D  -> enxerga padrões curtos de movimento (alguns frames vizinhos)
    GRU bidirecional -> aprende a ORDEM dos eventos no sinal inteiro
    Pooling média+máximo -> resume a sequência num vetor
    Linear -> uma pontuação por classe
    """

    def __init__(self, n_features, n_classes, oculto=128, dropout=0.3):
        super().__init__()
        self.entrada = nn.Sequential(
            nn.LayerNorm(n_features),
            nn.Linear(n_features, oculto), nn.GELU(), nn.Dropout(dropout))
        self.conv = nn.Sequential(
            nn.Conv1d(oculto, oculto, kernel_size=5, padding=2),
            nn.BatchNorm1d(oculto), nn.GELU())
        self.gru = nn.GRU(oculto, oculto // 2, batch_first=True, bidirectional=True)
        self.saida = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * oculto, n_classes))

    def forward(self, x):                       # x: (lote, tempo, features)
        h = self.entrada(x)
        h = h + self.conv(h.transpose(1, 2)).transpose(1, 2)   # conexão residual
        h, _ = self.gru(h)
        h = torch.cat([h.mean(dim=1), h.amax(dim=1)], dim=1)
        return self.saida(h)
