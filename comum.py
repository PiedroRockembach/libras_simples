"""
comum.py — código compartilhado entre extração, treino e tempo real.

REGRA DE OURO: o pré-processamento usado no treino e na câmera precisa ser
EXATAMENTE o mesmo. Se você normalizar de um jeito no treino e de outro na
inferência, o modelo vê "outro idioma" em tempo real e erra tudo
(isso tem nome: training-serving skew). Por isso tudo fica neste arquivo.
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

# Pontos da pose que usamos: nariz, ombros, cotovelos, punhos.
# Pernas e quadril não dizem nada sobre o sinal e só adicionam ruído.
POSE_IDX = [0, 11, 12, 13, 14, 15, 16]
# Pares esquerda/direita da pose (usados para espelhar o corpo na augmentation)
POSE_PARES = [(11, 12), (13, 14), (15, 16), (1, 4), (2, 5), (3, 6), (7, 8), (9, 10)]

MODELOS = {
    "mao": ("hand_landmarker.task",
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/latest/hand_landmarker.task"),
    "pose": ("pose_landmarker_lite.task",
             "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"),
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
    Roda HandLandmarker + PoseLandmarker em modo VIDEO.

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
        self._ultimo_ts = -1

    def processar(self, frame_bgr, timestamp_ms):
        """Retorna (pose[33,3], mao_esq[21,3], mao_dir[21,3]); NaN = não detectado."""
        import cv2
        ts = max(int(timestamp_ms), self._ultimo_ts + 1)   # garante ordem crescente
        self._ultimo_ts = ts
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)    # MediaPipe espera RGB
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)

        res_maos = self.maos.detect_for_video(img, ts)
        res_pose = self.pose.detect_for_video(img, ts)

        pose = np.full((33, 3), np.nan, np.float32)
        if res_pose.pose_landmarks:
            pose[:] = [[p.x, p.y, p.z] for p in res_pose.pose_landmarks[0]]

        # Cada mão vai para um "slot" fixo (esquerda/direita). Se a ordem das mãos
        # mudasse entre frames, o modelo veria a mão "pulando" de posição no vetor.
        slots = {"Left": np.full((21, 3), np.nan, np.float32),
                 "Right": np.full((21, 3), np.nan, np.float32)}
        for lms, lado in zip(res_maos.hand_landmarks, res_maos.handedness):
            nome = lado[0].category_name
            if not np.isnan(slots[nome][0, 0]):              # slot já ocupado:
                nome = "Right" if nome == "Left" else "Left"  # usa o outro
            slots[nome][:] = [[p.x, p.y, p.z] for p in lms]
        return pose, slots["Left"], slots["Right"]

    def fechar(self):
        self.maos.close()
        self.pose.close()


# ----------------------------------------------------------------------------
# 2) Limpeza da sequência bruta
# ----------------------------------------------------------------------------
def interpolar_lacunas(seq, max_lacuna=MAX_LACUNA):
    """
    seq: (T, N, D) com NaN onde não houve detecção.
    Com movimento rápido a mão borra e o MediaPipe "perde" ela por 1-3 frames.
    Buracos curtos com detecção dos dois lados são preenchidos por interpolação
    linear; buracos longos significam que a mão realmente saiu (ficam NaN).
    """
    seq = seq.copy()
    presente = ~np.isnan(seq[:, 0, 0])
    idx = np.where(presente)[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= max_lacuna + 1:
            for t in range(a + 1, b):
                w = (t - a) / (b - a)
                seq[t] = (1 - w) * seq[a] + w * seq[b]
    return seq


def recortar_sem_maos(pose, me, md, margem=2):
    """Remove o início/fim do vídeo em que nenhuma mão aparece (pessoa parada)."""
    alguma = ~np.isnan(me[:, 0, 0]) | ~np.isnan(md[:, 0, 0])
    idx = np.where(alguma)[0]
    if len(idx) == 0:
        return pose, me, md
    a, b = max(idx[0] - margem, 0), min(idx[-1] + margem + 1, len(pose))
    return pose[a:b], me[a:b], md[a:b]


def recortar_parado(pose, me, md, limiar=0.01, ini=0.05, fim=0.95, margem=3, suav=5):
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
    T = len(pose)
    if T < 8:
        return pose, me, md
    largura = np.nanmedian(np.linalg.norm(pose[:, 11, :2] - pose[:, 12, :2], axis=1))
    if not np.isfinite(largura) or largura <= 0:
        return pose, me, md
    p = pose[:, [15, 16], :2] / largura                      # (T, 2 punhos, xy)
    k = np.ones(suav) / suav
    p = np.stack([np.convolve(np.pad(p[:, i, j], suav // 2, mode="edge"), k, "valid")
                  for i in range(2) for j in range(2)], axis=1).reshape(T, 2, 2)
    v = np.linalg.norm(np.diff(p, axis=0), axis=2).max(axis=1)  # punho mais rápido em cada frame
    v = np.maximum(v - limiar, 0)
    if not np.isfinite(v).all() or v.sum() <= 0:
        return pose, me, md
    acum = np.concatenate([[0.0], np.cumsum(v)]) / v.sum()   # (T,) de 0 a 1
    a = max(int(np.searchsorted(acum, ini)) - margem, 0)
    b = min(int(np.searchsorted(acum, fim)) + 1 + margem, T)
    return pose[a:b], me[a:b], md[a:b]


def corrigir_aspecto(pose, me, md, largura, altura):
    """
    O MediaPipe devolve x em [0,1] da LARGURA e y em [0,1] da ALTURA.
    Num vídeo 16:9, uma distância de 0.1 em x é ~1.8x maior que 0.1 em y.
    Multiplicar x pela razão de aspecto deixa as duas direções na mesma escala,
    senão rotações e distâncias ficam distorcidas.
    """
    r = largura / altura
    out = []
    for a in (pose, me, md):
        a = a.copy()
        a[..., 0] *= r
        out.append(a)
    return out


# ----------------------------------------------------------------------------
# 3) Features — pensadas a partir dos parâmetros da Libras
# ----------------------------------------------------------------------------
def features_por_frame(pose, me, md):
    """
    Entrada: pose (T,33,3), mãos (T,21,3) já com aspecto corrigido.
    Saída: (T, 226)

    Os parâmetros fonológicos da Libras guiam o que extraímos:
      - CONFIGURAÇÃO DE MÃO  -> formato da mão relativo ao punho, escalado pelo
                                tamanho da mão (63 valores por mão)
      - PONTO DE ARTICULAÇÃO -> onde a mão está em relação ao CORPO (42 por mão)
      - MOVIMENTO            -> vem da sequência no tempo + velocidades (depois)
      - ORIENTAÇÃO           -> fica implícita no formato 3D da mão
      - Contexto do corpo    -> nariz, ombros, cotovelos, punhos (14)
    """
    T = len(pose)
    # Referencial do corpo: origem no meio dos ombros, unidade = largura dos ombros.
    # Assim a feature não muda se a pessoa está mais perto/longe ou mais à esquerda.
    oe, od = pose[:, 11, :2], pose[:, 12, :2]
    centro = (oe + od) / 2
    escala = np.linalg.norm(oe - od, axis=1)
    falta = np.isnan(escala)
    if falta.all():                       # sem pose nenhuma: valores padrão
        centro[:] = [0.5, 0.5]
        escala[:] = 0.3
    elif falta.any():                     # pose falhou em alguns frames
        centro[falta] = np.nanmedian(centro, axis=0)
        escala[falta] = np.nanmedian(escala)
    escala = np.maximum(escala, 1e-3)[:, None]

    f_pose = (pose[:, POSE_IDX, :2] - centro[:, None]) / escala[:, None]
    partes = [f_pose.reshape(T, -1)]

    for mao in (me, md):
        presente = ~np.isnan(mao[:, 0, 0])
        # ponto de articulação: posição da mão no referencial do corpo
        pos = (mao[:, :, :2] - centro[:, None]) / escala[:, None]
        # configuração de mão: forma relativa ao punho, escala = punho→base do dedo médio
        tam = np.linalg.norm(mao[:, 9, :2] - mao[:, 0, :2], axis=1)
        tam = np.maximum(np.nan_to_num(tam, nan=1.0), 1e-3)[:, None, None]
        forma = (mao - mao[:, :1]) / tam
        partes += [pos.reshape(T, -1), forma.reshape(T, -1),
                   presente[:, None].astype(np.float32)]

    # Mão ausente vira zero; a flag "presente" avisa o modelo que o zero é ausência.
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


def preparar_sequencia(pose, me, md):
    """Pipeline completo de features. Usado no treino E na câmera."""
    f = features_por_frame(pose, me, md)
    return adicionar_velocidade(reamostrar(f))


N_FEATURES = 2 * (14 + 2 * (42 + 63 + 1))   # 452


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
