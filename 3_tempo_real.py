"""
Passo 3 — reconhecimento em tempo real pela webcam.

Uso:  python 3_tempo_real.py --modelo modelo_sinais.pt --camera 0
Teclas: q = sair, c = limpar a frase

Ideia central: JANELA DESLIZANTE. Guardamos os landmarks dos últimos N segundos
(N = duração típica de um sinal no treino) e, a cada poucos frames, passamos
essa janela pelo MESMO pré-processamento do treino e pelo modelo.
"""
import argparse
import time
from collections import deque

import cv2
import numpy as np
import torch

from comum import (ExtratorLandmarks, ModeloSinais, corrigir_aspecto,
                   interpolar_lacunas, preparar_sequencia)

CONEXOES_MAO = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
                (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


def desenhar_mao(img, mao, cor):
    if np.isnan(mao[0, 0]):
        return
    h, w = img.shape[:2]
    pts = [(int(x * w), int(y * h)) for x, y, _ in mao]
    for a, b in CONEXOES_MAO:
        cv2.line(img, pts[a], pts[b], cor, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="modelo_sinais.pt")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--limiar", type=float, default=0.7,
                    help="confiança mínima para aceitar um sinal")
    ap.add_argument("--estabilidade", type=int, default=4,
                    help="predições seguidas iguais exigidas")
    args = ap.parse_args()

    # map_location="cpu": a inferência de 1 janela por vez é leve e roda bem na CPU;
    # também garante que um modelo treinado na GPU abra numa máquina sem GPU.
    ck = torch.load(args.modelo, map_location="cpu")
    classes = ck["classes"]
    modelo = ModeloSinais(ck["n_features"], len(classes))
    modelo.load_state_dict(ck["estado"])
    modelo.eval()
    janela_s = ck["duracao_janela_s"]

    ext = ExtratorLandmarks()
    cap = cv2.VideoCapture(args.camera)
    buffer = deque()                        # (tempo, pose, mao_esq, mao_dir)
    historico = deque(maxlen=args.estabilidade)
    frase, ultimo_aceito, t0, n_frame = [], None, time.time(), 0
    texto_atual = ""

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        agora = time.time() - t0
        h, w = frame.shape[:2]
        # Processa o frame SEM espelhar, igual aos vídeos de treino.
        # (Espelhamos só a imagem exibida, mais natural para quem se vê.)
        pose, me, md = ext.processar(frame, agora * 1000)
        buffer.append((agora, pose, me, md))
        while buffer and agora - buffer[0][0] > janela_s:
            buffer.popleft()

        n_frame += 1
        tem_mao = sum(not np.isnan(b[2][0, 0]) or not np.isnan(b[3][0, 0]) for b in buffer)
        # Só prevê a cada 2 frames, com janela cheia e mãos visíveis na maior parte
        if n_frame % 2 == 0 and len(buffer) >= 8 and tem_mao > 0.6 * len(buffer):
            P = np.array([b[1] for b in buffer]); E = np.array([b[2] for b in buffer])
            D = np.array([b[3] for b in buffer])
            P, E, D = corrigir_aspecto(P, E, D, w, h)
            E, D, P = interpolar_lacunas(E), interpolar_lacunas(D), interpolar_lacunas(P)
            x = torch.from_numpy(preparar_sequencia(P, E, D))[None]
            with torch.no_grad():
                prob = torch.softmax(modelo(x), dim=1)[0].numpy()
            k = int(prob.argmax())
            historico.append(k)
            texto_atual = f"{classes[k]} {prob[k]:.0%}"

            # Suavização: só aceita se a MESMA classe venceu várias vezes seguidas
            # com confiança alta. Uma predição isolada é muito instável.
            estavel = len(historico) == historico.maxlen and len(set(historico)) == 1
            if estavel and prob[k] >= args.limiar:
                nome = classes[k]
                if not nome.startswith("_") and nome != ultimo_aceito:
                    frase.append(nome)
                    buffer.clear()          # recomeça: evita contar o mesmo sinal 2x
                    historico.clear()
                ultimo_aceito = nome
        elif tem_mao == 0:
            ultimo_aceito = None            # mãos abaixadas: permite repetir o sinal

        tela = frame.copy()
        desenhar_mao(tela, me, (0, 200, 255))
        desenhar_mao(tela, md, (255, 120, 0))
        tela = cv2.flip(tela, 1)
        cv2.rectangle(tela, (0, 0), (w, 70), (0, 0, 0), -1)
        cv2.putText(tela, texto_atual, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(tela, " ".join(frase[-6:]), (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (255, 255, 255), 2)
        cv2.imshow("Libras", tela)
        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord("q"):
            break
        if tecla == ord("c"):
            frase.clear()

    cap.release()
    ext.fechar()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
