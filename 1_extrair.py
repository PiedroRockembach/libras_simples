"""
Passo 1 — roda o MediaPipe em todos os vídeos e salva os landmarks em cache.

Estrutura esperada:
    dados/videos/<classe>/<qualquer_nome>.mp4
    ex.: dados/videos/obrigado/v1.mp4, dados/videos/oi/pessoa2.mp4

Por que salvar landmarks "brutos" e não as features finais?
O MediaPipe é a etapa cara (segundos por vídeo). Se você quiser mudar a
normalização ou as features depois, só re-treina, sem reprocessar os vídeos.

Uso:  python 1_extrair.py --videos dados/videos --saida dados/landmarks
"""
import argparse
import os

import cv2
import numpy as np

from comum import ExtratorLandmarks

EXTENSOES = (".mp4", ".mov", ".avi", ".mkv", ".webm")


def extrair_video(caminho):
    cap = cv2.VideoCapture(caminho)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    largura = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    altura = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # Um extrator novo por vídeo: o rastreamento do vídeo anterior não deve
    # "vazar" para o primeiro frame do próximo.
    ext = ExtratorLandmarks()
    poses, mes, mds = [], [], []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        p, me, md = ext.processar(frame, i * 1000.0 / fps)
        poses.append(p); mes.append(me); mds.append(md)
        i += 1
    cap.release()
    ext.fechar()
    return (np.array(poses), np.array(mes), np.array(mds), fps, largura, altura)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="dados/videos")
    ap.add_argument("--saida", default="dados/landmarks")
    args = ap.parse_args()

    for classe in sorted(os.listdir(args.videos)):
        pasta = os.path.join(args.videos, classe)
        if not os.path.isdir(pasta):
            continue
        os.makedirs(os.path.join(args.saida, classe), exist_ok=True)
        for nome in sorted(os.listdir(pasta)):
            if not nome.lower().endswith(EXTENSOES):
                continue
            destino = os.path.join(args.saida, classe, os.path.splitext(nome)[0] + ".npz")
            if os.path.exists(destino):
                continue                     # cache: já processado
            pose, me, md, fps, w, h = extrair_video(os.path.join(pasta, nome))
            if len(pose) == 0:
                print(f"  [aviso] não consegui ler {nome}")
                continue
            np.savez_compressed(destino, pose=pose, mao_esq=me, mao_dir=md,
                                fps=fps, largura=w, altura=h)
            taxa = np.mean(~np.isnan(me[:, 0, 0]) | ~np.isnan(md[:, 0, 0]))
            print(f"{classe}/{nome}: {len(pose)} frames, mão detectada em {taxa:.0%}")
            if taxa < 0.5:
                print("  [aviso] poucas detecções de mão: confira iluminação/enquadramento")


if __name__ == "__main__":
    main()
