"""
Passo 1 — roda o MediaPipe em todos os vídeos e salva os landmarks em cache.

Estrutura esperada (a pasta unificada criada pelos scripts de preparacao/):
    dados/videos/<classe>/<fonte>__<id>.mp4

Por que salvar landmarks "brutos" e não as features finais?
O MediaPipe é a etapa cara (segundos por vídeo). Se você quiser mudar a
normalização ou as features depois, só re-treina, sem reprocessar os vídeos.

Por que vários processos?
O MediaPipe roda na CPU (no Windows ele não usa a GPU) e um vídeo não depende
do outro: é um problema "embaraçosamente paralelo". Cada processo pega um
vídeo por vez. Usamos PROCESSOS e não threads porque em Python as threads
disputam o GIL; processos rodam de fato em paralelo, cada um no seu núcleo.

Uso:  python 1_extrair.py --videos dados/videos --saida dados/landmarks [--processos 16]
"""
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np

from comum import ExtratorLandmarks, baixar_modelos

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


def processar(origem, destino):
    """Roda num processo separado: extrai, salva e devolve um resumo (ou o erro)."""
    try:
        pose, me, md, fps, w, h = extrair_video(origem)
        if len(pose) == 0:
            return origem, None, "não consegui ler o vídeo"
        # Escrita atômica: grava num temporário e renomeia. Se o processo morrer no
        # meio, não sobra um .npz pela metade que seria tratado como "já processado".
        tmp = destino + ".tmp"
        with open(tmp, "wb") as f:
            np.savez_compressed(f, pose=pose, mao_esq=me, mao_dir=md,
                                fps=fps, largura=w, altura=h)
        os.replace(tmp, destino)
        taxa = float(np.mean(~np.isnan(me[:, 0, 0]) | ~np.isnan(md[:, 0, 0])))
        return origem, (len(pose), taxa), None
    except Exception as e:                       # um vídeo ruim não derruba os outros
        return origem, None, f"{type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="dados/videos")
    ap.add_argument("--saida", default="dados/landmarks")
    # O MediaPipe já usa algumas threads por processo: metade dos núcleos evita
    # que os processos fiquem disputando a CPU.
    ap.add_argument("--processos", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    args = ap.parse_args()

    tarefas = []
    for classe in sorted(os.listdir(args.videos)):
        pasta = os.path.join(args.videos, classe)
        if not os.path.isdir(pasta):
            continue
        for nome in sorted(os.listdir(pasta)):
            if not nome.lower().endswith(EXTENSOES):
                continue
            destino = os.path.join(args.saida, classe, os.path.splitext(nome)[0] + ".npz")
            if os.path.exists(destino):
                continue                         # cache: já processado
            os.makedirs(os.path.dirname(destino), exist_ok=True)
            tarefas.append((os.path.join(pasta, nome), destino))

    if not tarefas:
        print("Nada a fazer: todos os vídeos já estão no cache.")
        return
    # Baixa os modelos ANTES de abrir os processos (senão todos baixariam juntos
    # e escreveriam o mesmo arquivo ao mesmo tempo).
    baixar_modelos()
    print(f"{len(tarefas)} vídeos a processar com {args.processos} processos...")

    t0, feitos, frames, erros = time.time(), 0, 0, 0
    with ProcessPoolExecutor(max_workers=args.processos) as pool:
        futuros = [pool.submit(processar, o, d) for o, d in tarefas]
        for fut in as_completed(futuros):
            origem, res, erro = fut.result()
            feitos += 1
            nome = os.path.relpath(origem, args.videos)
            if erro:
                erros += 1
                print(f"  [erro] {nome}: {erro}")
                continue
            n, taxa = res
            frames += n
            aviso = "  [aviso] poucas detecções de mão" if taxa < 0.3 else ""
            dt = time.time() - t0
            print(f"[{feitos}/{len(tarefas)}] {nome}: {n} frames, mão em {taxa:.0%} | "
                  f"{frames / dt:.0f} frames/s, faltam ~{dt / feitos * (len(tarefas) - feitos) / 60:.0f} min"
                  + aviso)
    print(f"\nConcluído em {(time.time() - t0) / 60:.1f} min: {feitos - erros} ok, {erros} erros.")


if __name__ == "__main__":
    main()
