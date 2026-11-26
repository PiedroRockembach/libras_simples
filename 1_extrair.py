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

from comum import VERSAO_CACHE, ExtratorLandmarks, baixar_modelos, empilhar

EXTENSOES = (".mp4", ".mov", ".avi", ".mkv", ".webm")
SEP = "__"

# Bases gravadas em CÂMERA LENTA: processa 1 de cada N frames. O V-LIBRASIL foi exportado a metade
# da velocidade (dissertação, p. 57; medido em 2026-11-24: punhos ~2,7x mais lentos que na Incluir
# nas mesmas palavras, o resto é ritmo da pessoa). Pulando 1 frame a cada 2, a sequência volta à
# velocidade real com o mesmo fps do arquivo: duração, velocidades e o recortar_parado ficam
# comparáveis às outras bases, sem mexer no vídeo nem no treino/câmera.
PASSO_FRAMES = {"vlibrasil": 2}


def extrair_video(caminho, passo=1):
    """Devolve (sequência, fps, largura, altura); sequência = dict com pose, mãos, rosto, expressão."""
    cap = cv2.VideoCapture(caminho)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    largura = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    altura = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # Um extrator novo por vídeo: o rastreamento do vídeo anterior não deve
    # "vazar" para o primeiro frame do próximo.
    ext = ExtratorLandmarks()
    frames, i = [], 0
    while True:
        ok = cap.grab()                      # grab sem decodificar: pular frame sai quase de graça
        if not ok:
            break
        if i % passo == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            frames.append(ext.processar(frame, i * 1000.0 / fps))
        i += 1
    cap.release()
    ext.fechar()
    return empilhar(frames), fps, largura, altura


def versao_do_cache(caminho):
    """Versão gravada num .npz do cache (1 = antigo, sem o campo). None = ilegível/corrompido."""
    try:
        with np.load(caminho) as d:
            return int(d["versao"]) if "versao" in d.files else 1
    except Exception:
        return None


def processar(origem, destino):
    """Roda num processo separado: extrai, salva e devolve um resumo (ou o erro)."""
    try:
        passo = PASSO_FRAMES.get(os.path.basename(origem).split(SEP, 1)[0], 1)
        seq, fps, w, h = extrair_video(origem, passo)
        if len(seq["pose"]) == 0:
            return origem, None, "não consegui ler o vídeo"
        # Escrita atômica: grava num temporário e renomeia. Se o processo morrer no
        # meio, não sobra um .npz pela metade que seria tratado como "já processado".
        tmp = destino + ".tmp"
        with open(tmp, "wb") as f:
            np.savez_compressed(f, **seq, fps=fps, largura=w, altura=h, versao=VERSAO_CACHE,
                                passo=passo)
        os.replace(tmp, destino)
        tem = lambda a: ~np.isnan(a.reshape(len(a), -1)[:, 0])   # noqa: E731
        taxa_mao = float(np.mean(tem(seq["mao_esq"]) | tem(seq["mao_dir"])))
        taxa_rosto = float(np.mean(tem(seq["rosto"])))
        return origem, (len(seq["pose"]), taxa_mao, taxa_rosto), None
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

    tarefas, desatualizados = [], 0
    for classe in sorted(os.listdir(args.videos)):
        pasta = os.path.join(args.videos, classe)
        if not os.path.isdir(pasta):
            continue
        for nome in sorted(os.listdir(pasta)):
            if not nome.lower().endswith(EXTENSOES):
                continue
            destino = os.path.join(args.saida, classe, os.path.splitext(nome)[0] + ".npz")
            if os.path.exists(destino):
                # Cache: pula o que já foi processado COM a versão atual. Arquivo de uma versão
                # antiga (ex.: sem rosto) é refeito, senão o treino ficaria sem a informação nova.
                v = versao_do_cache(destino)
                if v is not None and v >= VERSAO_CACHE:
                    continue
                desatualizados += 1
            os.makedirs(os.path.dirname(destino), exist_ok=True)
            tarefas.append((os.path.join(pasta, nome), destino))

    if not tarefas:
        print("Nada a fazer: todos os vídeos já estão no cache (versão atual).")
        return
    # Baixa os modelos ANTES de abrir os processos (senão todos baixariam juntos
    # e escreveriam o mesmo arquivo ao mesmo tempo).
    baixar_modelos()
    print(f"{len(tarefas)} vídeos a processar ({desatualizados} de cache antigo, versão < "
          f"{VERSAO_CACHE}) com {args.processos} processos...")

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
            n, taxa, taxa_rosto = res
            frames += n
            aviso = ("  [aviso] poucas detecções de mão" if taxa < 0.3 else "") + \
                    ("  [aviso] poucas detecções de rosto" if taxa_rosto < 0.5 else "")
            dt = time.time() - t0
            print(f"[{feitos}/{len(tarefas)}] {nome}: {n} frames, mão em {taxa:.0%}, "
                  f"rosto em {taxa_rosto:.0%} | "
                  f"{frames / dt:.0f} frames/s, faltam ~{dt / feitos * (len(tarefas) - feitos) / 60:.0f} min"
                  + aviso)
    print(f"\nConcluído em {(time.time() - t0) / 60:.1f} min: {feitos - erros} ok, {erros} erros.")


if __name__ == "__main__":
    main()
