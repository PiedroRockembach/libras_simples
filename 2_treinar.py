"""
Passo 2 — treina o modelo temporal a partir dos landmarks em cache.

Uso:  python 2_treinar.py --landmarks dados/landmarks --epocas 150

Dica: crie uma classe "_nada" com vídeos de você parado, coçando o rosto,
mexendo as mãos sem sinalizar. Sem ela o modelo é OBRIGADO a escolher um sinal
mesmo quando você não está sinalizando. Classes que começam com "_" são
treinadas normalmente mas nunca são exibidas como resultado na câmera.
"""
import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from comum import (N_FEATURES, POSE_PARES, ModeloSinais, corrigir_aspecto,
                   interpolar_lacunas, preparar_sequencia, recortar_sem_maos)


# ----------------------------------------------------------------------------
# Carregamento
# ----------------------------------------------------------------------------
def carregar(pasta):
    classes = sorted(c for c in os.listdir(pasta) if os.path.isdir(os.path.join(pasta, c)))
    amostras = []                     # (pose, me, md, rótulo, duração_s, nome)
    for rotulo, classe in enumerate(classes):
        for nome in sorted(os.listdir(os.path.join(pasta, classe))):
            if not nome.endswith(".npz"):
                continue
            d = np.load(os.path.join(pasta, classe, nome))
            pose, me, md = corrigir_aspecto(d["pose"], d["mao_esq"], d["mao_dir"],
                                            float(d["largura"]), float(d["altura"]))
            me, md = interpolar_lacunas(me), interpolar_lacunas(md)
            pose = interpolar_lacunas(pose)
            pose, me, md = recortar_sem_maos(pose, me, md)
            amostras.append((pose, me, md, rotulo, len(pose) / float(d["fps"]),
                             f"{classe}/{nome}"))
    return classes, amostras


def dividir_por_video(amostras, frac_val, seed=0):
    """
    Divide por VÍDEO, nunca por frame/janela. Janelas do mesmo vídeo são quase
    idênticas; se uma cai no treino e outra na validação, a validação fica
    otimista e mente sobre a generalização.
    Classes com 1 vídeo ficam só no treino (não há o que separar).
    """
    rng = random.Random(seed)
    por_classe = {}
    for i, a in enumerate(amostras):
        por_classe.setdefault(a[3], []).append(i)
    treino, val = [], []
    for idx in por_classe.values():
        rng.shuffle(idx)
        n_val = int(round(len(idx) * frac_val)) if len(idx) >= 2 else 0
        n_val = min(n_val, len(idx) - 1)
        val += idx[:n_val]
        treino += idx[n_val:]
    return treino, val


# ----------------------------------------------------------------------------
# Data augmentation — aplicada nos landmarks brutos, a cada época, aleatória
# ----------------------------------------------------------------------------
def espelhar(pose, me, md):
    """
    Inverte x e troca esquerda<->direita. Em Libras isso equivale à pessoa
    canhota fazendo o sinal: mesmo significado, mão dominante trocada.
    """
    pose, me, md = pose.copy(), me.copy(), md.copy()
    for a in (pose, me, md):
        a[..., 0] = -a[..., 0]
    for i, j in POSE_PARES:
        pose[:, [i, j]] = pose[:, [j, i]]
    return pose, md, me


def aumentar(pose, me, md, rng):
    if rng.random() < 0.5:
        pose, me, md = espelhar(pose, me, md)

    # Recorte temporal: na câmera a janela nunca pega o sinal perfeitamente
    # alinhado; treinar com pedaços de 75-100% do sinal ensina a tolerar isso.
    T = len(pose)
    if T > 8:
        n = max(int(T * rng.uniform(0.75, 1.0)), 4)
        a = rng.randint(0, T - n)
        pose, me, md = pose[a:a + n], me[a:a + n], md[a:a + n]

    # Rotação leve, esticamento e ruído: variações de câmera, corpo e tremor.
    ang = np.deg2rad(rng.uniform(-12, 12))
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]], np.float32)
    S = np.diag([rng.uniform(0.9, 1.1), rng.uniform(0.9, 1.1)]).astype(np.float32)
    M = R @ S
    centro = np.nanmean(pose[:, [11, 12], :2], axis=(0, 1))
    if np.isnan(centro).any():
        centro = np.array([0.5, 0.5], np.float32)
    out = []
    for arr, sigma in ((pose, 0.004), (me, 0.002), (md, 0.002)):
        arr = arr.copy()
        arr[..., :2] = (arr[..., :2] - centro) @ M.T + centro
        arr[..., :2] += np.random.normal(0, sigma, arr[..., :2].shape)
        out.append(arr)
    pose, me, md = out

    # Simula falhas de detecção: some com a mão em alguns frames aleatórios.
    for mao in (me, md):
        for t in range(len(mao)):
            if rng.random() < 0.05:
                mao[t] = np.nan
    return pose, me, md


class DatasetSinais(Dataset):
    def __init__(self, amostras, indices, aumentar_dados, repeticoes=1, seed=0):
        self.amostras = amostras
        self.aug = aumentar_dados
        self.rng = random.Random(seed)
        # Balanceamento: cada classe aparece o mesmo número de vezes por época
        # (`repeticoes`), ciclando pelos vídeos dela; senão a classe com mais vídeos
        # domina o treino. Antes o alvo era max(vídeos por classe) × repeticoes: uma
        # única classe com 4 vídeos fazia TODAS as 1069 terem 128 amostras (4x o custo).
        if aumentar_dados:
            por_classe = {}
            for i in indices:
                por_classe.setdefault(amostras[i][3], []).append(i)
            # max(): garante que todo vídeo aparece pelo menos uma vez por época
            alvo = max(repeticoes, max(len(v) for v in por_classe.values()))
            self.indices = [v[k % len(v)] for v in por_classe.values() for k in range(alvo)]
        else:
            self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, k):
        pose, me, md, rotulo, _, _ = self.amostras[self.indices[k]]
        if self.aug:
            pose, me, md = aumentar(pose, me, md, self.rng)
        x = preparar_sequencia(pose, me, md)
        return torch.from_numpy(x), rotulo


def semear_worker(_):
    """
    Cada processo do DataLoader recebe uma CÓPIA do dataset, inclusive do
    `self.rng`. Sem re-semear, todos os processos sorteariam exatamente as mesmas
    augmentations (e, sem persistent_workers, repetiriam as mesmas a cada época).
    torch.initial_seed() já é diferente por processo e por época.
    """
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = random.Random(torch.initial_seed() % 2**32)


# ----------------------------------------------------------------------------
# Treino
# ----------------------------------------------------------------------------
def avaliar(modelo, loader, n_classes, device="cpu"):
    modelo.eval()
    conf = np.zeros((n_classes, n_classes), int)
    with torch.no_grad():
        for x, y in loader:
            p = modelo(x.to(device)).argmax(1).cpu()
            for a, b in zip(y.numpy(), p.numpy()):
                conf[a, b] += 1
    return conf.trace() / max(conf.sum(), 1), conf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--landmarks", default="dados/landmarks")
    ap.add_argument("--saida", default="modelo_sinais.pt")
    ap.add_argument("--epocas", type=int, default=150)
    ap.add_argument("--frac-val", type=float, default=0.2)
    ap.add_argument("--repeticoes", type=int, default=32,
                    help="cópias aumentadas de cada vídeo por época")
    ap.add_argument("--paciencia", type=int, default=30)
    ap.add_argument("--lote", type=int, default=32, help="tamanho do lote (batch)")
    ap.add_argument("--workers", type=int, default=min(8, max(0, (os.cpu_count() or 1) - 1)),
                    help="processos preparando dados em paralelo (0 = no processo principal)")
    ap.add_argument("--dispositivo", default="auto", help="auto, cuda ou cpu")
    args = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0); random.seed(0)

    # GPU se houver. Com este modelo pequeno, a GPU calcula um lote em poucos ms;
    # o gargalo passa a ser preparar os dados (augmentation + features, em Python
    # na CPU). Por isso os `workers`: vários processos preparam lotes em paralelo
    # enquanto a GPU treina.
    device = torch.device("cuda" if args.dispositivo == "auto" and torch.cuda.is_available()
                          else "cpu" if args.dispositivo == "auto" else args.dispositivo)
    print(f"Dispositivo: {device}"
          + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else "")
          + f" | workers: {args.workers}")

    classes, amostras = carregar(args.landmarks)
    print(f"{len(classes)} classes, {len(amostras)} vídeos: {classes}")
    treino, val = dividir_por_video(amostras, args.frac_val)
    if not val:
        print("[aviso] nenhuma classe tem 2+ vídeos: sem validação. A acurácia de "
              "treino NÃO diz se o modelo generaliza. Grave mais vídeos por classe.")

    # persistent_workers: os processos sobrevivem entre épocas (no Windows, criar
    # processo é caro e cada um recebe uma cópia dos dados). pin_memory: a cópia
    # CPU -> GPU fica mais rápida e pode ocorrer em paralelo (non_blocking).
    extra = dict(num_workers=args.workers, persistent_workers=args.workers > 0,
                 pin_memory=device.type == "cuda",
                 worker_init_fn=semear_worker if args.workers > 0 else None)
    dl_treino = DataLoader(DatasetSinais(amostras, treino, True, args.repeticoes),
                           batch_size=args.lote, shuffle=True, **extra)
    dl_val = DataLoader(DatasetSinais(amostras, val, False), batch_size=256,
                        **{**extra, "num_workers": min(args.workers, 2),
                           "persistent_workers": args.workers > 0}) if val else None
    print(f"{len(dl_treino.dataset)} amostras de treino por época "
          f"({len(dl_treino)} lotes de {args.lote})")

    modelo = ModeloSinais(N_FEATURES, len(classes)).to(device)
    otim = torch.optim.AdamW(modelo.parameters(), lr=1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(otim, T_max=args.epocas)
    # label smoothing: não deixa o modelo ficar 100% confiante -> probabilidades
    # mais honestas, o que importa para o limiar de confiança na câmera.
    perda_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    melhor, sem_melhora = -1.0, 0
    duracao_janela = float(np.median([a[4] for a in amostras]))
    t_inicio = time.time()
    for ep in range(1, args.epocas + 1):
        t_ep = time.time()
        modelo.train()
        tot, acertos, n = 0.0, 0, 0
        for x, y in dl_treino:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = modelo(x)
            perda = perda_fn(logits, y)
            otim.zero_grad(); perda.backward(); otim.step()
            tot += perda.item() * len(y); n += len(y)
            acertos += (logits.argmax(1) == y).sum().item()
        sched.step()

        acc_val = avaliar(modelo, dl_val, len(classes), device)[0] if dl_val else acertos / n
        if ep % 5 == 0 or ep == 1:
            print(f"época {ep:3d}  perda {tot / n:.3f}  acc treino {acertos / n:.2%}"
                  + (f"  acc val {acc_val:.2%}" if dl_val else "")
                  + f"  ({time.time() - t_ep:.1f}s)")
        if acc_val > melhor:
            melhor, sem_melhora = acc_val, 0
            # Pesos salvos na CPU: o checkpoint abre em qualquer máquina, com ou sem GPU.
            estado = {k: v.detach().cpu() for k, v in modelo.state_dict().items()}
            torch.save({"estado": estado, "classes": classes,
                        "n_features": N_FEATURES, "duracao_janela_s": duracao_janela},
                       args.saida)
        else:
            sem_melhora += 1
            if sem_melhora >= args.paciencia:
                print(f"Parando: {args.paciencia} épocas sem melhora (early stopping).")
                break

    print(f"\nTempo total de treino: {(time.time() - t_inicio) / 60:.1f} min ({ep} épocas)")
    print(f"Melhor acurácia {'val' if dl_val else 'treino'}: {melhor:.2%} -> {args.saida}")
    print(f"Janela da câmera: {duracao_janela:.2f}s (mediana da duração dos sinais)")
    if dl_val:
        ck = torch.load(args.saida, map_location=device)
        modelo.load_state_dict(ck["estado"])
        _, conf = avaliar(modelo, dl_val, len(classes), device)
        print("\nMatriz de confusão (linha = verdadeiro, coluna = previsto):")
        larg = max(len(c) for c in classes)
        for i, c in enumerate(classes):
            print(f"{c:>{larg}} " + " ".join(f"{v:3d}" for v in conf[i]))


if __name__ == "__main__":
    main()
