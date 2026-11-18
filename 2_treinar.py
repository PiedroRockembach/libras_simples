"""
Passo 2 — treina o modelo temporal a partir dos landmarks em cache.

Uso:  python 2_treinar.py --landmarks dados/landmarks --epocas 150

Dica: crie uma classe "_nada" com vídeos de você parado, coçando o rosto,
mexendo as mãos sem sinalizar. Sem ela o modelo é OBRIGADO a escolher um sinal
mesmo quando você não está sinalizando. Classes que começam com "_" são
treinadas normalmente mas nunca são exibidas como resultado na câmera.
"""
import argparse
import collections
import csv
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from comum import (EXPRESSOES_ESPELHO, N_FEATURES, PONTOS, POSE_PARES, ROSTO_PARES,
                   VERSAO_CACHE, ModeloSinais, carregar_npz, fatiar, limpar,
                   preparar_sequencia, recortar_parado, recortar_sem_maos)

SEP = "__"   # dados/landmarks/<classe>/<fonte>__<id>.npz (ver preparacao/base_unificada.py)


# ----------------------------------------------------------------------------
# Carregamento
# ----------------------------------------------------------------------------
def ler_sinalizadores(pasta_manifestos):
    """
    {"<fonte>__<id>": sinalizador} a partir da coluna "sinalizador" dos manifestos de cada base
    (só bases que informam quem sinaliza, como o MINDS). Vídeos sem essa informação ficam fora.
    """
    mapa = {}
    if not os.path.isdir(pasta_manifestos):
        return mapa
    for nome in sorted(os.listdir(pasta_manifestos)):
        if not nome.endswith(".csv") or nome == "indice.csv" or nome.endswith("_original.csv"):
            continue
        with open(os.path.join(pasta_manifestos, nome), encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("sinalizador") and r.get("arquivo"):
                    mapa[os.path.splitext(os.path.basename(r["arquivo"]))[0]] = r["sinalizador"]
    return mapa


def carregar(pasta, excluir_fontes=(), sinalizadores=None):
    """
    Devolve (classes, amostras); amostra = (sequência, rótulo, duração_s, nome, sinalizador),
    sequência = dict com pose, mãos, rosto e expressão (ver comum.py).
    excluir_fontes: bases que ficam de fora (pelo prefixo "<fonte>__" do arquivo).
    Classes que ficarem sem nenhum vídeo não entram (não teriam como ser aprendidas).
    """
    sinalizadores = sinalizadores or {}
    brutas, antigos = [], 0
    for classe in sorted(c for c in os.listdir(pasta) if os.path.isdir(os.path.join(pasta, c))):
        for nome in sorted(os.listdir(os.path.join(pasta, classe))):
            if not nome.endswith(".npz") or nome.split(SEP, 1)[0] in excluir_fontes:
                continue
            seq, meta = carregar_npz(os.path.join(pasta, classe, nome))
            antigos += meta["versao"] < VERSAO_CACHE
            seq = limpar(seq, meta["largura"], meta["altura"])
            seq = recortar_parado(recortar_sem_maos(seq))
            brutas.append((seq, classe, len(seq["pose"]) / meta["fps"],
                           f"{classe}/{nome}", sinalizadores.get(os.path.splitext(nome)[0])))
    if antigos:
        print(f"[aviso] {antigos} vídeos com cache antigo (versão < {VERSAO_CACHE}, sem rosto): "
              f"treinam com o rosto 'ausente'. Rode o 1_extrair.py para atualizar.")
    classes = sorted({a[1] for a in brutas})
    idx = {c: i for i, c in enumerate(classes)}
    return classes, [(a[0], idx[a[1]], *a[2:]) for a in brutas]


def dividir(amostras, frac_val, val_sinalizadores=(), seed=0):
    """
    Divide por VÍDEO, nunca por frame/janela (janelas do mesmo vídeo são quase idênticas; uma
    no treino e outra na validação deixam a validação otimista).

    1) Vídeos com sinalizador conhecido: vão para a validação se a PESSOA está em
       `val_sinalizadores`, senão para o treino. Mede o que importa na câmera: acertar o sinal
       de alguém que o modelo nunca viu. Também mantém juntas as gravações da mesma execução por
       câmeras diferentes (Canon/Kinect do MINDS), que têm o mesmo sinalizador.
    2) Os demais (sem sinalizador): por classe, uma fração `frac_val` dos vídeos, sempre
       deixando pelo menos 1 no treino. Classes com 1 vídeo ficam só no treino.
    """
    rng = random.Random(seed)
    val_sin = set(val_sinalizadores)
    treino, val, por_classe = [], [], {}
    for i, a in enumerate(amostras):
        if a[4] is not None:
            (val if a[4] in val_sin else treino).append(i)
        else:
            por_classe.setdefault(a[1], []).append(i)
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
def espelhar(seq):
    """
    Inverte x e troca esquerda<->direita. Em Libras isso equivale à pessoa
    canhota fazendo o sinal: mesmo significado, mão dominante trocada.
    No rosto, troca também os pontos pareados (olho esq. <-> dir. etc.) e os blendshapes
    "...Left" <-> "...Right": senão a imagem espelhada teria o sorriso do lado errado.
    """
    seq = {k: v.copy() for k, v in seq.items()}
    for k in PONTOS:
        seq[k][..., 0] = -seq[k][..., 0]
    for i, j in POSE_PARES:
        seq["pose"][:, [i, j]] = seq["pose"][:, [j, i]]
    for i, j in ROSTO_PARES:
        seq["rosto"][:, [i, j]] = seq["rosto"][:, [j, i]]
    seq["expressao"] = seq["expressao"][:, EXPRESSOES_ESPELHO]
    seq["mao_esq"], seq["mao_dir"] = seq["mao_dir"], seq["mao_esq"]
    return seq


# Ruído (desvio padrão, em coordenadas de imagem) por parte: o tremor típico do detector.
RUIDO = {"pose": 0.004, "mao_esq": 0.002, "mao_dir": 0.002, "rosto": 0.001}


def aumentar(seq, rng):
    if rng.random() < 0.5:
        seq = espelhar(seq)

    # Recorte temporal: na câmera a janela nunca pega o sinal perfeitamente
    # alinhado; treinar com pedaços de 75-100% do sinal ensina a tolerar isso.
    T = len(seq["pose"])
    if T > 8:
        n = max(int(T * rng.uniform(0.75, 1.0)), 4)
        a = rng.randint(0, T - n)
        seq = fatiar(seq, a, a + n)

    # Rotação leve, esticamento e ruído: variações de câmera, corpo e tremor. A MESMA
    # transformação em todas as partes, para a mão continuar no mesmo lugar do rosto.
    ang = np.deg2rad(rng.uniform(-12, 12))
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]], np.float32)
    S = np.diag([rng.uniform(0.9, 1.1), rng.uniform(0.9, 1.1)]).astype(np.float32)
    M = R @ S
    centro = np.nanmean(seq["pose"][:, [11, 12], :2], axis=(0, 1))
    if np.isnan(centro).any():
        centro = np.array([0.5, 0.5], np.float32)
    seq = dict(seq)
    for k in PONTOS:
        arr = seq[k].copy()
        arr[..., :2] = (arr[..., :2] - centro) @ M.T + centro
        arr[..., :2] += np.random.normal(0, RUIDO[k], arr[..., :2].shape)
        seq[k] = arr
    # Blendshapes: um pouco de ruído (a estimativa da expressão também oscila), mantendo 0 a 1.
    seq["expressao"] = np.clip(seq["expressao"] + np.random.normal(0, 0.02, seq["expressao"].shape), 0, 1)

    # Simula falhas de detecção: some com a mão (ou o rosto) em alguns frames aleatórios.
    for k in ("mao_esq", "mao_dir"):
        for t in range(len(seq[k])):
            if rng.random() < 0.05:
                seq[k][t] = np.nan
    for t in range(len(seq["rosto"])):
        if rng.random() < 0.03:
            seq["rosto"][t] = np.nan
            seq["expressao"][t] = np.nan
    return seq


class DatasetSinais(Dataset):
    def __init__(self, amostras, indices, aumentar_dados, repeticoes=1, seed=0):
        self.amostras = amostras
        self.aug = aumentar_dados
        self.rng = random.Random(seed)
        # Balanceamento: cada classe aparece EXATAMENTE `repeticoes` vezes por época; senão a
        # classe com mais vídeos domina o treino (e o custo cresce com ela: antes o alvo era
        # max(vídeos por classe), e as ~48 gravações de uma classe do MINDS faziam TODAS as
        # ~1078 classes terem 48+ amostras). Classe com até `repeticoes` vídeos: cicla por eles.
        # Classe com mais: sorteia um vídeo a cada amostra, então todos aparecem ao longo das
        # épocas sem aumentar o tamanho da época.
        self.alvo = repeticoes
        if aumentar_dados:
            por_classe = {}
            for i in indices:
                por_classe.setdefault(amostras[i][1], []).append(i)
            self.itens = [(tuple(v), k) for v in por_classe.values() for k in range(repeticoes)]
        else:
            self.itens = [((i,), 0) for i in indices]

    def __len__(self):
        return len(self.itens)

    def __getitem__(self, k):
        videos, j = self.itens[k]
        i = videos[j % len(videos)] if len(videos) <= self.alvo else self.rng.choice(videos)
        seq, rotulo = self.amostras[i][:2]
        if self.aug:
            seq = aumentar(seq, self.rng)
        x = preparar_sequencia(seq)
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
def avaliar(modelo, loader, perda_fn, device="cpu"):
    """Devolve (perda média, top-5 previstos (N, 5), verdadeiros (N,)), na ordem do loader."""
    modelo.eval()
    perda, tops, ys = 0.0, [], []
    with torch.no_grad():
        for x, y in loader:
            logits = modelo(x.to(device))
            y = y.to(device)
            perda += perda_fn(logits, y).item() * len(y)
            tops.append(logits.topk(min(5, logits.shape[1]), dim=1).indices.cpu())
            ys.append(y.cpu())
    y = torch.cat(ys).numpy()
    return perda / max(len(y), 1), torch.cat(tops).numpy(), y


def acertos(top, y):
    """(top-1, top-5) em fração. Top-5 = o certo está entre os 5 mais prováveis."""
    if len(y) == 0:
        return 0.0, 0.0
    return float((top[:, 0] == y).mean()), float((top == y[:, None]).any(axis=1).mean())


def relatorio_final(top, y, grupos, classes):
    """Top-1/top-5 por grupo de validação e os pares mais confundidos (matriz cheia só se pequena)."""
    for nome, mascara in grupos.items():
        if mascara.any():
            a1, a5 = acertos(top[mascara], y[mascara])
            print(f"  {nome:34s} {mascara.sum():5d} vídeos  top-1 {a1:6.2%}  top-5 {a5:6.2%}")
    pares = collections.Counter((classes[a], classes[b]) for a, b in zip(y, top[:, 0]) if a != b)
    if pares:
        print("\nPares mais confundidos (verdadeiro -> previsto: vezes):")
        for (a, b), n in pares.most_common(15):
            print(f"  {a} -> {b}: {n}")
    if len(classes) <= 30:
        conf = np.zeros((len(classes), len(classes)), int)
        np.add.at(conf, (y, top[:, 0]), 1)
        print("\nMatriz de confusão (linha = verdadeiro, coluna = previsto):")
        larg = max(len(c) for c in classes)
        for i, c in enumerate(classes):
            print(f"{c:>{larg}} " + " ".join(f"{v:3d}" for v in conf[i]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--landmarks", default="dados/landmarks")
    ap.add_argument("--saida", default="modelo_sinais.pt")
    ap.add_argument("--epocas", type=int, default=150)
    ap.add_argument("--frac-val", type=float, default=0.2)
    ap.add_argument("--repeticoes", type=int, default=32,
                    help="amostras aumentadas de CADA CLASSE por época (teto, igual para todas)")
    ap.add_argument("--manifestos", default="dados/manifestos",
                    help="de onde vem a coluna 'sinalizador' usada na validação por pessoa")
    ap.add_argument("--val-sinalizadores", nargs="*", default=["minds_11", "minds_12"],
                    help="pessoas que ficam SÓ na validação (nunca vistas no treino)")
    ap.add_argument("--excluir-fontes", nargs="*", default=["minds_libras_rgbd"],
                    help="bases fora do treino (padrão: Kinect, adiado; passe a flag vazia p/ incluir)")
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

    classes, amostras = carregar(args.landmarks, args.excluir_fontes,
                                 ler_sinalizadores(args.manifestos))
    por_fonte = collections.Counter(a[3].split("/", 1)[1].split(SEP, 1)[0] for a in amostras)
    print(f"{len(classes)} classes, {len(amostras)} vídeos {dict(por_fonte)}"
          + (f" | fora: {args.excluir_fontes}" if args.excluir_fontes else ""))
    treino, val = dividir(amostras, args.frac_val, args.val_sinalizadores)
    vistos = {a[4] for a in amostras if a[4]}
    faltam = set(args.val_sinalizadores) - vistos
    if faltam:
        print(f"[aviso] sinalizadores de validação sem nenhum vídeo: {sorted(faltam)}")
    # Grupos de validação, na ordem de `val` (= ordem do dl_val, que não embaralha).
    tem_sin = np.array([amostras[i][4] is not None for i in val], bool)
    grupos = {f"pessoas novas {sorted(set(args.val_sinalizadores) & vistos)}": tem_sin,
              "outros (gravação nova, sem id de pessoa)": ~tem_sin,
              "total": np.ones(len(val), bool)}
    print(f"Treino: {len(treino)} vídeos | validação: {len(val)} "
          f"({tem_sin.sum()} de pessoas novas, {(~tem_sin).sum()} outros)")
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

    # Checkpoint e early stopping pela PERDA de validação, não pela acurácia: com poucos vídeos
    # de validação a acurácia anda em degraus e satura cedo; com o critério estrito ">" o arquivo
    # ficava sendo um modelo das primeiras épocas, quase sem treino. A perda é contínua e também
    # melhora quando a probabilidade do certo sobe, o que importa para o limiar da câmera.
    melhor, sem_melhora = float("inf"), 0
    duracao_janela = float(np.median([a[2] for a in amostras]))
    t_inicio = time.time()
    for ep in range(1, args.epocas + 1):
        t_ep = time.time()
        modelo.train()
        tot, certos, n = 0.0, 0, 0
        for x, y in dl_treino:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = modelo(x)
            perda = perda_fn(logits, y)
            otim.zero_grad(); perda.backward(); otim.step()
            tot += perda.item() * len(y); n += len(y)
            certos += (logits.argmax(1) == y).sum().item()
        sched.step()

        if dl_val:
            perda_val, top, yv = avaliar(modelo, dl_val, perda_fn, device)
            a1, a5 = acertos(top, yv)
        else:
            perda_val = tot / n                       # sem validação: só resta a perda de treino
        if ep % 5 == 0 or ep == 1:
            print(f"época {ep:3d}  perda {tot / n:.3f}  acc treino {certos / n:.2%}"
                  + (f"  | val: perda {perda_val:.3f}  top-1 {a1:.2%}  top-5 {a5:.2%}"
                     if dl_val else "")
                  + f"  ({time.time() - t_ep:.1f}s)")
        if perda_val < melhor:
            melhor, sem_melhora = perda_val, 0
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
    print(f"Melhor perda {'val' if dl_val else 'treino'}: {melhor:.3f} -> {args.saida}")
    print(f"Janela da câmera: {duracao_janela:.2f}s (mediana da duração dos sinais)")
    if dl_val:
        ck = torch.load(args.saida, map_location=device)
        modelo.load_state_dict(ck["estado"])
        _, top, yv = avaliar(modelo, dl_val, perda_fn, device)
        print("\nValidação do checkpoint salvo:")
        relatorio_final(top, yv, grupos, classes)


if __name__ == "__main__":
    main()
