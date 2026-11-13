"""
Adaptador da base "Incluir Tecnologia" (dicionário de Libras do YouTube) para a pasta unificada.

    incluir_tecnologia.tar.gz                    dados/videos/bom dia/incluir_tecnologia__m8F3o0Y39kA.mp4
      incluir_tecnologia/videos/Bom Dia.mp4  ->  dados/manifestos/incluir_tecnologia.csv (+ _original)
      incluir_tecnologia/MANIFEST.csv            dados/excluidos/incluir_tecnologia/ (excluídos, metadados)

Este arquivo só sabe LER o formato desta base (MANIFEST.csv com colunas file/class/video_id/sha256).
Normalização de rótulos, movimentação e manifestos ficam em base_unificada.py, iguais para
todas as bases. As regras e o porquê de cada uma estão documentados lá e no GUIA.md (seção 0).

Específico desta base:
  - EXCLUÍDOS: textos/músicas sinalizados inteiros (100-160 s), que não são sinais isolados.
  - METADADOS do pacote (info/, channel_listing.json): saem das classes.
  - id = video_id do YouTube (único e rastreável).
  - ^, ~, ´ (acentos do português) e "libras" são sinais válidos: entram normalmente.

Idempotente e com dry run por padrão. Uso (a partir da raiz do projeto):
    # do zero, a partir do pacote baixado:
    python preparacao/incluir_tecnologia_pre_tratamento.py --origem incluir_tecnologia.tar.gz --executar
    # a partir de uma pasta já extraída (vídeos em videos/, soltos na raiz ou em <classe>/<id>.mp4):
    python preparacao/incluir_tecnologia_pre_tratamento.py --origem incluir_tecnologia --executar
    # só conferir o que está na pasta unificada (inclusive o sha256):
    python preparacao/incluir_tecnologia_pre_tratamento.py --origem incluir_tecnologia.tar.gz --verificar-sha
"""
import argparse
import csv
import os
import shutil
import sys
import tarfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from base_unificada import (DADOS, PASTA_MANIFESTOS, PASTA_VIDEOS, Item,  # noqa: I001
                            arquivos_da_fonte, conferir, nome_arquivo, normalizar_rotulo,
                            organizar)

FONTE = "incluir_tecnologia"
EXCLUIR = {"A sua.mp4", "Simples desejo.mp4", "Amor Sincero.mp4"}
METADADOS = ["info", "channel_listing.json"]
MANIFESTO_ORIGINAL = os.path.join(PASTA_MANIFESTOS, f"{FONTE}_original.csv")


def extrair_tar(arquivo):
    """Extrai numa pasta temporária dentro de dados/ e devolve a pasta que contém o MANIFEST.csv."""
    tmp = os.path.join(DADOS, f"_extraindo_{FONTE}")
    shutil.rmtree(tmp, ignore_errors=True)              # sobra de uma execução interrompida
    print(f"Extraindo {arquivo} ...")
    with tarfile.open(arquivo) as tar:
        # filter="data" bloqueia caminhos absolutos, "../" e links perigosos dentro do tar
        tar.extractall(tmp, filter="data")
    for raiz, _, arquivos in os.walk(tmp):
        if "MANIFEST.csv" in arquivos:
            return tmp, raiz
    sys.exit(f"[erro] MANIFEST.csv não encontrado dentro de {arquivo}")


def ler_manifesto_original(base):
    """O manifesto ORIGINAL da base, de onde estiver (o já salvo em dados/manifestos tem prioridade)."""
    for caminho in (MANIFESTO_ORIGINAL, os.path.join(base or "", "MANIFEST_original.csv"),
                    os.path.join(base or "", "MANIFEST.csv")):
        if os.path.isfile(caminho):
            with open(caminho, encoding="utf-8") as f:
                linhas = list(csv.DictReader(f))
            if "classe_original" in linhas[0] or "rotulo_original" in linhas[0]:
                continue                                # manifesto já organizado: não é o original
            return caminho, linhas
    sys.exit("[erro] manifesto original não encontrado")


def montar_itens(base, linhas):
    """Uma linha do MANIFEST -> um Item, com os lugares onde o vídeo pode estar."""
    excl_legado = os.path.join(os.path.dirname(base), f"excluidos_{os.path.basename(base)}") \
        if base else None
    itens = []
    for r in linhas:
        arq, vid = r["file"], r["video_id"]
        candidatos = []
        if base:
            candidatos += [os.path.join(base, "videos", arq), os.path.join(base, arq)]
            if arq not in EXCLUIR:                      # layout <classe>/<id>.mp4 (organização antiga)
                candidatos.append(os.path.join(base, normalizar_rotulo(r["class"]), f"{vid}.mp4"))
            else:
                candidatos.append(os.path.join(excl_legado, arq))
        itens.append(Item(rotulo_original=r["class"], id=vid, arquivo_original=arq,
                          candidatos=candidatos, excluir=arq in EXCLUIR, sha256=r["sha256"],
                          linha={k: v for k, v in r.items() if k not in ("file", "class", "video_id")}))
    return itens


def limpar_origem(base, tmp):
    """Depois de mover: apaga a pasta temporária ou a pasta de origem, se só sobrou o que já foi salvo."""
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)
        return
    excl_legado = os.path.join(os.path.dirname(base), f"excluidos_{os.path.basename(base)}")
    for pasta in (base, excl_legado):
        if not os.path.isdir(pasta):
            continue
        for raiz, dirs, arqs in os.walk(pasta, topdown=False):
            for d in dirs:
                p = os.path.join(raiz, d)
                if not os.listdir(p):
                    os.rmdir(p)
        sobra = set(os.listdir(pasta))
        # MANIFEST*.csv: o original já está em dados/manifestos; o "organizado" antigo é obsoleto.
        if sobra <= {"MANIFEST.csv", "MANIFEST_original.csv"}:
            shutil.rmtree(pasta)
            print(f"Pasta de origem removida (vazia): {pasta}")
        else:
            print(f"[atenção] {pasta} ainda tem arquivos não reconhecidos: {sorted(sobra)[:5]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--origem", default="incluir_tecnologia.tar.gz",
                    help="o pacote .tar.gz ou uma pasta já extraída")
    ap.add_argument("--executar", action="store_true", help="move de fato (sem isso: dry run)")
    ap.add_argument("--verificar-sha", action="store_true", help="confere o sha256 dos vídeos")
    ap.add_argument("--remover-orfaos", action="store_true",
                    help="apaga vídeos desta base na pasta unificada que o plano atual não gera")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    tmp, base = None, None
    eh_tar = args.origem.endswith((".tar.gz", ".tgz"))
    if not eh_tar:
        base = os.path.abspath(args.origem)
    elif args.executar and not os.path.isfile(MANIFESTO_ORIGINAL):
        tmp, base = extrair_tar(args.origem)            # primeira vez: precisa do conteúdo do pacote
    elif args.executar:
        # Já organizado antes: só extrai se faltar algum vídeo na pasta unificada.
        _, linhas = ler_manifesto_original(None)
        presentes = {os.path.basename(f) for f in arquivos_da_fonte(FONTE)}  # em qualquer classe
        faltam = [r for r in linhas if r["file"] not in EXCLUIR
                  and nome_arquivo(FONTE, r["video_id"]) not in presentes]
        if faltam:
            print(f"{len(faltam)} vídeos faltando na pasta unificada: extraindo o pacote.")
            tmp, base = extrair_tar(args.origem)
    elif not os.path.isfile(MANIFESTO_ORIGINAL):
        print(f"(dry run) extrairia {args.origem} e organizaria em {PASTA_VIDEOS}; rode com --executar")
        return

    caminho_manifesto, linhas = ler_manifesto_original(base)
    if args.executar and not os.path.isfile(MANIFESTO_ORIGINAL):
        os.makedirs(PASTA_MANIFESTOS, exist_ok=True)
        shutil.copy2(caminho_manifesto, MANIFESTO_ORIGINAL)   # backup ANTES de mover qualquer coisa

    metadados = [os.path.join(base, m) for m in METADADOS] if base else []
    linhas_org = organizar(FONTE, montar_itens(base, linhas), metadados, args.executar,
                           args.verificar_sha, args.remover_orfaos)
    if args.executar and base:
        limpar_origem(base, tmp)
    if not args.executar and args.verificar_sha and os.path.isdir(PASTA_VIDEOS):
        conferir(FONTE, linhas_org, verificar_sha=True)


if __name__ == "__main__":
    main()
