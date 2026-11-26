"""
Adaptador do V-LIBRASIL (Rodrigues e Zanchettin, CIn-UFPE, 2021) para a pasta unificada.

1364 palavras/expressões x 3 articuladores (4086 vídeos; 3 da base original estão no error.csv),
gravados em chroma key, 1080p. Fonte usada: a cópia completa do Kaggle (davimedio01/v-librasil),
porque o site da UFPE ficou fora do ar em 2026-11 (ver preparacao/baixar_videos.py):

    videos UFPE (V-LIBRASIL)/data/Abacaxi_Articulador1.mp4
      -> dados/videos/abacaxi/vlibrasil__20210127091036_6011583c87073.mp4

Específico desta base:
  - CLASSE PELO CATÁLOGO: o annotations.csv do Kaggle traz, para cada vídeo, o link original da UFPE
    (url_download), o mesmo do dados/catalogo_treino.csv. O catálogo já decidiu a classe de cada um
    (variante única conhecida -> classe existente, "palavra (vN)" ou palavra nova) e deixou de fora
    as ~200 palavras com várias formas conhecidas (ambíguas). Este adaptador só segue essa decisão:
    vídeo cujo link não está no catálogo fica em dados/brutos.
  - SINALIZADOR = ARTICULADOR: o nome do arquivo diz quem sinaliza (Articulador1..3) e vira a coluna
    "sinalizador" = vlibrasil_<N> (global entre bases), usada na validação por pessoa do 2_treinar.py.
  - ID = nome do vídeo na UFPE (ex.: 20210127091036_6011583c87073): único e rastreável até o site.
  - NOMES NO WINDOWS: os arquivos trocam ? \\ " por caracteres da faixa privada do Unicode
    (U+F03F, U+F05C, U+F022); o annotations.csv usa os originais. Os dois lados são comparados
    depois de desfazer a troca.
  - CÂMERA LENTA: segundo a dissertação (p. 57), os vídeos foram exportados a METADE da velocidade.
    Isso é tratado na extração (1_extrair.py), não aqui: o vídeo fica intacto.

Idempotente e com dry run por padrão. Uso (a partir da raiz do projeto):
    python preparacao/vlibrasil_pre_tratamento.py                 # dry run
    python preparacao/vlibrasil_pre_tratamento.py --executar      # move para dados/videos
"""
import argparse
import collections
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from base_unificada import DADOS, Item, organizar  # noqa: I001

FONTE = "vlibrasil"
CATALOGO = os.path.join(DADOS, "catalogo_treino.csv")


def desfazer_troca_windows(nome):
    """'O quê\\uf03f' -> 'O quê?': a faixa U+F000-U+F0FF guarda caracteres proibidos em nomes."""
    return "".join(chr(ord(c) - 0xF000) if 0xF000 <= ord(c) <= 0xF0FF else c for c in nome)


def achar_base(origem):
    """Pasta que contém o annotations.csv (o zip do Kaggle tem uma pasta a mais)."""
    for raiz, _, arquivos in os.walk(origem):
        if "annotations.csv" in arquivos and os.path.isdir(os.path.join(raiz, "data")):
            return raiz
    sys.exit(f"[erro] annotations.csv + data/ não encontrados em {origem} (o download terminou?)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--origem", default=os.path.join(DADOS, "brutos", FONTE),
                    help="pasta onde o dataset do Kaggle foi extraído")
    ap.add_argument("--catalogo", default=CATALOGO, help="dados/catalogo_treino.csv (decide as classes)")
    ap.add_argument("--modo", choices=["mover", "copiar", "link"], default="mover")
    ap.add_argument("--executar", action="store_true", help="move de fato (sem isso: dry run)")
    ap.add_argument("--remover-orfaos", action="store_true",
                    help="apaga vídeos desta base na pasta unificada que o plano atual não gera")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    base = achar_base(os.path.abspath(args.origem))
    pasta_data = os.path.join(base, "data")
    # Arquivos já movidos para dados/videos não estão mais em data/: o organizar() os acha pelo nome.
    arquivos = {desfazer_troca_windows(f): os.path.join(pasta_data, f) for f in os.listdir(pasta_data)}

    with open(args.catalogo, encoding="utf-8") as f:
        classe_do_link = {r["link"]: r["classe"] for r in csv.DictReader(f) if r["fonte"] == FONTE}
    if not classe_do_link:
        sys.exit("[erro] o catálogo não tem linhas do V-LIBRASIL: rode preparacao/catalogo_unificado.py")

    with open(os.path.join(base, "annotations.csv"), encoding="utf-8-sig") as f:
        anotacoes = list(csv.DictReader(f))

    itens, fora_do_catalogo = [], collections.Counter()
    for r in anotacoes:
        classe = classe_do_link.get(r["url_download"])
        if classe is None:
            fora_do_catalogo[r["class"]] += 1
            continue
        art = r["user_id"].replace("Articulador", "")
        itens.append(Item(
            rotulo_original=r["class"], classe=classe,
            id=os.path.splitext(r["video_id"])[0],
            arquivo_original=r["video_name"],
            candidatos=[arquivos.get(r["video_name"], os.path.join(pasta_data, r["video_name"]))],
            linha={"sinalizador": f"vlibrasil_{art}", "articulador": int(art),
                   "largura": r["width"], "altura": r["height"], "n_frames": r["fps"],
                   "url_page": r["url_page"], "url_download": r["url_download"]}))

    print(f"{len(anotacoes)} vídeos no annotations.csv | {len(itens)} com classe no catálogo | "
          f"{sum(fora_do_catalogo.values())} fora ({len(fora_do_catalogo)} palavras ambíguas ou sem "
          f"exemplos suficientes; ficam em {os.path.relpath(pasta_data)})")
    organizar(FONTE, itens, executar=args.executar, remover_orfaos=args.remover_orfaos, modo=args.modo)


if __name__ == "__main__":
    main()
