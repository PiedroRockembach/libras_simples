"""
Adaptador da base MINDS-Libras (Rezende et al., UFMG; CC-BY-4.0) para a pasta unificada.

A base tem 20 sinais x 12 sinalizantes x 5 repetições, gravados AO MESMO TEMPO por duas câmeras,
publicadas como dois pacotes no Zenodo. Cada pacote vira uma fonte:

  minds_libras_rgb  (Canon 1080p, vídeos recortados no sinal; DOI 10.5281/zenodo.2667329)
    Sinalizador01/Canon/15MacaSinalizador01-1.mp4       -> dados/videos/maçã/minds_libras_rgb__s01_maca_r1.mp4
  minds_libras_rgbd (Kinect: RGB 1080p de 5 s fixos + profundidade + esqueleto; DOI 10.5281/zenodo.4322984)
    Sinalizador01/15Maca/1-15Maca_1RGB.mp4              -> dados/videos/maçã/minds_libras_rgbd__s01_maca_r1.mp4
    (*Depth.mp4, *Body.txt e *Face.txt são ignorados: o pipeline usa só o RGB, pelo MediaPipe)

Este arquivo só sabe LER o formato desta base. Normalização de rótulos, manifestos e conferência
ficam em base_unificada.py, iguais para todas as bases.

Específico desta base:
  - ORIGEM INTACTA: os dados brutos ficam num repositório compartilhado (/mnt/hd1tb/libras-data/raw,
    usado também por outros projetos), então os vídeos são LINKADOS (padrão) ou copiados, nunca movidos.
  - RÓTULOS: os nomes da base vêm sem acento ("Maca", "America"). ROTULOS corrige a grafia para cair
    na classe que já existe na Incluir Tecnologia; "Cinco" vai para a classe "5" (o número; conferido
    visualmente contra o "5" da Incluir: mesma configuração de mão e ponto de articulação).
  - MESMA TOMADA, MESMO ID: o id "s<sinalizador>_<sinal>_r<repetição>" é igual nas duas fontes, porque
    os dois vídeos são a MESMA execução vista por câmeras diferentes. Na divisão treino/validação, os
    dois precisam ficar do mesmo lado; senão a validação vê uma execução que o treino já viu.
  - Lacunas da base (normais, não são erro): no pacote Canon faltam alguns sinais dos sinalizadores 03,
    04 e 09, e o sinalizador 07 tem repetições extras (r6, r7) de "Barulho" e "Filho".

Idempotente e com dry run por padrão. Uso (a partir da raiz do projeto):
    python preparacao/minds_libras_pre_tratamento.py                 # dry run das duas fontes
    python preparacao/minds_libras_pre_tratamento.py --executar      # cria os links
    python preparacao/minds_libras_pre_tratamento.py --fontes rgb --copiar --executar
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from base_unificada import Item, organizar  # noqa: I001

RAW = "/mnt/hd1tb/libras-data/raw"
FONTES = {
    "rgb": ("minds_libras_rgb", os.path.join(RAW, "minds_libras_rgb", "extracted")),
    "rgbd": ("minds_libras_rgbd", os.path.join(RAW, "minds_libras_rgbd", "extracted")),
}

# Nome do sinal no arquivo da base -> rótulo em português (vira a classe após normalizar_rotulo).
# Os 11 primeiros já existem como classe na Incluir Tecnologia; os outros 9 são classes novas.
ROTULOS = {
    "Amarelo": "Amarelo", "America": "América", "Aproveitar": "Aproveitar", "Bala": "Bala",
    "Banheiro": "Banheiro", "Cinco": "5", "Conhecer": "Conhecer", "Espelho": "Espelho",
    "Filho": "Filho", "Maca": "Maçã", "Ruim": "Ruim",
    "Acontecer": "Acontecer", "Aluno": "Aluno", "Banco": "Banco", "Barulho": "Barulho",
    "Esquina": "Esquina", "Medo": "Medo", "Sapo": "Sapo", "Vacina": "Vacina", "Vontade": "Vontade",
}

# Sinalizador01/Canon/15MacaSinalizador01-1.mp4
PADRAO_RGB = re.compile(r"Sinalizador(\d{2})/Canon/(\d{2})([A-Za-z]+)Sinalizador(\d{2})-(\d+)\.mp4")
# Sinalizador01/15Maca/1-15Maca_1RGB.mp4
PADRAO_RGBD = re.compile(r"Sinalizador(\d{2})/(\d{2})([A-Za-z]+)/(\d+)-(\d{2})([A-Za-z]+)_(\d+)RGB\.mp4")
IGNORAR_RGBD = re.compile(r".*(Depth\.mp4|Body\.txt|Face\.txt)$")


def ler_base(chave, origem):
    """Percorre a pasta extraída e devolve (itens, arquivos não reconhecidos)."""
    itens, estranhos = [], []
    for raiz, _, arquivos in os.walk(origem):
        for nome in sorted(arquivos):
            caminho = os.path.join(raiz, nome)
            rel = os.path.relpath(caminho, origem).replace(os.sep, "/")
            if chave == "rgb":
                m = PADRAO_RGB.fullmatch(rel)
                ok = m and m.group(1) == m.group(4)
                sin, num, sinal, rep = (m.group(1), m.group(2), m.group(3), m.group(5)) if ok else (None,) * 4
            else:
                if IGNORAR_RGBD.fullmatch(nome):
                    continue
                m = PADRAO_RGBD.fullmatch(rel)
                ok = (m and int(m.group(1)) == int(m.group(4)) and m.group(2) == m.group(5)
                      and m.group(3) == m.group(6))
                sin, num, sinal, rep = (m.group(1), m.group(2), m.group(3), m.group(7)) if ok else (None,) * 4
            if not ok or sinal not in ROTULOS:
                estranhos.append(rel)
                continue
            itens.append(Item(
                rotulo_original=ROTULOS[sinal],
                id=f"s{sin}_{sinal.lower()}_r{int(rep)}",
                arquivo_original=rel,
                candidatos=[caminho],
                # "minds_03": global entre bases (o "03" de outra base é outra pessoa) e igual nas
                # duas câmeras. O 2_treinar.py lê esta coluna para separar validação por pessoa.
                linha={"sinalizador": f"minds_{sin}", "sinal_minds": f"{num}{sinal}", "repeticao": int(rep),
                       "camera": "canon" if chave == "rgb" else "kinect"}))
    return itens, estranhos


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fontes", nargs="+", choices=list(FONTES), default=list(FONTES),
                    help="quais pacotes organizar (padrão: os dois)")
    ap.add_argument("--origem-rgb", default=FONTES["rgb"][1], help="pasta extraída do pacote Canon")
    ap.add_argument("--origem-rgbd", default=FONTES["rgbd"][1], help="pasta extraída do pacote Kinect")
    ap.add_argument("--copiar", action="store_true",
                    help="copia os vídeos em vez de criar links simbólicos (~85 GB a mais em disco)")
    ap.add_argument("--modo", choices=["link", "copiar", "mover"], default=None,
                    help="link (padrão: origem compartilhada intacta), copiar, ou mover (para vídeos "
                         "baixados só para este projeto, ex.: pelo preparacao/baixar_videos.py; no "
                         "Windows, link simbólico exige permissão de administrador)")
    ap.add_argument("--executar", action="store_true", help="cria os links/cópias (sem isso: dry run)")
    ap.add_argument("--remover-orfaos", action="store_true",
                    help="apaga vídeos desta base na pasta unificada que o plano atual não gera")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    for chave in args.fontes:
        fonte = FONTES[chave][0]
        origem = os.path.abspath(args.origem_rgb if chave == "rgb" else args.origem_rgbd)
        if not os.path.isdir(origem):
            sys.exit(f"[erro] pasta não encontrada: {origem} (o HD de dados está montado?)")
        itens, estranhos = ler_base(chave, origem)
        if estranhos:
            print(f"[atenção] {fonte}: {len(estranhos)} arquivos fora do padrão ignorados: {estranhos[:5]}")
        organizar(fonte, itens, executar=args.executar, remover_orfaos=args.remover_orfaos,
                  modo=args.modo or ("copiar" if args.copiar else "link"))


if __name__ == "__main__":
    main()
