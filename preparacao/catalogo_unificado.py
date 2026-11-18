"""
Catálogo unificado de treino: UM CSV com todos os vídeos (nossos + fontes do MALTA-LIBRAS +
V-LIBRASIL), restrito às classes com quantidade significativa de dados para treinar.

    python preparacao/catalogo_unificado.py --islr ../ISLR_LIBRAS [--min-videos 3]
    -> dados/catalogo_treino.csv   (uma linha por vídeo)

FONTES
  Nossas (já em dados/videos): incluir_tecnologia, minds_libras_rgb, minds_libras_rgbd.
  MALTA-LIBRAS (repositório ISLR_LIBRAS, Malta Lab), arquivo curado MALTA_LIBRAS_original.csv:
    malta_acessibilidade_v2 (INES v2), malta_acessibilidade_v3 (INES v3), malta_spreadthesign,
    malta_ufsc_corpus, malta_ufsc_signbank, malta_ufv, malta_usp. O dicionário "Youtube" do MALTA
    é a NOSSA Incluir Tecnologia: não entra de novo (seria duplicata e vazaria entre treino e
    validação), mas é usado para descobrir a variante e o sinalizante dos nossos vídeos.
  V-LIBRASIL (UFPE): links_videos_vlibrasil.csv; ~3 sinalizantes por sinal (identidade desconhecida).

POR QUE CADA REGRA (foco: qualidade do reconhecimento, e o máximo de classes que ela permitir)
  - Rótulo = coluna `checked` do MALTA, NÃO `lemma`: a `lemma` é de um lematizador automático e erra
    ("açaí"->"acair", "água"->"aguo", "a disposição"->"a"). Normalizada com normalizar_rotulo().
  - Classe = (palavra, VARIANTE). O MALTA anotou à mão a variante (`group`): "abrir" tem 6 formas.
    Juntar só pela palavra misturaria sinais diferentes na mesma classe (ruído de rótulo).
      * vídeo com variante anotada (MALTA) -> classe daquela variante;
      * nossa classe ganha a variante pelo vídeo correspondente no "Youtube" do MALTA;
      * fonte sem anotação (V-LIBRASIL, nossa classe sem par) só é unida se a palavra tiver UMA
        variante conhecida; com várias, fica de fora (ambígua).
    Nomes: classe nossa mantém o nome (o cache de landmarks é indexado por ele); variante nova de
    uma palavra que tem outras vira "palavra (vN)", N = group do MALTA.
  - Significativa = >= --min-videos EXEMPLOS distintos obteníveis (padrão 3: o mínimo para treinar
    com dois e validar com um). Vídeos diferentes da mesma pessoa contam (decisão do usuário).
    Canon e Kinect do MINDS gravaram a MESMA execução: contam como um exemplo só.
  - confianca_classe: "alta" = atinge o mínimo sem o V-LIBRASIL; "media" = precisa dele (variante
    não conferida entre os 3 sinalizantes, identidade das pessoas desconhecida).
  - n_sinalizadores_classe (informativo, para validar com pessoa nunca vista):
      * IDs: MALTA `actor` (global entre dicionários) -> malta_<id>; MINDS -> minds_<NN>.
      * Incluir sem par no MALTA: desconhecido, mas o canal tem os atores 41/51/57; só conta como
        pessoa nova se nenhum deles já estiver na classe (contagem conservadora).
      * V-LIBRASIL: cada vídeo do sinal conta como uma pessoa (até 3); a ordem no CSV é
        cronológica e as datas não separam as pessoas, então a identidade é desconhecida.
  - Obtenível = já temos (local) ou tem link. Acessibilidade v2 e UFSC Corpus não têm links nos
    CSVs (o Corpus usa os mesmos nomes de arquivo do SignBank, mas é outra gravação: casar pelos
    links do SignBank baixaria a duplicata errada). Ficam listados como `sem_link` e não contam.
"""
import argparse
import collections
import csv
import os
import re
import sys
import unicodedata
import urllib.parse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from base_unificada import DADOS, PASTA_MANIFESTOS, normalizar_rotulo  # noqa: E402

DICIONARIOS = {   # dicionário no MALTA -> nome da fonte aqui
    "Acessibilidades2": "malta_acessibilidade_v2", "Acessibilidades3": "malta_acessibilidade_v3",
    "SpreadTheSign": "malta_spreadthesign", "UFSC": "malta_ufsc_corpus",
    "UFSC_V2": "malta_ufsc_signbank", "UFV": "malta_ufv", "USP": "malta_usp",
}
ATORES_INCLUIR = {"malta_41", "malta_51", "malta_57"}   # atores do "Youtube" (= Incluir) no MALTA
SUFIXO_VARIANTE = re.compile(r"^(.*) \((\d+)\)$")      # "feliz (2)" -> ("feliz", "2")

COLUNAS = ["classe", "palavra", "variante_malta", "origem_variante", "fonte", "dicionario",
           "sinalizador", "status", "arquivo_local", "link", "rotulo_original",
           "n_exemplos_classe", "n_sinalizadores_classe", "n_videos_classe", "fontes_classe",
           "confianca_classe", "id"]


def norm(rotulo):
    try:
        return normalizar_rotulo(rotulo.replace("_", " ").strip())
    except ValueError:
        return None


def nome_base(caminho):
    """Nome do arquivo comparável entre o CSV do MALTA e os links (sem %xx, NFC, minúsculas)."""
    b = urllib.parse.unquote(os.path.basename(caminho.strip("[]'\" ")))
    return unicodedata.normalize("NFC", b).lower()


def ler_csv(caminho):
    with open(caminho, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def carregar_links(pasta):
    """{fonte: {chave: {links}}} com a chave de casamento própria de cada dicionário."""
    L = collections.defaultdict(lambda: collections.defaultdict(set))
    for x in ler_csv(os.path.join(pasta, "links_videos_acessibilidade_brasil.csv")):
        L["malta_acessibilidade_v3"][nome_base(x["Link"])].add(x["Link"])
    for x in ler_csv(os.path.join(pasta, "links_videos_ufsc_signbank.csv")):
        L["malta_ufsc_signbank"][nome_base(x["Link"])].add(x["Link"])
    for x in ler_csv(os.path.join(pasta, "links_videos_ufv.csv")):
        b = nome_base(x["Link"])
        L["malta_ufv"][b].add(x["Link"])
        L["malta_ufv"][re.sub(r"^\d+-", "", b)].add(x["Link"])          # "01553-abrir.mp4" -> "abrir.mp4"
    for x in ler_csv(os.path.join(pasta, "links_videos_usp.csv")):
        b = re.sub(r"^libras_glossario_aula\d+_(.*)_stream(\.\w+)$", r"\1\2", nome_base(x["Link"]))
        L["malta_usp"][b].add(x["Link"])
    for x in ler_csv(os.path.join(pasta, "links_videos_spreadthesign.csv")):
        c = norm(x["Palavra"])                       # o MALTA numera os arquivos: casa pela palavra
        if c:
            L["malta_spreadthesign"][c].add(x["Link"])
    return L


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--islr", default=os.path.join(os.path.dirname(os.path.dirname(DADOS)), "ISLR_LIBRAS"),
                    help="pasta do repositório ISLR_LIBRAS (Malta Lab)")
    ap.add_argument("--min-videos", type=int, default=3,
                    help="exemplos distintos por classe (Canon+Kinect da mesma execução = 1)")
    ap.add_argument("--saida", default=os.path.join(DADOS, "catalogo_treino.csv"))
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    malta = ler_csv(os.path.join(args.islr, "dataset_intersections", "MALTA_LIBRAS_original.csv"))
    links = carregar_links(os.path.join(args.islr, "video_downloads"))
    for x in malta:
        x["palavra"] = norm(x["checked"])

    # ------------------------------------------------------------ 1) nossos vídeos
    nossos = []
    for fonte in ("incluir_tecnologia", "minds_libras_rgb", "minds_libras_rgbd"):
        p = os.path.join(PASTA_MANIFESTOS, f"{fonte}.csv")
        if not os.path.exists(p):
            print(f"[aviso] manifesto ausente: {p}")
            continue
        for x in ler_csv(p):
            if x["status"] == "ok" and x["classe"]:
                nossos.append({"classe": x["classe"], "fonte": fonte, "dicionario": fonte, "id": x["id"],
                               "sinalizador": x.get("sinalizador", ""), "status": "local",
                               "arquivo_local": f"dados/videos/{x['arquivo']}", "link": "",
                               "rotulo_original": x["rotulo_original"], "titulo": x.get("title", ""),
                               "altura": x.get("height", "")})

    # Vídeos do "Youtube" do MALTA -> nossos vídeos da Incluir, por TÍTULO + ALTURA. O canal tem
    # títulos repetidos em uploads diferentes ("Abacate" em 360p e em 1080p, às vezes outra pessoa):
    # o arquivo do MALTA traz a resolução no nome. Se ainda empatar, não atribui (não sabemos qual).
    def chave_titulo(t):
        return re.sub(r"\s+", " ", t.replace("/", "_")).strip().lower()
    por_titulo = collections.defaultdict(list)
    for v in nossos:
        if v["fonte"] == "incluir_tecnologia":
            por_titulo[chave_titulo(v["titulo"])].append(v)
    casados, empates = 0, 0
    for x in malta:
        if x["dictionary"] != "Youtube":
            continue
        arq = os.path.basename(x["path"].strip("[]'\" "))
        m = re.search(r"\s*\((\d{3,4})p_[^)]*\)\.\w+$", arq)
        t = arq[:m.start()] if m else os.path.splitext(arq)[0]
        cand = por_titulo.get(chave_titulo(t), [])
        if len(cand) > 1 and m:
            cand = [v for v in cand if v["altura"] == m.group(1)]
        if len(cand) == 1:
            cand[0]["sinalizador"] = f"malta_{x['actor']}"
            cand[0]["chave_malta"] = (x["palavra"], x["group"])
            casados += 1
        elif cand:
            empates += 1

    # ------------------------------------------------------------ 2) variante de cada classe nossa
    grupos_malta = collections.defaultdict(set)          # palavra -> {group} (fora do Youtube)
    for x in malta:
        if x["dictionary"] != "Youtube" and x["palavra"]:
            grupos_malta[x["palavra"]].add(x["group"])
    classes_nossas = collections.defaultdict(list)
    for v in nossos:
        classes_nossas[v["classe"]].append(v)
    bases_nossas = collections.defaultdict(set)          # "feliz" -> {"feliz (1)", "feliz (2)"}
    for c in classes_nossas:
        m = SUFIXO_VARIANTE.match(c)
        bases_nossas[m.group(1) if m else c].add(c)

    chave_da_classe, origem_da_classe = {}, {}
    for c, vids in classes_nossas.items():
        chaves = {v["chave_malta"] for v in vids if "chave_malta" in v}
        base = SUFIXO_VARIANTE.match(c).group(1) if SUFIXO_VARIANTE.match(c) else c
        if len(chaves) == 1:
            chave_da_classe[c], origem_da_classe[c] = chaves.pop(), "anotada_malta"
        elif not chaves and c == base and len(grupos_malta.get(c, ())) == 1 and len(bases_nossas[c]) == 1:
            chave_da_classe[c], origem_da_classe[c] = (c, next(iter(grupos_malta[c]))), "inferida_unica"
        else:
            origem_da_classe[c] = "propria" if not grupos_malta.get(base) else "ambigua"
    classe_da_chave = {k: c for c, k in chave_da_classe.items()}

    # ------------------------------------------------------------ 3) vídeos do MALTA
    def nome_classe(palavra, grupo):
        if (palavra, grupo) in classe_da_chave:
            return classe_da_chave[(palavra, grupo)]
        if len(grupos_malta[palavra]) == 1 and palavra not in bases_nossas:
            return palavra
        return f"{palavra} (v{grupo})"

    externos, sem_par = [], collections.Counter()
    for x in malta:
        if x["dictionary"] == "Youtube" or not x["palavra"]:
            continue
        fonte = DICIONARIOS[x["dictionary"]]
        if fonte == "malta_spreadthesign":
            ls = links[fonte].get(x["palavra"], set())
            ls = ls if len(ls) == 1 else set()              # palavra com 2+ vídeos: não sei qual é
        else:
            ls = links[fonte].get(nome_base(x["path"]), set()) if fonte in links else set()
        if not ls:
            sem_par[fonte] += 1
        externos.append({"classe": nome_classe(x["palavra"], x["group"]), "fonte": fonte,
                         "dicionario": x["dictionary"], "sinalizador": f"malta_{x['actor']}",
                         "status": "baixar" if ls else "sem_link", "arquivo_local": "",
                         "link": sorted(ls)[0] if ls else "", "rotulo_original": x["checked"],
                         "palavra": x["palavra"], "variante_malta": x["group"],
                         "origem_variante": "anotada_malta"})

    # ------------------------------------------------------------ 4) V-LIBRASIL
    ambiguas_vl = collections.Counter()
    for x in ler_csv(os.path.join(args.islr, "video_downloads", "links_videos_vlibrasil.csv")):
        w = norm(x["sign"])
        if not w:
            continue
        # formas conhecidas = variantes do MALTA + nossas classes dessa palavra sem variante casada
        n_var = len(grupos_malta.get(w, ())) + len(bases_nossas.get(w, set()) - set(classe_da_chave))
        if n_var > 1 or origem_da_classe.get(w) == "ambigua":
            ambiguas_vl[w] += 1                            # várias formas conhecidas: não dá para unir
            continue
        if w in grupos_malta:
            classe, var, origem = nome_classe(w, next(iter(grupos_malta[w]))), next(iter(grupos_malta[w])), "inferida_unica"
        else:
            classe, var, origem = w, "", "propria" if w not in classes_nossas else "inferida_unica"
        externos.append({"classe": classe, "fonte": "vlibrasil", "dicionario": "V-LIBRASIL",
                         "sinalizador": "", "status": "baixar", "arquivo_local": "",
                         "link": x["link"], "rotulo_original": x["sign"], "palavra": w,
                         "variante_malta": var, "origem_variante": origem})

    for v in nossos:
        c = v["classe"]
        k = chave_da_classe.get(c)
        m = SUFIXO_VARIANTE.match(c)
        v.update(palavra=k[0] if k else (m.group(1) if m else c), variante_malta=k[1] if k else "",
                 origem_variante=origem_da_classe[c])
    todos = nossos + externos

    # ------------------------------------------------------------ 5) contagem e filtro
    por_classe = collections.defaultdict(list)
    for v in todos:
        por_classe[v["classe"]].append(v)

    def execucao(v):
        """Canon e Kinect do MINDS gravaram a MESMA execução: contam como um exemplo só."""
        return ("minds", v["id"]) if v["fonte"].startswith("minds_libras") else (v["fonte"], v["link"] or v["arquivo_local"])

    def contar(vids):
        """(exemplos distintos, exemplos sem o V-LIBRASIL, pessoas distintas, vídeos obteníveis)."""
        ok = [v for v in vids if v["status"] != "sem_link"]
        ex = {execucao(v) for v in ok}
        ex_sem_vl = {e for e in ex if e[0] != "vlibrasil"}
        # Pessoas: só informativo (útil para validar com pessoa nunca vista).
        ids = {v["sinalizador"] for v in ok if v["sinalizador"]}
        n = len(ids)
        if any(v["fonte"] == "incluir_tecnologia" and not v["sinalizador"] for v in ok) and not ids & ATORES_INCLUIR:
            n += 1                                     # Incluir sem par: conta 1 se nenhum ator do canal
        n += min(3, sum(v["fonte"] == "vlibrasil" for v in ok))
        return len(ex), len(ex_sem_vl), n, len(ok)

    stats = {c: contar(v) for c, v in por_classe.items()}
    mantidas = {c for c, s in stats.items() if s[0] >= args.min_videos}
    # alta: significativa mesmo sem o V-LIBRASIL (variante anotada ou fonte própria).
    # media: precisa do V-LIBRASIL para chegar ao mínimo (variante não conferida).
    confianca = {c: "alta" if stats[c][1] >= args.min_videos else "media" for c in mantidas}

    linhas = []
    for c in sorted(mantidas):
        ne, _, n, nv = stats[c]
        fontes = ";".join(sorted({v["fonte"] for v in por_classe[c] if v["status"] != "sem_link"}))
        for v in sorted(por_classe[c], key=lambda v: (v["status"] != "local", v["fonte"])):
            linhas.append({**{k: v.get(k, "") for k in COLUNAS}, "n_sinalizadores_classe": n,
                           "n_exemplos_classe": ne, "n_videos_classe": nv, "fontes_classe": fontes,
                           "confianca_classe": confianca[c]})
    with open(args.saida, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUNAS)
        w.writeheader()
        w.writerows(linhas)

    # ------------------------------------------------------------ relatório
    print(f"Incluir casados com o 'Youtube' do MALTA: {casados} vídeos ({empates} empates não atribuídos)")
    print(f"Classes nossas: {len(classes_nossas)} | origem da variante: "
          f"{dict(collections.Counter(origem_da_classe.values()))}")
    print(f"MALTA sem link (não contam): {dict(sem_par)}")
    print(f"V-LIBRASIL descartados por variante ambígua: {sum(ambiguas_vl.values())} vídeos "
          f"({len(ambiguas_vl)} palavras)")
    print(f"\nClasses candidatas: {len(por_classe)}")
    for lim in (2, 3, 4, 5, 6, 8, 10):
        print(f"  >= {lim:2d} exemplos: {sum(s[0] >= lim for s in stats.values()):5d} classes "
              f"(sem o V-LIBRASIL: {sum(s[1] >= lim for s in stats.values()):4d})")
    ob = [v for v in linhas if v["status"] != "sem_link"]
    print(f"\nMANTIDAS (>= {args.min_videos} exemplos): {len(mantidas)} classes, {len(ob)} vídeos obteníveis "
          f"({sum(v['status'] == 'local' for v in ob)} já temos, {sum(v['status'] == 'baixar' for v in ob)} a baixar)"
          f" + {len(linhas) - len(ob)} listados sem link")
    print("  vídeos por fonte:", dict(collections.Counter(v["fonte"] for v in ob).most_common()))
    print("  exemplos/classe:", dict(sorted(collections.Counter(min(stats[c][0], 20) for c in mantidas).items())))
    print("  classes com >= 2 / >= 3 pessoas (permitem validar com pessoa nova):",
          sum(stats[c][2] >= 2 for c in mantidas), "/", sum(stats[c][2] >= 3 for c in mantidas))
    so_vl = [c for c in mantidas if {v["fonte"] for v in por_classe[c] if v["status"] != "sem_link"} == {"vlibrasil"}]
    print(f"  confiança: {dict(collections.Counter(confianca.values()))} | só V-LIBRASIL: {len(so_vl)}")
    for nivel in ("alta", "media"):
        cs = [c for c in mantidas if confianca[c] == nivel]
        ja = sum(c in classes_nossas for c in cs)
        print(f"  {nivel}: {len(cs)} classes ({ja} já existem em dados/videos, {len(cs) - ja} novas), "
              f"{sum(stats[c][3] for c in cs)} vídeos")
    print(f"\n-> {args.saida}")


if __name__ == "__main__":
    main()
