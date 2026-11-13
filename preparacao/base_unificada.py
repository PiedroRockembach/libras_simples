"""
Núcleo comum da preparação: junta TODAS as bases de dados numa pasta única de classes.

    dados/videos/<classe>/<fonte>__<id>.mp4    <- única pasta com classes (o 1_extrair.py lê daqui)
    dados/manifestos/<fonte>.csv               <- manifesto organizado de cada base
    dados/manifestos/<fonte>_original.csv      <- manifesto original da base (backup, permite refazer)
    dados/manifestos/indice.csv                <- índice geral (todas as bases), regenerado sempre
    dados/excluidos/<fonte>/                   <- vídeos excluídos e metadados, FORA das classes

Ideia central: cada base tem um ADAPTADOR (preparacao/<fonte>_pre_tratamento.py) que só sabe ler o
formato daquela base e devolve uma lista de `Item`. Todo o resto (normalizar rótulos, mover,
escrever manifestos, conferir) fica aqui, igual para todas. Assim "Bom Dia" de uma base e
"bom dia" de outra caem na MESMA pasta, e o modelo os trata como o mesmo sinal.

Por que o prefixo "<fonte>__" no nome do arquivo:
  - evita colisão entre bases (dois datasets podem ter um vídeo "001.mp4");
  - a origem continua rastreável até o cache (dados/landmarks/<classe>/<fonte>__<id>.npz), o que
    permite, por exemplo, validar com uma base e treinar com outra (= outros sinalizantes);
  - permite re-rodar uma base sem tocar nas outras (cada base só mexe nos próprios arquivos).

Para adicionar uma base nova: copie o incluir_tecnologia_pre_tratamento.py, troque a leitura do
formato e as exclusões, e confira os avisos de "possível duplicata" que o organizar() imprime.
"""
import collections
import csv
import hashlib
import os
import re
import shutil
import unicodedata
from dataclasses import dataclass, field

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DADOS = os.path.join(RAIZ, "dados")
PASTA_VIDEOS = os.path.join(DADOS, "videos")
PASTA_MANIFESTOS = os.path.join(DADOS, "manifestos")
PASTA_EXCLUIDOS = os.path.join(DADOS, "excluidos")
SEP = "__"                                   # separa fonte e id no nome do arquivo

# Grafias diferentes da mesma palavra (vale para TODAS as bases; aplicado após normalizar).
UNIR = {
    "quarta feira": "quarta-feira", "quinta feira": "quinta-feira",
    "segunda feira": "segunda-feira", "sexta feira": "sexta-feira",
    "terça feira": "terça-feira", "cachorro quente": "cachorro-quente",
}

# Nomes que o Windows reserva e não aceita como nome de pasta.
RESERVADOS = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(10)),
              *(f"lpt{i}" for i in range(10))}

COLUNAS = ["arquivo", "classe", "fonte", "id", "rotulo_original", "arquivo_original", "status"]


def normalizar_rotulo(rotulo):
    """
    Rótulo original -> nome de pasta canônico. As regras valem para todas as bases:
      - minúsculas: o Windows não diferencia "Bom dia/" de "Bom Dia/", e para o modelo seriam
        dois rótulos para o MESMO sinal (ele seria punido por acertar);
      - "/" vira " - ", e ?, aspas e ponto final são removidos/trocados (inválidos em pastas);
      - variantes numeradas ficam num formato único "(n)": "azul 2" -> "azul (2)".
        Variantes são sinais DIFERENTES para a mesma palavra: ficam em pastas separadas.
    """
    c = unicodedata.normalize("NFC", rotulo).strip().lower()  # NFC: "é" sempre no mesmo código
    c = re.sub(r'"([^"]*)"', r"(\1)", c)                       # cheio "comida" -> cheio (comida)
    c = c.replace("?", "")                                     # posso? -> posso
    c = " - ".join(p.strip() for p in c.split("/"))            # ele/você -> ele - você
    c = re.sub(r"^(.*\D) (\d+)$", r"\1 (\2)", c)               # azul 2 -> azul (2)
    c = re.sub(r"\s+", " ", c).strip().rstrip(".").strip()     # arroz. -> arroz
    c = UNIR.get(c, c)
    if not c or re.search(r'[\\/:*?"<>|]', c) or c in RESERVADOS or c.startswith("_"):
        raise ValueError(f"rótulo inválido após normalizar: {rotulo!r} -> {c!r}")
    return c


def chave_parecida(classe):
    """Sem acentos, espaços e pontuação: 'Cachorro-Quente' e 'cachorro quente' viram a mesma."""
    s = unicodedata.normalize("NFD", classe.lower())
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return re.sub(r"[^a-z0-9]", "", s)


@dataclass
class Item:
    """Um vídeo de uma base, como o adaptador o encontrou."""
    rotulo_original: str
    id: str                          # único dentro da base (ex.: id do YouTube)
    arquivo_original: str            # nome no pacote original (para rastrear)
    candidatos: list                 # caminhos onde o vídeo pode estar agora (o 1º que existir)
    linha: dict = field(default_factory=dict)   # linha do manifesto original (colunas extras)
    excluir: bool = False            # True = vai para dados/excluidos/<fonte>/
    sha256: str = ""                 # se a base informar, permite conferir integridade


def nome_arquivo(fonte, id_, ext=".mp4"):
    return f"{fonte}{SEP}{id_}{ext}"


def sha256(caminho):
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def arquivos_da_fonte(fonte):
    """Todos os vídeos desta fonte que já estão na pasta unificada."""
    if not os.path.isdir(PASTA_VIDEOS):
        return set()
    return {os.path.join(PASTA_VIDEOS, c, f)
            for c in os.listdir(PASTA_VIDEOS) if os.path.isdir(os.path.join(PASTA_VIDEOS, c))
            for f in os.listdir(os.path.join(PASTA_VIDEOS, c)) if f.startswith(fonte + SEP)}


def _classes_de_outras_fontes(fonte):
    """{classe: {fontes}} considerando só as OUTRAS bases já presentes na pasta unificada."""
    out = collections.defaultdict(set)
    if os.path.isdir(PASTA_VIDEOS):
        for c in os.listdir(PASTA_VIDEOS):
            p = os.path.join(PASTA_VIDEOS, c)
            if os.path.isdir(p):
                for f in os.listdir(p):
                    if SEP in f and not f.startswith(fonte + SEP):
                        out[c].add(f.split(SEP, 1)[0])
    return out


def organizar(fonte, itens, metadados=(), executar=False, verificar_sha=False,
              remover_orfaos=False):
    """
    Move os vídeos da base `fonte` para a pasta unificada. Idempotente: o que já está no
    destino é só conferido. Sem `executar`, apenas mostra o plano (dry run).
    metadados: caminhos (arquivos/pastas) da base que não são vídeos -> dados/excluidos/<fonte>/metadados
    """
    if not re.fullmatch(r"[a-z0-9]+(_[a-z0-9]+)*", fonte):
        raise ValueError(f"nome de fonte inválido: {fonte!r} (use minúsculas, dígitos e _ simples)")
    pasta_excl = os.path.join(PASTA_EXCLUIDOS, fonte)
    # Vídeos desta fonte que JÁ estão na pasta unificada, pelo nome do arquivo. Se uma regra de
    # rótulo mudou, o vídeo está na pasta da classe antiga: é só movê-lo, sem re-extrair o pacote.
    existentes = {}
    for f in arquivos_da_fonte(fonte):
        existentes.setdefault(os.path.basename(f), f)

    plano, linhas, ja_ok = [], [], 0
    grupos = collections.defaultdict(set)
    for it in itens:
        if it.excluir:
            classe, destino = "", os.path.join(pasta_excl, it.arquivo_original)
            rel, status = "", "excluido"
        else:
            classe = normalizar_rotulo(it.rotulo_original)
            ext = os.path.splitext(it.arquivo_original)[1].lower() or ".mp4"
            rel = f"{classe}/{nome_arquivo(fonte, it.id, ext)}"
            destino = os.path.join(PASTA_VIDEOS, classe, nome_arquivo(fonte, it.id, ext))
            status = "ok"
            grupos[classe].add(it.rotulo_original)
        candidatos = list(it.candidatos)
        if not it.excluir and os.path.basename(destino) in existentes:
            candidatos.insert(0, existentes[os.path.basename(destino)])
        origem = next((c for c in candidatos if os.path.isfile(c)), None)
        if os.path.isfile(destino):
            ja_ok += 1
        elif origem:
            plano.append((origem, destino))
        else:
            print(f"[ausente] {it.arquivo_original} (não encontrado: marcado como 'ausente')")
            classe, rel, status = "", "", "ausente"
        linhas.append({"arquivo": rel, "classe": classe, "fonte": fonte, "id": it.id,
                       "rotulo_original": it.rotulo_original,
                       "arquivo_original": it.arquivo_original, "status": status,
                       **{k: v for k, v in it.linha.items() if k not in COLUNAS}})
        it._destino = destino

    destinos = [d.lower() for _, d in plano] + \
               [it._destino.lower() for it in itens if os.path.isfile(it._destino)]
    repetidos = [d for d, n in collections.Counter(destinos).items() if n > 1]
    if repetidos:
        raise SystemExit(f"[erro] vídeos com o mesmo destino (id repetido?): {repetidos[:5]}")

    # Órfãos: arquivos desta fonte na pasta unificada que o plano atual não usa
    # (ex.: um id que saiu do manifesto, ou passou a ser excluído).
    usados = {os.path.normcase(it._destino) for it in itens if not it.excluir} | \
             {os.path.normcase(o) for o, _ in plano}
    orfaos = sorted(f for f in existentes.values() if os.path.normcase(f) not in usados)

    # ---------------------------------------------------------------- relatório
    print(f"=== {fonte}")
    unidas = {c: o for c, o in grupos.items() if len(o) > 1}
    if unidas:
        print("Rótulos unidos (nomes originais diferentes -> mesma classe):")
        for c, origs in sorted(unidas.items()):
            print(f"  {c!r:28s} <- {sorted(origs)}")
    cont = collections.Counter(l["classe"] for l in linhas if l["classe"])
    outras = _classes_de_outras_fontes(fonte)
    em_comum = sorted(set(cont) & set(outras))
    print(f"{len(plano)} vídeos a mover, {ja_ok} já no lugar | {len(cont)} classes "
          f"({len(em_comum)} já existiam em outras bases) | vídeos por classe: "
          f"{dict(sorted(collections.Counter(cont.values()).items()))} | "
          f"excluídos: {sum(l['status'] == 'excluido' for l in linhas)}")
    # Mesma palavra escrita diferente em bases diferentes = provável classe duplicada.
    chaves = collections.defaultdict(set)
    for c in outras:
        chaves[chave_parecida(c)].add(c)
    suspeitas = [(c, sorted(chaves[chave_parecida(c)] - {c})) for c in sorted(cont)
                 if chaves.get(chave_parecida(c), set()) - {c}]
    if suspeitas:
        print("[atenção] possíveis duplicatas com outras bases (ajuste UNIR se forem o mesmo sinal):")
        for c, parecidas in suspeitas:
            print(f"  {c!r} ~ {parecidas}")
    meta = [m for m in metadados if os.path.exists(m)]
    if meta:
        print(f"Metadados a tirar das classes (-> {pasta_excl}{os.sep}metadados): "
              f"{[os.path.basename(m) for m in meta]}")
    if orfaos:
        print(f"[atenção] {len(orfaos)} arquivos órfãos desta fonte "
              f"({'serão removidos' if remover_orfaos else 'use --remover-orfaos'}): "
              f"{[os.path.relpath(o, PASTA_VIDEOS) for o in orfaos[:5]]}")

    if not executar:
        print("(dry run: nada foi movido; rode com --executar)")
        return linhas

    # ---------------------------------------------------------------- execução
    os.makedirs(PASTA_MANIFESTOS, exist_ok=True)
    erros = 0
    for origem, destino in plano:
        try:
            os.makedirs(os.path.dirname(destino), exist_ok=True)
            shutil.move(origem, destino)
        except OSError as e:
            erros += 1
            print(f"[erro] {origem} -> {destino}: {e}")
    for m in meta:
        os.makedirs(os.path.join(pasta_excl, "metadados"), exist_ok=True)
        alvo = os.path.join(pasta_excl, "metadados", os.path.basename(m))
        if not os.path.exists(alvo):
            shutil.move(m, alvo)
    if remover_orfaos:
        for o in orfaos:
            os.remove(o)        # seguro: tudo é reproduzível a partir do pacote original
    for c in os.listdir(PASTA_VIDEOS):                 # pastas de classe que ficaram vazias
        p = os.path.join(PASTA_VIDEOS, c)
        if os.path.isdir(p) and not os.listdir(p):
            os.rmdir(p)

    escrever_csv(os.path.join(PASTA_MANIFESTOS, f"{fonte}.csv"), linhas)
    atualizar_indice()
    print(f"Movidos: {len(plano) - erros}, erros: {erros}")
    conferir(fonte, linhas, verificar_sha)
    return linhas


def escrever_csv(caminho, linhas):
    campos = list(COLUNAS)
    for l in linhas:
        campos += [k for k in l if k not in campos]
    with open(caminho, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(linhas)


def atualizar_indice():
    """Regenera dados/manifestos/indice.csv juntando as colunas comuns de todas as bases."""
    todas = []
    for nome in sorted(os.listdir(PASTA_MANIFESTOS)):
        if nome.endswith(".csv") and nome != "indice.csv" and not nome.endswith("_original.csv"):
            with open(os.path.join(PASTA_MANIFESTOS, nome), encoding="utf-8") as f:
                todas += [{k: r.get(k, "") for k in COLUNAS} for r in csv.DictReader(f)]
    escrever_csv(os.path.join(PASTA_MANIFESTOS, "indice.csv"), todas)


def conferir(fonte=None, linhas=None, verificar_sha=False):
    """Mostra o que o 1_extrair.py vai enxergar e, opcionalmente, confere o sha256."""
    pastas = [d for d in os.listdir(PASTA_VIDEOS) if os.path.isdir(os.path.join(PASTA_VIDEOS, d))]
    por_fonte = collections.Counter()
    fontes_por_classe = collections.defaultdict(set)
    for c in pastas:
        for f in os.listdir(os.path.join(PASTA_VIDEOS, c)):
            fo = f.split(SEP, 1)[0] if SEP in f else "(sem fonte)"
            por_fonte[fo] += 1
            fontes_por_classe[c].add(fo)
    soltos = [f for f in os.listdir(PASTA_VIDEOS) if os.path.isfile(os.path.join(PASTA_VIDEOS, f))]
    multi = sum(len(v) > 1 for v in fontes_por_classe.values())
    print(f"Pasta unificada: {len(pastas)} classes, {sum(por_fonte.values())} vídeos "
          f"{dict(por_fonte)}, {multi} classes com 2+ bases, {len(soltos)} arquivos soltos")
    if verificar_sha and linhas:
        com_sha = [l for l in linhas if l["arquivo"] and l.get("sha256")]
        ruins = [l["arquivo"] for l in com_sha
                 if sha256(os.path.join(PASTA_VIDEOS, l["arquivo"])) != l["sha256"]]
        print(f"sha256 ({fonte}): {len(com_sha)} conferidos, {len(ruins)} diferentes {ruins[:5]}")
