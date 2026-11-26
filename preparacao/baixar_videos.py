"""
Baixa automaticamente os vídeos brutos de todas as bases, para reproduzir o projeto em outra máquina.

    python preparacao/baixar_videos.py                          # plano: o que falta e quanto espaço
    python preparacao/baixar_videos.py --executar               # baixa tudo o que falta
    python preparacao/baixar_videos.py --executar --organizar   # baixa e leva para dados/videos/
    python preparacao/baixar_videos.py --fontes incluir_tecnologia minds_libras_rgb --executar

FONTES (o que cada uma baixa, de onde e para onde)
  incluir_tecnologia  YouTube, pelos video_id do manifesto versionado
                      (dados/manifestos/incluir_tecnologia_original.csv) -> dados/brutos/incluir_tecnologia/videos/
                      Usa yt-dlp (pip install yt-dlp): só o vídeo, sem áudio (o pipeline não usa som),
                      em mp4 até 1080p, como no pacote original.
  minds_libras_rgb    Zenodo 10.5281/zenodo.2667329 (Canon; 12 zips, ~65 GB)
  minds_libras_rgbd   Zenodo 10.5281/zenodo.4322984 (Kinect; 12 zips, ~30 GB)
                      -> dados/brutos/<fonte>/extracted/. Lista os arquivos pela API do Zenodo, retoma
                      download interrompido, confere o MD5 e extrai SÓ os vídeos RGB que o adaptador
                      reconhece (profundidade/esqueleto do Kinect ficam de fora). O zip é apagado depois
                      de extraído (--manter-zips para guardar).
  vlibrasil           Kaggle davimedio01/v-librasil (cópia completa do V-LIBRASIL da UFPE, ~10,9 GB),
                      via kagglehub (pip install kagglehub; dataset público, não precisa de login)
                      -> dados/brutos/vlibrasil/videos UFPE (V-LIBRASIL)/{data/*.mp4, annotations.csv}.
                      O site da UFPE (fonte dos links do catálogo) ficou fora do ar em 2026-11. Manual:
                      curl -L -o v-librasil.zip https://www.kaggle.com/api/v1/datasets/download/davimedio01/v-librasil
                      e extrair o zip em dados/brutos/vlibrasil/.
  catalogo            os links com status "baixar" de dados/catalogo_treino.csv (fontes do MALTA-LIBRAS)
                      -> dados/brutos/<fonte>/<nome do arquivo no link>. Também dá para pedir uma fonte
                      só: malta_acessibilidade_v3, malta_spreadthesign, malta_ufsc_signbank, malta_ufv,
                      malta_usp. (O V-LIBRASIL vem do Kaggle, acima, e não dos links da UFPE.)

POR QUE ASSIM
  - Idempotente: o que já está baixado (ou já organizado em dados/videos/) é pulado; um download
    interrompido continua de onde parou (HTTP Range) e só vira arquivo final quando completo
    (".part" -> nome final), então nunca sobra um vídeo pela metade que pareceria pronto.
  - Educado com os servidores: no máximo 2 conexões simultâneas por site, novas tentativas com
    espera crescente; 404/410 não são repetidos (link quebrado não volta sozinho).
  - Rastreável: cada execução grava dados/brutos/_logs/<fonte>.csv com link, arquivo, status,
    tamanho e sha256 de cada vídeo.
  - --organizar chama os ADAPTADORES de preparacao/ (a lógica de organização continua num lugar só).
    As fontes do catálogo ainda não têm adaptador: ficam baixadas em dados/brutos/.

AVISO SOBRE O SHA256 DA INCLUIR: o YouTube reprocessa os vídeos, então um download novo pode não ter
o mesmo sha256 do pacote original (incluir_tecnologia.tar.gz). O conteúdo é o mesmo sinal; se o
pacote original estiver disponível, prefira-o (o adaptador aceita o .tar.gz direto).
"""
import argparse
import collections
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, AQUI)
import minds_libras_pre_tratamento as minds  # noqa: E402  (padrões de nome dos vídeos do MINDS)
from base_unificada import DADOS, PASTA_MANIFESTOS, PASTA_VIDEOS  # noqa: E402
from incluir_tecnologia_pre_tratamento import EXCLUIR as EXCLUIR_INCLUIR  # noqa: E402

BRUTOS = os.path.join(DADOS, "brutos")
LOGS = os.path.join(BRUTOS, "_logs")
CATALOGO = os.path.join(DADOS, "catalogo_treino.csv")
# Identificação genérica (sem dados pessoais): alguns servidores recusam clientes sem User-Agent.
USER_AGENT = "Mozilla/5.0 (compatible; libras-pesquisa-academica/1.0)"
ZENODO = {"minds_libras_rgb": ("2667329", "rgb"), "minds_libras_rgbd": ("4322984", "rgbd")}
FONTES_CATALOGO = ["malta_acessibilidade_v3", "malta_spreadthesign", "malta_ufsc_signbank",
                   "malta_ufv", "malta_usp"]
KAGGLE_VLIBRASIL = "davimedio01/v-librasil"
PASTA_VLIBRASIL = os.path.join(BRUTOS, "vlibrasil")
BYTES_VLIBRASIL = 10_853_185_748                  # tamanho informado pela API do Kaggle (2026-11-24)
N_VIDEOS_VLIBRASIL = 4086                         # 4089 da UFPE menos os 3 do error.csv
TODAS = ["incluir_tecnologia", *ZENODO, "vlibrasil", "catalogo"]
# YouTube: só vídeo (sem áudio = sem precisar juntar trilhas), mp4 até 1080p; senão o melhor que houver.
FORMATO_YT = "bv*[ext=mp4][height<=1080]/bv*[height<=1080]/b[height<=1080]/b"
POR_SITE = 2                       # conexões simultâneas por site
FALHAS_PARA_DESISTIR = 5           # falhas SEGUIDAS num site para pular o resto dele nesta execução

_semaforos = collections.defaultdict(lambda: threading.Semaphore(POR_SITE))
_falhas_seguidas = collections.Counter()
_trava = threading.Lock()


def _contexto_ssl():
    """
    Verificação de certificado pelo sistema operacional (truststore), quando instalado. Alguns
    servidores (ex.: o SignBank da UFSC) não enviam o certificado intermediário: o navegador e o
    verificador do Windows/macOS o buscam sozinhos, o OpenSSL do Python não, e a conexão falha.
    Com o truststore ela continua VERIFICADA; desligar a verificação seria aceitar conteúdo de quem
    estiver no meio do caminho.
    """
    try:
        import ssl
        import truststore
        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    except ImportError:
        return None


CONTEXTO_SSL = _contexto_ssl()


def site(url):
    return urllib.parse.urlparse(url).netloc


def semaforo(url):
    with _trava:
        return _semaforos[site(url)]


def registrar(url, ok):
    """'Disjuntor' por site: um servidor fora do ar faria cada link esperar todas as tentativas."""
    with _trava:
        _falhas_seguidas[site(url)] = 0 if ok else _falhas_seguidas[site(url)] + 1


def site_desistido(url):
    with _trava:
        return _falhas_seguidas[site(url)] >= FALHAS_PARA_DESISTIR


def hash_arquivo(caminho, algoritmo="sha256"):
    h = hashlib.new(algoritmo)
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def mb(n):
    return f"{n / 2**30:.1f} GB" if n >= 2**30 else f"{n / 2**20:.0f} MB"


# ----------------------------------------------------------------------------- HTTP
def baixar_http(url, destino, md5=None, tentativas=4, timeout=60):
    """Baixa `url` em `destino` com retomada e novas tentativas. Devolve o nº de bytes."""
    if site_desistido(url):
        raise RuntimeError(f"pulado: {site(url)} falhou {FALHAS_PARA_DESISTIR}+ vezes seguidas "
                           "(fora do ar?); rode de novo mais tarde")
    try:
        n = _baixar_http(url, destino, md5, tentativas, timeout)
    except Exception:
        registrar(url, False)
        raise
    registrar(url, True)
    return n


def _baixar_http(url, destino, md5, tentativas, timeout):
    os.makedirs(os.path.dirname(destino), exist_ok=True)
    parte = destino + ".part"
    erro = None
    for t in range(tentativas):
        if t and site_desistido(url):
            break
        try:
            ini = os.path.getsize(parte) if os.path.exists(parte) else 0
            cab = {"User-Agent": USER_AGENT}
            if ini:
                cab["Range"] = f"bytes={ini}-"
            with semaforo(url), urllib.request.urlopen(urllib.request.Request(url, headers=cab),
                                                       timeout=timeout, context=CONTEXTO_SSL) as r:
                if ini and r.status != 206:          # servidor ignorou o Range: recomeça do zero
                    ini = 0
                tipo = r.headers.get("Content-Type", "")
                if "text/html" in tipo:
                    raise ValueError(f"o link devolveu uma página HTML ({tipo}), não um vídeo")
                with open(parte, "ab" if ini else "wb") as f:
                    shutil.copyfileobj(r, f, 1 << 20)
            if os.path.getsize(parte) < 1024:
                raise ValueError("arquivo baixado com menos de 1 KB")
            if md5 and hash_arquivo(parte, "md5") != md5:
                os.remove(parte)
                raise ValueError("MD5 diferente do informado pela fonte (arquivo corrompido)")
            os.replace(parte, destino)
            return os.path.getsize(destino)
        except urllib.error.HTTPError as e:
            erro = e
            if e.code in (404, 410):                 # link quebrado: repetir não adianta
                break
            if e.code == 416 and os.path.exists(parte):   # Range inválido: recomeça
                os.remove(parte)
        except (urllib.error.URLError, OSError, ValueError) as e:
            erro = e
            if "CERTIFICATE_VERIFY_FAILED" in str(e):   # não é passageiro: repetir não adianta
                break
        if t < tentativas - 1:
            time.sleep(min(30, 3 * 2 ** t))
    raise RuntimeError(f"{type(erro).__name__}: {erro}")


def escrever_log(fonte, linhas):
    if not linhas:
        return
    os.makedirs(LOGS, exist_ok=True)
    with open(os.path.join(LOGS, f"{fonte}.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0]))
        w.writeheader()
        w.writerows(linhas)


# ----------------------------------------------------------------------------- Incluir (YouTube)
def plano_incluir():
    """[(video_id, arquivo_original, destino)] do que falta; e quantos já existem."""
    manif = os.path.join(PASTA_MANIFESTOS, "incluir_tecnologia_original.csv")
    with open(manif, encoding="utf-8") as f:
        linhas = [r for r in csv.DictReader(f) if r["file"] not in EXCLUIR_INCLUIR]
    organizados = set()
    p = os.path.join(PASTA_MANIFESTOS, "incluir_tecnologia.csv")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            organizados = {r["arquivo_original"] for r in csv.DictReader(f)
                           if r["arquivo"] and os.path.isfile(os.path.join(PASTA_VIDEOS, r["arquivo"]))}
    pasta = os.path.join(BRUTOS, "incluir_tecnologia", "videos")
    falta = [(r["video_id"], r["file"], os.path.join(pasta, r["file"]), int(r["size"] or 0))
             for r in linhas if r["file"] not in organizados and not os.path.isfile(os.path.join(pasta, r["file"]))]
    return falta, len(linhas) - len(falta), manif


def baixar_incluir(falta, manif, paralelo):
    try:
        import yt_dlp
    except ImportError:
        sys.exit("[erro] a Incluir Tecnologia vem do YouTube e precisa do yt-dlp: pip install yt-dlp")
    os.makedirs(os.path.join(BRUTOS, "incluir_tecnologia", "videos"), exist_ok=True)
    # Cópia do manifesto ao lado dos vídeos: o adaptador também sabe ler a pasta sozinha.
    shutil.copy2(manif, os.path.join(BRUTOS, "incluir_tecnologia", "MANIFEST.csv"))
    tem_ffmpeg = shutil.which("ffmpeg") is not None

    def um(item):
        vid, arq, destino, _ = item
        base = os.path.splitext(destino)[0].replace("%", "%%")   # % é especial no molde do yt-dlp
        opts = {"format": FORMATO_YT, "outtmpl": base + ".%(ext)s", "quiet": True, "no_warnings": True,
                "noprogress": True, "retries": 5, "fragment_retries": 5, "overwrites": False}
        if tem_ffmpeg:   # se só houver webm, troca o contêiner para mp4 (sem recodificar)
            opts["postprocessors"] = [{"key": "FFmpegVideoRemuxer", "preferedformat": "mp4"}]
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={vid}"])
        if not os.path.isfile(destino):
            raise RuntimeError("baixou, mas não em mp4 (instale o ffmpeg para converter o contêiner)")
        return os.path.getsize(destino)

    return executar_lote("incluir_tecnologia", falta, um, paralelo,
                         lambda it: {"video_id": it[0], "arquivo": it[1]})


# ----------------------------------------------------------------------------- MINDS (Zenodo)
def arquivos_zenodo(record):
    url = f"https://zenodo.org/api/records/{record}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60, context=CONTEXTO_SSL) as r:
        dados = json.load(r)
    return [{"nome": f["key"], "bytes": int(f["size"]), "md5": f["checksum"].split(":", 1)[1],
             "url": f["links"]["self"]} for f in dados["files"]]


def plano_minds(fonte):
    record, _ = ZENODO[fonte]
    zips = arquivos_zenodo(record)
    pasta_zips = os.path.join(BRUTOS, fonte, "zips")
    falta = [z for z in zips if not os.path.exists(os.path.join(pasta_zips, z["nome"] + ".ok"))]
    return falta, len(zips) - len(falta)


def extrair_minds(caminho_zip, chave, destino):
    """Extrai do zip só os vídeos que o adaptador do MINDS reconhece. Devolve quantos."""
    padrao = minds.PADRAO_RGB if chave == "rgb" else minds.PADRAO_RGBD
    n = 0
    with zipfile.ZipFile(caminho_zip) as z:
        for info in z.infolist():
            m = re.search(r"Sinalizador\d{2}/.*", info.filename.replace("\\", "/"))
            if info.is_dir() or not m or not padrao.fullmatch(m.group(0)):
                continue                    # profundidade, esqueleto, outros arquivos: ficam de fora
            alvo = os.path.join(destino, *m.group(0).split("/"))
            if os.path.isfile(alvo) and os.path.getsize(alvo) == info.file_size:
                n += 1
                continue
            os.makedirs(os.path.dirname(alvo), exist_ok=True)
            with z.open(info) as src, open(alvo + ".tmp", "wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
            os.replace(alvo + ".tmp", alvo)
            n += 1
    return n


def baixar_minds(fonte, falta, manter_zips):
    _, chave = ZENODO[fonte]
    pasta_zips = os.path.join(BRUTOS, fonte, "zips")
    destino = os.path.join(BRUTOS, fonte, "extracted")

    def um(z):
        caminho = os.path.join(pasta_zips, z["nome"])
        if not os.path.isfile(caminho):
            baixar_http(z["url"], caminho, md5=z["md5"], timeout=120)
        n = extrair_minds(caminho, chave, destino)
        if n == 0:
            raise RuntimeError("nenhum vídeo reconhecido no zip (o formato da base mudou?)")
        # Marcador: este zip já foi baixado, conferido e extraído (mesmo que os vídeos depois sejam
        # movidos para dados/videos pelo adaptador, não baixa de novo).
        open(caminho + ".ok", "w").write(f"{n} vídeos extraídos\n")
        if not manter_zips:
            os.remove(caminho)
        return z["bytes"]

    # Zips grandes: um de cada vez (o Zenodo limita a banda por conexão e o disco agradece).
    return executar_lote(fonte, falta, um, 1, lambda z: {"zip": z["nome"], "md5": z["md5"]})


# ----------------------------------------------------------------------------- V-LIBRASIL (Kaggle)
def vlibrasil_presentes():
    """Quantos vídeos do V-LIBRASIL já existem: em dados/brutos ou já organizados em dados/videos."""
    brutos = sum(1 for r, _, fs in os.walk(PASTA_VLIBRASIL) for f in fs if f.endswith(".mp4"))
    org = 0
    p = os.path.join(PASTA_MANIFESTOS, "vlibrasil.csv")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            org = sum(1 for r in csv.DictReader(f)
                      if r["arquivo"] and os.path.isfile(os.path.join(PASTA_VIDEOS, r["arquivo"])))
    return brutos + org


def baixar_vlibrasil():
    """Baixa e extrai o dataset inteiro do Kaggle em dados/brutos/vlibrasil (um zip só)."""
    try:
        import kagglehub
    except ImportError:
        sys.exit("[erro] o V-LIBRASIL vem do Kaggle e precisa do kagglehub: pip install kagglehub")
    os.makedirs(PASTA_VLIBRASIL, exist_ok=True)
    caminho = kagglehub.dataset_download(KAGGLE_VLIBRASIL, output_dir=PASTA_VLIBRASIL)
    n = vlibrasil_presentes()
    escrever_log("vlibrasil", [{"origem": f"kaggle:{KAGGLE_VLIBRASIL}", "pasta": caminho, "videos": n,
                                "status": "ok" if n >= N_VIDEOS_VLIBRASIL else "incompleto"}])
    print(f"  [vlibrasil] {n} vídeos em {caminho}")
    return (1, 0) if n >= N_VIDEOS_VLIBRASIL else (0, 1)


# ----------------------------------------------------------------------------- catálogo (links)
def nome_do_link(link):
    """Nome do arquivo no link (único dentro de cada fonte, conferido em 2026-11-18)."""
    nome = urllib.parse.unquote(os.path.basename(urllib.parse.urlparse(link).path))
    return re.sub(r'[\\/:*?"<>|]', "_", nome)


def plano_catalogo(fontes):
    if not os.path.exists(CATALOGO):
        sys.exit(f"[erro] {CATALOGO} não existe: rode preparacao/catalogo_unificado.py antes")
    vistos, falta, ja = set(), collections.defaultdict(list), collections.Counter()
    with open(CATALOGO, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["status"] != "baixar" or r["fonte"] not in fontes or r["link"] in vistos:
                continue
            vistos.add(r["link"])
            destino = os.path.join(BRUTOS, r["fonte"], nome_do_link(r["link"]))
            if os.path.isfile(destino):
                ja[r["fonte"]] += 1
            else:
                falta[r["fonte"]].append((r["link"], destino, r["classe"]))
    return falta, ja


def baixar_catalogo(fonte, falta, paralelo):
    return executar_lote(fonte, falta, lambda it: baixar_http(it[0], it[1]), paralelo,
                         lambda it: {"link": it[0], "arquivo": os.path.basename(it[1]), "classe": it[2]})


# ----------------------------------------------------------------------------- execução
def executar_lote(fonte, itens, funcao, paralelo, descrever):
    """Roda `funcao` em cada item (em paralelo), mostra o progresso e grava o log. Devolve (ok, erros)."""
    log, ok, erros, t0, total_bytes = [], 0, 0, time.time(), 0
    with ThreadPoolExecutor(max_workers=max(1, paralelo)) as pool:
        futuros = {pool.submit(funcao, it): it for it in itens}
        for i, fut in enumerate(as_completed(futuros), 1):
            it = futuros[fut]
            linha = {**descrever(it), "status": "ok", "bytes": "", "erro": ""}
            try:
                n = fut.result()
                total_bytes += n or 0
                linha["bytes"] = n
                ok += 1
            except Exception as e:                      # um item ruim não para os outros
                linha.update(status="erro", erro=str(e)[:300])
                erros += 1
            log.append(linha)
            if linha["status"] == "erro" or i % 25 == 0 or i == len(itens):
                dt = time.time() - t0
                print(f"  [{fonte}] {i}/{len(itens)} | ok {ok}, erros {erros} | {mb(total_bytes)} em "
                      f"{dt / 60:.1f} min" + (f" | ERRO {list(descrever(it).values())[0]}: {linha['erro']}"
                                              if linha["status"] == "erro" else ""))
    escrever_log(fonte, log)
    return ok, erros


def organizar(fontes):
    """Chama os adaptadores (mesma lógica de sempre) sobre os vídeos baixados."""
    py = sys.executable
    if "incluir_tecnologia" in fontes and os.path.isdir(os.path.join(BRUTOS, "incluir_tecnologia")):
        subprocess.run([py, os.path.join(AQUI, "incluir_tecnologia_pre_tratamento.py"),
                        "--origem", os.path.join(BRUTOS, "incluir_tecnologia"), "--executar"], check=True)
    chaves = [ZENODO[f][1] for f in ZENODO if f in fontes and os.path.isdir(os.path.join(BRUTOS, f, "extracted"))]
    if chaves:
        # mover: os vídeos foram baixados só para este projeto (no Windows, link exige administrador).
        subprocess.run([py, os.path.join(AQUI, "minds_libras_pre_tratamento.py"), "--fontes", *chaves,
                        "--origem-rgb", os.path.join(BRUTOS, "minds_libras_rgb", "extracted"),
                        "--origem-rgbd", os.path.join(BRUTOS, "minds_libras_rgbd", "extracted"),
                        "--modo", "mover", "--executar"], check=True)
    if "vlibrasil" in fontes and os.path.isdir(PASTA_VLIBRASIL):
        subprocess.run([py, os.path.join(AQUI, "vlibrasil_pre_tratamento.py"),
                        "--origem", PASTA_VLIBRASIL, "--executar"], check=True)
    if any(f in fontes for f in FONTES_CATALOGO):
        print("[aviso] as fontes do MALTA-LIBRAS ainda não têm adaptador: os vídeos "
              f"ficam em {BRUTOS}/<fonte>/ até a integração em dados/videos/ ser implementada.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fontes", nargs="+", default=TODAS, choices=TODAS + FONTES_CATALOGO,
                    help="o que baixar (padrão: tudo). 'catalogo' = todas as fontes do catálogo")
    ap.add_argument("--executar", action="store_true", help="baixa de fato (sem isso: só o plano)")
    ap.add_argument("--organizar", action="store_true",
                    help="depois de baixar, roda os adaptadores e leva os vídeos para dados/videos/")
    ap.add_argument("--paralelo", type=int, default=6, help="downloads simultâneos (no máx. 2 por site)")
    ap.add_argument("--manter-zips", action="store_true", help="não apaga os zips do MINDS após extrair")
    ap.add_argument("--limite", type=int, default=None,
                    help="no máximo N itens por fonte (teste rápido numa máquina nova)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    fontes = set(args.fontes)
    if "catalogo" in fontes:
        fontes |= set(FONTES_CATALOGO)

    # ------------------------------------------------------------ plano
    planos, total = {}, 0
    if "incluir_tecnologia" in fontes:
        falta, ja, manif = plano_incluir()
        planos["incluir_tecnologia"] = (falta, manif)
        est = sum(x[3] for x in falta)
        total += est
        print(f"incluir_tecnologia (YouTube): {len(falta)} a baixar (~{mb(est)}), {ja} já presentes")
    for fonte in ZENODO:
        if fonte in fontes:
            try:
                falta, ja = plano_minds(fonte)
            except Exception as e:
                print(f"[erro] não consegui consultar o Zenodo para {fonte}: {e}")
                continue
            planos[fonte] = falta
            est = sum(z["bytes"] for z in falta)
            total += est
            print(f"{fonte} (Zenodo {ZENODO[fonte][0]}): {len(falta)} zips a baixar ({mb(est)}), {ja} já extraídos")
    if "vlibrasil" in fontes:
        ja = vlibrasil_presentes()
        planos["vlibrasil"] = ja < N_VIDEOS_VLIBRASIL
        if planos["vlibrasil"]:
            total += BYTES_VLIBRASIL
        print(f"vlibrasil (Kaggle {KAGGLE_VLIBRASIL}): "
              + (f"dataset inteiro a baixar ({mb(BYTES_VLIBRASIL)}; {ja} vídeos já presentes)"
                 if planos["vlibrasil"] else f"completo ({ja} vídeos)"))
    cat = [f for f in FONTES_CATALOGO if f in fontes]
    if cat:
        falta, ja = plano_catalogo(cat)
        for f in cat:
            planos[f] = falta.get(f, [])
            print(f"{f} (catálogo): {len(planos[f])} a baixar, {ja[f]} já presentes")

    if args.limite is not None:
        for k, v in planos.items():
            if k == "incluir_tecnologia":
                planos[k] = (v[0][:args.limite], v[1])
            elif k != "vlibrasil":                    # o Kaggle entrega um zip só: não dá para limitar
                planos[k] = v[:args.limite]
        print(f"(--limite {args.limite}: no máximo {args.limite} itens por fonte)")

    os.makedirs(BRUTOS, exist_ok=True)
    livre = shutil.disk_usage(BRUTOS).free
    print(f"\nEspaço: ~{mb(total)} a baixar (sem contar o catálogo, de tamanho desconhecido), "
          f"{mb(livre)} livres em {BRUTOS}")
    if total > livre:
        print("[atenção] o espaço livre não basta para tudo: baixe por partes com --fontes")
    if not args.executar:
        print("(plano apenas: nada foi baixado; rode com --executar)")
        return

    # ------------------------------------------------------------ download
    resumo = {}
    if "incluir_tecnologia" in planos and planos["incluir_tecnologia"][0]:
        resumo["incluir_tecnologia"] = baixar_incluir(*planos["incluir_tecnologia"], min(args.paralelo, 3))
    for fonte in ZENODO:
        if planos.get(fonte):
            resumo[fonte] = baixar_minds(fonte, planos[fonte], args.manter_zips)
    if planos.get("vlibrasil"):
        resumo["vlibrasil"] = baixar_vlibrasil()
    for f in cat:
        if planos.get(f):
            resumo[f] = baixar_catalogo(f, planos[f], args.paralelo)
    print("\nResumo (ok, erros):", resumo or "nada a baixar", f"| logs em {LOGS}")

    if args.organizar:
        organizar(fontes)


if __name__ == "__main__":
    main()
