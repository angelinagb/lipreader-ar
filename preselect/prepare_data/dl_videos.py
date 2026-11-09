import argparse
import random
import re
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
 
import yt_dlp
 
ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
 
# Errores que indican que el video ya no existe / no es accesible nunca.
# OJO: "This video is not available" NO esta aqui a proposito; yt-dlp lo muestra
# tambien cuando falla el solver de JavaScript (falta deno / yt-dlp desactualizado).
PERMANENT_ERRORS = (
    "private video",
    "removed by the uploader",
    "has been terminated",
    "copyright claim",
    "no longer available",
    "video has been removed",
)
BLOCK_ERRORS = ("not a bot", "http error 429", "too many requests")
 
 
def load_ids(path: Path) -> list[str]:
    ids, seen = [], set()
    for line in path.read_text(encoding="utf-8").splitlines():
        tok = line.strip().split(",")[0].split("\t")[0].split(" ")[0]
        if ID_RE.match(tok) and tok not in seen:
            seen.add(tok)
            ids.append(tok)
    return ids
 
 
def build_opts(args, out_dir: Path) -> dict:
    h = args.max_height
    # Prefiere H.264 + AAC (lo lee cualquier cosa: OpenCV, decord, ffmpeg).
    fmt = (
    f"bv*[height<={h}][vcodec^=avc1]+ba[ext=m4a]/"
    f"bv*[height<={h}]+ba[ext=m4a]/"
    f"bv*[height<={h}]+ba/"
    )
    opts = {
        "format": fmt,
        "merge_output_format": "mp4",
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": False,
        "no_warnings": False,
        "noprogress": False,
        "retries": 5,
        "fragment_retries": 5,
        "concurrent_fragment_downloads": 2,
    }
    if args.cookies:
        opts["cookiefile"] = args.cookies
    if args.cookies_from_browser:
        opts["cookiesfrombrowser"] = (args.cookies_from_browser,)
    return opts
 
 
def download_one(vid: str, opts: dict, out_dir: Path, stop: threading.Event, jitter: float):
    if stop.is_set():
        return vid, "skipped", ""
    time.sleep(random.uniform(0, jitter))  # escalona las peticiones
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={vid}"])
    except yt_dlp.utils.DownloadError as e:
        msg = " ".join(str(e).split())
        low = msg.lower()
        if any(b in low for b in BLOCK_ERRORS):
            stop.set()
            return vid, "blocked", msg
        if any(p in low for p in PERMANENT_ERRORS):
            return vid, "unavailable", msg
        return vid, "error", msg
    except Exception as e:  # noqa: BLE001
        return vid, "error", repr(e)
 
    final = out_dir / f"{vid}.mp4"
    if final.exists() and final.stat().st_size > 0:
        return vid, "ok", ""
    others = [p.name for p in out_dir.glob(f"{vid}.*")]
    return vid, "error", f"no se genero {final.name}; archivos encontrados: {others}"
 
 
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids", required=True, type=Path, help="txt con un ID de YouTube por linea")
    ap.add_argument("--out", required=True, type=Path, help="carpeta de salida, ej. Data/Spanish")
    ap.add_argument("--workers", type=int, default=3, help="descargas en paralelo (default 3; no abuses)")
    ap.add_argument("--max-height", type=int, default=720, help="altura maxima del video (default 720)")
    ap.add_argument("--limit", type=int, default=None, help="descargar solo N videos (piloto)")
    ap.add_argument("--shuffle", action="store_true", help="mezclar la lista antes de aplicar --limit")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jitter", type=float, default=3.0, help="espera aleatoria (s) antes de cada descarga")
    ap.add_argument("--cookies", type=str, default=None, help="archivo cookies.txt")
    ap.add_argument("--cookies-from-browser", type=str, default=None, help="firefox, chrome, etc.")
    ap.add_argument("--retry-unavailable", action="store_true", help="reintentar IDs marcados como eliminados")
    args = ap.parse_args()
 
    for tool in ("ffmpeg", "deno"):
        if shutil.which(tool) is None:
            print(f"[aviso] no encuentro '{tool}' en el PATH. "
                  f"{'Sin ffmpeg no se pueden unir audio y video.' if tool == 'ffmpeg' else 'Sin deno (u otro runtime JS) YouTube suele fallar con errores engañosos.'}",
                  file=sys.stderr)
 
    args.out.mkdir(parents=True, exist_ok=True)
    unavailable_file = args.out / "_unavailable.txt"
    failed_file = args.out / "_failed.tsv"
 
    ids = load_ids(args.ids)
    print(f"IDs validos en la lista: {len(ids)}")
 
    known_bad = set()
    if unavailable_file.exists() and not args.retry_unavailable:
        known_bad = set(unavailable_file.read_text().split())
 
    pending = [
        v for v in ids
        if not (args.out / f"{v}.mp4").exists() and v not in known_bad
    ]
    print(f"Ya descargados: {sum((args.out / f'{v}.mp4').exists() for v in ids)} | "
          f"marcados como no disponibles: {len(known_bad & set(ids))} | pendientes: {len(pending)}")
 
    if args.shuffle:
        random.Random(args.seed).shuffle(pending)
    if args.limit:
        pending = pending[: args.limit]
    if not pending:
        print("Nada que hacer")
        return
 
    opts = build_opts(args, args.out)
    stop = threading.Event()
    counts = {"ok": 0, "unavailable": 0, "error": 0, "blocked": 0, "skipped": 0}
    t0 = time.time()
 
    with ThreadPoolExecutor(max_workers=args.workers) as pool, \
            open(unavailable_file, "a") as f_unav, open(failed_file, "a") as f_fail:
        futs = [pool.submit(download_one, v, opts, args.out, stop, args.jitter) for v in pending]
        for i, fut in enumerate(as_completed(futs), 1):
            vid, status, msg = fut.result()
            counts[status] += 1
            if status == "unavailable":
                f_unav.write(vid + "\n"); f_unav.flush()
            elif status in ("error", "blocked"):
                f_fail.write(f"{vid}\t{status}\t{msg}\n"); f_fail.flush()
            if i % 10 == 0 or status in ("blocked",):
                el = time.time() - t0
                print(f"[{i}/{len(pending)}] ok={counts['ok']} no_disp={counts['unavailable']} "
                      f"err={counts['error']} ({el/60:.1f} min)")
 
    print("\nResumen:", counts)
    if counts["blocked"]:
        print("YouTube te esta limitando / pide verificacion. Baja --workers, sube --jitter, "
              "espera un rato y usa --cookies-from-browser. Vuelve a correr: retoma donde quedo.")
    done = counts["ok"] + counts["unavailable"] + counts["error"]
    if done and counts["error"] / done > 0.5:
        print("Mas de la mitad fallaron con errores no clasificados: revisa _failed.tsv. "
              "Si dicen 'not available', casi seguro es yt-dlp desactualizado o falta deno.")
 
 
if __name__ == "__main__":
    main()
 