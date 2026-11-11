
import argparse
import concurrent.futures
import json
import random
import re
import subprocess
import threading
import time
from pathlib import Path

import yt_dlp


# Rutas del proyecto
IDS_FILE = Path(
    r"C:\Users\Angelina\Documents\GitHub\lipreader-ar"
    r"\preselect\metadata\extracted\Spanish\Train\videos_ids_train.txt"
)

DOWNLOAD_DIR = Path(r"D:\ViSpeR\downloads")
VIDEO_DIR = Path(r"D:\ViSpeR\videos")

FPS = 25
MAX_HEIGHT = 720
SEED = 18

stop_event = threading.Event()


def load_ids(path):
    """Lee IDs de YouTube y elimina duplicados."""
    ids = []
    seen = set()

    with open(path, encoding="utf-8-sig") as file:
        for line in file:
            parts = line.strip().split()
            if not parts:
                continue

            video_id = parts[0]

            if re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
                if video_id not in seen:
                    ids.append(video_id)
                    seen.add(video_id)

    return ids


def is_valid_video(path):
    """Comprueba que el video exista y tenga 25 FPS."""
    if not path.exists() or path.stat().st_size == 0:
        return False

    command = [
        "ffprobe",
        "-v", "error",
        "-show_entries",
        "stream=codec_type,r_frame_rate,avg_frame_rate",
        "-of", "json",
        str(path)
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True
        )

        data = json.loads(result.stdout)
        video_streams = [
            stream for stream in data["streams"]
            if stream["codec_type"] == "video"
        ]

        if not video_streams:
            return False

        stream = video_streams[0]

        return (
            stream["r_frame_rate"] == "25/1"
            and stream["avg_frame_rate"] == "25/1"
        )

    except Exception:
        return False


def build_download_options():
    """Configuración de yt-dlp."""
    return {
        "format": (
            f"bv*[height<={MAX_HEIGHT}][vcodec^=avc1]+ba[ext=m4a]/"
            f"bv*[height<={MAX_HEIGHT}]+ba[ext=m4a]/"
            f"bv*[height<={MAX_HEIGHT}]+ba/"
            f"b[height<={MAX_HEIGHT}]"
        ),
        "merge_output_format": "mp4",
        "outtmpl": str(DOWNLOAD_DIR / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "retries": 5,
        "fragment_retries": 5,
        "concurrent_fragment_downloads": 2,
        "js_runtimes": {"deno": {}},
        "quiet": True,
        "no_warnings": False,
    }


def download_video(video_id, options):
    """Descarga un video individual."""
    url = f"https://www.youtube.com/watch?v={video_id}"

    time.sleep(random.uniform(0, 3))

    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            ydl.download([url])

        candidates = list(DOWNLOAD_DIR.glob(f"{video_id}.*"))

        candidates = [
            path for path in candidates
            if path.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}
            and not path.name.endswith(".part")
        ]

        if not candidates:
            raise FileNotFoundError("No se encontró el video descargado")

        return max(candidates, key=lambda p: p.stat().st_size)

    except Exception as error:
        raise RuntimeError(f"Error descargando {video_id}: {error}")


def convert_video(source, destination):
    """Convierte el video a CFR 25 FPS y conserva el audio."""
    temporary = destination.with_name(
        destination.stem + ".tmp.mp4"
    )

    base_command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-i", str(source),
        "-map", "0:v:0",
        "-map", "0:a?",
        "-vf", f"fps={FPS}",
        "-vsync", "cfr",
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "medium",
        "-movflags", "+faststart",
    ]

    # Primero intenta copiar el audio sin recodificarlo.
    command = base_command + [
        "-c:a", "copy",
        str(temporary)
    ]

    try:
        subprocess.run(command, check=True)

        if not is_valid_video(temporary):
            raise RuntimeError("El archivo convertido no pasó la validación")

    except Exception:
        temporary.unlink(missing_ok=True)

        # Si el audio no es compatible, lo convierte a AAC.
        command = base_command + [
            "-c:a", "aac",
            "-b:a", "192k",
            str(temporary)
        ]

        subprocess.run(command, check=True)

        if not is_valid_video(temporary):
            temporary.unlink(missing_ok=True)
            raise RuntimeError("Falló la validación de FPS")

    temporary.replace(destination)


def process_video(video_id, options):
    """Descarga, convierte y valida un video."""
    if stop_event.is_set():
        return "CANCELADO", video_id

    destination = VIDEO_DIR / f"{video_id}.mp4"

    # Si ya existe un resultado válido, no lo procesa otra vez.
    if is_valid_video(destination):
        return "YA EXISTE", video_id

    try:
        source = download_video(video_id, options)

        convert_video(source, destination)

        # El temporal se elimina solamente después de validar el final.
        source.unlink(missing_ok=True)

        return "OK", video_id

    except Exception as error:
        return f"ERROR: {error}", video_id



def download_batch(batch_ids, workers=3):
    """Descarga y normaliza una tanda de videos."""

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    stop_event.clear()

    options = build_download_options()

    results = {
        "OK": 0,
        "YA EXISTE": 0,
        "ERROR": 0,
        "CANCELADO": 0,
    }

    ready_ids = []
    failed = []

    print(f"Videos de esta tanda: {len(batch_ids)}")

    try:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=workers
        ) as executor:

            futures = {
                executor.submit(process_video, video_id, options): video_id
                for video_id in batch_ids
            }

            for future in concurrent.futures.as_completed(futures):
                status, video_id = future.result()

                if status.startswith("ERROR"):
                    results["ERROR"] += 1
                    failed.append((video_id, status))
                    print(f"[ERROR] {video_id}: {status}")

                else:
                    results[status] += 1
                    print(f"[{status}] {video_id}")

                    if status in ("OK", "YA EXISTE"):
                        ready_ids.append(video_id)

    except KeyboardInterrupt:
        stop_event.set()
        print("\nInterrupción solicitada.")

    if failed:
        failed_file = VIDEO_DIR / "_failed.tsv"

        with open(failed_file, "a", encoding="utf-8") as file:
            for video_id, error in failed:
                file.write(f"{video_id}\t{error}\n")

    print("\n--- RESUMEN ---")
    for status, count in results.items():
        print(f"{status}: {count}")

    return ready_ids, failed


def main():
    parser = argparse.ArgumentParser(
        description="Descarga videos y los normaliza a 25 FPS."
    )

    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)

    args = parser.parse_args()

    ids = load_ids(IDS_FILE)

    if args.shuffle:
        random.Random(args.seed).shuffle(ids)

    if args.limit > 0:
        ids = ids[:args.limit]

    download_batch(ids, workers=args.workers)


if __name__ == "__main__":
    main()
