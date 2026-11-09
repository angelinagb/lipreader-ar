#!/usr/bin/env python3
"""
prepare_pilot.py - Deja listo un piloto pequeño para correr crop_videos.py de ViSpeR.

crop_videos.py tiene tres requisitos que un piloto con pocos videos incumple por defecto:
  1. Los videos tienen que estar a 25 fps (los landmarks estan indexados por frame).
  2. El script recorre TODOS los IDs de los JSON y revienta con IndexError en el primer
     video que no tengas descargado -> hay que darle un JSON filtrado.
  3. Los landmarks/bboxes estan en coordenadas de algun video concreto; si el tuyo tiene
     otra resolucion, el recorte de boca sale desplazado.

Este script:
  - lee todos los *.json de --json_path (uno a uno, para no cargar todo en memoria)
    y se queda solo con los videos que existen en --video_dir
  - convierte a 25 fps esos videos -> --out_video_dir  (copia si ya estaban a 25)
  - compara el rango de coordenadas de los landmarks con la resolucion de cada video
  - escribe --out_json_dir/subset.json

Uso:
  python prepare_pilot.py --video_dir Data/Spanish --json_path jsons/ \
      --out_video_dir Data/Spanish_25fps --out_json_dir jsons_pilot

Luego:
  python crop_videos.py --video_dir Data/Spanish_25fps --json_path jsons_pilot \
      --save_path clips_pilot --merge_audio True
"""
import argparse
import json
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np


def probe(path: Path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    s = json.loads(out)["streams"][0]
    rate = s.get("avg_frame_rate") or "0/0"
    if rate in ("0/0", "0"):
        rate = s["r_frame_rate"]
    return int(s["width"]), int(s["height"]), float(Fraction(rate))


def to_25fps(src: Path, dst: Path):
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(src),
         "-vf", "fps=25", "-c:v", "libx264", "-preset", "fast", "-crf", "18",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", str(dst)],
        check=True,
    )


def check_landmarks(clips, w: int, h: int) -> str:
    """Compara el rango de coordenadas de los landmarks con el tamaño del video."""
    for clip in clips[:5]:
        try:
            lm = np.asarray(clip["landmarks"], dtype=float)
        except (KeyError, ValueError, TypeError):
            continue
        if lm.size == 0 or lm.ndim < 2 or lm.shape[-1] != 2:
            return f"forma de landmarks inesperada {getattr(lm, 'shape', None)}: revisalos a mano"
        xmax, ymax = float(lm[..., 0].max()), float(lm[..., 1].max())
        info = f"landmarks {lm.shape}, x_max={xmax:.1f}, y_max={ymax:.1f} (video {w}x{h})"
        if xmax <= 1.5 and ymax <= 1.5:
            return info + " -> parecen normalizados (0-1), no deberia haber problema de resolucion"
        if xmax > w * 1.02 or ymax > h * 1.02:
            return (info + f"\n    [!] LANDMARKS FUERA DEL FRAME: se calcularon sobre un video de mayor "
                    f"resolucion (aprox. {max(xmax / w, ymax / h):.2f}x). Descarga con mas resolucion "
                    f"(--max-height) o reescala los landmarks.")
        return info + " -> dentro del frame (verifica visualmente que el recorte caiga en la boca)"
    return "no pude leer landmarks de los primeros clips"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video_dir", required=True, type=Path)
    ap.add_argument("--json_path", required=True, type=Path)
    ap.add_argument("--out_video_dir", required=True, type=Path)
    ap.add_argument("--out_json_dir", required=True, type=Path)
    args = ap.parse_args()

    videos = {p.stem: p for p in sorted(args.video_dir.glob("*.mp4"))}
    print(f"Videos descargados en {args.video_dir}: {len(videos)}")
    if not videos:
        sys.exit("No hay .mp4 en --video_dir")

    jfiles = sorted(args.json_path.glob("*.json"))
    if not jfiles:
        sys.exit(f"No encontre *.json en {args.json_path}")
    print(f"Leyendo {len(jfiles)} JSON...")

    subset, n_ids_total, sample_keys = {}, 0, []
    for jf in jfiles:
        with open(jf, encoding="utf-8") as f:
            chunk = json.load(f)
        n_ids_total += len(chunk)
        if len(sample_keys) < 3:
            sample_keys += list(chunk.keys())[: 3 - len(sample_keys)]
        for vid, clips in chunk.items():
            if vid in videos:
                subset[vid] = clips
        del chunk

    print(f"IDs en los JSON: {n_ids_total} | de tus videos, con anotaciones: {len(subset)}")
    sin_json = [v for v in videos if v not in subset]
    if sin_json:
        print(f"Descargados pero SIN anotaciones en los JSON ({len(sin_json)}): {sin_json}")
    if not subset:
        sys.exit(f"Ningun video coincide con los JSON. Ejemplos de claves del JSON: {sample_keys}")

    args.out_video_dir.mkdir(parents=True, exist_ok=True)
    args.out_json_dir.mkdir(parents=True, exist_ok=True)

    ok = {}
    for vid, clips in subset.items():
        src = videos[vid]
        dst = args.out_video_dir / f"{vid}.mp4"
        w, h, fps = probe(src)
        print(f"\n{vid}: {w}x{h} @ {fps:.3f} fps, {len(clips)} clips")
        print("   ", check_landmarks(clips, w, h))
        try:
            if abs(fps - 25.0) < 0.01:
                if not dst.exists():
                    shutil.copy2(src, dst)
                print("    ya estaba a 25 fps")
            else:
                if not dst.exists():
                    print("    convirtiendo a 25 fps...")
                    to_25fps(src, dst)
        except subprocess.CalledProcessError as e:
            print(f"    [x] fallo la conversion: {e}")
            continue
        ok[vid] = clips

    out_json = args.out_json_dir / "subset.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(ok, f, ensure_ascii=False)
    n_clips = sum(len(c) for c in ok.values())
    print(f"\nListo: {len(ok)} videos, {n_clips} clips esperados -> {out_json}")


if __name__ == "__main__":
    main()