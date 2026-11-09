
import json
from pathlib import Path
import ijson

VIDEO_DIR = Path("Data/Spanish/videos")
CHUNKS_DIR = Path("metadata/extracted/Spanish/Train")
OUTPUT_DIR = Path("metadata_pilot")
OUTPUT_FILE = OUTPUT_DIR / "pilot_10.json"


def main():
    targets = {p.stem for p in VIDEO_DIR.glob("*.mp4")}

    if not targets:
        print("No se encontraron videos.")
        return

    print(f"Videos a buscar: {len(targets)}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    found = set()

    tmp_file = OUTPUT_DIR / "pilot_10.tmp"

    with tmp_file.open("w", encoding="utf-8") as out:
        out.write("{")
        first = True

        for chunk in sorted(CHUNKS_DIR.glob("chunk_*.json")):
            print(f"\nInspeccionando {chunk.name}...")

            with chunk.open("rb") as f:
                for video_id, segments in ijson.kvitems(
                    f, "", use_float=True
                ):
                    if video_id not in targets:
                        continue

                    if not first:
                        out.write(",")

                    json.dump(video_id, out)
                    out.write(":")
                    json.dump(segments, out, ensure_ascii=False)

                    first = False
                    found.add(video_id)

                    print(
                        f"Encontrado: {video_id} "
                        f"({len(segments)} segmentos)"
                    )

                    if found == targets:
                        break

            if found == targets:
                break

        out.write("}")

    tmp_file.replace(OUTPUT_FILE)

    missing = targets - found

    print("\n--- RESUMEN ---")
    print(f"Encontrados: {len(found)}/{len(targets)}")
    print(f"Faltantes: {sorted(missing)}")
    print(f"Resultado: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
