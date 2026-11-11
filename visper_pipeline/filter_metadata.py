
import json
import ijson
from pathlib import Path

# Rutas
JSON_DIR = Path(
    r"C:\Users\Angelina\Documents\GitHub\lipreader-ar\preselect\metadata\extracted\Spanish\Train"
)
OUTPUT_DIR = Path(r"D:\ViSpeR\metadata_20")


def filter_metadata(batch_ids):
    wanted_ids = set(batch_ids)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Videos de esta tanda: {len(wanted_ids)}")

    found_ids = set()

    # Procesar los chunks sin cargarlos completos en RAM
    for chunk in sorted(JSON_DIR.glob("chunk_*.json")):
        print(f"\nProcesando: {chunk.name}")

        with open(chunk, "rb") as f:
            for video_id, segments in ijson.kvitems(f, ""):
                if video_id not in wanted_ids:
                    continue

                if video_id in found_ids:
                    continue

                output_file = OUTPUT_DIR / f"{video_id}.json"

                with open(output_file, "w", encoding="utf-8") as out:
                    json.dump(
                        {video_id: segments},
                        out,
                        ensure_ascii=False,
                        default=float
                    )

                found_ids.add(video_id)
                print(f"  Encontrado: {video_id}")
                if found_ids == wanted_ids:
                    break
        if found_ids == wanted_ids:
            print("\nTodas las anotaciones encontradas. Se detiene la búsqueda.")
            break

    # Resumen
    missing = wanted_ids - found_ids

    print("\n========== RESUMEN ==========")
    print(f"Encontrados: {len(found_ids)}")
    print(f"Faltantes: {len(missing)}")

    if missing:
        print("IDs faltantes:")
        for video_id in sorted(missing):
            print(video_id)

    print(f"\nMetadatos guardados en: {OUTPUT_DIR}")

    return found_ids, missing
