
import json
from pathlib import Path

json_path = Path("metadata/extracted/Spanish/Train/chunk_1.json")

with json_path.open("r", encoding="utf-8") as f:
    data = json.load(f)

print("Tipo de estructura:", type(data).__name__)
print("Cantidad de videos:", len(data))

# Primer video
video_id = next(iter(data))
clips = data[video_id]

print("\nPrimer video:", video_id)
print("Tipo de anotaciones:", type(clips).__name__)
print("Cantidad de clips:", len(clips))

# Primera anotación
if clips:
    first_clip = clips[0]

    print("\nCampos del clip:")
    for key, value in first_clip.items():
        if isinstance(value, list):
            print(f"{key}: lista de {len(value)} elementos")
        else:
            print(f"{key}: {value}")
