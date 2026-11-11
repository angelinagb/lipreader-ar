import json
from pathlib import Path

metadata_path = Path(r"D:\ViSpeR\clips\metadata.json")

with open(metadata_path, "r", encoding="utf-8") as f:
    metadata = json.load(f)

print("Clips registrados:", len(metadata))
print("Videos registrados:", len({
    Path(path).parent.name for path in metadata
}))
print("Primer registro:", next(iter(metadata.items())))