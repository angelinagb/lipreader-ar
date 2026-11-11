
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "data_prepare"))
from data_prepare.utils import remove_characters

VIDEO_DIR = Path(r"D:\ViSpeR\videos")
JSON_DIR = Path(r"D:\ViSpeR\metadata_20")
OUTPUT = Path(r"D:\ViSpeR\clips\metadata.json")

labels = {}
missing = []

for json_file in sorted(JSON_DIR.glob("*.json")):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for video_id, segments in data.items():
        for k, clip in enumerate(segments):
            video_path = (
                VIDEO_DIR / video_id / f"{k:05d}.mp4"
            )

            if video_path.is_file():
                labels[str(video_path)] = {
                    "label": remove_characters(clip["label"], "<ar>"),
                    "length": len(clip["landmarks"])
                }
            else:
                missing.append(str(video_path))

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(labels, f, ensure_ascii=False)

print("\n========== RESULTADO ==========")
print("Clips etiquetados:", len(labels))
print("Clips faltantes:", len(missing))
print("Metadata:", OUTPUT)
