from pathlib import Path

ids = Path("videos_ids_train_es.txt").read_text().splitlines()

sample = ids[:300]

Path("sample_urls.txt").write_text(
    "\n".join(
        f"https://www.youtube.com/watch?v={video_id}"
        for video_id in sample
    )
)