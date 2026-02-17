import csv
import subprocess
from pathlib import Path

CSV_PATH = "./real_songs.csv"
OUT_DIR = Path("./real_songs")
OUT_DIR.mkdir(exist_ok=True)

YTDLP = "yt-dlp"

with open(CSV_PATH) as f:
    reader = csv.DictReader(f)
    for row in reader:
        ytid = row["youtube_id"]
        out_file = OUT_DIR / f"{ytid}.wav"

        if out_file.exists():
            continue

        url = f"https://www.youtube.com/watch?v={ytid}"

        cmd = [
            YTDLP,
            "-f", "bestaudio/best",
            "--extract-audio",
            "--audio-format", "wav",
            "--audio-quality", "0",
            "--postprocessor-args", "-ac 1 -ar 16000",
            "-o", str(out_file),
            url
        ]

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            print(f"[FAILED] {ytid}")
