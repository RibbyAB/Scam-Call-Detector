"""
Ubah rekaman audio buatan sendiri menjadi dataset teks (CSV) pakai Whisper.

Struktur folder yang dibutuhkan:
    data/raw/audio/
    ├── scam/
    │   ├── cs_bank_01.wav
    │   ├── undian_01.mp3
    │   └── ...
    └── normal/
        ├── keluarga_01.wav
        └── ...

Nama folder = label (scam -> 1, normal -> 0).
Nama file sebaiknya diawali modusnya, contoh: cs_bank_01.wav -> modus "cs_bank".

Cara pakai:
    pip install openai-whisper pandas
    (butuh ffmpeg terinstall, lihat catatan di bawah)
    python audio_to_dataset.py

Output: audio_calls.csv  (kolom: file, text, label, modus, sumber, durasi_detik)
"""

import os
import re

import pandas as pd
import whisper

AUDIO_DIR = "data/raw/audio"
OUTPUT = "data/interim/audio_calls.csv"
MODEL_SIZE = "small"      # tiny / base / small / medium. Makin besar makin akurat tapi makin lambat
LABELS = {"scam": 1, "normal": 0}
EXTENSIONS = (".wav", ".mp3", ".m4a", ".ogg", ".flac")

print(f"Loading Whisper '{MODEL_SIZE}'...")
model = whisper.load_model(MODEL_SIZE)

rows = []
for folder, label in LABELS.items():
    folder_path = os.path.join(AUDIO_DIR, folder)
    if not os.path.isdir(folder_path):
        print(f"Folder {folder_path} tidak ada, dilewati")
        continue

    files = sorted(f for f in os.listdir(folder_path) if f.lower().endswith(EXTENSIONS))
    for i, filename in enumerate(files, 1):
        path = os.path.join(folder_path, filename)
        print(f"[{folder}] {i}/{len(files)}  {filename}")

        result = model.transcribe(path, language="id", fp16=False)
        text = " ".join(result["text"].split())

        # ambil modus dari nama file: "cs_bank_01.wav" -> "cs_bank"
        name = os.path.splitext(filename)[0]
        modus = re.sub(r"_?\d+$", "", name) or "unknown"

        durasi = result["segments"][-1]["end"] if result["segments"] else 0

        rows.append({
            "file": f"{folder}/{filename}",
            "text": text,
            "label": label,
            "modus": modus,
            "sumber": "rekaman",
            "durasi_detik": round(durasi, 1),
        })

df = pd.DataFrame(rows)
df.to_csv(OUTPUT, index=False)
print(f"\nSelesai: {len(df)} rekaman -> {OUTPUT}")
print(df["label"].value_counts())
