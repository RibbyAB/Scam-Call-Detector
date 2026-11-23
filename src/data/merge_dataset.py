"""Gabungkan semua sumber data jadi satu CSV."""
import pandas as pd

sources = [
    "data/interim/synthetic_calls.csv",
    "data/interim/audio_calls.csv",
    "data/raw/manual/manual_calls.csv",
]

dfs = []
for path in sources:
    try:
        dfs.append(pd.read_csv(path))
        print("OK  ", path)
    except FileNotFoundError:
        print("SKIP", path, "(belum ada)")

df = pd.concat(dfs, ignore_index=True)
df = df.dropna(subset=["text", "label"]).drop_duplicates(subset="text")
df.to_csv("data/processed/dataset_final.csv", index=False)
print(len(df), "baris ->", "data/processed/dataset_final.csv")
print(df["label"].value_counts())
