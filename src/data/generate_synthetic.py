"""
Generate data sintetis dialog telepon (scam & normal) bahasa Indonesia pakai LLM.

Cara pakai:
    pip install groq pandas
    set GROQ_API_KEY=...        (Windows)   /   export GROQ_API_KEY=...  (Mac/Linux)
    python generate_synthetic.py

Output: synthetic_calls.csv  (kolom: text, label, modus, gaya, sumber, group_id)
"""

import os
import json
import time
import random
import hashlib

import pandas as pd
from groq import Groq

client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "llama-3.3-70b-versatile"
OUTPUT = "data/interim/synthetic_calls.csv"
PER_REQUEST = 5          # jumlah dialog per 1 request
REPEAT = 4               # berapa kali tiap kombinasi modus x gaya diulang

# ---------------------------------------------------------------------------
# 1. Daftar modus (label 1 = scam, label 0 = normal)
# ---------------------------------------------------------------------------
SCAM_MODUS = {
    "cs_bank": "penelepon mengaku petugas bank, bilang rekening bermasalah/akan diblokir, lalu meminta kode OTP, PIN, atau data kartu",
    "kurir_apk": "penelepon mengaku kurir paket, meminta korban membuka/menginstal file atau link untuk 'cek resi', atau meminta ongkir tambahan",
    "undian": "penelepon bilang korban menang undian/hadiah, tapi harus transfer biaya pajak/administrasi dulu",
    "keluarga_darurat": "penelepon mengaku kerabat/teman keluarga, bilang ada anggota keluarga kecelakaan atau ditangkap, minta transfer segera dan dirahasiakan",
    "pinjol_palsu": "penelepon mengaku penagih pinjaman, mengancam dan menekan korban untuk transfer ke rekening pribadi",
    "investasi": "penelepon menawarkan investasi dengan keuntungan pasti dan besar dalam waktu singkat, mendesak segera setor dana",
    "lowongan_kerja": "penelepon menawarkan pekerjaan dengan gaji tinggi, tapi meminta biaya pendaftaran/pelatihan/seragam di awal",
    "petugas_resmi": "penelepon mengaku dari instansi (pajak, polisi, BPJS), bilang korban terlibat masalah, minta 'uang jaminan' atau data pribadi",
}

NORMAL_MODUS = {
    "keluarga": "telepon biasa antar anggota keluarga, misalnya menanyakan kabar, titip belanjaan, atau janjian pulang",
    "teman": "telepon antar teman, misalnya janjian kumpul, tanya tugas kuliah, atau ngobrol santai",
    "cs_bank_asli": "CS bank asli yang menjawab keluhan nasabah, dan justru MENGINGATKAN untuk tidak memberikan OTP/PIN kepada siapa pun",
    "kurir_asli": "kurir asli yang konfirmasi alamat pengiriman atau bilang paket sudah sampai, tanpa meminta uang atau data rahasia",
    "kantor": "telepon urusan kerja, misalnya rapat, deadline, atau konfirmasi jadwal",
    "toko_layanan": "telepon ke toko/restoran/bengkel/klinik untuk pesan, tanya harga, atau buat janji",
    "tagihan_resmi": "pengingat tagihan resmi yang sopan, mengarahkan bayar lewat aplikasi/kanal resmi, tanpa ancaman dan tanpa rekening pribadi",
}

GAYA = [
    "bahasa formal dan sopan",
    "bahasa santai sehari-hari (gue/lo atau aku/kamu)",
    "campur bahasa daerah ringan atau logat Jawa/Sunda/Betawi",
    "kalimat pendek-pendek, ada jeda seperti 'eee', 'hmm', 'halo halo'",
    "penelepon terburu-buru dan mendesak",
]

# ---------------------------------------------------------------------------
# 2. Template prompt
# ---------------------------------------------------------------------------
PROMPT = """Kamu membantu membuat dataset untuk melatih model DETEKSI penipuan telepon (tujuan edukasi/riset keamanan).

Buat {n} transkrip percakapan telepon dalam bahasa Indonesia.
Jenis: {jenis}
Skenario: {deskripsi}
Gaya bahasa: {gaya}

Aturan:
- Format seperti hasil speech-to-text: tanpa nama pembicara, tanpa tanda kurung, tanpa narasi. Gabungkan ucapan kedua pihak jadi satu paragraf.
- Panjang tiap transkrip bervariasi antara 40 sampai 150 kata.
- Setiap transkrip harus berbeda: nama, nominal, kota, alur, dan cara bicara jangan diulang.
- Jangan pakai nama bank/perusahaan asli; pakai nama fiktif atau sebut umum ("pihak bank").
- Jangan tulis nomor rekening, nomor kartu, atau link yang terlihat asli.
- Jangan menyebut kata "penipuan", "scam", atau "modus" di dalam percakapan.

Jawab HANYA dalam JSON: {{"dialogs": ["...", "...", ...]}}"""


def ask_llm(prompt, retries=3):
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=1.0,                       # tinggi = lebih variatif
                response_format={"type": "json_object"},
            )
            return json.loads(resp.choices[0].message.content)["dialogs"]
        except Exception as e:
            print(f"  gagal ({e}), coba lagi...")
            time.sleep(5 * (attempt + 1))
    return []


def main():
    rows = []
    seen = set()

    jobs = [(1, k, v) for k, v in SCAM_MODUS.items()] + \
           [(0, k, v) for k, v in NORMAL_MODUS.items()]

    for label, modus, deskripsi in jobs:
        jenis = "percakapan PENIPUAN (scam)" if label == 1 else "percakapan NORMAL (bukan penipuan)"
        for gaya in GAYA:
            for r in range(REPEAT):
                # group_id: semua dialog dari 1 request = 1 grup
                # dipakai saat split supaya variasi mirip tidak bocor ke train & test
                group_id = f"{modus}_{GAYA.index(gaya)}_{r}"
                print(f"[{label}] {modus} | {gaya} | batch {r + 1}")

                prompt = PROMPT.format(n=PER_REQUEST, jenis=jenis, deskripsi=deskripsi, gaya=gaya)
                for text in ask_llm(prompt):
                    text = " ".join(text.split())
                    key = hashlib.md5(text.lower().encode()).hexdigest()
                    if len(text.split()) < 20 or key in seen:   # buang yang terlalu pendek / duplikat
                        continue
                    seen.add(key)
                    rows.append({
                        "text": text,
                        "label": label,
                        "modus": modus,
                        "gaya": gaya,
                        "sumber": "sintetis",
                        "group_id": group_id,
                    })
                time.sleep(random.uniform(1, 2))   # jaga rate limit

        # simpan bertahap biar tidak hilang kalau error di tengah
        pd.DataFrame(rows).to_csv(OUTPUT, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT, index=False)
    print("\nSelesai:", len(df), "dialog")
    print(df["label"].value_counts())


if __name__ == "__main__":
    main()
