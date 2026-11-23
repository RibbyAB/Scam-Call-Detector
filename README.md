# Scam Call Detector

Deteksi telepon penipuan berbahasa Indonesia: audio -> Whisper -> IndoBERT -> scam / normal.

## Alur
1. Data: `src/data/generate_synthetic.py`, `src/data/audio_to_dataset.py`, data manual di `data/raw/manual/`
2. Gabung: `python src/data/merge_dataset.py`
3. EDA & preprocessing: `notebooks/01`, `02`
4. Model: `notebooks/03` (baseline), `04` (IndoBERT)
5. Evaluasi: `notebooks/05`
6. Demo: `streamlit run app/streamlit_app.py`

## Setup
    python -m venv .venv
    .venv\Scripts\activate        (Windows)
    pip install -r requirements.txt

Semua script dijalankan dari folder root project.
