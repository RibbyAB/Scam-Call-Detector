"""Fungsi pembersihan teks."""
import re


def clean_for_bert(text: str) -> str:
    """Ringan saja: rapikan spasi dan buang karakter aneh. Tokenizer IndoBERT yang urus sisanya."""
    text = re.sub(r"[^\w\s.,!?'-]", " ", str(text))
    return " ".join(text.split())


def clean_for_tfidf(text: str) -> str:
    """Lebih agresif untuk baseline TF-IDF."""
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return " ".join(text.split())
