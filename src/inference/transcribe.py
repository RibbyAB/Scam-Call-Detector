"""Audio -> teks pakai Whisper."""
import whisper

_model = None


def transcribe(path: str, size: str = "small") -> str:
    global _model
    if _model is None:
        _model = whisper.load_model(size)
    result = _model.transcribe(path, language="id", fp16=False)
    return " ".join(result["text"].split())
