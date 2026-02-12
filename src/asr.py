"""
ASR wrapper: Whisper / faster-whisper. Input: float32 16 kHz mono. Output: text.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ASR:
    """Transcribe float32 16 kHz mono audio to text."""

    def __init__(
        self,
        backend: str = "faster_whisper",
        model_size: str = "small",
        device: str = "cpu",
        language: Optional[str] = None,
        model_path: Optional[str] = None,
        no_speech_threshold: float = 0.4,
    ):
        self.backend = backend
        self.model_size = model_size
        self.device = device
        self.language = language
        self.model_path = model_path
        self.no_speech_threshold = no_speech_threshold
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if self.backend == "faster_whisper":
            logging.getLogger("faster_whisper").setLevel(logging.WARNING)
            from faster_whisper import WhisperModel
            model_path = self.model_path or self.model_size
            device = self.device.split(":")[0]
            try:
                self._model = WhisperModel(
                    model_path,
                    device=device,
                    compute_type="float16" if device == "cuda" else "int8",
                )
            except ValueError as e:
                if "CUDA" in str(e) or "cuda" in str(e).lower():
                    logger.warning(
                        "ASR: CUDA not available (%s), falling back to CPU",
                        e,
                    )
                    self.device = "cpu"
                    self._model = WhisperModel(
                        model_path,
                        device="cpu",
                        compute_type="int8",
                    )
                else:
                    raise
        elif self.backend == "openai_whisper":
            import whisper
            self._model = whisper.load_model(
                self.model_path or self.model_size,
                device=self.device,
            )
        else:
            raise ValueError(f"Unknown ASR backend: {self.backend}")

    def transcribe(
        self,
        audio: "np.ndarray",
        sample_rate: int = 16000,
    ) -> str:
        """Return transcribed text. audio: float32, 1D, 16 kHz preferred."""
        import numpy as np
        self._ensure_loaded()
        audio = np.asarray(audio, dtype=np.float32).ravel()
        if audio.size == 0:
            return ""

        if self.backend == "faster_whisper":
            # Audio must be float32, 16 kHz (we already resample in the pipeline).
            # no_speech_threshold: lower = less likely to drop segments as "silence" (default 0.6).
            # condition_on_previous_text=False reduces "You" / repetitive hallucinations on CPU.
            segments, _ = self._model.transcribe(
                audio,
                language=self.language,
                vad_parameters=None,
                no_speech_threshold=self.no_speech_threshold,
                condition_on_previous_text=False,
            )
            return " ".join(s.text.strip() for s in segments).strip()
        else:
            import whisper
            result = self._model.transcribe(audio, fp16=("cuda" in self.device), language=self.language)
            return (result.get("text") or "").strip()
