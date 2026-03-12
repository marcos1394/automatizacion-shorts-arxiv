"""
tts_engine.py — Text-to-speech using Kokoro-82M (local, Apple Silicon).

Install:
    pip install "kokoro>=0.9.4" soundfile
    brew install espeak-ng
    brew install ffmpeg   (optional: WAV → MP3)

Run with:
    PYTORCH_ENABLE_MPS_FALLBACK=1 python main.py --shorts
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_VOICE = os.getenv("TTS_VOICE", "af_heart")
SAMPLE_RATE = 24000

# Available voices:
#   af_heart  — warm American female (default)
#   af_bella  — bright American female
#   am_echo   — calm American male
#   am_fenrir — deep American male
#   bf_emma   — British female
#   bm_george — British male


class TTSEngine:
    """Kokoro-82M TTS wrapper. Loads once (singleton)."""

    _pipeline = None

    def __init__(self, voice: str = DEFAULT_VOICE):
        self.voice = voice
        self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        if TTSEngine._pipeline is None:
            try:
                from kokoro import KPipeline
                logger.info("Loading Kokoro-82M TTS pipeline...")
                TTSEngine._pipeline = KPipeline(lang_code="a")
                logger.info("TTS pipeline ready.")
            except ImportError:
                logger.error(
                    "Kokoro not installed.\n"
                    "Run: pip install 'kokoro>=0.9.4' soundfile && brew install espeak-ng"
                )
                raise
            except Exception as e:
                logger.error(f"Error loading Kokoro: {e}", exc_info=True)
                raise

    def synthesize(self, text: str, output_path: str | Path) -> Path | None:
        """Convert text to WAV. Returns path or None on failure."""
        try:
            import numpy as np
            import soundfile as sf

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            logger.info(f"Synthesizing {len(text.split())} words → voice '{self.voice}'...")

            audio_chunks = []
            for _gs, _ps, audio in TTSEngine._pipeline(text, voice=self.voice):
                audio_chunks.append(audio)

            if not audio_chunks:
                logger.error("Kokoro returned no audio.")
                return None

            full_audio = np.concatenate(audio_chunks)
            duration = len(full_audio) / SAMPLE_RATE
            sf.write(str(output_path), full_audio, SAMPLE_RATE)

            logger.info(
                f"Audio saved: {output_path.name} "
                f"({duration:.1f}s, {output_path.stat().st_size // 1024} KB)"
            )
            return output_path

        except Exception as e:
            logger.error(f"TTS error: {e}", exc_info=True)
            return None

    def synthesize_to_mp3(self, text: str, output_path: str | Path) -> Path | None:
        """Synthesize to WAV then convert to MP3 via ffmpeg."""
        import subprocess

        output_path = Path(output_path)
        wav_path = output_path.with_suffix(".wav")

        result = self.synthesize(text, wav_path)
        if not result:
            return None

        try:
            mp3_path = output_path.with_suffix(".mp3")
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(wav_path), "-q:a", "2", str(mp3_path)],
                check=True, capture_output=True,
            )
            wav_path.unlink()
            logger.info(f"MP3 saved: {mp3_path.name}")
            return mp3_path
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("ffmpeg not available, keeping WAV.")
            return wav_path

    @staticmethod
    def cleanup() -> None:
        """Release model from memory. Call before program exit to avoid bus error."""
        if TTSEngine._pipeline is not None:
            try:
                del TTSEngine._pipeline
                TTSEngine._pipeline = None
                import torch, gc
                gc.collect()
                if hasattr(torch, "mps"):
                    torch.mps.empty_cache()
                logger.debug("TTS pipeline released from memory.")
            except Exception:
                pass