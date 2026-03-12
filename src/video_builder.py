"""
video_builder.py — Assembles slides + audio into a YouTube Short MP4.

Fix: audio attached AFTER concatenate_videoclips (MoviePy v2 known issue).

Install:
    pip install "moviepy>=2.0.0"
    brew install ffmpeg
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def build_video(
    slides: list[dict],
    audio_path: Path,
    output_path: Path,
) -> Path | None:
    """
    Assembles slides and audio into a single MP4.

    Args:
        slides:      List of {"path": Path, "duration": int/float} dicts.
        audio_path:  Path to WAV narration file.
        output_path: Where to save the final MP4.

    Returns:
        Path to output MP4, or None if failed.
    """
    try:
        from moviepy import ImageClip, AudioFileClip, concatenate_videoclips

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 1. Load audio
        audio = AudioFileClip(str(audio_path))
        total_duration = audio.duration
        logger.info(f"Building video | {len(slides)} slides | {total_duration:.1f}s audio")

        # 2. Normalize slide durations to match audio exactly
        slides = _normalize_durations(slides, total_duration)

        # 3. Build individual slide clips with Ken Burns
        clips = []
        for i, slide in enumerate(slides):
            clip = ImageClip(str(slide["path"])).with_duration(slide["duration"])
            clip = _ken_burns(clip, slide["duration"], direction=i % 2)
            clips.append(clip)
            logger.info(f"  Slide {i+1}/{len(slides)}: {Path(slide['path']).name} ({slide['duration']:.1f}s)")

        # 4. Concatenate video (NO audio yet — MoviePy v2 loses audio through concatenate)
        video = concatenate_videoclips(clips, method="compose")

        # 5. Attach audio AFTER concatenation (this is the fix)
        final = video.with_audio(audio)

        # 6. Export
        logger.info(f"Rendering → {output_path.name} ...")
        final.write_videofile(
            str(output_path),
            fps=30,
            codec="libx264",
            audio_codec="aac",
            audio_bitrate="192k",
            temp_audiofile=str(output_path.parent / "_temp_audio.m4a"),
            remove_temp=True,
            logger=None,
        )

        # Close clips to free memory
        audio.close()
        final.close()

        size_mb = output_path.stat().st_size / 1_000_000
        logger.info(f"Video saved: {output_path.name} ({size_mb:.1f} MB)")
        return output_path

    except ImportError:
        logger.error("MoviePy not installed. Run: pip install 'moviepy>=2.0.0'")
        return None
    except Exception as e:
        logger.error(f"Video build error: {e}", exc_info=True)
        return None


def _normalize_durations(slides: list[dict], total: float) -> list[dict]:
    """Scale durations to sum exactly to total audio duration."""
    raw_total = sum(s["duration"] for s in slides)
    scale = total / raw_total
    result = []
    running = 0.0
    for i, s in enumerate(slides):
        if i < len(slides) - 1:
            d = round(s["duration"] * scale, 3)
            running += d
        else:
            d = round(total - running, 3)
        result.append({**s, "duration": d})
    return result


def _ken_burns(clip, duration: float, direction: int = 0):
    """Slow zoom in or out for visual interest."""
    try:
        from moviepy import vfx
        if direction == 0:
            return clip.with_effects([vfx.Resize(lambda t: 1 + 0.06 * (t / duration))])
        else:
            return clip.with_effects([vfx.Resize(lambda t: 1.06 - 0.06 * (t / duration))])
    except Exception:
        return clip