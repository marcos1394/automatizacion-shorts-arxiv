#!/usr/bin/env python3
"""
arxiv-social-automation — Entry point.

Usage:
    python main.py                          # Post thread to X
    python main.py --dry-run                # Simulate without posting
    python main.py --shorts                 # Full pipeline + auto-upload a YouTube
    python main.py --shorts --no-upload       # Solo genera el video, no sube
    python main.py --shorts --privacy private  # Sube como privado para revisar primero
    python main.py --shorts --no-ai-images  # Pipeline with gradient slides (faster)
    python main.py --shorts --audio-only    # Script + audio only
    python main.py --category ml            # Use cs.LG category
    python main.py --verify                 # Check X credentials
"""

import sys
import argparse
from pathlib import Path

# ── Fix sys.path ──────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if _ROOT.name == "src":
    _ROOT = _ROOT.parent
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv(dotenv_path=_ROOT / ".env")

from src.utils import setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="arXiv → X threads & YouTube Shorts bot")
    p.add_argument("--dry-run",      action="store_true", help="Generate thread but don't post to X")
    p.add_argument("--shorts",       action="store_true", help="Run YouTube Shorts pipeline")
    p.add_argument("--audio-only",   action="store_true", help="With --shorts: script+audio only")
    p.add_argument("--no-ai-images", action="store_true", help="With --shorts: use gradients instead of FLUX")
    p.add_argument("--category",     default="ai", choices=["ai", "ml", "cv", "nlp", "robotics"])
    p.add_argument("--verify",       action="store_true", help="Verify X credentials")
    p.add_argument("--no-upload",    action="store_true", help="Skip YouTube upload (solo genera el video)")
    p.add_argument("--privacy",      default="public",  choices=["public", "private", "unlisted"])
    p.add_argument("--log-level",    default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(level=args.log_level)

    if args.verify:
        from src.publisher import XPublisher
        sys.exit(0 if XPublisher().verify_credentials() else 1)

    if args.shorts:
        sys.exit(0 if run_shorts_pipeline(args) else 1)

    from src.bot import run_bot
    sys.exit(0 if run_bot(dry_run=args.dry_run) else 1)


def run_shorts_pipeline(args) -> bool:
    import logging
    logger = logging.getLogger(__name__)

    from src.arxiv_client import fetch_latest_paper
    from src.llm_engine import LLMEngine
    from src.script_generator import generate_script
    from src.tts_engine import TTSEngine
    from src.slides_generator import generate_slides, parse_script_to_slides
    from src.video_builder import build_video

    output_dir = _ROOT / "output" / "shorts"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # ── 1. Fetch + score papers by trend ─────────────────────────────
        logger.info("━" * 52)
        logger.info("STEP 1/5 — Fetching & scoring papers by trend...")
        from src.arxiv_client import fetch_recent_papers
        from src.trend_selector import select_trending_paper

        papers = fetch_recent_papers(category=args.category, max_results=50)
        if not papers:
            logger.error("No papers fetched.")
            return False

        paper = select_trending_paper(papers, category=args.category, top_n=15)
        if not paper:
            return False
        logger.info(f"Selected: {paper.title[:70]}...")

        safe_name = "".join(
            c if c.isalnum() or c in "-_ " else "" for c in paper.title[:40]
        ).strip().replace(" ", "_")
        paper_dir = output_dir / safe_name
        paper_dir.mkdir(exist_ok=True)

        # ── 2. Generate script ────────────────────────────────────────────
        logger.info("━" * 52)
        logger.info("STEP 2/5 — Generating narration script...")
        llm = LLMEngine()
        script = generate_script(paper, llm)
        if not script:
            return False

        print(f"\n{'═' * 55}\nNARRATION SCRIPT:\n{'─' * 55}")
        print(script)
        print(f"{'─' * 55}\nWords: {len(script.split())} (~{len(script.split())/2.5:.0f}s)\n")
        (paper_dir / "script.txt").write_text(script, encoding="utf-8")

        # ── 3. Synthesize audio ────────────────────────────────────────────
        logger.info("━" * 52)
        logger.info("STEP 3/5 — Synthesizing audio with Kokoro-82M...")
        tts = TTSEngine()
        audio_path = paper_dir / f"{safe_name}.wav"
        audio_result = tts.synthesize(script, audio_path)
        if not audio_result:
            return False

        if args.audio_only:
            print(f"\n✅ Audio saved: {audio_result}")
            return True

        # ── 4. Generate slides ─────────────────────────────────────────────
        logger.info("━" * 52)
        logger.info("STEP 4/5 — Generating slides...")
        slides_dir = paper_dir / "slides"
        slide_content = parse_script_to_slides(script, llm)

        backgrounds = None
        if not args.no_ai_images:
            logger.info("Generating AI backgrounds with FLUX.1-schnell...")
            logger.info("(~60-120s per image at 576x1024, upscaled to 1080x1920)")
            # Liberar Kokoro antes de cargar FLUX — ambos son pesados en RAM
            TTSEngine.cleanup()
            import gc, mlx.core as mx
            gc.collect()
            try:
                mx.metal.clear_cache()
            except Exception:
                pass
            try:
                from src.image_generator import ImageGenerator
                # Extraer keywords del título para prompts únicos
                import re
                stopwords = {"the","a","an","of","in","on","for","with","and","or",
                             "is","are","we","our","via","using","based","towards",
                             "new","novel","large","model","paper","study"}
                kw = [w for w in re.findall(r'\b[a-zA-Z]{4,}\b', paper.title.lower())
                      if w not in stopwords][:5]

                ig = ImageGenerator()
                backgrounds = ig.generate_all_backgrounds(
                    slides_dir / "backgrounds",
                    paper_id=paper.arxiv_id,
                    paper_title=paper.title,
                    paper_keywords=kw,
                )
                logger.info("AI backgrounds ready.")
            except Exception as e:
                logger.warning(f"FLUX failed ({e}), falling back to gradients.")
                backgrounds = None

        slides = generate_slides(paper, slide_content, slides_dir, backgrounds=backgrounds)

        # ── 5. Build video ─────────────────────────────────────────────────
        logger.info("━" * 52)
        logger.info("STEP 5/5 — Assembling video with MoviePy...")
        video_path = paper_dir / f"{safe_name}_short.mp4"
        result = build_video(slides, audio_result, video_path)

        if result:
            size_mb = result.stat().st_size / 1_000_000
            print(f"\n{'═' * 55}")
            print(f"✅ YouTube Short ready!")
            print(f"   Video : {result}")
            print(f"   Size  : {size_mb:.1f} MB")
            print(f"   Next  : Phase 5 — auto-upload to YouTube")
            print(f"{'═' * 55}\n")
            return True
        return False

    finally:
        TTSEngine.cleanup()
        try:
            from src.image_generator import ImageGenerator
            ImageGenerator.cleanup()
        except Exception:
            pass
        import gc
        gc.collect()


if __name__ == "__main__":
    main()