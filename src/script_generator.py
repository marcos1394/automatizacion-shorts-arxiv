"""
script_generator.py — Generates a 60-90 second narration script from a paper.
"""

import logging
import re

logger = logging.getLogger(__name__)

SCRIPT_PROMPT_TEMPLATE = """You are a science YouTuber who creates viral short-form videos \
about AI research. Your style is clear, energetic, and accessible.

Write a voiceover narration script for a YouTube Short about this paper:

TITLE: {title}
AUTHORS: {authors}
DATE: {published}
ABSTRACT: {summary}

REQUIREMENTS:
- EXACTLY 160-200 words (this is critical — count carefully before finishing)
- Hook in the first 5 words — bold statement or surprising fact
- Explain the core problem in plain English
- State the 2-3 most important findings with concrete numbers if available
- End with a punchy "why this matters" statement
- DO NOT say "In this paper..." or "The authors found..." — be direct and vivid
- DO NOT include stage directions, labels, or formatting markers
- Write as ONE flowing paragraph — no bullet points, no line breaks
- Write ONLY the narration text, nothing else
- If your draft is under 160 words, expand with more context and detail"""


def generate_script(paper, llm_engine) -> str | None:
    """
    Generates a YouTube Short narration script from a paper.
    Retries once if the word count is too short.
    """
    logger.info("Generating narration script...")

    prompt = SCRIPT_PROMPT_TEMPLATE.format(
        title=paper.title,
        authors=", ".join(paper.authors) if paper.authors else "N/A",
        published=paper.published,
        summary=paper.summary[:2000],
    )

    for attempt in range(1, 3):  # max 2 attempts
        script = llm_engine.generate_custom(
            system_prompt=(
                "You are a science YouTuber. Output only the narration script. "
                "It must be 160-200 words — count carefully."
            ),
            user_message=prompt,
            max_tokens=600,      # increased from 400
            temperature=0.75,
        )

        if not script:
            logger.error("LLM returned empty script.")
            return None

        script = _clean_script(script)
        word_count = len(script.split())
        duration = word_count / 2.5

        logger.info(f"Attempt {attempt}: {word_count} words (~{duration:.0f}s)")

        if word_count >= 140:
            break
        elif attempt == 1:
            logger.warning(f"Script too short ({word_count} words), retrying with stronger prompt...")
            prompt = prompt + "\n\nYour previous response was too short. Write MORE detail — at least 160 words."

    return script


def _clean_script(text: str) -> str:
    text = re.sub(r"\*+", "", text)
    text = re.sub(r"^[\w\s]{1,20}:\s*\n", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()