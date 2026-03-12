#!/usr/bin/env python3
"""
Bot principal — orquesta fetch → generate → publish.
"""

import logging
from datetime import datetime

from .arxiv_client import fetch_latest_paper
from .llm_engine import LLMEngine
from .publisher import XPublisher
from .utils import save_run_record

logger = logging.getLogger(__name__)


def run_bot(dry_run: bool = False) -> bool:
    logger.info("=" * 60)
    logger.info(f"Starting bot | dry_run={dry_run} | {datetime.now().isoformat()}")

    # 1. Fetch paper
    logger.info("Fetching latest paper from arXiv...")
    paper = fetch_latest_paper()
    if not paper:
        logger.error("Could not fetch any paper.")
        return False
    logger.info(f"Paper found: {paper.title[:80]}...")

    # 2. Generate thread
    logger.info("Generating thread with local LLM...")
    llm = LLMEngine()
    thread_text = llm.generate_thread(paper)
    if not thread_text:
        logger.error("LLM did not generate any content.")
        return False

    tweets = _parse_thread(thread_text)
    logger.info(f"Thread generated: {len(tweets)} tweets")
    for i, tweet in enumerate(tweets, 1):
        logger.debug(f"  Tweet {i} ({len(tweet)} chars): {tweet[:60]}...")

    # 3. Publish or simulate
    if dry_run:
        print("\n--- DRY RUN ---")
        for i, tweet in enumerate(tweets, 1):
            print(f"\n[Tweet {i}/{len(tweets)}]\n{tweet}\n{'─' * 40}")
        logger.info("Dry run complete. Nothing was published.")
        return True

    logger.info("Publishing thread to X...")
    publisher = XPublisher()
    success = publisher.publish_thread(tweets)

    if success:
        save_run_record(paper, tweets)
        logger.info("Thread published successfully.")
    else:
        logger.error("Failed to publish thread.")

    return success


def _parse_thread(text: str) -> list[str]:
    """
    Converts raw LLM output into a clean list of valid tweets.
    - Splits on '---'
    - Strips leading dashes/labels the model sometimes adds (e.g. '-hook:')
    - Truncates tweets that exceed 280 chars
    """
    raw_tweets = [t.strip() for t in text.split("---") if t.strip()]
    tweets = []
    for tweet in raw_tweets:
        # Remove leading dashes and label artifacts like "-hook:" or "-Hallazgo 1:"
        tweet = tweet.lstrip("-").strip()
        # Remove label patterns at the start: "hook:", "Finding 1:", etc.
        import re
        tweet = re.sub(r"^[\w\s]+:\s*", "", tweet, count=1) if re.match(r"^-?[\w\s]{1,20}:\s", tweet) else tweet
        tweet = tweet.strip()

        if len(tweet) > 280:
            truncated = tweet[:277].rsplit(" ", 1)[0] + "..."
            tweets.append(truncated)
        else:
            tweets.append(tweet)
    return tweets