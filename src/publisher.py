"""
Publisher — posts threads to X using API v2 with OAuth 1.0a credentials.
"""

import logging
import os
import time

import tweepy

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY = 10
TWEET_DELAY = 3


class XPublisher:

    def __init__(self):
        self.client = self._build_client()

    def _build_client(self) -> tweepy.Client:
        required = [
            "X_API_KEY", "X_API_SECRET",
            "X_ACCESS_TOKEN", "X_ACCESS_TOKEN_SECRET",
        ]
        missing = [v for v in required if not os.getenv(v)]
        if missing:
            raise EnvironmentError(f"Missing env vars: {', '.join(missing)}")

        return tweepy.Client(
            consumer_key=os.getenv("X_API_KEY"),
            consumer_secret=os.getenv("X_API_SECRET"),
            access_token=os.getenv("X_ACCESS_TOKEN"),
            access_token_secret=os.getenv("X_ACCESS_TOKEN_SECRET"),
        )

    def verify_credentials(self) -> bool:
        try:
            me = self.client.get_me()
            logger.info(f"Credentials OK. User: @{me.data.username}")
            return True
        except Exception as e:
            logger.error(f"Invalid credentials: {e}")
            return False

    def _post_tweet(self, text: str, reply_to: int | None = None) -> int | None:
        kwargs = {"text": text}
        if reply_to:
            kwargs["reply"] = {"in_reply_to_tweet_id": reply_to}

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = self.client.create_tweet(**kwargs)
                tweet_id = resp.data["id"]
                logger.debug(f"Tweet posted (id={tweet_id})")
                return tweet_id

            except tweepy.errors.Forbidden as e:
                logger.error(f"403 Forbidden: {e}")
                return None

            except tweepy.errors.TooManyRequests:
                wait = RETRY_DELAY * attempt
                logger.warning(f"Rate limit hit. Waiting {wait}s...")
                time.sleep(wait)

            except tweepy.errors.TwitterServerError as e:
                logger.warning(f"503 from X (attempt {attempt}/{MAX_RETRIES}). Retrying in {RETRY_DELAY * attempt}s...")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY * attempt)

            except Exception as e:
                logger.error(f"Unexpected error: {e}", exc_info=True)
                return None

        logger.error(f"All {MAX_RETRIES} attempts failed.")
        return None

    def publish_thread(self, tweets: list[str]) -> bool:
        if not tweets:
            return False

        logger.info(f"Publishing thread of {len(tweets)} tweets...")
        prev_id = None

        for i, tweet in enumerate(tweets, 1):
            logger.info(f"  Posting tweet {i}/{len(tweets)}...")
            tweet_id = self._post_tweet(tweet, reply_to=prev_id)

            if tweet_id is None:
                logger.error(f"Thread interrupted at tweet {i}.")
                return False

            prev_id = tweet_id
            if i < len(tweets):
                time.sleep(TWEET_DELAY)

        logger.info("Thread fully published.")
        return True