"""
LLM engine — local inference via MLX + Qwen 2.5.
"""

import logging
import os
import re

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.getenv(
    "LLM_MODEL",
    "mlx-community/Qwen2.5-7B-Instruct-4bit",
)

THREAD_PROMPT_TEMPLATE = """You are a science communicator who makes cutting-edge AI research \
accessible and exciting on social media.

Write a Twitter/X thread about the following research paper:

TITLE: {title}
AUTHORS: {authors}
DATE: {published}
ABSTRACT: {summary}
LINK: {url}

REQUIREMENTS:
- Thread must have between 4 and 6 tweets
- Tweet 1: Powerful hook with an emoji that sparks curiosity (max 280 chars)
- Tweet 2: The problem this research solves
- Tweet 3-4: The 2-3 most important findings, with concrete data if available
- Tweet 5: Practical implications or future applications
- Tweet 6 (optional): Thought-provoking question to drive engagement + paper link
- Use emojis strategically (not excessively)
- Clear language, avoid excessive jargon
- Each tweet must work as a standalone post
- Do NOT add labels like "Tweet 1:" or "Hook:" before the content
- IMPORTANT: Separate each tweet with exactly "---" on its own line

Output ONLY the tweets, nothing else."""


class LLMEngine:
    """
    Wrapper around MLX + Qwen for text generation.
    Model is loaded once (singleton) to avoid wasting RAM.
    """

    _model = None
    _tokenizer = None

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        if LLMEngine._model is None:
            try:
                from mlx_lm import load
                logger.info(f"Loading model: {self.model_name}")
                logger.info("(First load may take ~30s, instant afterwards)")
                LLMEngine._model, LLMEngine._tokenizer = load(self.model_name)
                logger.info("Model loaded into memory.")
            except ImportError:
                logger.error("mlx_lm not installed. Run: pip install mlx-lm")
                raise
            except Exception as e:
                logger.error(f"Error loading model: {e}", exc_info=True)
                raise

    def generate_thread(self, paper) -> "str | None":
        try:
            from mlx_lm import generate
            from mlx_lm.sample_utils import make_sampler

            prompt = THREAD_PROMPT_TEMPLATE.format(
                title=paper.title,
                authors=", ".join(paper.authors) if paper.authors else "N/A",
                published=paper.published,
                summary=paper.summary[:1500],
                url=paper.url,
            )
            messages = [{"role": "user", "content": prompt}]

            # Qwen3 accepts enable_thinking; Qwen2.5 does not — handle both
            try:
                prompt_text = LLMEngine._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
            except TypeError:
                prompt_text = LLMEngine._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

            sampler = make_sampler(temp=0.7)
            response = generate(
                LLMEngine._model, LLMEngine._tokenizer,
                prompt=prompt_text, max_tokens=1200, sampler=sampler, verbose=False,
            )
            response = _strip_thinking_tags(response)
            logger.info(f"Generation complete ({len(response)} chars).")
            return response

        except Exception as e:
            logger.error(f"LLM generation error: {e}", exc_info=True)
            return None

    def generate_custom(self, system_prompt: str, user_message: str,
                        max_tokens: int = 800, temperature: float = 0.7) -> "str | None":
        try:
            from mlx_lm import generate
            from mlx_lm.sample_utils import make_sampler

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]
            try:
                prompt_text = LLMEngine._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
            except TypeError:
                prompt_text = LLMEngine._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            sampler = make_sampler(temp=temperature)
            result = generate(
                LLMEngine._model, LLMEngine._tokenizer,
                prompt=prompt_text, max_tokens=max_tokens, sampler=sampler, verbose=False,
            )
            return _strip_thinking_tags(result)
        except Exception as e:
            logger.error(f"Custom generation error: {e}", exc_info=True)
            return None


def _strip_thinking_tags(text: str) -> str:
    """Remove <think>...</think> blocks that Qwen 3 may insert."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()