"""
trend_selector.py — Selección inteligente de papers por tendencia real.

Estrategia multicapa:
  1. Obtener 50 papers recientes de arXiv (últimas 48h)
  2. Consultar Semantic Scholar para cada paper → influenceScore + citationVelocity
  3. Cruzar con Google Trends de los keywords del abstract → trendScore
  4. Score compuesto → elegir el paper con mayor potencial viral

APIs usadas:
  - arXiv (ya integrado)
  - Semantic Scholar (gratis, sin key, 100 req/5min)
  - pytrends (Google Trends, sin key)

Install:
    pip install pytrends requests semanticscholar
"""

import logging
import time
import re
from dataclasses import dataclass, field
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

SS_BASE = "https://api.semanticscholar.org/graph/v1"
SS_FIELDS = "title,year,citationCount,influentialCitationCount,publicationDate"

CATEGORY_KEYWORDS = {
    "ai":       ["artificial intelligence", "LLM", "foundation model", "reasoning"],
    "ml":       ["machine learning", "deep learning", "neural network", "transformer"],
    "cv":       ["computer vision", "image recognition", "diffusion model", "ViT"],
    "nlp":      ["NLP", "language model", "text generation", "RLHF"],
    "robotics": ["robot", "autonomous", "reinforcement learning", "manipulation"],
}


@dataclass
class ScoredPaper:
    paper: object             # arxiv Paper object
    arxiv_id: str = ""
    citation_count: int = 0
    influential_citations: int = 0
    trend_score: float = 0.0
    recency_score: float = 0.0
    total_score: float = 0.0
    ss_found: bool = False


def _extract_keywords(text: str, max_kw: int = 4) -> list[str]:
    """Extract meaningful keywords from title/abstract for Google Trends."""
    # Remove common academic stopwords
    stopwords = {
        "the", "a", "an", "of", "in", "on", "for", "with", "and", "or",
        "is", "are", "we", "our", "this", "that", "from", "to", "by",
        "via", "using", "based", "towards", "new", "novel", "large", "model",
        "paper", "study", "method", "approach", "framework", "learning",
    }
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    seen, keywords = set(), []
    for w in words:
        if w not in stopwords and w not in seen:
            seen.add(w)
            keywords.append(w)
        if len(keywords) >= max_kw:
            break
    return keywords


def _get_semantic_scholar_score(arxiv_id: str) -> dict:
    """Query Semantic Scholar for citation metrics. Returns empty dict if not found."""
    clean_id = arxiv_id.replace("http://arxiv.org/abs/", "").strip()
    url = f"{SS_BASE}/paper/arXiv:{clean_id}"
    try:
        r = requests.get(url, params={"fields": SS_FIELDS}, timeout=8)
        if r.status_code == 200:
            data = r.json()
            return {
                "citation_count": data.get("citationCount", 0) or 0,
                "influential": data.get("influentialCitationCount", 0) or 0,
                "found": True,
            }
        elif r.status_code == 404:
            return {"citation_count": 0, "influential": 0, "found": False}
        else:
            logger.debug(f"SS API {r.status_code} for {clean_id}")
            return {"citation_count": 0, "influential": 0, "found": False}
    except Exception as e:
        logger.debug(f"SS lookup failed for {clean_id}: {e}")
        return {"citation_count": 0, "influential": 0, "found": False}


def _get_google_trends_score(keywords: list[str]) -> float:
    """
    Get Google Trends interest score for keywords (last 7 days).
    Returns 0.0-1.0. Falls back to 0.0 if pytrends unavailable.
    """
    if not keywords:
        return 0.0
    try:
        from pytrends.request import TrendReq
        pt = TrendReq(hl="en-US", tz=360, timeout=(10, 25))
        # Use top 3 keywords max (pytrends limit is 5)
        kw_list = keywords[:3]
        pt.build_payload(kw_list, timeframe="now 7-d", geo="")
        df = pt.interest_over_time()
        if df.empty:
            return 0.0
        # Average interest across keywords and time
        score = float(df[kw_list].mean().mean()) / 100.0
        return round(score, 3)
    except ImportError:
        logger.debug("pytrends not installed — skipping Google Trends score")
        return 0.0
    except Exception as e:
        logger.debug(f"Google Trends error: {e}")
        return 0.0


def _recency_score(paper) -> float:
    """
    Score based on how recent the paper is.
    Papers from today/yesterday get 1.0, older papers decay.
    """
    from datetime import datetime, timezone
    try:
        pub = paper.published
        if hasattr(pub, "replace"):
            pub = pub.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        days_old = (now - pub).days
        # Decay: 1.0 today, 0.5 at 3 days, 0.1 at 14 days
        return max(0.0, round(1.0 / (1.0 + days_old * 0.3), 3))
    except Exception:
        return 0.5


def select_trending_paper(papers: list, category: str = "ai",
                           top_n: int = 5) -> object:
    """
    Score and rank papers by trend potential.
    Returns the highest-scoring paper.

    Scoring weights:
      - Semantic Scholar citations:       35%
      - Semantic Scholar influential:     25%
      - Google Trends keyword interest:   25%
      - Recency (days since publication): 15%
    """
    logger.info(f"Scoring {len(papers)} papers for trend potential...")

    scored = []
    for i, paper in enumerate(papers[:top_n * 3]):  # check top 3x candidates
        sp = ScoredPaper(paper=paper, arxiv_id=str(paper.arxiv_id))

        # ── Semantic Scholar ──────────────────────────────────────────────
        ss = _get_semantic_scholar_score(sp.arxiv_id)
        sp.citation_count = ss["citation_count"]
        sp.influential_citations = ss["influential"]
        sp.ss_found = ss["found"]

        # ── Google Trends ─────────────────────────────────────────────────
        keywords = _extract_keywords(paper.title + " " + paper.summary[:300])
        sp.trend_score = _get_google_trends_score(keywords)

        # ── Recency ───────────────────────────────────────────────────────
        sp.recency_score = _recency_score(paper)

        # ── Composite score ───────────────────────────────────────────────
        # Normalize citation counts (log scale, cap at 100)
        import math
        cit_norm = min(math.log1p(sp.citation_count) / math.log1p(100), 1.0)
        inf_norm  = min(math.log1p(sp.influential_citations) / math.log1p(20), 1.0)

        sp.total_score = round(
            0.35 * cit_norm +
            0.25 * inf_norm +
            0.25 * sp.trend_score +
            0.15 * sp.recency_score,
            4
        )

        scored.append(sp)

        logger.info(
            f"  [{i+1:2d}] score={sp.total_score:.3f} | "
            f"cit={sp.citation_count} inf={sp.influential_citations} "
            f"trend={sp.trend_score:.2f} rec={sp.recency_score:.2f} | "
            f"{paper.title[:55]}..."
        )

        # Respect Semantic Scholar rate limit (100 req/5min = 1.2 req/s)
        time.sleep(0.9)

    if not scored:
        logger.warning("No papers scored — returning first paper")
        return papers[0]

    # Sort by score descending
    scored.sort(key=lambda x: x.total_score, reverse=True)

    winner = scored[0]
    logger.info(
        f"\n🏆 Selected paper (score={winner.total_score:.3f}):\n"
        f"   {winner.paper.title}\n"
        f"   Citations: {winner.citation_count} | "
        f"Influential: {winner.influential_citations} | "
        f"Trend: {winner.trend_score:.2f}"
    )
    return winner.paper


def save_trend_log(paper, score_data: dict, log_path: Path) -> None:
    """Append paper selection to trend log JSON for analytics."""
    import json
    from datetime import datetime
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "title": paper.title,
        "arxiv_id": str(paper.arxiv_id),
        **score_data,
    }
    entries = []
    if log_path.exists():
        try:
            entries = json.loads(log_path.read_text())
        except Exception:
            entries = []
    entries.append(entry)
    log_path.write_text(json.dumps(entries, indent=2))