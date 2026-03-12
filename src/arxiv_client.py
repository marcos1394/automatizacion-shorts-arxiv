"""
arxiv_client.py — Fetch recent papers from arXiv by category.
Returns a single Paper (fetch_latest_paper) or list (fetch_recent_papers).
"""

import logging
from dataclasses import dataclass

import arxiv

logger = logging.getLogger(__name__)

CATEGORIES = {
    "ai":       "cs.AI",
    "ml":       "cs.LG",
    "cv":       "cs.CV",
    "nlp":      "cs.CL",
    "robotics": "cs.RO",
}


@dataclass
class Paper:
    title:     str
    authors:   list[str]
    abstract:  str
    url:       str
    arxiv_id:  str
    published: str
    summary:   str = ""

    @classmethod
    def from_arxiv(cls, result) -> "Paper":
        return cls(
            title=result.title.strip(),
            authors=[str(a) for a in result.authors],
            abstract=result.summary.strip(),
            url=str(result.entry_id),
            arxiv_id=str(result.entry_id).split("/abs/")[-1],
            published=result.published.strftime("%B %d, %Y"),
            summary=result.summary.strip(),
        )


def fetch_recent_papers(category: str = "ai", max_results: int = 50) -> list[Paper]:
    """
    Fetch up to max_results recent papers for trend scoring.
    Returns list of Paper objects sorted by submission date (newest first).
    """
    cat = CATEGORIES.get(category, "cs.AI")
    logger.info(f"Buscando {max_results} papers en categoría: {cat}")

    client = arxiv.Client()
    search = arxiv.Search(
        query=f"cat:{cat}",
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    papers = []
    for result in client.results(search):
        papers.append(Paper.from_arxiv(result))

    logger.info(f"Obtenidos {len(papers)} papers de arXiv")
    return papers


def fetch_latest_paper(category: str = "ai") -> Paper | None:
    """Fetch the single most recent paper (legacy — used without trend scoring)."""
    papers = fetch_recent_papers(category=category, max_results=1)
    return papers[0] if papers else None