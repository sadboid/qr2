"""Citation network expansion + Research Rabbit-style discovery tools.

Two layers:
1. Network expansion (async)  — fetch forward/backward citations via SS API
2. Local graph analysis (sync) — anchor papers, themed groups, reading paths
   These work entirely on the already-fetched corpus dict list, so they are
   fast and need no extra network calls.
"""

import asyncio
import logging
from collections import defaultdict
from typing import List, Dict, Any, Tuple
import httpx

logger = logging.getLogger(__name__)

_SS_BASE = "https://api.semanticscholar.org/graph/v1"
_SS_API_KEY = __import__("os").environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
_ss_semaphore = asyncio.Semaphore(1)  # 1 req/1.1s with API key
_SS_DELAY = 1.1


async def expand_via_citation_network(
    top_papers: List[Dict[str, Any]],
    max_papers_per_direction: int = 3,
) -> List[Dict[str, Any]]:
    """
    Expand paper corpus by finding papers cited by and citing the top papers.

    For each of the top 5 papers:
    - Get papers it cites (backward direction) → add 2-3 most cited ones
    - Get papers citing it (forward direction) → add 2-3 most recent ones

    Args:
        top_papers: List of top-ranked papers (with Semantic Scholar paperId)
        max_papers_per_direction: Max papers to add per direction (citations + references)

    Returns:
        List of new paper dicts discovered via citation network
    """
    expanded = []
    seen_ids = {p.get("paperId") for p in top_papers}

    # Process top 5 papers
    for paper in top_papers[:5]:
        paper_id = paper.get("paperId", "")
        if not paper_id:
            continue

        logger.info(f"[Citation Network] Expanding via {paper.get('title', '')[:50]}...")

        # Get references (papers this paper cites)
        references = await _get_references(paper_id)
        for ref in references[:max_papers_per_direction]:
            if ref.get("paperId") not in seen_ids:
                expanded.append(ref)
                seen_ids.add(ref.get("paperId"))

        # Get citations (papers citing this paper)
        citations = await _get_citations(paper_id)
        for cite in citations[:max_papers_per_direction]:
            if cite.get("paperId") not in seen_ids:
                expanded.append(cite)
                seen_ids.add(cite.get("paperId"))

    logger.info(f"[Citation Network] Discovered {len(expanded)} papers via network expansion")
    return expanded


async def _get_references(paper_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Get papers cited by the given paper (backward direction)."""
    async with _ss_semaphore:
        await asyncio.sleep(_SS_DELAY)
        try:
            headers = {"x-api-key": _SS_API_KEY} if _SS_API_KEY else {}
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.get(
                    f"{_SS_BASE}/paper/{paper_id}/references",
                    params={"limit": limit, "fields": "paperId,title,abstract,authors,year,citationCount,venue,url"},
                    headers=headers,
                    timeout=30,
                )
                if r.status_code != 200:
                    logger.debug(f"Failed to get references for {paper_id}: {r.status_code}")
                    return []

                data = r.json().get("data", [])
                papers = []

                for item in data:
                    ref = item.get("citedPaper", {})
                    if ref.get("paperId"):
                        papers.append(_normalize_ss_paper(ref))

                # Sort by citation count (most cited = most influential)
                papers.sort(key=lambda p: p.get("citationCount", 0), reverse=True)
                return papers

        except Exception as e:
            logger.debug(f"Error getting references: {e}")
            return []


async def _get_citations(paper_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Get papers citing the given paper (forward direction)."""
    async with _ss_semaphore:
        await asyncio.sleep(_SS_DELAY)
        try:
            headers = {"x-api-key": _SS_API_KEY} if _SS_API_KEY else {}
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.get(
                    f"{_SS_BASE}/paper/{paper_id}/citations",
                    params={"limit": limit, "fields": "paperId,title,abstract,authors,year,citationCount,venue,url"},
                    headers=headers,
                    timeout=30,
                )
                if r.status_code != 200:
                    logger.debug(f"Failed to get citations for {paper_id}: {r.status_code}")
                    return []

                data = r.json().get("data", [])
                papers = []

                for item in data:
                    citing = item.get("citingPaper", {})
                    if citing.get("paperId"):
                        papers.append(_normalize_ss_paper(citing))

                # Sort by year (most recent = newest citations)
                papers.sort(key=lambda p: p.get("year", 0), reverse=True)
                return papers

        except Exception as e:
            logger.debug(f"Error getting citations: {e}")
            return []


def _normalize_ss_paper(paper: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize Semantic Scholar paper format for consistency."""
    return {
        "paperId": paper.get("paperId", ""),
        "title": paper.get("title", ""),
        "abstract": paper.get("abstract", ""),
        "authors": paper.get("authors", []),
        "year": paper.get("year", 2026),
        "citationCount": paper.get("citationCount", 0),
        "venue": paper.get("venue", ""),
        "url": paper.get("url", ""),
        "_source": "semantic_scholar",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Research Rabbit–style local graph analysis (no extra API calls)
# ─────────────────────────────────────────────────────────────────────────────

def find_anchor_papers(
    papers: List[Dict[str, Any]],
    top_n: int = 5,
) -> List[Dict[str, Any]]:
    """
    Find anchor papers — high-citation hubs that serve as conceptual entry points
    (analogous to Research Rabbit's 'highly connected' papers).

    Ranking: weighted sum of citation count + recency bonus.
    """
    _CURRENT_YEAR = 2026

    def _anchor_score(p: Dict[str, Any]) -> float:
        cites = p.get("citationCount", 0)
        year = p.get("year", 2000)
        age = max(1, _CURRENT_YEAR - year)
        recency_bonus = 1.0 if age <= 3 else 0.0
        return cites / age + recency_bonus * 20

    return sorted(papers, key=_anchor_score, reverse=True)[:top_n]


def group_by_theme(
    papers: List[Dict[str, Any]],
    keywords: List[str],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group papers into themed collections (Research Rabbit 'collections').
    Each paper is assigned to its best-matching keyword theme.
    Papers that match no keyword go into 'General'.
    """
    groups: Dict[str, List] = defaultdict(list)

    for paper in papers:
        text = (
            (paper.get("title") or "") + " " + (paper.get("abstract") or "")
        ).lower()
        best_kw = None
        best_count = 0
        for kw in keywords[:6]:
            count = text.count(kw.lower())
            if count > best_count:
                best_count = count
                best_kw = kw
        groups[best_kw or "General"].append(paper)

    # Sort each group internally by citation count
    return {k: sorted(v, key=lambda p: -p.get("citationCount", 0)) for k, v in groups.items()}


def generate_reading_path(
    papers: List[Dict[str, Any]],
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Generate a prioritised reading path (Research Rabbit 'reading list' concept):

    Stage 1 — Foundational: highly cited (≥50 citations), read first
    Stage 2 — Bridge: moderate citations (10–49), conceptual links
    Stage 3 — Frontier: recent papers (≥2023), newest advances

    Returns list of (stage_label, paper_dict) tuples.
    """
    _CURRENT_YEAR = 2026
    _RECENT_CUTOFF = _CURRENT_YEAR - 3

    foundational = [p for p in papers if p.get("citationCount", 0) >= 50]
    frontier = [
        p for p in papers
        if (p.get("year", 0) >= _RECENT_CUTOFF and p.get("citationCount", 0) < 50)
    ]
    bridge = [
        p for p in papers
        if p not in foundational and p not in frontier
    ]

    path: List[Tuple[str, Dict]] = []
    for p in sorted(foundational, key=lambda x: -x.get("citationCount", 0))[:4]:
        path.append(("Foundational", p))
    for p in sorted(bridge, key=lambda x: -x.get("citationCount", 0))[:4]:
        path.append(("Bridge", p))
    for p in sorted(frontier, key=lambda x: -x.get("year", 0))[:4]:
        path.append(("Frontier", p))

    return path


def compute_co_citation_clusters(
    papers: List[Dict[str, Any]],
    keywords: List[str],
) -> List[Dict[str, Any]]:
    """
    Simple co-citation cluster summary (Research Rabbit network clusters).

    Groups papers by theme and computes cluster statistics:
    - cluster name (keyword)
    - paper count
    - anchor paper (highest cited)
    - average year (recency)

    Returns list of cluster summary dicts suitable for serialisation.
    """
    groups = group_by_theme(papers, keywords)
    clusters = []
    for kw, group_papers in groups.items():
        if not group_papers:
            continue
        anchor = max(group_papers, key=lambda p: p.get("citationCount", 0))
        avg_year = (
            sum(p.get("year", 2020) for p in group_papers) / len(group_papers)
            if group_papers else 2020
        )
        clusters.append({
            "theme": kw,
            "paper_count": len(group_papers),
            "anchor_paper": anchor.get("title", ""),
            "anchor_year": anchor.get("year", ""),
            "anchor_citations": anchor.get("citationCount", 0),
            "avg_year": round(avg_year, 1),
            "papers": [
                {
                    "title": p.get("title", ""),
                    "year": p.get("year", ""),
                    "citations": p.get("citationCount", 0),
                    "url": p.get("url", ""),
                }
                for p in group_papers[:5]
            ],
        })
    return sorted(clusters, key=lambda c: -c["paper_count"])
