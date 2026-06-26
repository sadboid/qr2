"""Citation network expansion: find related papers via citation relationships."""

import asyncio
import logging
from typing import List, Dict, Any
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
