"""Crossref API integration for additional paper discovery."""

import asyncio
import logging
from typing import List, Dict, Any
import httpx

logger = logging.getLogger(__name__)

_CROSSREF_BASE = "https://api.crossref.org/v1"
_crossref_semaphore = asyncio.Semaphore(10)  # Crossref allows 50 req/sec, be conservative


async def search_crossref(
    query: str,
    limit: int = 20,
    filters: Dict[str, Any] = None,
) -> List[Dict[str, Any]]:
    """
    Search Crossref for papers by query with optional filters.

    Free API, no authentication required.

    Args:
        query: Search query (keywords)
        limit: Number of results to return
        filters: Optional filters like {'type': 'journal-article', 'from-pub-date': '2020-01-01'}

    Returns:
        List of paper metadata dicts with DOI, title, authors, year, abstract, venue
    """
    async with _crossref_semaphore:
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                params = {
                    "query": query,
                    "rows": limit,
                    "sort": "relevance",
                }

                # Add filters if provided
                if filters:
                    for key, value in filters.items():
                        params[f"filter"] = f"{key}:{value}"

                r = await client.get(
                    f"{_CROSSREF_BASE}/works",
                    params=params,
                    headers={"User-Agent": "QR2-Research-Machine (research@example.com)"},
                )

                if r.status_code != 200:
                    logger.warning(f"Crossref search failed: {r.status_code}")
                    return []

                data = r.json()
                papers = []

                for item in data.get("message", {}).get("items", []):
                    try:
                        paper = _parse_crossref_item(item)
                        if paper:
                            papers.append(paper)
                    except Exception as e:
                        logger.debug(f"Failed to parse Crossref item: {e}")
                        continue

                logger.info(f"Crossref '{query[:50]}': {len(papers)} papers")
                return papers

        except Exception as e:
            logger.warning(f"Crossref request failed: {e}")
            return []


def _parse_crossref_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Parse a Crossref API response item into standardized format."""
    title = item.get("title", [""])[0] if item.get("title") else ""
    if not title:
        return None

    # Extract authors
    authors = []
    for author in item.get("author", []):
        name_parts = []
        if author.get("given"):
            name_parts.append(author.get("given"))
        if author.get("family"):
            name_parts.append(author.get("family"))
        if name_parts:
            authors.append(" ".join(name_parts))

    # Extract year from published date
    year = None
    if item.get("published-online"):
        try:
            year = int(item["published-online"]["date-parts"][0][0])
        except (IndexError, TypeError, KeyError):
            pass
    if not year and item.get("issued"):
        try:
            year = int(item["issued"]["date-parts"][0][0])
        except (IndexError, TypeError, KeyError):
            pass

    year = year or 2026

    # Extract abstract (may not be available)
    abstract = item.get("abstract", "")
    if not abstract or len(abstract) < 50:
        # If no abstract, use title as fallback
        abstract = title

    # Extract venue (journal name)
    venue = item.get("container-title", [""])[0] if item.get("container-title") else ""

    # Get DOI as paper_id
    doi = item.get("DOI", "")

    # Citation count (crossref provides this via cite count)
    citation_count = item.get("is-referenced-by-count", 0)

    # URL (use DOI link)
    url = f"https://doi.org/{doi}" if doi else ""

    return {
        "paperId": doi or title[:30],
        "doi": doi,
        "title": title,
        "abstract": abstract,
        "authors": [{"name": a} for a in authors],
        "year": year,
        "citationCount": citation_count,
        "venue": venue,
        "url": url,
        "_source": "crossref",
    }
