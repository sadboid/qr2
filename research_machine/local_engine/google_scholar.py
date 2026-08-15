"""Google Scholar API integration (via SerpAPI) — OPTIONAL paid service."""

import asyncio
import logging
from typing import List, Dict, Any
import os

logger = logging.getLogger(__name__)

# Note: Google Scholar requires SerpAPI (https://serpapi.com/)
# Free tier: 100 searches/month
# Paid: $0.01 per search
# Set SERPAPI_KEY in .env to enable
_SERPAPI_KEY = os.environ.get("SERPAPI_KEY", "")


async def search_google_scholar(
    query: str,
    limit: int = 30,
) -> List[Dict[str, Any]]:
    """
    Search Google Scholar via SerpAPI (OPTIONAL paid service).

    Google Scholar covers 389M+ papers, better than Semantic Scholar for:
    - Citation tracking (more complete citation counts)
    - Real-time indexing
    - Broader coverage of non-CS papers

    Cost: $0.01 per query (SerpAPI pricing)

    Args:
        query: Search query (keywords)
        limit: Number of results to return

    Returns:
        List of paper metadata dicts
    """
    if not _SERPAPI_KEY:
        logger.debug("SERPAPI_KEY not set — skipping Google Scholar search")
        return []

    try:
        import httpx

        async with httpx.AsyncClient(timeout=30) as client:
            # SerpAPI endpoint for Google Scholar
            params = {
                "q": query,
                "engine": "google_scholar",
                "api_key": _SERPAPI_KEY,
                "num": limit,
            }

            r = await client.get(
                "https://serpapi.com/search",
                params=params,
            )

            if r.status_code != 200:
                logger.warning(f"Google Scholar search failed: {r.status_code}")
                return []

            data = r.json()
            papers = []

            for result in data.get("organic_results", []):
                try:
                    paper = _parse_scholar_result(result)
                    if paper:
                        papers.append(paper)
                except Exception as e:
                    logger.debug(f"Failed to parse Scholar result: {e}")
                    continue

            logger.info(f"Google Scholar '{query[:50]}': {len(papers)} papers (cost: ${limit * 0.01:.2f})")
            return papers

    except ImportError:
        logger.warning("httpx not available for Google Scholar search")
        return []
    except Exception as e:
        logger.warning(f"Google Scholar request failed: {e}")
        return []


def _parse_scholar_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Parse a Google Scholar search result."""
    title = result.get("title", "")
    if not title:
        return None

    # Extract authors from publication info
    publication = result.get("publication_info", {})
    author_str = publication.get("authors", "")
    authors = [a.strip() for a in author_str.split(",") if a.strip()]

    # Citation count from the snippet
    inline_links = result.get("inline_links", {})
    citation_count = 0
    if "Cited by" in str(inline_links):
        # Try to extract count
        cited_by = str(inline_links).split("Cited by")[-1]
        try:
            citation_count = int("".join(filter(str.isdigit, cited_by.split()[0])))
        except (IndexError, ValueError):
            pass

    # Year from publication date
    year = 2026
    snippet = result.get("snippet", "")
    import re

    year_match = re.search(r"(\d{4})", snippet)
    if year_match:
        year = int(year_match.group(1))

    # Construct abstract from snippet
    abstract = snippet or title

    # Venue from publication
    venue = publication.get("publication", "")

    # Link as URL
    url = result.get("link", "")

    return {
        "paperId": url or title[:30],
        "title": title,
        "abstract": abstract,
        "authors": [{"name": a} for a in authors],
        "year": year,
        "citationCount": citation_count,
        "venue": venue,
        "url": url,
        "_source": "google_scholar",
    }


def print_cost_estimate(num_queries: int) -> str:
    """Print estimated cost for Google Scholar searches."""
    cost = num_queries * 0.01
    return f"Estimated cost: ${cost:.2f} ({num_queries} queries × $0.01)"
