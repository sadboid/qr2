"""OpenAlex API integration — 250M+ open academic works, free, no auth required.

OpenAlex polite pool: add mailto= for higher rate limits (100k req/day).
Abstract is stored as an inverted index (word → [positions]) — must reconstruct.

Docs: https://docs.openalex.org/api-entities/works/search-works
"""

import asyncio
import logging
import os
from typing import List, Dict, Any, Optional

import httpx

logger = logging.getLogger(__name__)

_OA_BASE = "https://api.openalex.org/works"
_MAILTO = os.environ.get("OPENALEX_MAILTO", "research@qr2.ai")
_CURRENT_YEAR = 2026

_oa_semaphore = asyncio.Semaphore(5)  # polite pool: up to 10/sec, stay conservative


def _reconstruct_abstract(inverted_index: Optional[Dict[str, List[int]]]) -> str:
    """Reconstruct abstract text from OpenAlex inverted index format."""
    if not inverted_index:
        return ""
    positions: List = []
    for word, locs in inverted_index.items():
        for pos in locs:
            positions.append((pos, word))
    positions.sort(key=lambda x: x[0])
    return " ".join(w for _, w in positions)


def _parse_oa_work(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Map an OpenAlex work object to the standard corpus dict format."""
    title = (item.get("title") or "").strip()
    if not title:
        return None

    # Abstract
    abstract = _reconstruct_abstract(item.get("abstract_inverted_index"))
    if len(abstract) < 50:
        # Some works have no abstract in OA (paywalled) — skip rather than fabricate
        return None

    # Authors
    authors = []
    for auth in item.get("authorships", []):
        name = auth.get("author", {}).get("display_name", "")
        if name:
            authors.append(name)

    # Year
    year = item.get("publication_year") or _CURRENT_YEAR

    # Venue — prefer journal name, fall back to source type
    primary = item.get("primary_location") or {}
    source = primary.get("source") or {}
    venue = source.get("display_name") or item.get("type", "")

    # Citation count
    citation_count = item.get("cited_by_count") or 0

    # URL — prefer landing page, else DOI, else OA ID
    url = (primary.get("landing_page_url")
           or f"https://doi.org/{item['doi']}" if item.get("doi") else ""
           or item.get("id", ""))

    # Open-access PDF
    oa_info = item.get("open_access") or {}
    oa_url = oa_info.get("oa_url") or None

    # Concept scores (OpenAlex-assigned domain tags)
    concepts = item.get("concepts") or []
    concept_names = [c.get("display_name", "") for c in concepts if c.get("score", 0) > 0.3]

    oa_id = item.get("id", "").replace("https://openalex.org/", "")

    return {
        "paperId": oa_id or title[:30],
        "title": title,
        "abstract": abstract,
        "authors": [{"name": a} for a in authors],
        "year": year,
        "citationCount": citation_count,
        "venue": venue,
        "url": url,
        "openAccessPdf": {"url": oa_url} if oa_url else None,
        "_source": "openalex",
        "_concepts": concept_names,
    }


async def search_openalex(
    query: str,
    limit: int = 25,
    from_year: Optional[int] = None,
    filter_concepts: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Search OpenAlex for works matching query.

    Args:
        query: Natural-language search query
        limit: Max results (OpenAlex max per-page: 200)
        from_year: Optional year filter (e.g. 2018)
        filter_concepts: Optional list of OpenAlex concept IDs to filter by,
                         e.g. ['C41008148'] for Computer Science

    Returns:
        List of standardised paper dicts (same format as SS/Crossref)
    """
    async with _oa_semaphore:
        params: Dict[str, Any] = {
            "search": query,
            "per-page": min(limit, 50),
            "select": (
                "id,title,abstract_inverted_index,authorships,"
                "publication_year,cited_by_count,primary_location,"
                "open_access,concepts,doi,type"
            ),
            "sort": "relevance_score:desc",
            "mailto": _MAILTO,
        }

        # Optional year filter
        filters = []
        if from_year:
            filters.append(f"publication_year:>{from_year}")
        # Skip retracted works
        filters.append("is_retracted:false")
        # Only include journal articles, conference papers, preprints
        filters.append("type:article|preprint|proceedings-article")
        if filter_concepts:
            filters.append(f"concepts.id:{'|'.join(filter_concepts)}")
        if filters:
            params["filter"] = ",".join(filters)

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.get(
                    _OA_BASE,
                    params=params,
                    headers={"User-Agent": f"QR2-ResearchMachine/1.0 (mailto:{_MAILTO})"},
                )
                if r.status_code == 429:
                    logger.warning("OpenAlex rate limited — waiting 5s")
                    await asyncio.sleep(5)
                    r = await client.get(_OA_BASE, params=params)
                if r.status_code != 200:
                    logger.warning(f"OpenAlex search failed: {r.status_code}")
                    return []

                items = r.json().get("results", [])
                papers = []
                for item in items:
                    try:
                        p = _parse_oa_work(item)
                        if p:
                            papers.append(p)
                    except Exception as e:
                        logger.debug(f"OpenAlex parse error: {e}")
                        continue

                logger.info(f"OpenAlex '{query[:50]}': {len(papers)} papers")
                return papers

        except Exception as e:
            logger.warning(f"OpenAlex request failed: {e}")
            return []


async def fetch_openalex_batch(
    queries: List[str],
    limit_per_query: int = 25,
) -> List[Dict[str, Any]]:
    """Run multiple OpenAlex queries concurrently and merge results."""
    tasks = [search_openalex(q, limit=limit_per_query) for q in queries]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    merged = []
    for r in results:
        if isinstance(r, list):
            merged.extend(r)
    return merged
