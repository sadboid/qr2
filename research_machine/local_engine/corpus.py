"""Literature corpus: search, deduplicate, rank real papers."""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime

import os

import httpx

from . import venue_ranking
from . import crossref_search
from . import citation_network

logger = logging.getLogger(__name__)

_SS_API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")

_CURRENT_YEAR = 2026
_SS_BASE = "https://api.semanticscholar.org/graph/v1"
_SS_FIELDS = "paperId,title,abstract,authors,year,citationCount,venue,externalIds,url"
_ARXIV_BASE = "http://export.arxiv.org/api/query"


@dataclass
class Paper:
    paper_id: str
    title: str
    abstract: str
    authors: List[str]
    year: int
    citation_count: int
    venue: str
    url: str
    source: str  # "semantic_scholar" | "arxiv"
    keywords_matched: List[str] = field(default_factory=list)
    author_h_index: int = 0  # Lead author h-index (from Semantic Scholar)

    @property
    def is_recent(self) -> bool:
        return self.year >= _CURRENT_YEAR - 3

    @property
    def relevance_score(self) -> float:
        """Advanced reputation scoring with venue tiers + h-index + recency decay."""
        # Keyword match ratio (0.0-1.0)
        kw_score = min(len(self.keywords_matched) / 3, 1.0)

        # Use advanced reputation formula from venue_ranking
        score = venue_ranking.calculate_reputation_score(
            keywords_match=kw_score,
            citation_count=self.citation_count,
            venue=self.venue,
            year=self.year,
            author_h_index=self.author_h_index,
            current_year=_CURRENT_YEAR,
        )
        return score

    def short_ref(self) -> str:
        first_author = self.authors[0].split(",")[0] if self.authors else "Unknown"
        return f"{first_author}, {self.year}"

    def apa_ref(self, idx: int) -> str:
        author_str = "; ".join(self.authors[:3])
        if len(self.authors) > 3:
            author_str += " et al."
        venue = f" *{self.venue}*." if self.venue else "."
        return f"{idx}. {author_str} ({self.year}). {self.title}{venue}"


_ss_semaphore = asyncio.Semaphore(1)  # one SS request at a time
_SS_DELAY = 1.1  # seconds between SS requests (1 req/s with API key)


async def _search_semantic_scholar(
    query: str, limit: int = 30, retries: int = 4
) -> List[Dict[str, Any]]:
    async with _ss_semaphore:
        await asyncio.sleep(_SS_DELAY)
        delay = 3.0
        for attempt in range(retries):
            try:
                headers = {"x-api-key": _SS_API_KEY} if _SS_API_KEY else {}
                async with httpx.AsyncClient() as client:
                    r = await client.get(
                        f"{_SS_BASE}/paper/search",
                        params={"query": query, "limit": limit, "fields": _SS_FIELDS},
                        headers=headers,
                        timeout=30,
                    )
                    if r.status_code == 429:
                        logger.warning(f"SS rate limit, waiting {delay}s (attempt {attempt+1})...")
                        await asyncio.sleep(delay)
                        delay = min(delay * 2, 30)
                        continue
                    r.raise_for_status()
                    data = r.json().get("data", [])
                    logger.info(f"SS '{query[:50]}': {len(data)} papers")
                    return data
            except Exception as e:
                logger.warning(f"SS attempt {attempt+1} failed: {e}")
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30)
        return []


async def _search_arxiv(query: str, max_results: int = 20) -> List[Dict[str, Any]]:
    import xml.etree.ElementTree as ET

    await asyncio.sleep(3)  # arXiv rate limit
    arxiv_url = "https://export.arxiv.org/api/query"  # use https directly
    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            r = await client.get(
                arxiv_url,
                params={
                    "search_query": f"all:{query}",
                    "max_results": max_results,
                    "sortBy": "submittedDate",
                    "sortOrder": "descending",
                },
                timeout=30,
            )
            r.raise_for_status()
    except Exception as e:
        logger.warning(f"arXiv search failed: {e}")
        return []

    papers = []
    try:
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(r.text)
        for entry in root.findall("atom:entry", ns):
            title_el = entry.find("atom:title", ns)
            abstract_el = entry.find("atom:summary", ns)
            published_el = entry.find("atom:published", ns)
            id_el = entry.find("atom:id", ns)

            if title_el is None or abstract_el is None:
                continue

            year = _CURRENT_YEAR
            if published_el is not None and published_el.text:
                try:
                    year = int(published_el.text[:4])
                except ValueError:
                    pass

            authors = [
                a.find("atom:name", ns).text
                for a in entry.findall("atom:author", ns)
                if a.find("atom:name", ns) is not None
            ]

            papers.append({
                "paperId": id_el.text if id_el is not None else "",
                "title": title_el.text.strip().replace("\n", " "),
                "abstract": abstract_el.text.strip().replace("\n", " "),
                "authors": [{"name": a} for a in authors],
                "year": year,
                "citationCount": 0,
                "venue": "arXiv",
                "url": id_el.text if id_el is not None else "",
                "_source": "arxiv",
            })
    except ET.ParseError as e:
        logger.warning(f"arXiv XML parse error: {e}")

    return papers


def _extract_keywords(text: str, keywords: List[str]) -> List[str]:
    text_lower = text.lower()
    return [kw for kw in keywords if kw.lower() in text_lower]


def _deduplicate(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: Dict[str, bool] = {}
    result = []
    for p in papers:
        key = re.sub(r"\W+", "", (p.get("title") or "").lower())[:60]
        if key and key not in seen:
            seen[key] = True
            result.append(p)
    return result


def _to_paper(raw: Dict[str, Any], keywords: List[str]) -> Optional[Paper]:
    title = (raw.get("title") or "").strip()
    abstract = (raw.get("abstract") or "").strip()
    if not title or not abstract or len(abstract) < 50:
        return None

    authors = [
        a.get("name", "Unknown") for a in (raw.get("authors") or [])
    ]
    if not authors:
        authors = ["Unknown"]

    # Extract lead author h-index from Semantic Scholar data
    author_h_index = 0
    author_list = raw.get("authors") or []
    if author_list and isinstance(author_list, list) and len(author_list) > 0:
        lead_author = author_list[0]
        if isinstance(lead_author, dict):
            author_h_index = lead_author.get("hIndex") or 0

    return Paper(
        paper_id=raw.get("paperId") or raw.get("url") or title[:20],
        title=title,
        abstract=abstract,
        authors=authors,
        year=raw.get("year") or _CURRENT_YEAR,
        citation_count=raw.get("citationCount") or 0,
        venue=raw.get("venue") or "",
        url=raw.get("url") or "",
        source=raw.get("_source", "semantic_scholar"),
        keywords_matched=_extract_keywords(title + " " + abstract, keywords),
        author_h_index=author_h_index,
    )


async def fetch_corpus(
    queries: List[str],
    keywords: List[str],
    target_size: int = 50,
    enable_citation_network: bool = True,
) -> List[Paper]:
    """
    Fetch and rank papers from multiple sources: SS, arXiv, Crossref, and citation networks.

    Multi-source strategy:
    1. Fetch from Semantic Scholar (primary, most comprehensive)
    2. Fetch from arXiv (preprints, recent work)
    3. Fetch from Crossref (additional publisher metadata)
    4. Expand via citation networks (papers citing/cited by top papers)
    5. Deduplicate and re-rank by reputation score
    6. Return top N papers
    """
    raw_all: List[Dict[str, Any]] = []

    # Phase 1: Parallel multi-source search
    logger.info(f"[Corpus] Fetching from 3 sources: Semantic Scholar + arXiv + Crossref")

    # Run SS searches sequentially (rate limit: 1 req/1.1s via semaphore)
    for q in queries[:3]:
        res = await _search_semantic_scholar(q, limit=25)
        raw_all.extend(res)

    # arXiv: use second query for better relevance
    arxiv_q = queries[1] if len(queries) > 1 else queries[0]
    arxiv_raw = await _search_arxiv(arxiv_q, max_results=20)
    raw_all.extend(arxiv_raw)

    # Crossref: use first query for additional papers
    crossref_q = queries[0]
    crossref_raw = await crossref_search.search_crossref(crossref_q, limit=20)
    raw_all.extend(crossref_raw)

    logger.info(f"[Corpus] Phase 1 (multi-source): {len(raw_all)} raw papers")

    # Deduplicate early to avoid redundant network expansion
    deduped = _deduplicate(raw_all)
    papers = [p for raw in deduped if (p := _to_paper(raw, keywords)) is not None]

    # Sort by relevance to get top candidates for network expansion
    papers.sort(key=lambda p: p.relevance_score, reverse=True)

    # Phase 2: Citation network expansion (optional, add 15-25 related papers)
    if enable_citation_network and len(papers) > 5:
        logger.info(f"[Corpus] Phase 2 (citation network): Expanding via top 5 papers...")
        try:
            # Only expand from papers that have Semantic Scholar paper IDs
            ss_papers = [p for p in papers[:5] if p.source == "semantic_scholar"]
            if ss_papers:
                # Convert to dict format for network expansion
                top_dicts = [
                    {
                        "paperId": p.paper_id,
                        "title": p.title,
                        "abstract": p.abstract,
                        "year": p.year,
                    }
                    for p in ss_papers
                ]

                network_papers = await citation_network.expand_via_citation_network(
                    top_dicts, max_papers_per_direction=3
                )

                # Convert back to Paper objects
                for raw in network_papers:
                    if (p := _to_paper(raw, keywords)) is not None:
                        papers.append(p)

                logger.info(
                    f"[Corpus] Phase 2: Added {len(network_papers)} papers via citation network"
                )
        except Exception as e:
            logger.warning(f"[Corpus] Citation network expansion failed: {e}")

    # Phase 3: Final deduplication and ranking
    # Re-deduplicate in case network expansion added duplicates
    final_deduped = _deduplicate([_paper_to_dict(p) for p in papers])
    final_papers = [
        p
        for raw in final_deduped
        if (p := _to_paper(raw, keywords)) is not None
    ]

    # Sort by reputation score (includes venue tiers, h-index, recency)
    final_papers.sort(key=lambda p: p.relevance_score, reverse=True)
    final_papers = final_papers[:target_size]

    logger.info(
        f"[Corpus] Final: {len(final_papers)} papers (from {len(raw_all)} initial + network expansion)"
    )
    return final_papers


def _paper_to_dict(paper: Paper) -> Dict[str, Any]:
    """Convert Paper object back to dict format for deduplication."""
    return {
        "paperId": paper.paper_id,
        "title": paper.title,
        "abstract": paper.abstract,
        "authors": [{"name": a} for a in paper.authors],
        "year": paper.year,
        "citationCount": paper.citation_count,
        "venue": paper.venue,
        "url": paper.url,
        "_source": paper.source,
    }
