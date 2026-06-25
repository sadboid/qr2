"""Literature corpus: search, deduplicate, rank real papers."""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime

import httpx

logger = logging.getLogger(__name__)

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

    @property
    def is_recent(self) -> bool:
        return self.year >= _CURRENT_YEAR - 3

    @property
    def relevance_score(self) -> float:
        recency = 1.0 if self.is_recent else 0.5
        cite_score = min(self.citation_count / 100, 1.0)
        kw_score = min(len(self.keywords_matched) / 3, 1.0)
        return 0.4 * kw_score + 0.3 * recency + 0.3 * cite_score

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
_SS_DELAY = 3.5  # seconds between SS requests (avoid rate limit)


async def _search_semantic_scholar(
    query: str, limit: int = 30, retries: int = 4
) -> List[Dict[str, Any]]:
    async with _ss_semaphore:
        await asyncio.sleep(_SS_DELAY)
        delay = 3.0
        for attempt in range(retries):
            try:
                async with httpx.AsyncClient() as client:
                    r = await client.get(
                        f"{_SS_BASE}/paper/search",
                        params={"query": query, "limit": limit, "fields": _SS_FIELDS},
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
    )


async def fetch_corpus(
    queries: List[str],
    keywords: List[str],
    target_size: int = 40,
) -> List[Paper]:
    """Fetch and rank papers from SS + arXiv for the given queries."""
    raw_all: List[Dict[str, Any]] = []

    # Run SS searches sequentially (rate limit: 1 req/1.5s via semaphore)
    for q in queries[:3]:
        res = await _search_semantic_scholar(q, limit=25)
        raw_all.extend(res)

    # arXiv: use second query (keyword-focused) for better relevance
    arxiv_q = queries[1] if len(queries) > 1 else queries[0]
    arxiv_raw = await _search_arxiv(arxiv_q, max_results=20)
    raw_all.extend(arxiv_raw)

    deduped = _deduplicate(raw_all)
    papers = [p for raw in deduped if (p := _to_paper(raw, keywords)) is not None]

    # Sort by relevance
    papers.sort(key=lambda p: p.relevance_score, reverse=True)
    papers = papers[:target_size]

    logger.info(f"Corpus: {len(papers)} papers (from {len(raw_all)} raw)")
    return papers
