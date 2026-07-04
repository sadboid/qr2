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
from . import openalex_search

logger = logging.getLogger(__name__)

_SS_API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")

_CURRENT_YEAR = 2026
_SS_BASE = "https://api.semanticscholar.org/graph/v1"
_SS_FIELDS = "paperId,title,abstract,authors,year,citationCount,venue,externalIds,url,openAccessPdf"
_ARXIV_BASE = "http://export.arxiv.org/api/query"

_FULLTEXT_SEMAPHORE = asyncio.Semaphore(3)  # max 3 concurrent full-text fetches
_UNPAYWALL_EMAIL = os.environ.get("UNPAYWALL_EMAIL", "duy.bui@eiu.edu.vn")


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
    open_access_url: Optional[str] = None  # PDF URL from SS openAccessPdf field
    doi: Optional[str] = None  # DOI (for Unpaywall full-text fallback)
    full_text: Optional[str] = None  # Full text fetched from arXiv HTML or open-access PDF

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
        """APA 7 author-year key: 'Smith, 2023' / 'Smith & Jones, 2023' / 'Smith et al., 2023'."""
        def _last(a): return a.split(",")[0].strip().split()[-1]
        if not self.authors:
            return f"Unknown, {self.year}"
        if len(self.authors) == 1:
            return f"{_last(self.authors[0])}, {self.year}"
        if len(self.authors) == 2:
            return f"{_last(self.authors[0])} & {_last(self.authors[1])}, {self.year}"
        return f"{_last(self.authors[0])} et al., {self.year}"

    def apa_ref(self) -> str:
        """APA 7th edition reference entry (no number, alphabetical order by caller)."""
        def _fmt(a: str) -> str:
            a = a.strip()
            if "," in a:
                last, rest = a.split(",", 1)
                initials = " ".join(p[0].upper() + "." for p in rest.split() if p)
                return f"{last.strip()}, {initials}" if initials else last.strip()
            words = a.split()
            if not words:
                return "Unknown"
            last = words[-1]
            initials = " ".join(w[0].upper() + "." for w in words[:-1] if w)
            return f"{last}, {initials}" if initials else last

        authors = self.authors or ["Unknown Author"]
        if len(authors) == 1:
            author_str = _fmt(authors[0])
        elif len(authors) <= 20:
            fmts = [_fmt(a) for a in authors]
            author_str = ", ".join(fmts[:-1]) + ", & " + fmts[-1]
        else:
            fmts = [_fmt(a) for a in authors[:19]]
            author_str = ", ".join(fmts) + ", . . . " + _fmt(authors[-1])

        # Sentence case: capitalize first word + first word after colon
        t = self.title or "Untitled"
        title_sc = t[0].upper() + t[1:].lower() if len(t) > 1 else t.upper()
        title_sc = re.sub(r'(:\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), title_sc)

        venue_str = f" *{self.venue}*." if self.venue else "."
        url_str = f" {self.url}" if self.url and self.url.startswith("http") else ""
        return f"{author_str} ({self.year}). {title_sc}.{venue_str}{url_str}"


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
        # repr(e) so empty-message errors (ConnectError, timeouts) are diagnosable
        logger.warning(f"arXiv search failed: {e!r}")
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


def _deduplicate_papers(papers: List[Paper]) -> List[Paper]:
    """Dedup a list of Paper objects by normalized title, preserving ALL fields.

    Used for final dedup instead of round-tripping through dicts, which silently
    dropped open_access_url / author_h_index / doi / full_text and killed the
    full-text pipeline.
    """
    seen: Dict[str, bool] = {}
    result: List[Paper] = []
    for p in papers:
        key = re.sub(r"\W+", "", (p.title or "").lower())[:60]
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

    # Extract open-access PDF URL from Semantic Scholar / OpenAlex response
    oa_pdf = raw.get("openAccessPdf")
    open_access_url: Optional[str] = None
    if isinstance(oa_pdf, dict) and oa_pdf.get("url"):
        open_access_url = oa_pdf["url"]

    # Extract DOI (SS externalIds.DOI, explicit doi field, or a doi.org url)
    doi: Optional[str] = None
    ext = raw.get("externalIds")
    if isinstance(ext, dict) and ext.get("DOI"):
        doi = ext["DOI"]
    elif raw.get("doi"):
        doi = str(raw["doi"]).replace("https://doi.org/", "")
    else:
        url = raw.get("url") or ""
        m = re.search(r"doi\.org/(10\.\S+)", url)
        if m:
            doi = m.group(1)

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
        open_access_url=open_access_url,
        doi=doi,
    )


async def fetch_corpus(
    queries: List[str],
    keywords: List[str],
    target_size: int = 50,
    enable_citation_network: bool = True,
) -> List[Paper]:
    """
    Fetch and rank papers from multiple sources: SS, arXiv, Crossref, OpenAlex,
    and citation networks.

    Multi-source strategy:
    1. Fetch from Semantic Scholar (primary, most comprehensive)
    2. Fetch from arXiv (preprints, recent work)
    3. Fetch from Crossref (additional publisher metadata)
    4. Fetch from OpenAlex (250M+ works, strong for management/business journals)
    5. Expand via citation networks (papers citing/cited by top papers)
    6. Deduplicate and re-rank by reputation score
    7. Return top N papers
    """
    raw_all: List[Dict[str, Any]] = []

    # Phase 1: Parallel multi-source search
    logger.info("[Corpus] Fetching from 4 sources: Semantic Scholar + arXiv + Crossref + OpenAlex")

    # Run SS searches sequentially (rate limit: 1 req/1.1s via semaphore)
    for q in queries[:3]:
        res = await _search_semantic_scholar(q, limit=25)
        raw_all.extend(res)

    # arXiv + Crossref + OpenAlex run concurrently
    arxiv_q = queries[1] if len(queries) > 1 else queries[0]
    oa_queries = queries[:3]  # use all queries for OpenAlex (fast, high rate limit)

    arxiv_task = asyncio.create_task(_search_arxiv(arxiv_q, max_results=20))
    crossref_task = asyncio.create_task(
        crossref_search.search_crossref(queries[0], limit=20)
    )
    oa_task = asyncio.create_task(
        openalex_search.fetch_openalex_batch(oa_queries, limit_per_query=25)
    )

    arxiv_raw, crossref_raw, oa_raw = await asyncio.gather(
        arxiv_task, crossref_task, oa_task, return_exceptions=True
    )

    if isinstance(arxiv_raw, list):
        raw_all.extend(arxiv_raw)
    if isinstance(crossref_raw, list):
        raw_all.extend(crossref_raw)
    if isinstance(oa_raw, list):
        raw_all.extend(oa_raw)
        logger.info(f"[Corpus] OpenAlex contributed {len(oa_raw) if isinstance(oa_raw, list) else 0} raw papers")

    logger.info(f"[Corpus] Phase 1 (multi-source): {len(raw_all)} raw papers")
    n_raw = len(raw_all)

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

    # Phase 3: Final deduplication and ranking.
    # Dedup on Paper objects directly — NOT via _paper_to_dict round-trip, which
    # dropped open_access_url / author_h_index / doi / full_text and broke full-text.
    final_papers = _deduplicate_papers(papers)

    # Sort by reputation score (includes venue tiers, h-index, recency)
    final_papers.sort(key=lambda p: p.relevance_score, reverse=True)
    final_papers = final_papers[:target_size]

    logger.info(
        f"[Corpus] Final: {len(final_papers)} papers (from {n_raw} initial across 4 sources + network expansion)"
    )
    return final_papers, n_raw


def _extract_arxiv_id(paper: Paper) -> Optional[str]:
    """Extract arXiv ID from paper URL or paper_id field."""
    for text in [paper.url, paper.paper_id]:
        if not text:
            continue
        m = re.search(r'arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)', text, re.IGNORECASE)
        if m:
            return re.sub(r'v\d+$', '', m.group(1))  # strip version suffix
    return None


async def _fetch_arxiv_html(arxiv_id: str) -> Optional[str]:
    """Fetch full text for an arXiv paper from the HTML export endpoint."""
    url = f"https://arxiv.org/html/{arxiv_id}"
    try:
        async with _FULLTEXT_SEMAPHORE:
            await asyncio.sleep(3.0)  # arXiv rate limit
            async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
                r = await client.get(url, headers={"User-Agent": "research-machine/1.0 (academic use)"})
                if r.status_code != 200:
                    logger.debug(f"arXiv HTML {arxiv_id}: HTTP {r.status_code}")
                    return None
                html = r.text
    except Exception as e:
        logger.debug(f"arXiv HTML fetch failed for {arxiv_id}: {e}")
        return None

    from .fulltext_extractor import FullTextExtractor
    text = FullTextExtractor().clean_html(html)
    # Keep up to 12,000 chars (intro + methods + results covers most papers)
    return text[:12000] if len(text) > 200 else None


async def _fetch_pdf_text(pdf_url: str) -> Optional[str]:
    """Download a PDF and extract plain text (first 12KB of content)."""
    try:
        import pypdf  # type: ignore
        import io
    except Exception:
        logger.debug("pypdf unavailable — PDF full-text extraction skipped")
        return None

    try:
        async with _FULLTEXT_SEMAPHORE:
            await asyncio.sleep(1.0)
            async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
                r = await client.get(
                    pdf_url,
                    headers={"User-Agent": "research-machine/1.0 (academic use)"},
                )
                if r.status_code != 200:
                    return None
                content_length = int(r.headers.get("content-length", 0))
                if content_length > 10 * 1024 * 1024:  # skip > 10MB
                    return None
                pdf_bytes = r.content
    except Exception as e:
        logger.debug(f"PDF download failed ({pdf_url[:60]}): {e}")
        return None

    try:
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        pages_text = []
        for page in reader.pages[:15]:  # first 15 pages cover intro+methods+results
            try:
                pages_text.append(page.extract_text() or "")
            except Exception:
                pass
        text = "\n".join(pages_text)
        return text[:12000] if len(text) > 200 else None
    except Exception as e:
        logger.debug(f"PDF parse error: {e}")
        return None


async def _unpaywall_pdf_url(doi: str) -> Optional[str]:
    """Resolve an open-access PDF URL for a DOI via Unpaywall."""
    if not doi:
        return None
    doi = doi.replace("https://doi.org/", "").strip()
    try:
        async with _FULLTEXT_SEMAPHORE:
            async with httpx.AsyncClient(follow_redirects=True, timeout=20) as client:
                r = await client.get(
                    f"https://api.unpaywall.org/v2/{doi}",
                    params={"email": _UNPAYWALL_EMAIL},
                )
                if r.status_code != 200:
                    return None
                loc = r.json().get("best_oa_location") or {}
                return loc.get("url_for_pdf") or loc.get("url")
    except Exception as e:
        logger.debug(f"Unpaywall lookup failed for {doi}: {e!r}")
        return None


async def fetch_full_texts(papers: List[Paper], max_papers: int = 30) -> List[Paper]:
    """
    Fetch full text for papers that have open-access content.

    For each candidate, try in order until one yields text:
    1. arXiv HTML (arxiv.org/html/{id}) — free, clean, always available
    2. Known open-access PDF URL (from Semantic Scholar / OpenAlex)
    3. Unpaywall-resolved PDF URL (DOI → OA PDF)

    Candidates = every arXiv paper + top-ranked papers that have an OA url or a DOI,
    up to max_papers. Modifies papers in-place (sets paper.full_text).
    """
    arxiv_candidates = [p for p in papers if _extract_arxiv_id(p)]

    # Any paper with an OA url OR a DOI is a full-text candidate (Unpaywall covers DOIs).
    top_by_relevance = sorted(papers, key=lambda p: p.relevance_score, reverse=True)
    oa_budget = max(0, max_papers - len(arxiv_candidates))
    oa_candidates = [
        p for p in top_by_relevance
        if (p.open_access_url or p.doi) and p not in arxiv_candidates
    ][:oa_budget]

    logger.info(
        f"[Corpus] Full-text fetch: {len(arxiv_candidates)} arXiv + {len(oa_candidates)} OA/DOI candidates"
    )

    async def _fetch_one(paper: Paper) -> None:
        # 1. arXiv HTML
        arxiv_id = _extract_arxiv_id(paper)
        if arxiv_id:
            text = await _fetch_arxiv_html(arxiv_id)
            if text:
                paper.full_text = text
                return
        # 2. Known OA PDF
        if paper.open_access_url:
            text = await _fetch_pdf_text(paper.open_access_url)
            if text:
                paper.full_text = text
                return
        # 3. Unpaywall fallback via DOI
        if paper.doi:
            pdf_url = await _unpaywall_pdf_url(paper.doi)
            if pdf_url:
                text = await _fetch_pdf_text(pdf_url)
                if text:
                    paper.full_text = text

    tasks = [_fetch_one(p) for p in arxiv_candidates + oa_candidates]
    await asyncio.gather(*tasks, return_exceptions=True)

    full_text_count = sum(1 for p in papers if p.full_text)
    logger.info(f"[Corpus] Full text fetched for {full_text_count}/{len(papers)} papers")
    return papers


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
