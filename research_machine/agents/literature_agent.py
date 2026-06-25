"""Literature Agent — orchestrates multi-source paper search and summarisation.

Search sources: Semantic Scholar (primary) + arXiv (secondary).
Deduplication is done by title normalisation before embedding.
Partial source failures are tolerated — whichever sources respond are used.
Uses Claude Haiku 4.5 for per-paper relevance summaries (~$0.15/run).
"""

import asyncio
import logging
import re
from typing import List, Optional
from pydantic import BaseModel
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from research_machine.config import settings
from research_machine.search.semantic_scholar import SemanticScholarClient
from research_machine.search.arxiv_search import ArxivClient

logger = logging.getLogger(__name__)


class PaperSummary(BaseModel):
    title: str
    abstract: str
    authors: List[str]
    year: Optional[int]
    citation_count: int
    source: str          # 'semantic_scholar' | 'arxiv'
    url: str
    relevance_summary: str  # 1-sentence Haiku summary


def _normalise_title(title: str) -> str:
    """Lowercase, strip punctuation for deduplication matching."""
    return re.sub(r"[^a-z0-9\s]", "", title.lower()).strip()


def _extract_authors(raw_authors: list) -> List[str]:
    names = []
    for a in raw_authors:
        if isinstance(a, dict):
            names.append(a.get("name", ""))
        elif isinstance(a, str):
            names.append(a)
    return [n for n in names if n]


def _extract_year(paper: dict) -> Optional[int]:
    if paper.get("year"):
        try:
            return int(paper["year"])
        except (ValueError, TypeError):
            pass
    if paper.get("published"):
        try:
            return int(str(paper["published"])[:4])
        except (ValueError, TypeError):
            pass
    return None


class LiteratureOutput(BaseModel):
    papers: List[PaperSummary]
    total_found: int
    search_queries_used: List[str]


class LiteratureAgent:
    """Orchestrates literature search across Semantic Scholar and arXiv."""

    def __init__(
        self,
        ss_client: Optional[SemanticScholarClient] = None,
        arxiv_client: Optional[ArxivClient] = None,
        model: str = "claude-haiku-4-5-20251001",
        papers_per_query: int = 20,
    ):
        self.ss = ss_client or SemanticScholarClient()
        self.arxiv = arxiv_client or ArxivClient()
        self.llm = ChatAnthropic(
            model=model,
            api_key=settings.anthropic_api_key,
            max_tokens=512,
        )
        self.papers_per_query = papers_per_query

    async def search(
        self,
        queries: List[str],
        domain: str,
    ) -> LiteratureOutput:
        """
        Run all queries against both sources, deduplicate, and summarise.

        Args:
            queries: List of search queries from HypothesisAgent
            domain: 'startup' | 'enterprise' — used to tune relevance summaries

        Returns:
            LiteratureOutput with deduplicated, summarised papers
        """
        all_raw: List[dict] = []

        # Fan out: query both sources in parallel for each query
        tasks = []
        for q in queries:
            tasks.append(self._safe_ss_search(q))
            tasks.append(self._safe_arxiv_search(q))

        results = await asyncio.gather(*tasks)
        for batch in results:
            all_raw.extend(batch)

        logger.info(f"Raw papers fetched: {len(all_raw)} across {len(queries)} queries")

        # Deduplicate by normalised title
        seen_titles: set = set()
        unique: List[dict] = []
        for p in all_raw:
            norm = _normalise_title(p.get("title", ""))
            if norm and norm not in seen_titles:
                seen_titles.add(norm)
                unique.append(p)

        logger.info(f"After deduplication: {len(unique)} unique papers")

        # Sort by citation count descending (quality signal)
        unique.sort(key=lambda p: p.get("citationCount", 0) or 0, reverse=True)

        # Keep top 50 for summarisation (cost control)
        top = unique[:50]

        # Batch-summarise with Haiku
        summaries = await self._batch_summarise(top, domain)

        papers = []
        for raw, summary in zip(top, summaries):
            authors = _extract_authors(raw.get("authors", []))
            papers.append(PaperSummary(
                title=raw.get("title", ""),
                abstract=raw.get("abstract", "") or "",
                authors=authors,
                year=_extract_year(raw),
                citation_count=raw.get("citationCount", 0) or 0,
                source=raw.get("_source", "unknown"),
                url=raw.get("url", "") or raw.get("externalIds", {}).get("DOI", ""),
                relevance_summary=summary,
            ))

        return LiteratureOutput(
            papers=papers,
            total_found=len(all_raw),
            search_queries_used=queries,
        )

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    async def _safe_ss_search(self, query: str) -> List[dict]:
        """Semantic Scholar search with error tolerance."""
        try:
            results = await self.ss.search_papers(query, limit=self.papers_per_query)
            for r in results:
                r["_source"] = "semantic_scholar"
            return results
        except Exception as e:
            logger.warning(f"Semantic Scholar search failed for '{query}': {e}")
            return []

    async def _safe_arxiv_search(self, query: str) -> List[dict]:
        """arXiv search with error tolerance."""
        try:
            results = await self.arxiv.search_papers(query, max_results=self.papers_per_query)
            for r in results:
                r["_source"] = "arxiv"
                r["citationCount"] = 0  # arXiv doesn't provide citation counts
            return results
        except Exception as e:
            logger.warning(f"arXiv search failed for '{query}': {e}")
            return []

    async def _batch_summarise(self, papers: List[dict], domain: str) -> List[str]:
        """Generate 1-sentence relevance summaries for each paper using Haiku."""
        if not papers:
            return []

        # Build a single batched prompt to minimise API calls
        items = []
        for i, p in enumerate(papers):
            title = p.get("title", "")[:120]
            abstract = (p.get("abstract", "") or "")[:300]
            items.append(f"[{i}] Title: {title}\nAbstract: {abstract[:300]}")

        batch_text = "\n\n".join(items)

        system = (
            f"You are a research assistant summarising papers for a {domain} + AI researcher. "
            "For each paper, write exactly ONE sentence explaining its relevance to business AI research. "
            "Output a JSON array of strings, one per paper, in the same order. No other text."
        )
        user = f"Papers:\n\n{batch_text}\n\nOutput JSON array of {len(papers)} relevance sentences:"

        try:
            response = await self.llm.ainvoke([
                SystemMessage(content=system),
                HumanMessage(content=user),
            ])
            raw = response.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            import json
            summaries = json.loads(raw)
            if len(summaries) == len(papers):
                return summaries
            # Pad or trim if length mismatch
            return (summaries + [""] * len(papers))[:len(papers)]
        except Exception as e:
            logger.warning(f"Batch summarisation failed: {e}. Using empty summaries.")
            return [""] * len(papers)
