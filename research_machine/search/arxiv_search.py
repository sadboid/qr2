"""arXiv API integration for literature search.

arXiv recommends max 1 request per 3 seconds — enforced via a module-level semaphore.
"""

import httpx
import asyncio
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Module-level semaphore: only 1 concurrent arXiv request allowed
_arxiv_semaphore = asyncio.Semaphore(1)
_ARXIV_RATE_DELAY = 3.0  # seconds between requests


class ArxivClient:
    """Client for arXiv API with built-in rate limiting."""

    BASE_URL = "http://export.arxiv.org/api/query"
    MAX_RESULTS_PER_QUERY = 300

    async def search_papers(
        self,
        query: str,
        max_results: int = 50,
        sort_by: str = "submittedDate",
        sort_order: str = "descending",
    ) -> List[Dict[str, Any]]:
        """Search arXiv for papers. Respects the 1-req/3s rate limit."""
        async with _arxiv_semaphore:
            await asyncio.sleep(_ARXIV_RATE_DELAY)
            params = {
                "search_query": query,
                "start": 0,
                "max_results": min(max_results, self.MAX_RESULTS_PER_QUERY),
                "sortBy": sort_by,
                "sortOrder": sort_order,
            }

            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        self.BASE_URL, params=params, timeout=30.0
                    )
                    response.raise_for_status()

                papers = self._parse_feed(response.text)
                logger.info(f"arXiv: found {len(papers)} papers for '{query}'")
                return papers

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    logger.warning("arXiv rate limited — backing off 10s")
                    await asyncio.sleep(10.0)
                    raise
                logger.error(f"arXiv HTTP error: {e}")
                raise
            except httpx.HTTPError as e:
                logger.error(f"arXiv request error: {e}")
                raise

    def _parse_feed(self, xml_text: str) -> List[Dict[str, Any]]:
        NS = {
            "atom": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom",
        }
        root = ET.fromstring(xml_text)
        papers = []
        for entry in root.findall("atom:entry", NS):
            def txt(tag):
                el = entry.find(tag, NS)
                return el.text.strip() if el is not None and el.text else ""

            url = txt("atom:id")
            arxiv_id = url.split("/abs/")[-1] if url else None
            published = txt("atom:published")
            updated = txt("atom:updated")
            authors = [
                {"name": a.find("atom:name", NS).text.strip()}
                for a in entry.findall("atom:author", NS)
                if a.find("atom:name", NS) is not None
            ]
            primary_cat = entry.find("arxiv:primary_category", NS)
            category = primary_cat.get("term", "") if primary_cat is not None else ""
            papers.append({
                "arxivId": arxiv_id,
                "title": txt("atom:title"),
                "abstract": txt("atom:summary"),
                "authors": authors,
                "published": published,
                "updated": updated,
                "url": url,
                "categories": category,
                "year": int(published[:4]) if published else None,
                "_source": "arxiv",
            })
        return papers

    async def search_by_category(
        self, category: str, max_results: int = 50
    ) -> List[Dict[str, Any]]:
        return await self.search_papers(
            f"cat:{category}", max_results=max_results, sort_by="submittedDate"
        )

    async def search_by_author(
        self, author_name: str, max_results: int = 50
    ) -> List[Dict[str, Any]]:
        return await self.search_papers(
            f'au:"{author_name}"', max_results=max_results
        )


# Category mappings for common domains
ARXIV_CATEGORIES = {
    "cs.AI": "Artificial Intelligence",
    "cs.LG": "Machine Learning",
    "cs.NE": "Neural and Evolutionary Computing",
    "econ.GN": "Economics - General",
    "econ.TH": "Theoretical Economics",
    "q-fin.ST": "Quantitative Finance - Statistical Finance",
}
