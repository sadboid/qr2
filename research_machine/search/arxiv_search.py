"""arXiv API integration for literature search"""

import httpx
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
import feedparser

logger = logging.getLogger(__name__)


class ArxivClient:
    """Client for arXiv API"""

    BASE_URL = "http://export.arxiv.org/api/query"
    MAX_RESULTS_PER_QUERY = 300  # arXiv soft limit

    async def search_papers(
        self,
        query: str,
        max_results: int = 50,
        sort_by: str = "submittedDate",
        sort_order: str = "descending",
    ) -> List[Dict[str, Any]]:
        """
        Search arXiv for papers.

        Args:
            query: Search query (can use arXiv format: cat:cs.AI, all:deep learning)
            max_results: Maximum results to return
            sort_by: Sort field ('submittedDate' or 'relevance')
            sort_order: 'ascending' or 'descending'

        Returns:
            List of paper objects
        """
        async with httpx.AsyncClient() as client:
            params = {
                "search_query": query,
                "start": 0,
                "max_results": min(max_results, self.MAX_RESULTS_PER_QUERY),
                "sortBy": sort_by,
                "sortOrder": sort_order
            }

            try:
                response = await client.get(
                    self.BASE_URL,
                    params=params,
                    timeout=30.0
                )
                response.raise_for_status()

                feed = feedparser.parse(response.text)
                papers = self._parse_feed(feed)

                logger.info(f"Found {len(papers)} papers on arXiv for query: {query}")
                return papers

            except httpx.HTTPError as e:
                logger.error(f"arXiv API error: {e}")
                raise

    def _parse_feed(self, feed: Any) -> List[Dict[str, Any]]:
        """Parse arXiv Atom feed into paper objects"""
        papers = []

        for entry in feed.entries:
            # Extract authors
            authors = [{"name": author.name} for author in entry.get("authors", [])]

            # Extract arXiv ID from entry
            arxiv_id = entry.id.split("/abs/")[-1] if entry.id else None

            paper = {
                "arxivId": arxiv_id,
                "title": entry.title,
                "abstract": entry.summary,
                "authors": authors,
                "published": entry.published,
                "updated": entry.updated,
                "url": entry.id,
                "categories": entry.get("arxiv_primary_category", {}).get("term", ""),
            }

            papers.append(paper)

        return papers

    async def search_by_category(
        self,
        category: str,
        max_results: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Search papers by arXiv category.

        Args:
            category: arXiv category (e.g., 'cs.AI', 'econ.GN')
            max_results: Maximum results

        Returns:
            List of papers in category
        """
        query = f"cat:{category}"
        return await self.search_papers(
            query,
            max_results=max_results,
            sort_by="submittedDate"
        )

    async def search_by_author(
        self,
        author_name: str,
        max_results: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Search papers by author name.

        Args:
            author_name: Author name
            max_results: Maximum results

        Returns:
            List of papers by author
        """
        query = f'au:"{author_name}"'
        return await self.search_papers(query, max_results=max_results)


async def search_recent_papers(
    query: str,
    days_back: int = 30,
    max_results: int = 50
) -> List[Dict[str, Any]]:
    """
    Convenience function to search for recent papers.

    Args:
        query: Search query
        days_back: How many days back to search
        max_results: Max results

    Returns:
        List of recent papers
    """
    client = ArxivClient()

    papers = await client.search_papers(
        query,
        max_results=max_results,
        sort_by="submittedDate",
        sort_order="descending"
    )

    return papers


# Category mappings for common domains
ARXIV_CATEGORIES = {
    "cs.AI": "Artificial Intelligence",
    "cs.LG": "Machine Learning",
    "cs.NE": "Neural and Evolutionary Computing",
    "econ.GN": "Economics - General",
    "econ.TH": "Theoretical Economics",
    "q-fin.ST": "Quantitative Finance - Statistical Finance",
}
