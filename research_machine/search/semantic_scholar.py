"""Semantic Scholar API integration for literature search"""

import httpx
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class SemanticScholarClient:
    """Client for Semantic Scholar API v1"""

    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    BATCH_SIZE = 100

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.headers = {}
        if api_key:
            self.headers["x-api-key"] = api_key

    async def search_papers(
        self,
        query: str,
        limit: int = 50,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        venue: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for papers using Semantic Scholar API.

        Args:
            query: Search query string
            limit: Maximum number of results (default 50)
            year_min: Minimum publication year filter
            year_max: Maximum publication year filter
            venue: Venue filter (conference/journal name)

        Returns:
            List of paper objects with metadata
        """
        async with httpx.AsyncClient() as client:
            params = {
                "query": query,
                "limit": min(limit, 100),  # API limit is 100
                "fields": ",".join([
                    "paperId", "title", "abstract", "authors",
                    "year", "citationCount", "influentialCitationCount",
                    "venue", "url", "externalIds", "publicationDate"
                ])
            }

            if year_min:
                params["minYear"] = year_min
            if year_max:
                params["maxYear"] = year_max
            if venue:
                params["venue"] = venue

            try:
                response = await client.get(
                    f"{self.BASE_URL}/paper/search",
                    params=params,
                    headers=self.headers,
                    timeout=30.0
                )
                response.raise_for_status()
                data = response.json()

                papers = data.get("data", [])
                logger.info(f"Found {len(papers)} papers for query: {query}")
                return papers

            except httpx.HTTPError as e:
                logger.error(f"Semantic Scholar API error: {e}")
                raise

    async def get_paper(self, paper_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific paper.

        Args:
            paper_id: Semantic Scholar paper ID

        Returns:
            Paper object with full metadata
        """
        async with httpx.AsyncClient() as client:
            fields = ",".join([
                "paperId", "title", "abstract", "authors", "year",
                "citationCount", "influentialCitationCount", "venue",
                "url", "externalIds", "references", "publicationDate"
            ])

            try:
                response = await client.get(
                    f"{self.BASE_URL}/paper/{paper_id}",
                    params={"fields": fields},
                    headers=self.headers,
                    timeout=30.0
                )
                response.raise_for_status()
                return response.json()

            except httpx.HTTPError as e:
                logger.error(f"Failed to get paper {paper_id}: {e}")
                raise

    async def get_author_papers(
        self,
        author_id: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get papers published by an author.

        Args:
            author_id: Semantic Scholar author ID
            limit: Maximum number of papers to return

        Returns:
            List of papers by the author
        """
        async with httpx.AsyncClient() as client:
            fields = ",".join([
                "paperId", "title", "year", "citationCount",
                "venue", "url", "externalIds"
            ])

            try:
                response = await client.get(
                    f"{self.BASE_URL}/author/{author_id}/papers",
                    params={"limit": limit, "fields": fields},
                    headers=self.headers,
                    timeout=30.0
                )
                response.raise_for_status()
                return response.json().get("data", [])

            except httpx.HTTPError as e:
                logger.error(f"Failed to get papers for author {author_id}: {e}")
                raise

    def extract_citations(self, paper: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract citation information from a paper.

        Returns:
            List of citation dictionaries with title, year, authors
        """
        citations = []

        if "references" in paper:
            for ref in paper["references"]:
                citations.append({
                    "title": ref.get("title"),
                    "year": ref.get("year"),
                    "authors": ref.get("authors", []),
                    "paperId": ref.get("paperId"),
                    "citationCount": ref.get("citationCount", 0),
                })

        return citations

    def calculate_h_index(self, papers: List[Dict[str, Any]]) -> int:
        """
        Calculate H-index from a list of papers.
        H-index = largest h such that the author has h papers with >= h citations
        """
        if not papers:
            return 0

        citation_counts = sorted(
            [p.get("citationCount", 0) for p in papers],
            reverse=True
        )

        h_index = 0
        for i, count in enumerate(citation_counts):
            if count >= (i + 1):
                h_index = i + 1
            else:
                break

        return h_index


async def search_recent_papers(
    query: str,
    years_back: int = 3,
    limit: int = 50
) -> List[Dict[str, Any]]:
    """
    Convenience function to search for recent papers in a domain.

    Args:
        query: Search query
        years_back: How many years back to search (default 3 years)
        limit: Max results

    Returns:
        List of papers from recent years
    """
    client = SemanticScholarClient()
    current_year = datetime.now().year
    year_min = current_year - years_back

    return await client.search_papers(
        query,
        limit=limit,
        year_min=year_min
    )
