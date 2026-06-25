"""Citation verification against Semantic Scholar API."""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher

from research_machine.db.models import Citation
from research_machine.search.semantic_scholar import SemanticScholarClient
from research_machine.config import settings

logger = logging.getLogger(__name__)


@dataclass
class VerificationReport:
    """Report from citation verification."""
    verified_count: int
    total_citations: int
    failures: List[Dict[str, Any]] = field(default_factory=list)
    verified_citations: List[Citation] = field(default_factory=list)
    confidence: float = 0.0

    def __post_init__(self):
        """Calculate confidence score."""
        if self.total_citations > 0:
            self.confidence = self.verified_count / self.total_citations
        else:
            self.confidence = 1.0


class CitationVerifier:
    """Verifies citations against Semantic Scholar API with rate limiting and caching."""

    def __init__(self, api_key: Optional[str] = None):
        self.client = SemanticScholarClient(api_key=api_key or settings.semantic_scholar_api_key)
        self.rate_limiter = asyncio.Semaphore(1)  # 1 request at a time
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.request_delay = 1.0  # 1 second between requests

    def _make_cache_key(self, title: str, year: Optional[int]) -> str:
        """Generate cache key from title and year."""
        return f"{title.lower().strip()}_{year}"

    async def verify_citations(
        self,
        citations: List[Citation],
    ) -> VerificationReport:
        """
        Verify citations against Semantic Scholar API.

        Args:
            citations: List of Citation objects to verify

        Returns:
            VerificationReport with verification results
        """
        if not citations:
            return VerificationReport(
                verified_count=0,
                total_citations=0,
                confidence=1.0
            )

        verified_citations = []
        failures = []

        for citation in citations:
            try:
                result = await self._verify_single_citation(citation)

                if result["verified"]:
                    # Update citation with verified metadata
                    citation.h_index = result.get("h_index", citation.h_index)
                    citation.external_id = result.get("external_id", citation.external_id)
                    verified_citations.append(citation)
                else:
                    failures.append({
                        "title": citation.title,
                        "year": citation.year,
                        "reason": result.get("reason", "Paper not found in Semantic Scholar"),
                    })

                # Rate limiting: wait before next request
                await asyncio.sleep(self.request_delay)

            except Exception as e:
                logger.error(f"Error verifying citation '{citation.title}': {e}")
                failures.append({
                    "title": citation.title,
                    "year": citation.year,
                    "reason": f"API error: {str(e)}",
                })

        return VerificationReport(
            verified_count=len(verified_citations),
            total_citations=len(citations),
            verified_citations=verified_citations,
            failures=failures,
        )

    async def _verify_single_citation(self, citation: Citation) -> Dict[str, Any]:
        """
        Verify a single citation.

        Args:
            citation: Citation object to verify

        Returns:
            Dict with verified (bool), h_index, external_id, and reason
        """
        cache_key = self._make_cache_key(citation.title or "", citation.year)

        # Check cache first
        if cache_key in self.cache:
            logger.debug(f"Using cached result for citation: {citation.title}")
            return self.cache[cache_key]

        try:
            async with self.rate_limiter:
                # Try to find paper by title and year
                search_results = await self.client.search_papers(
                    query=citation.title or "",
                    limit=3,  # Get top 3 results
                )

                if not search_results:
                    result = {
                        "verified": False,
                        "reason": "No papers found with this title",
                    }
                    self.cache[cache_key] = result
                    return result

                # Find best matching paper
                best_match = self._find_best_match(
                    citation.title or "",
                    citation.year,
                    search_results
                )

                if best_match:
                    # Extract author h-index
                    h_index = await self._get_author_h_index(best_match)

                    result = {
                        "verified": True,
                        "h_index": h_index,
                        "external_id": best_match.get("paperId"),
                        "reason": "Paper verified in Semantic Scholar",
                    }
                else:
                    result = {
                        "verified": False,
                        "reason": "No matching paper found (title/year mismatch)",
                    }

                self.cache[cache_key] = result
                return result

        except Exception as e:
            logger.error(f"Error in citation verification: {e}")
            return {
                "verified": False,
                "reason": f"API error: {str(e)}",
            }

    def _find_best_match(
        self,
        query_title: str,
        query_year: Optional[int],
        search_results: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Find best matching paper from search results.

        Uses title similarity and year matching.

        Args:
            query_title: Citation title to match
            query_year: Citation year to match
            search_results: List of papers from Semantic Scholar

        Returns:
            Best matching paper dict or None
        """
        best_match = None
        best_score = 0.0

        for paper in search_results:
            paper_title = paper.get("title", "")
            paper_year = paper.get("year")

            # Title similarity (0-1)
            title_similarity = SequenceMatcher(
                None,
                query_title.lower(),
                paper_title.lower()
            ).ratio()

            # Year matching bonus
            year_match = 1.0 if (query_year and paper_year and abs(query_year - paper_year) <= 1) else 0.5

            # Combined score: 70% title similarity + 30% year match
            combined_score = (title_similarity * 0.7) + (year_match * 0.3)

            if combined_score > best_score and title_similarity >= 0.6:  # Minimum 60% title match
                best_score = combined_score
                best_match = paper

        return best_match

    async def _get_author_h_index(self, paper: Dict[str, Any]) -> int:
        """
        Get H-index of lead author (first author).

        Args:
            paper: Paper dict from Semantic Scholar

        Returns:
            H-index of lead author, or 0 if not found
        """
        authors = paper.get("authors", [])

        if not authors:
            return 0

        lead_author = authors[0]
        author_id = lead_author.get("authorId")

        if not author_id:
            return 0

        try:
            # Fetch all papers by lead author to calculate h-index
            author_papers = await self.client.get_author_papers(author_id, limit=100)

            if not author_papers:
                return 0

            h_index = self.client.calculate_h_index(author_papers)
            logger.debug(f"Author {lead_author.get('name', 'unknown')} h-index: {h_index}")
            return h_index

        except Exception as e:
            logger.warning(f"Could not fetch author h-index: {e}")
            return 0
