"""Citation linker — maps inline citations to database Citation IDs."""

import re
import logging
from typing import List, Tuple
from sqlalchemy.orm import Session

from research_machine.agents.literature_agent import PaperSummary
from research_machine.db.models import Citation, Paper

logger = logging.getLogger(__name__)


class CitationLinker:
    """Maps [Title, Year] citations to database Citation IDs."""

    def __init__(self, db: Session):
        self.db = db

    def link_citations(
        self, paper_content: str, papers: List[PaperSummary]
    ) -> Tuple[str, List[int]]:
        """Map inline citations to DB IDs.

        Args:
            paper_content: Paper text with [Title, Year] citations
            papers: Available papers to link to

        Returns:
            (updated_content with [1], [2], [3] notation, citation_ids list)
        """
        # Extract all [Title, Year] patterns
        pattern = r"\[([^\]]+),\s*(\d{4})\]"
        matches = list(re.finditer(pattern, paper_content))

        citation_mapping = {}  # (title, year) -> citation_id
        citation_ids = []

        updated_content = paper_content

        # Process matches in reverse order to maintain indices
        for match in reversed(matches):
            title, year = match.group(1), int(match.group(2))

            # Skip if already processed
            if (title, year) in citation_mapping:
                citation_id = citation_mapping[(title, year)]
            else:
                # Find matching paper
                matching_paper = self._find_paper(title, year, papers)

                if matching_paper:
                    # Get or create Citation record
                    citation_id = self._get_or_create_citation(matching_paper)
                    citation_mapping[(title, year)] = citation_id
                    citation_ids.insert(0, citation_id)
                else:
                    # Paper not found, keep original citation
                    logger.warning(f"Paper not found: [{title}, {year}]")
                    continue

            # Replace [Title, Year] with [citation_id]
            start, end = match.span()
            updated_content = (
                updated_content[:start]
                + f"[{citation_id}]"
                + updated_content[end:]
            )

        logger.info(f"[CitationLinker] linked {len(citation_ids)} citations")

        return updated_content, citation_ids

    def _find_paper(self, title: str, year: int, papers: List[PaperSummary]) -> PaperSummary:
        """Find paper by title and year (fuzzy match)."""
        for paper in papers:
            if paper.year == year and self._title_match(title, paper.title):
                return paper
        return None

    def _title_match(self, title1: str, title2: str, threshold: float = 0.7) -> bool:
        """Simple title matching (normalize and check substring overlap)."""
        t1 = title1.lower().replace(" ", "")
        t2 = title2.lower().replace(" ", "")

        # Simple substring check
        if t1 in t2 or t2 in t1:
            return True

        # Levenshtein-style: count matching words
        words1 = set(t1.split())
        words2 = set(t2.split())
        if words1 and words2:
            overlap = len(words1 & words2) / max(len(words1), len(words2))
            return overlap >= threshold

        return False

    def _get_or_create_citation(self, paper: PaperSummary) -> int:
        """Get or create Citation DB record and return ID.

        For now, return a hash of paper title + year as ID.
        (Full DB integration in Phase 2)
        """
        # Generate deterministic ID from paper
        citation_key = f"{paper.title}_{paper.year}".encode()
        citation_id = hash(citation_key) % 1000000

        # TODO: Store in DB Citation table
        # For now, just return ID
        return abs(citation_id)
