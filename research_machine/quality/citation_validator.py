"""Citation validation gate — checks citation count, h-index, and recency."""

import logging
from datetime import datetime, timedelta
from typing import List, Optional

from sqlalchemy.orm import Session

from research_machine.config import settings
from research_machine.quality.novelty_check import QualityGateResult

logger = logging.getLogger(__name__)


class CitationValidator:
    """Validates citation quality and quantity."""

    def __init__(self, db: Session):
        self.db = db
        self.min_citations = settings.min_citations
        self.min_h_index = settings.min_h_index
        self.min_recency_ratio = settings.min_recency_ratio

    def evaluate(self, citations: List[dict]) -> QualityGateResult:
        """Evaluate citation quality.

        Args:
            citations: List of citation dicts with 'year' and 'h_index' fields

        Returns:
            QualityGateResult
        """
        try:
            if not citations:
                return QualityGateResult(
                    passed=False,
                    score=0.0,
                    feedback=f"No citations found (required: {self.min_citations}+)",
                )

            # Check count
            citation_count = len(citations)
            count_ok = citation_count >= self.min_citations

            # Check h-index
            h_indices = [c.get("h_index", 0) for c in citations if c.get("h_index")]
            avg_h_index = sum(h_indices) / len(h_indices) if h_indices else 0
            h_index_ok = avg_h_index >= self.min_h_index

            # Check recency (% from last 3 years)
            current_year = datetime.now().year
            recent_count = sum(
                1 for c in citations if c.get("year", 0) >= current_year - 3
            )
            recency_ratio = recent_count / citation_count if citation_count > 0 else 0
            recency_ok = recency_ratio >= self.min_recency_ratio

            # Overall pass
            passed = count_ok and h_index_ok and recency_ok

            # Build feedback
            feedback_parts = []
            if not count_ok:
                feedback_parts.append(
                    f"Citation count ({citation_count}) < {self.min_citations}"
                )
            if not h_index_ok:
                feedback_parts.append(
                    f"Avg h-index ({avg_h_index:.1f}) < {self.min_h_index}"
                )
            if not recency_ok:
                feedback_parts.append(
                    f"Recency ratio ({recency_ratio:.1%}) < {self.min_recency_ratio:.0%}"
                )

            feedback = (
                "All citation criteria met"
                if passed
                else "Citation issues: " + "; ".join(feedback_parts)
            )

            logger.info(
                f"[CitationValidator] count={citation_count}, h-index={avg_h_index:.1f}, recency={recency_ratio:.1%}, passed={passed}"
            )

            return QualityGateResult(
                passed=passed,
                score=(
                    (count_ok + h_index_ok + recency_ok) / 3.0
                ),  # Fraction of criteria met
                feedback=feedback,
            )

        except Exception as e:
            logger.error(f"[CitationValidator] error: {e}")
            return QualityGateResult(
                passed=False,
                score=0.0,
                feedback=f"Error validating citations: {e}",
            )
