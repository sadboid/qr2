"""Source quality scoring — rank papers by multiple quality dimensions."""

import logging
from dataclasses import dataclass
from typing import List, Dict, Any

from . import venue_ranking

logger = logging.getLogger(__name__)


@dataclass
class SourceQualityScore:
    """Multi-dimensional quality score for a paper."""
    title: str
    year: int
    venue: str

    # Individual quality dimensions (0.0-1.0)
    venue_score: float          # Tier-based venue ranking
    h_index_score: float        # Lead author credibility
    citation_score: float       # Citation influence (log-scaled)
    recency_score: float        # How recent the paper is
    peer_review_score: float    # Peer-reviewed vs preprint

    # Composite scores
    overall_quality: float      # Weighted average of all dimensions
    fit_for_lit_review: bool   # Suitable for literature review
    quality_tier: str          # "excellent", "good", "acceptable", "poor"

    def __post_init__(self):
        """Calculate composite scores."""
        # Overall quality: 40% venue + 25% citations + 20% h-index + 15% recency
        self.overall_quality = (
            0.40 * self.venue_score +
            0.25 * self.citation_score +
            0.20 * self.h_index_score +
            0.15 * self.recency_score
        )

        # Peer review is mandatory threshold (not averaged)
        if self.peer_review_score < 0.5:  # Preprint
            self.overall_quality *= 0.8  # Penalty for preprints

        # Determine quality tier
        if self.overall_quality >= 0.85:
            self.quality_tier = "excellent"
            self.fit_for_lit_review = True
        elif self.overall_quality >= 0.70:
            self.quality_tier = "good"
            self.fit_for_lit_review = True
        elif self.overall_quality >= 0.50:
            self.quality_tier = "acceptable"
            self.fit_for_lit_review = self.citation_score >= 0.4  # Must have some citations
        else:
            self.quality_tier = "poor"
            self.fit_for_lit_review = False


class SourceQualityScorer:
    """Score sources across multiple quality dimensions."""

    def score_paper(self,
                   title: str,
                   year: int,
                   venue: str,
                   citation_count: int,
                   author_h_index: int = 0,
                   current_year: int = 2026) -> SourceQualityScore:
        """
        Score a single paper across quality dimensions.

        Returns: SourceQualityScore with overall_quality and fit_for_lit_review
        """

        # 1. Venue Score (0.0-1.0)
        tier, tier_boost = venue_ranking.get_venue_tier(venue)
        venue_score = tier_boost  # 0.03, 0.08, or 0.15

        # 2. H-Index Score (0.0-1.0)
        h_boost = venue_ranking.author_h_index_boost(author_h_index)
        h_index_score = h_boost  # 0.0 to 0.10

        # 3. Citation Score (0.0-1.0)
        import math
        citation_score = min(0.20, (math.log(max(1, citation_count) + 1) / 10.0) * 0.20)

        # 4. Recency Score (0.0-1.0)
        recency_score = venue_ranking.recency_bonus(year, current_year)

        # 5. Peer Review Score (0.0-1.0)
        peer_review_score = 1.0 if self._is_peer_reviewed(venue) else 0.5

        return SourceQualityScore(
            title=title,
            year=year,
            venue=venue,
            venue_score=venue_score,
            h_index_score=h_index_score,
            citation_score=citation_score,
            recency_score=recency_score,
            peer_review_score=peer_review_score,
        )

    def score_corpus(self, papers: List[Dict[str, Any]]) -> tuple[List[SourceQualityScore], Dict[str, Any]]:
        """
        Score all papers in corpus and return statistics.

        Returns: (scores, stats_dict)
        """
        scores = []

        for paper in papers:
            score = self.score_paper(
                title=paper.get("title", ""),
                year=paper.get("year", 2026),
                venue=paper.get("venue", ""),
                citation_count=paper.get("citationCount", 0),
                author_h_index=paper.get("author_h_index", 0),
            )
            scores.append(score)

        # Calculate statistics
        overall_scores = [s.overall_quality for s in scores]
        fit_count = sum(1 for s in scores if s.fit_for_lit_review)

        stats = {
            "total_papers": len(scores),
            "avg_quality": sum(overall_scores) / len(overall_scores) if overall_scores else 0.0,
            "min_quality": min(overall_scores) if overall_scores else 0.0,
            "max_quality": max(overall_scores) if overall_scores else 0.0,
            "fit_for_lit_review": fit_count,
            "fit_ratio": fit_count / len(scores) if scores else 0.0,
            "by_tier": {
                "excellent": sum(1 for s in scores if s.quality_tier == "excellent"),
                "good": sum(1 for s in scores if s.quality_tier == "good"),
                "acceptable": sum(1 for s in scores if s.quality_tier == "acceptable"),
                "poor": sum(1 for s in scores if s.quality_tier == "poor"),
            }
        }

        logger.info(f"[SourceQuality] Corpus stats: {fit_count}/{len(scores)} fit for lit review (avg quality: {stats['avg_quality']:.2f})")

        return scores, stats

    def _is_peer_reviewed(self, venue: str) -> bool:
        """Check if venue indicates peer review."""
        if not venue:
            return False

        venue_lower = venue.lower()

        if any(x in venue_lower for x in ["arxiv", "preprint", "technical report", "working paper"]):
            return False

        if any(x in venue_lower for x in ["conference", "journal", "proceedings", "transactions", "review"]):
            return True

        return len(venue) > 5
