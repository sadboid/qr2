"""Source quality scoring — rank papers by multiple quality dimensions."""

import logging
import math
from dataclasses import dataclass, field
from typing import List, Dict, Any

from . import venue_ranking

logger = logging.getLogger(__name__)

# Normalisation constants — max possible raw value from each venue_ranking helper
_MAX_VENUE_BOOST = 0.15   # Tier 1 boost
_MAX_H_BOOST = 0.10       # h-index >= 50
_MAX_RECENCY = 0.15       # 0-1 year old paper
_LOG_CITATION_BASE = math.log(1001)  # citations=1000 → score=1.0


@dataclass
class SourceQualityScore:
    """Multi-dimensional quality score for a paper. All sub-scores are 0.0-1.0."""
    title: str
    year: int
    venue: str

    # Individual quality dimensions — all normalised to 0.0-1.0
    venue_score: float          # Tier-based venue ranking (0.20 / 0.53 / 1.0)
    h_index_score: float        # Lead author credibility (0.0-1.0)
    citation_score: float       # Citation influence, log-scaled (0.0-1.0)
    recency_score: float        # How recent the paper is (0.0-1.0)
    peer_review_score: float    # Peer-reviewed vs preprint (0.5 or 1.0)

    # Composite scores (calculated in __post_init__)
    overall_quality: float = field(init=False, default=0.0)
    fit_for_lit_review: bool = field(init=False, default=False)
    quality_tier: str = field(init=False, default="poor")

    def __post_init__(self):
        """Calculate composite scores from normalised sub-scores."""
        # Weighted average: 40% venue + 25% citations + 20% h-index + 15% recency
        self.overall_quality = (
            0.40 * self.venue_score +
            0.25 * self.citation_score +
            0.20 * self.h_index_score +
            0.15 * self.recency_score
        )

        # Preprints get a 20% penalty (not peer-reviewed)
        if self.peer_review_score < 0.8:
            self.overall_quality *= 0.8

        self.overall_quality = round(min(1.0, self.overall_quality), 4)

        # Quality tiers
        if self.overall_quality >= 0.70:
            self.quality_tier = "excellent"
            self.fit_for_lit_review = True
        elif self.overall_quality >= 0.50:
            self.quality_tier = "good"
            self.fit_for_lit_review = True
        elif self.overall_quality >= 0.30:
            self.quality_tier = "acceptable"
            # Acceptable papers need at least a few citations to be used
            self.fit_for_lit_review = self.citation_score >= 0.10
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
        All returned sub-scores are normalised to 0.0-1.0.
        """

        # 1. Venue Score — normalise raw boost (0.03/0.08/0.15) to 0-1 scale
        tier, tier_boost = venue_ranking.get_venue_tier(venue)
        venue_score = tier_boost / _MAX_VENUE_BOOST  # 0.20, 0.53, or 1.0

        # 2. H-Index Score — normalise raw boost (0.0-0.10) to 0-1 scale
        h_boost = venue_ranking.author_h_index_boost(author_h_index)
        h_index_score = h_boost / _MAX_H_BOOST if _MAX_H_BOOST > 0 else 0.0

        # 3. Citation Score — log-scale normalised to 0-1
        # 1 cite→0.10, 10→0.35, 100→0.67, 1000+→1.0
        citation_score = min(1.0, math.log(max(1, citation_count) + 1) / _LOG_CITATION_BASE)

        # 4. Recency Score — normalise raw bonus (0.0-0.15) to 0-1 scale
        raw_recency = venue_ranking.recency_bonus(year, current_year)
        recency_score = raw_recency / _MAX_RECENCY if _MAX_RECENCY > 0 else 0.0

        # 5. Peer Review Score
        peer_review_score = 1.0 if self._is_peer_reviewed(venue) else 0.5

        return SourceQualityScore(
            title=title,
            year=year,
            venue=venue,
            venue_score=round(venue_score, 3),
            h_index_score=round(h_index_score, 3),
            citation_score=round(citation_score, 3),
            recency_score=round(recency_score, 3),
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
