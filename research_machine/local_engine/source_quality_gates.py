"""Quality gates for source verification and corpus validation."""

import logging
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SourceQualityGateResult:
    """Result of source quality gate validation."""
    passed: bool
    score: float                    # 0.0-1.0
    total_papers: int
    verified_papers: int
    high_quality_papers: int        # Excellent + Good tier
    verification_rate: float        # % of papers verified
    quality_rate: float             # % of papers in high-quality tiers
    issues: List[str]
    recommendations: List[str]


class SourceQualityGates:
    """Automated quality gates for source corpus validation."""

    def __init__(self):
        # Minimum thresholds for paper acceptance
        self.min_verification_rate = 0.80       # 80% of papers must be verified
        self.min_high_quality_ratio = 0.40      # 40% should be excellent/good
        self.min_peer_reviewed_ratio = 0.60     # 60% should be peer-reviewed
        self.min_total_papers = 20              # At least 20 papers in corpus

    def validate_corpus(self,
                       papers: List[Dict[str, Any]],
                       verifications: List[Any],
                       quality_scores: List[Any]) -> SourceQualityGateResult:
        """
        Validate entire paper corpus against quality gates.

        Args:
            papers: Paper list
            verifications: SourceVerification objects (from SourceVerifier)
            quality_scores: SourceQualityScore objects (from SourceQualityScorer)

        Returns:
            SourceQualityGateResult with pass/fail decision
        """
        issues = []
        recommendations = []

        total = len(papers)
        if total < self.min_total_papers:
            issues.append(f"Corpus too small: {total} papers (min: {self.min_total_papers})")

        # Calculate verification rate
        verified_count = sum(1 for v in verifications if v.is_verified())
        verification_rate = verified_count / total if total > 0 else 0.0

        if verification_rate < self.min_verification_rate:
            issues.append(
                f"Low verification rate: {verification_rate:.0%} (min: {self.min_verification_rate:.0%})"
            )
            recommendations.append("Check metadata consistency for unverified papers")
            recommendations.append("Remove papers with mismatched title/year/authors")

        # Calculate high-quality ratio
        high_quality_count = sum(
            1 for q in quality_scores
            if q.quality_tier in ["excellent", "good"]
        )
        high_quality_ratio = high_quality_count / total if total > 0 else 0.0

        if high_quality_ratio < self.min_high_quality_ratio:
            issues.append(
                f"Low high-quality paper ratio: {high_quality_ratio:.0%} (target: {self.min_high_quality_ratio:.0%})"
            )
            recommendations.append("Add more papers from top-tier venues (ICML, NeurIPS, ACL, etc.)")
            recommendations.append("Prioritize recent papers with high citation counts")

        # Calculate peer-reviewed ratio
        peer_reviewed_count = sum(
            1 for q in quality_scores
            if q.peer_review_score >= 0.8
        )
        peer_reviewed_ratio = peer_reviewed_count / total if total > 0 else 0.0

        if peer_reviewed_ratio < self.min_peer_reviewed_ratio:
            issues.append(
                f"High preprint ratio: {1 - peer_reviewed_ratio:.0%} preprints (max: {1 - self.min_peer_reviewed_ratio:.0%})"
            )
            recommendations.append("Replace low-citation preprints with published papers")
            recommendations.append("Preprints acceptable if they have 5+ citations")

        # Check for citation count distribution
        citation_counts = [p.get("citationCount", 0) for p in papers]
        avg_citations = sum(citation_counts) / len(citation_counts) if citation_counts else 0
        if avg_citations < 5:
            recommendations.append(f"Average citations is low ({avg_citations:.1f}) — prioritize cited work")

        # Calculate composite score
        # 40% verification + 40% high-quality ratio + 20% peer-reviewed ratio
        composite_score = (
            0.40 * verification_rate +
            0.40 * high_quality_ratio +
            0.20 * peer_reviewed_ratio
        )

        # Pass if: no critical issues AND composite score >= 0.60
        passed = len(issues) == 0 and composite_score >= 0.60

        if not passed:
            logger.warning(f"[SourceQualityGates] FAILED: {len(issues)} issues, score {composite_score:.2f}")
        else:
            logger.info(f"[SourceQualityGates] PASSED: verification {verification_rate:.0%}, quality {high_quality_ratio:.0%}, peer-reviewed {peer_reviewed_ratio:.0%}")

        return SourceQualityGateResult(
            passed=passed,
            score=composite_score,
            total_papers=total,
            verified_papers=verified_count,
            high_quality_papers=high_quality_count,
            verification_rate=verification_rate,
            quality_rate=high_quality_ratio,
            issues=issues,
            recommendations=recommendations,
        )

    def get_recommendations(self, gate_result: SourceQualityGateResult) -> str:
        """Generate human-readable recommendations from gate result."""
        if gate_result.passed:
            return "✓ Corpus passes all quality gates. Ready for literature review generation."

        lines = ["✗ Corpus does not meet quality standards. Recommendations:"]
        for i, rec in enumerate(gate_result.recommendations, 1):
            lines.append(f"  {i}. {rec}")

        return "\n".join(lines)
