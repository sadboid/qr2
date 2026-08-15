"""Quality gates and validation modules for research papers."""

from research_machine.quality.novelty_check import NoveltyGate, QualityGateResult
from research_machine.quality.citation_validator import CitationValidator
from research_machine.quality.peer_review import PeerReviewGate, QuickReviewResult, DetailedReviewResult

__all__ = [
    "NoveltyGate",
    "CitationValidator",
    "PeerReviewGate",
    "QualityGateResult",
    "QuickReviewResult",
    "DetailedReviewResult",
]
