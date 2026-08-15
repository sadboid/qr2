"""Metrics collection and reporting for generated papers."""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class PaperMetrics:
    """Quality metrics for a generated paper."""
    paper_id: str
    title: str
    novelty_score: float  # 0-1 (< 0.70 passes)
    citation_count: int
    avg_h_index: float
    recency_ratio: float  # % from last 3 years (>= 0.30 passes)
    peer_review_score: float  # 0-10 (>= 7.0 passes)
    generation_cost_usd: float
    generation_time_seconds: float
    status: str  # "accepted", "revision_requested", "rejected"
    generated_at: datetime = field(default_factory=datetime.utcnow)

    def passes_novelty_gate(self, threshold: float = 0.70) -> bool:
        """Check if paper passes novelty gate."""
        return self.novelty_score < threshold

    def passes_citation_gate(
        self,
        min_citations: int = 15,
        min_h_index: float = 5.0,
        min_recency: float = 0.30,
    ) -> bool:
        """Check if paper passes citation gate."""
        return (
            self.citation_count >= min_citations
            and self.avg_h_index >= min_h_index
            and self.recency_ratio >= min_recency
        )

    def passes_peer_review_gate(self, threshold: float = 7.0) -> bool:
        """Check if paper passes peer review gate."""
        return self.peer_review_score >= threshold

    def all_gates_pass(
        self,
        novelty_threshold: float = 0.70,
        min_citations: int = 15,
        min_h_index: float = 5.0,
        min_recency: float = 0.30,
        peer_review_threshold: float = 7.0,
    ) -> bool:
        """Check if all quality gates pass."""
        return (
            self.passes_novelty_gate(novelty_threshold)
            and self.passes_citation_gate(min_citations, min_h_index, min_recency)
            and self.passes_peer_review_gate(peer_review_threshold)
        )


class MetricsCollector:
    """Collects and reports metrics across all generated papers."""

    def __init__(self):
        self.papers: List[PaperMetrics] = []

    def add_paper(self, metrics: PaperMetrics) -> None:
        """Add metrics for a generated paper."""
        self.papers.append(metrics)
        logger.info(f"Added metrics for paper: {metrics.title} (status: {metrics.status})")

    def summary_report(self) -> Dict[str, Any]:
        """Generate summary report across all papers."""
        if not self.papers:
            return {
                "total_papers": 0,
                "status_breakdown": {},
                "average_metrics": {},
                "total_cost": 0.0,
            }

        # Count by status
        status_counts = {}
        for paper in self.papers:
            status_counts[paper.status] = status_counts.get(paper.status, 0) + 1

        # Calculate averages
        avg_novelty = sum(p.novelty_score for p in self.papers) / len(self.papers)
        avg_citations = sum(p.citation_count for p in self.papers) / len(self.papers)
        avg_h_index = sum(p.avg_h_index for p in self.papers) / len(self.papers)
        avg_recency = sum(p.recency_ratio for p in self.papers) / len(self.papers)
        avg_peer_review = sum(p.peer_review_score for p in self.papers) / len(self.papers)
        avg_time = sum(p.generation_time_seconds for p in self.papers) / len(self.papers)
        total_cost = sum(p.generation_cost_usd for p in self.papers)

        return {
            "total_papers": len(self.papers),
            "status_breakdown": status_counts,
            "average_metrics": {
                "novelty_score": round(avg_novelty, 3),
                "citation_count": round(avg_citations, 1),
                "h_index": round(avg_h_index, 2),
                "recency_ratio": round(avg_recency, 3),
                "peer_review_score": round(avg_peer_review, 2),
                "generation_time_seconds": round(avg_time, 1),
            },
            "total_cost_usd": round(total_cost, 2),
            "cost_per_paper": round(total_cost / len(self.papers), 2) if self.papers else 0.0,
        }

    def get_papers_by_status(self, status: str) -> List[PaperMetrics]:
        """Get all papers with a specific status."""
        return [p for p in self.papers if p.status == status]

    def get_high_quality_papers(
        self,
        novelty_threshold: float = 0.70,
        min_citations: int = 15,
        min_h_index: float = 5.0,
        min_recency: float = 0.30,
        peer_review_threshold: float = 7.0,
    ) -> List[PaperMetrics]:
        """Get papers that pass all quality gates."""
        return [
            p for p in self.papers
            if p.all_gates_pass(
                novelty_threshold,
                min_citations,
                min_h_index,
                min_recency,
                peer_review_threshold,
            )
        ]

    def detailed_report(self) -> Dict[str, Any]:
        """Generate detailed report with all papers."""
        return {
            "summary": self.summary_report(),
            "papers": [
                {
                    "id": p.paper_id,
                    "title": p.title,
                    "metrics": {
                        "novelty_score": p.novelty_score,
                        "citation_count": p.citation_count,
                        "avg_h_index": p.avg_h_index,
                        "recency_ratio": p.recency_ratio,
                        "peer_review_score": p.peer_review_score,
                    },
                    "cost_usd": p.generation_cost_usd,
                    "generation_time_seconds": p.generation_time_seconds,
                    "status": p.status,
                    "gates": {
                        "novelty": p.passes_novelty_gate(),
                        "citations": p.passes_citation_gate(),
                        "peer_review": p.passes_peer_review_gate(),
                    },
                }
                for p in self.papers
            ],
        }
