"""Novelty gate — checks if paper is sufficiently different from existing literature."""

import logging
from typing import Optional
from pydantic import BaseModel

from research_machine.config import settings
from research_machine.rag.embeddings import PaperSummaryEmbedder
from research_machine.rag.vector_store import QdrantVectorStore

logger = logging.getLogger(__name__)


class QualityGateResult(BaseModel):
    """Standard result format for quality gates."""

    passed: bool
    score: float  # 0-1 scale (meaning varies per gate)
    feedback: str  # Human-readable feedback


class NoveltyGate:
    """Novelty gate: checks semantic similarity to indexed papers."""

    def __init__(
        self, vector_store: QdrantVectorStore, embedder: PaperSummaryEmbedder
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.threshold = settings.novelty_threshold

    def evaluate(self, paper_title: str, paper_abstract: str) -> QualityGateResult:
        """Evaluate novelty of a paper.

        Args:
            paper_title: Paper title
            paper_abstract: Paper abstract

        Returns:
            QualityGateResult with novelty assessment
        """
        try:
            # Embed the paper
            query_vector = self.embedder.embed_paper(paper_title, paper_abstract)

            # Search for nearest neighbors in Qdrant
            results = self.vector_store.search(
                collection_name="papers",
                query_vector=query_vector,
                limit=1,
                score_threshold=0.5,  # Return if similarity >= 0.5
            )

            if not results:
                # No similar papers found
                return QualityGateResult(
                    passed=True,
                    score=0.0,
                    feedback="No similar papers found in corpus — highly novel.",
                )

            nearest = results[0]
            similarity = nearest["score"]
            nearest_title = nearest.get("metadata", {}).get("title", "Unknown")

            # Check against threshold
            is_novel = similarity < self.threshold
            feedback = (
                f"Novel (similarity {similarity:.3f} < {self.threshold})"
                if is_novel
                else f"Too similar to '{nearest_title}' (similarity {similarity:.3f} >= {self.threshold})"
            )

            logger.info(f"[NoveltyGate] score={similarity:.3f}, novel={is_novel}")

            return QualityGateResult(
                passed=is_novel,
                score=similarity,
                feedback=feedback,
            )

        except Exception as e:
            logger.error(f"[NoveltyGate] error: {e}")
            # Return graceful failure
            return QualityGateResult(
                passed=False,
                score=1.0,
                feedback=f"Error checking novelty: {e}",
            )
