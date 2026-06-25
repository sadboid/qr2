"""Indexing Agent — embeds papers into Qdrant and checks novelty of draft abstracts.

Novelty check: query Qdrant with the draft abstract embedding; if the nearest
neighbour scores >= threshold the draft is too similar to existing work.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from research_machine.agents.literature_agent import PaperSummary
from research_machine.rag.embeddings import MockEmbeddingsProvider, PaperSummaryEmbedder
from research_machine.rag.vector_store import QdrantVectorStore
from research_machine.config import settings

logger = logging.getLogger(__name__)

COLLECTION_NAME = "papers"


@dataclass
class NoveltyResult:
    score: float                           # Similarity to nearest indexed paper (0–1)
    is_novel: bool                         # True if score < threshold
    nearest_paper_title: Optional[str]
    feedback: str                          # Actionable feedback if not novel


class IndexingAgent:
    """Embeds and indexes papers; checks novelty of generated abstracts."""

    def __init__(
        self,
        vector_store: Optional[QdrantVectorStore] = None,
        embedder: Optional[PaperSummaryEmbedder] = None,
        novelty_threshold: float = None,
    ):
        self.vector_store = vector_store or QdrantVectorStore(
            url=settings.qdrant_url, api_key=settings.qdrant_api_key
        )
        self.embedder = embedder or PaperSummaryEmbedder(MockEmbeddingsProvider())
        self.novelty_threshold = novelty_threshold or settings.novelty_threshold

        # Ensure the collection exists
        self.vector_store.create_collection(COLLECTION_NAME, vector_size=384)

    def index_papers(self, papers: List[PaperSummary]) -> int:
        """
        Embed and upsert papers into Qdrant.

        Returns:
            Number of papers successfully indexed.
        """
        if not papers:
            return 0

        vectors = self.embedder.embed_papers(
            [{"title": p.title, "abstract": p.abstract} for p in papers]
        )

        metadata = [
            {
                "title": p.title,
                "authors": p.authors[:5],   # Trim to keep payload small
                "year": p.year,
                "citation_count": p.citation_count,
                "source": p.source,
                "url": p.url,
            }
            for p in papers
        ]

        self.vector_store.add_vectors(COLLECTION_NAME, vectors, metadata)
        logger.info(f"Indexed {len(papers)} papers into Qdrant collection '{COLLECTION_NAME}'")
        return len(papers)

    def check_novelty(self, abstract: str, title: str = "") -> NoveltyResult:
        """
        Check whether a draft abstract is sufficiently novel.

        Args:
            abstract: The generated abstract text
            title: Optional title (combined with abstract for embedding)

        Returns:
            NoveltyResult with score and actionable feedback
        """
        query_vec = self.embedder.embed_paper(title, abstract)

        try:
            results = self.vector_store.search(
                COLLECTION_NAME, query_vec, limit=3
            )
        except Exception as e:
            logger.warning(f"Novelty check search failed: {e}. Assuming novel.")
            return NoveltyResult(
                score=0.0,
                is_novel=True,
                nearest_paper_title=None,
                feedback="Could not check novelty (vector store unavailable). Proceeding.",
            )

        if not results:
            return NoveltyResult(
                score=0.0,
                is_novel=True,
                nearest_paper_title=None,
                feedback="No indexed papers to compare against — likely novel.",
            )

        top = results[0]
        score = top["score"]
        nearest_title = top["metadata"].get("title", "unknown")

        is_novel = score < self.novelty_threshold

        if is_novel:
            feedback = (
                f"Paper is sufficiently novel (similarity {score:.2f} < "
                f"threshold {self.novelty_threshold:.2f})."
            )
        else:
            feedback = (
                f"Too similar to existing work (similarity {score:.2f} >= "
                f"threshold {self.novelty_threshold:.2f}). "
                f"Nearest match: '{nearest_title}'. "
                "Suggestion: narrow the scope, add a cross-domain angle, or focus on "
                "a specific industry/geography not covered by the existing paper."
            )

        return NoveltyResult(
            score=score,
            is_novel=is_novel,
            nearest_paper_title=nearest_title,
            feedback=feedback,
        )
