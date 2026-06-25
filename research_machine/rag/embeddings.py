"""Embeddings for semantic search and similarity"""

from typing import List, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


class EmbeddingsProvider:
    """Base class for embedding providers"""

    def embed_text(self, text: str) -> List[float]:
        """Convert text to embedding vector"""
        raise NotImplementedError

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Convert multiple texts to embeddings"""
        raise NotImplementedError

    def similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors"""
        return self._cosine_similarity(np.array(vec1), np.array(vec2))

    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Cosine similarity between two vectors"""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(vec1, vec2) / (norm1 * norm2))


class MockEmbeddingsProvider(EmbeddingsProvider):
    """
    Mock embeddings for development/testing.
    Generates random embeddings for each unique text.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.cache = {}

    def embed_text(self, text: str) -> List[float]:
        """Generate mock embedding (same for same text)"""
        if text not in self.cache:
            # Use hash for deterministic but pseudo-random embeddings
            seed = hash(text) % (2**32)
            np.random.seed(seed)
            self.cache[text] = np.random.randn(self.dimension).tolist()

        return self.cache[text]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate mock embeddings for multiple texts"""
        return [self.embed_text(text) for text in texts]


class LocalEmbeddingsProvider(EmbeddingsProvider):
    """
    Uses sentence-transformers for local embeddings.
    This allows running embeddings offline without API calls.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize with a sentence-transformers model.

        Args:
            model_name: HuggingFace model name
        """
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.model_name = model_name
            logger.info(f"Loaded embeddings model: {model_name}")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            self.model = None

    def embed_text(self, text: str) -> List[float]:
        """Embed a single text"""
        if self.model is None:
            raise RuntimeError("sentence-transformers not installed")

        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts efficiently"""
        if self.model is None:
            raise RuntimeError("sentence-transformers not installed")

        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()


class PaperSummaryEmbedder:
    """
    Generates embeddings for papers using title + abstract.
    """

    def __init__(self, embeddings_provider: Optional[EmbeddingsProvider] = None):
        self.embeddings_provider = embeddings_provider or MockEmbeddingsProvider()

    def embed_paper(self, title: str, abstract: str) -> List[float]:
        """
        Generate embedding for a paper.

        Args:
            title: Paper title
            abstract: Paper abstract

        Returns:
            Embedding vector
        """
        # Combine title and abstract with appropriate weighting
        text = f"{title}. {abstract}"
        return self.embeddings_provider.embed_text(text)

    def embed_papers(self, papers: List[dict]) -> List[List[float]]:
        """
        Generate embeddings for multiple papers.

        Args:
            papers: List of paper dicts with 'title' and 'abstract' keys

        Returns:
            List of embedding vectors
        """
        texts = [
            f"{paper.get('title', '')}. {paper.get('abstract', '')}"
            for paper in papers
        ]
        return self.embeddings_provider.embed_texts(texts)


def compute_similarity_matrix(embeddings: List[List[float]]) -> np.ndarray:
    """
    Compute pairwise cosine similarity matrix.

    Args:
        embeddings: List of embedding vectors

    Returns:
        NxN similarity matrix where N is number of embeddings
    """
    embeddings_array = np.array(embeddings)
    norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)

    # Avoid division by zero
    norms = np.where(norms == 0, 1, norms)

    normalized = embeddings_array / norms
    similarity_matrix = np.dot(normalized, normalized.T)

    return similarity_matrix


def find_most_similar(
    query_embedding: List[float],
    candidate_embeddings: List[List[float]],
    top_k: int = 5
) -> List[tuple]:
    """
    Find most similar embeddings to a query.

    Args:
        query_embedding: Query vector
        candidate_embeddings: List of candidate vectors
        top_k: Return top K matches

    Returns:
        List of (index, similarity_score) tuples, sorted by similarity
    """
    query_array = np.array(query_embedding)
    candidates_array = np.array(candidate_embeddings)

    # Compute cosine similarities
    query_norm = np.linalg.norm(query_array)
    candidates_norms = np.linalg.norm(candidates_array, axis=1, keepdims=True)

    if query_norm == 0:
        return []

    normalized_query = query_array / query_norm
    normalized_candidates = candidates_array / np.where(candidates_norms == 0, 1, candidates_norms)

    similarities = np.dot(normalized_candidates, normalized_query)

    # Get top K
    top_indices = np.argsort(similarities)[::-1][:top_k]
    results = [(int(idx), float(similarities[idx])) for idx in top_indices]

    return results
