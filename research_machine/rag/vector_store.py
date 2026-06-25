"""Qdrant vector database integration for semantic search"""

from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, PointIdsList
import logging

logger = logging.getLogger(__name__)


class QdrantVectorStore:
    """Vector store using Qdrant for similarity search.

    All methods are synchronous — QdrantClient is a sync library.
    Run in executor if calling from async context.
    """

    def __init__(self, url: str = "http://localhost:6333", api_key: Optional[str] = None):
        self.client = QdrantClient(url=url, api_key=api_key)
        self._known_collections: set = set()

    def create_collection(
        self,
        collection_name: str,
        vector_size: int = 384,
        distance_metric: str = "cosine",
    ) -> bool:
        """Create a new collection. Returns True if created, False if already exists."""
        distance_map = {
            "cosine": Distance.COSINE,
            "euclidean": Distance.EUCLID,
            "dot": Distance.DOT,
        }

        try:
            self.client.get_collection(collection_name)
            logger.info(f"Collection '{collection_name}' already exists")
            self._known_collections.add(collection_name)
            return False
        except Exception:
            pass  # Does not exist — create it

        try:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=distance_map.get(distance_metric, Distance.COSINE),
                ),
            )
            logger.info(f"Created collection '{collection_name}' (dim={vector_size})")
            self._known_collections.add(collection_name)
            return True
        except Exception as e:
            logger.error(f"Failed to create collection '{collection_name}': {e}")
            raise

    def add_vectors(
        self,
        collection_name: str,
        vectors: List[List[float]],
        metadata: List[Dict[str, Any]],
        ids: Optional[List[int]] = None,
    ) -> List[int]:
        """Add vectors with metadata. Returns list of integer point IDs."""
        if len(vectors) != len(metadata):
            raise ValueError("vectors and metadata must have same length")

        points = []
        assigned_ids: List[int] = []

        for i, (vector, meta) in enumerate(zip(vectors, metadata)):
            if ids:
                point_id = ids[i]
            else:
                # Deterministic ID from content hash
                raw = str(meta.get("title", "")) + str(meta.get("url", "")) + str(i)
                point_id = abs(hash(raw)) % (2**31)

            assigned_ids.append(point_id)
            points.append(PointStruct(id=point_id, vector=vector, payload=meta))

        try:
            self.client.upsert(collection_name=collection_name, points=points)
            logger.info(f"Upserted {len(points)} vectors into '{collection_name}'")
            return assigned_ids
        except Exception as e:
            logger.error(f"Failed to add vectors: {e}")
            raise

    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Return top-K similar vectors with metadata and similarity scores."""
        try:
            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True,
            )
            return [
                {"id": r.id, "score": r.score, "metadata": r.payload}
                for r in results
            ]
        except Exception as e:
            logger.error(f"Search failed in '{collection_name}': {e}")
            raise

    def delete_vectors(self, collection_name: str, ids: List[int]) -> bool:
        """Delete vectors by integer ID."""
        try:
            self.client.delete(
                collection_name=collection_name,
                points_selector=PointIdsList(points=ids),
            )
            logger.info(f"Deleted {len(ids)} vectors from '{collection_name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to delete vectors: {e}")
            raise

    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Return basic collection statistics."""
        try:
            info = self.client.get_collection(collection_name)
            return {
                "name": collection_name,
                "vectors_count": info.points_count,
                "dimension": info.config.params.vectors.size,
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            raise

    def list_collections(self) -> List[str]:
        """List all collection names."""
        try:
            return [c.name for c in self.client.get_collections().collections]
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            raise
