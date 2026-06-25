"""Qdrant vector database integration for semantic search"""

from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import logging

logger = logging.getLogger(__name__)


class QdrantVectorStore:
    """Vector store using Qdrant for similarity search"""

    def __init__(self, url: str = "http://localhost:6333", api_key: Optional[str] = None):
        """
        Initialize Qdrant client.

        Args:
            url: Qdrant server URL
            api_key: Optional API key
        """
        self.client = QdrantClient(url=url, api_key=api_key)
        self.collections = {}

    def create_collection(
        self,
        collection_name: str,
        vector_size: int = 384,
        distance_metric: str = "cosine"
    ) -> bool:
        """
        Create a new collection for storing vectors.

        Args:
            collection_name: Name of collection
            vector_size: Dimension of vectors
            distance_metric: 'cosine', 'euclidean', or 'dot'

        Returns:
            True if created, False if already exists
        """
        try:
            # Check if collection exists
            try:
                self.client.get_collection(collection_name)
                logger.info(f"Collection '{collection_name}' already exists")
                return False
            except:
                pass  # Collection doesn't exist, create it

            distance_map = {
                "cosine": Distance.COSINE,
                "euclidean": Distance.EUCLID,
                "dot": Distance.DOT,
            }

            self.client.recreate_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=distance_map.get(distance_metric, Distance.COSINE)
                )
            )

            logger.info(f"Created collection '{collection_name}'")
            self.collections[collection_name] = True
            return True

        except Exception as e:
            logger.error(f"Failed to create collection '{collection_name}': {e}")
            raise

    async def add_vectors(
        self,
        collection_name: str,
        vectors: List[List[float]],
        metadata: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add vectors to collection.

        Args:
            collection_name: Target collection
            vectors: List of vector embeddings
            metadata: List of metadata dicts (one per vector)
            ids: Optional list of IDs; auto-generated if not provided

        Returns:
            List of point IDs
        """
        if len(vectors) != len(metadata):
            raise ValueError("vectors and metadata must have same length")

        points = []
        generated_ids = []

        for i, (vector, meta) in enumerate(zip(vectors, metadata)):
            point_id = ids[i] if ids else str(hash((tuple(vector), frozenset(meta.items()))) % (2**31))
            generated_ids.append(point_id)

            points.append(
                PointStruct(
                    id=int(point_id) if isinstance(point_id, str) else point_id,
                    vector=vector,
                    payload=meta
                )
            )

        try:
            self.client.upsert(
                collection_name=collection_name,
                points=points
            )
            logger.info(f"Added {len(points)} vectors to '{collection_name}'")
            return generated_ids

        except Exception as e:
            logger.error(f"Failed to add vectors: {e}")
            raise

    async def search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.

        Args:
            collection_name: Collection to search
            query_vector: Query embedding
            limit: Number of results to return
            score_threshold: Minimum similarity score

        Returns:
            List of results with metadata and scores
        """
        try:
            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True,
            )

            formatted_results = [
                {
                    "id": result.id,
                    "score": result.score,
                    "metadata": result.payload
                }
                for result in results
            ]

            return formatted_results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise

    async def search_by_id(
        self,
        collection_name: str,
        point_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find similar vectors to a specific point.

        Args:
            collection_name: Collection to search
            point_id: ID of reference point
            limit: Number of results

        Returns:
            List of similar results
        """
        try:
            results = self.client.recommend(
                collection_name=collection_name,
                positive=[int(point_id)],
                limit=limit,
                with_payload=True,
            )

            formatted_results = [
                {
                    "id": result.id,
                    "score": result.score,
                    "metadata": result.payload
                }
                for result in results
            ]

            return formatted_results

        except Exception as e:
            logger.error(f"Recommendation search failed: {e}")
            raise

    async def delete_vectors(
        self,
        collection_name: str,
        ids: List[str]
    ) -> bool:
        """Delete vectors by ID"""
        try:
            self.client.delete(
                collection_name=collection_name,
                points_selector=[int(id_) for id_ in ids]
            )
            logger.info(f"Deleted {len(ids)} vectors from '{collection_name}'")
            return True

        except Exception as e:
            logger.error(f"Failed to delete vectors: {e}")
            raise

    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            collection_info = self.client.get_collection(collection_name)
            return {
                "name": collection_name,
                "vectors_count": collection_info.points_count,
                "dimension": collection_info.config.params.vectors.size,
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            raise

    def list_collections(self) -> List[str]:
        """List all collections"""
        try:
            collections = self.client.get_collections()
            return [col.name for col in collections.collections]
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            raise
