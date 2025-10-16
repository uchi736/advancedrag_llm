"""
Vector Store Adapter - PGVector only
"""
from __future__ import annotations

from typing import List, Any, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import PGVector, DistanceStrategy


class PGVectorAdapter:
    """Adapter for PGVector (PostgreSQL with vector extension)"""

    def __init__(
        self,
        connection_string: str,
        collection_name: str,
        embedding_function: Embeddings,
        use_jsonb: bool = True,
        distance_strategy: str = "cosine"
    ):
        # Map string to DistanceStrategy enum
        strategy_map = {
            "cosine": DistanceStrategy.COSINE,
            "euclidean": DistanceStrategy.EUCLIDEAN,
            "max_inner_product": DistanceStrategy.MAX_INNER_PRODUCT
        }

        self.vector_store = PGVector(
            connection_string=connection_string,
            collection_name=collection_name,
            embedding_function=embedding_function,
            use_jsonb=use_jsonb,
            distance_strategy=strategy_map.get(distance_strategy.lower(), DistanceStrategy.COSINE)
        )
        self.collection_name = collection_name

    def add_documents(self, documents: List[Document], ids: Optional[List[str]] = None) -> List[str]:
        return self.vector_store.add_documents(documents, ids=ids)

    def similarity_search_with_score(self, query: str, k: int = 4) -> List[Tuple[Document, float]]:
        return self.vector_store.similarity_search_with_score(query, k=k)

    def delete(self, ids: List[str]) -> None:
        self.vector_store.delete(ids=ids)

    def get_by_ids(self, ids: List[str]) -> List[Document]:
        # PGVector doesn't have direct get_by_ids
        # This method isn't used in the current codebase
        return []


class VectorStoreFactory:
    """Factory class to create PGVector adapter"""

    @staticmethod
    def create_vector_store(
        config: Any,
        embedding_function: Embeddings,
        connection_string: str,
        **kwargs
    ) -> PGVectorAdapter:
        """
        Create a PGVector adapter

        Args:
            config: Configuration object
            embedding_function: Embedding function to use
            connection_string: Database connection string
            **kwargs: Additional arguments

        Returns:
            PGVectorAdapter instance
        """
        return PGVectorAdapter(
            connection_string=connection_string,
            collection_name=config.collection_name,
            embedding_function=embedding_function,
            use_jsonb=kwargs.get("use_jsonb", True),
            distance_strategy=kwargs.get("distance_strategy", getattr(config, "distance_strategy", "cosine"))
        )
