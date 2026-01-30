import json
import logging
import re
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from typing import List, Dict, Any, Optional, Tuple

from langchain_community.vectorstores import PGVector
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.runnables import RunnableConfig

from .config import Config
from .text_processor import JapaneseTextProcessor

logger = logging.getLogger(__name__)


def similarity_search_without_jargon(
    engine: Engine,
    embeddings,
    query: str,
    collection_name: str,
    k: int = 5,
    with_score: bool = False,
) -> list:
    """SQLレベルでjargon_termを除外したベクトル検索。

    Args:
        with_score: Trueなら List[Tuple[Document, float]] を返す
    """
    query_embedding = embeddings.embed_query(query)
    embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT e.document, e.cmetadata,
                   e.embedding <=> :embedding ::vector AS distance
            FROM langchain_pg_embedding e
            JOIN langchain_pg_collection c ON e.collection_id = c.uuid
            WHERE c.name = :collection
              AND (e.cmetadata->>'type' IS NULL OR e.cmetadata->>'type' != 'jargon_term')
            ORDER BY distance
            LIMIT :k
        """), {"embedding": embedding_str, "collection": collection_name, "k": k})
        results = rows.fetchall()

    docs = []
    for r in results:
        meta = r.cmetadata if isinstance(r.cmetadata, dict) else json.loads(r.cmetadata or "{}")
        doc = Document(page_content=r.document, metadata=meta)
        if with_score:
            docs.append((doc, float(r.distance)))
        else:
            docs.append(doc)
    return docs


class JapaneseHybridRetriever(BaseRetriever):
    """
    A retriever that combines vector search and BM25 keyword search
    with Reciprocal Rank Fusion (RRF) for hybrid search, optimized for
    Japanese language.
    """
    vector_store: PGVector
    connection_string: str
    config_params: Config
    text_processor: JapaneseTextProcessor = None
    search_type: str = "hybrid"
    engine: Optional[Engine] = None
    bm25_retriever: Optional[Any] = None
    bm25_collection: Optional[str] = None
    bm25_doc_count: int = 0

    def __init__(self, engine: Optional[Engine] = None, **kwargs):
        super().__init__(**kwargs)
        self.text_processor = JapaneseTextProcessor()
        # Reuse shared engine when provided to avoid recreating connections.
        object.__setattr__(self, "engine", engine or create_engine(self.connection_string))

    def _vector_search(self, q: str, config: Optional[RunnableConfig] = None) -> List[Tuple[Document, float]]:
        if not self.vector_store:
            return []
        try:
            search_k = self.config_params.vector_search_k
            embeddings = self.vector_store.embedding_function
            return similarity_search_without_jargon(
                engine=self.engine,
                embeddings=embeddings,
                query=q,
                collection_name=self.config_params.collection_name,
                k=search_k,
                with_score=True,
            )
        except Exception as exc:
            print(f"[HybridRetriever] vector search error: {exc}")
            return []

    def _build_bm25_retriever(self) -> Optional[BM25Retriever]:
        """BM25Retrieverを構築（キャッシュ付き）"""
        if not self.engine:
            logger.warning("[HybridRetriever] No engine available for BM25 search")
            return None

        collection = self.config_params.collection_name

        try:
            with self.engine.connect() as conn:
                # チャンク数を確認（キャッシュ判定用）
                result = conn.execute(text(
                    "SELECT COUNT(*) FROM document_chunks WHERE collection_name = :coll"
                ), {"coll": collection})
                doc_count = result.scalar() or 0

            # キャッシュが有効ならそのまま返す
            if (self.bm25_retriever is not None
                    and self.bm25_collection == collection
                    and self.bm25_doc_count == doc_count):
                return self.bm25_retriever

            if doc_count == 0:
                logger.warning(f"[BM25] No document chunks for collection: {collection}")
                return None

            # document_chunksから全件取得
            with self.engine.connect() as conn:
                result = conn.execute(text(
                    "SELECT chunk_id, content, metadata FROM document_chunks WHERE collection_name = :coll"
                ), {"coll": collection})
                rows = result.fetchall()

            docs = []
            for row in rows:
                md = row.metadata if isinstance(row.metadata, dict) else json.loads(row.metadata or "{}")
                md["chunk_id"] = row.chunk_id
                docs.append(Document(page_content=row.content, metadata=md))

            # SudachiPyトークン化をpreprocess_funcに渡す
            tp = self.text_processor

            def tokenize_for_bm25(text: str) -> List[str]:
                normalized = tp.normalize_text(text)
                if tp.is_japanese(normalized):
                    tokens = tp.tokenize(normalized)
                    return [t for t in tokens if len(t) >= 2]
                return normalized.split()

            retriever = BM25Retriever.from_documents(
                docs,
                preprocess_func=tokenize_for_bm25,
                k=self.config_params.keyword_search_k
            )

            # キャッシュ更新
            self.bm25_retriever = retriever
            self.bm25_collection = collection
            self.bm25_doc_count = doc_count
            logger.info(f"[BM25] Built index: {doc_count} chunks for collection '{collection}'")

            return retriever

        except Exception as exc:
            logger.error(f"[BM25] Failed to build retriever: {exc}")
            return None

    def _keyword_search(self, q: str, config: Optional[RunnableConfig] = None) -> List[Tuple[Document, float]]:
        """BM25によるキーワード検索"""
        retriever = self._build_bm25_retriever()
        if not retriever:
            return []

        try:
            docs = retriever.invoke(q)
            # BM25Retrieverはスコアを返さないので、順位ベースのスコアを付与
            return [(doc, 1.0 / (i + 1)) for i, doc in enumerate(docs)]
        except Exception as exc:
            logger.error(f"[BM25] keyword search error: {exc}")
            return []

    @staticmethod
    def _rrf_hybrid(rank: int, k: int = 60) -> float:
        return 1.0 / (k + rank)

    def _reciprocal_rank_fusion_hybrid(self, vres: List[Tuple[Document, float]], kres: List[Tuple[Document, float]]) -> List[Document]:
        score_map: Dict[str, Dict[str, Any]] = {}
        _id = lambda d: d.metadata.get("chunk_id", d.page_content[:100])
        
        for r, (d, _) in enumerate(vres, 1):
            doc_id_val = _id(d)
            score_map.setdefault(doc_id_val, {"doc": d, "s": 0.0})["s"] += self._rrf_hybrid(r, self.config_params.rrf_k_for_fusion)
            
        for r, (d, _) in enumerate(kres, 1):
            doc_id_val = _id(d)
            score_map.setdefault(doc_id_val, {"doc": d, "s": 0.0})["s"] += self._rrf_hybrid(r, self.config_params.rrf_k_for_fusion)
            
        ranked = sorted(score_map.values(), key=lambda x: x["s"], reverse=True)
        return [x["doc"] for x in ranked[:self.config_params.final_k]]

    def _get_relevant_documents(self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None, **kwargs: Any) -> List[Document]:
        config = kwargs.get("config")

        if self.search_type == 'vector':
            vres = self._vector_search(query, config=config)
            retrieved_docs = [doc for doc, score in vres]
        elif self.search_type == 'keyword':
            kres = self._keyword_search(query, config=config)
            retrieved_docs = [doc for doc, score in kres]
        else:  # hybrid (default)
            vres = self._vector_search(query, config=config)
            kres = self._keyword_search(query, config=config)
            retrieved_docs = self._reciprocal_rank_fusion_hybrid(vres, kres)

        return retrieved_docs[:self.config_params.final_k]

    async def _aget_relevant_documents(self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None, **kwargs: Any) -> List[Document]:
        # For simplicity, using the sync version. For production, implement async I/O.
        return self._get_relevant_documents(query, run_manager=run_manager, **kwargs)
