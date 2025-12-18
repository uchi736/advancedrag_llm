import json
import re
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from typing import List, Dict, Any, Optional, Tuple

from langchain_community.vectorstores import PGVector
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.runnables import RunnableConfig

from .config import Config
from .text_processor import JapaneseTextProcessor

class JapaneseHybridRetriever(BaseRetriever):
    """
    A retriever that combines vector search and keyword search (FTS)
    with Reciprocal Rank Fusion (RRF) for hybrid search, optimized for
    Japanese language.
    """
    vector_store: PGVector
    connection_string: str
    config_params: Config
    text_processor: JapaneseTextProcessor
    search_type: str = "hybrid"
    engine: Optional[Engine] = None
    
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

            # フィルタなしでk*2件取得 → 手動でjargon_term除外 → k件
            # (メタデータフィルタはtype未設定のドキュメントも除外してしまうため使わない)
            if hasattr(self.vector_store, 'similarity_search_with_score'):
                results = self.vector_store.similarity_search_with_score(q, k=search_k * 2)
            elif hasattr(self.vector_store, 'vector_store'):
                results = self.vector_store.vector_store.similarity_search_with_score(q, k=search_k * 2)
            else:
                return []

            # 手動でjargon_termを除外
            filtered = [(doc, score) for doc, score in results if doc.metadata.get('type') != 'jargon_term']
            return filtered[:search_k]

        except Exception as exc:
            print(f"[HybridRetriever] vector search error: {exc}")
            return []

    def _keyword_search(self, q: str, config: Optional[RunnableConfig] = None) -> List[Tuple[Document, float]]:
        """Performs keyword-based search with Japanese tokenization support."""
        res: List[Tuple[Document, float]] = []

        # Check if engine is available
        if not self.engine:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning("[HybridRetriever] No engine available for keyword search")
            return []

        normalized_query = self.text_processor.normalize_text(q)
        is_japanese = self.text_processor.is_japanese(normalized_query)

        try:
            with self.engine.connect() as conn:
                if is_japanese and self.config_params.enable_japanese_search:
                    tokens = self.text_processor.tokenize(normalized_query)
                    if not tokens: return []

                    # Filter tokens by minimum length
                    filtered_tokens = [t for t in tokens[:5] if len(t) >= self.config_params.japanese_min_token_length]
                    if not filtered_tokens: return []

                    # 記号を空白にして安全なクエリにする
                    safe_tokens = []
                    for t in filtered_tokens:
                        cleaned = re.sub(r"[^\\wぁ-んァ-ヶ一-龠ー\\s]", " ", t)
                        cleaned = cleaned.strip()
                        if cleaned:
                            safe_tokens.append(cleaned)

                    if not safe_tokens:
                        return []

                    # plainto_tsqueryで記号崩れを防ぐ（ORよりAND寄りになるが安全性優先）
                    tsquery_str = " ".join(safe_tokens)

                    sql = """
                        SELECT chunk_id, content, metadata,
                               ts_rank(to_tsvector('simple', tokenized_content), plainto_tsquery('simple', :tsquery)) AS score
                        FROM document_chunks
                        WHERE to_tsvector('simple', tokenized_content) @@ plainto_tsquery('simple', :tsquery)
                              AND collection_name = :collection_name
                        ORDER BY score DESC LIMIT :k;
                    """

                    db_result = conn.execute(text(sql), {
                        "tsquery": tsquery_str,
                        "collection_name": self.config_params.collection_name,
                        "k": self.config_params.keyword_search_k
                    })
                else:
                    sql = f"""
                        SELECT chunk_id, content, metadata, 
                               ts_rank(to_tsvector('{self.config_params.fts_language}', content), plainto_tsquery('{self.config_params.fts_language}', :q)) AS score 
                        FROM document_chunks 
                        WHERE to_tsvector('{self.config_params.fts_language}', content) @@ plainto_tsquery('{self.config_params.fts_language}', :q) 
                        AND collection_name = :collection_name 
                        ORDER BY score DESC LIMIT :k;
                    """
                    db_result = conn.execute(text(sql), {"q": normalized_query, "k": self.config_params.keyword_search_k, "collection_name": self.config_params.collection_name})
                
                for row in db_result:
                    md = row.metadata if isinstance(row.metadata, dict) else json.loads(row.metadata or "{}")
                    res.append((Document(page_content=row.content, metadata=md), float(row.score)))
                    
        except Exception as exc:
            print(f"[HybridRetriever] keyword search error: {exc}")
            return []
        return res

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
