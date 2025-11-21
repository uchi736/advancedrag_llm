"""
rag_system_enhanced.py
======================
This file acts as a facade for the RAG system, assembling components
from the `rag` subdirectory into a cohesive `RAGSystem` class.
"""
from __future__ import annotations

import os
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


try:
    import psycopg
    _PG_DIALECT = "psycopg"
except ModuleNotFoundError:
    _PG_DIALECT = "psycopg2"

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.runnables import RunnableConfig
from langchain_community.callbacks.manager import get_openai_callback
from langchain_community.vectorstores import PGVector

# --- Refactored Module Imports ---
from src.rag.config import Config
from src.rag.text_processor import JapaneseTextProcessor
from src.rag.term_extraction import JargonDictionaryManager  # 統合されたモジュールから
from src.rag.retriever import JapaneseHybridRetriever
from src.rag.ingestion import IngestionHandler
from src.rag.sql_handler import SQLHandler
from src.rag.evaluator import RAGEvaluator, EvaluationResults, EvaluationMetrics
from src.rag.prompts import (
    get_jargon_extraction_prompt,
    get_query_augmentation_prompt,
    get_query_expansion_prompt,
    get_reranking_prompt,
    get_answer_generation_prompt
)
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableBranch, Runnable

# load_dotenv()  # Commented out - loaded in main script

# スキーマ定義: 各テーブルの期待されるカラムと型
EXPECTED_SCHEMAS = {
    'document_chunks': [
        ('id', 'SERIAL PRIMARY KEY'),
        ('collection_name', 'VARCHAR(255)'),
        ('document_id', 'VARCHAR(255)'),
        ('chunk_id', 'VARCHAR(255)'),
        ('content', 'TEXT'),
        ('tokenized_content', 'TEXT'),
        ('metadata', 'JSONB'),
        ('created_at', 'TIMESTAMP')
    ],
    'parent_child_chunks': [
        ('id', 'SERIAL PRIMARY KEY'),
        ('collection_name', 'VARCHAR(255)'),
        ('parent_chunk_id', 'VARCHAR(255)'),
        ('parent_content', 'TEXT'),
        ('child_chunk_ids', 'TEXT[]'),
        ('metadata', 'JSONB'),
        ('created_at', 'TIMESTAMP')
    ]
}

def format_docs(docs: List[Any]) -> str:
    """Helper function to format documents for context."""
    if not docs:
        return "(コンテキスト無し)"
    return "\n\n".join([f"[ソース {i+1} ChunkID: {d.metadata.get('chunk_id', 'N/A')}]\n{d.page_content}" for i, d in enumerate(docs)])

def _reciprocal_rank_fusion(results_list: List[List], k: int = 60) -> List:
    """
    Reciprocal Rank Fusion (RRF) algorithm to combine multiple ranked lists.

    Args:
        results_list: List of ranked document lists
        k: Constant for RRF formula (default 60)

    Returns:
        Re-ranked list of documents
    """
    from collections import defaultdict

    doc_scores = defaultdict(float)
    doc_objects = {}

    for doc_list in results_list:
        for rank, doc in enumerate(doc_list, start=1):
            # Use chunk_id as unique identifier
            doc_id = doc.metadata.get('chunk_id', str(hash(doc.page_content)))
            doc_scores[doc_id] += 1 / (rank + k)
            if doc_id not in doc_objects:
                doc_objects[doc_id] = doc

    # Sort by RRF score
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_objects[doc_id] for doc_id, score in sorted_docs]

def _rerank_documents_with_llm(docs: List, question: str, llm, top_k: int = 5) -> List:
    """
    Re-rank documents using LLM-based relevance scoring.

    Args:
        docs: List of documents to re-rank
        question: User question
        llm: Language model for scoring
        top_k: Number of top documents to return

    Returns:
        Re-ranked list of top_k documents
    """
    import logging
    logger = logging.getLogger(__name__)

    if not docs:
        return []

    try:
        from src.rag.prompts import get_reranking_prompt

        reranking_prompt = get_reranking_prompt()

        # Format documents for reranking
        doc_texts = [f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)]
        docs_combined = "\n\n".join(doc_texts)

        # Get relevance scores from LLM
        response = llm.invoke(
            reranking_prompt.format(question=question, documents=docs_combined)
        )

        # Parse scores from response
        import re
        scores = re.findall(r'\d+', response.content if hasattr(response, 'content') else str(response))
        scores = [int(s) for s in scores[:len(docs)]]

        # Pad with zeros if fewer scores than docs
        while len(scores) < len(docs):
            scores.append(0)

        # Sort documents by score
        doc_score_pairs = list(zip(docs, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

        return [doc for doc, score in doc_score_pairs[:top_k]]

    except Exception as e:
        logger.warning(f"Reranking failed: {e}. Returning original order.")
        return docs[:top_k]

class RAGSystem:
    def __init__(self, cfg: Config):
        self.config = cfg
        self.text_processor = JapaneseTextProcessor()

        # PostgreSQL connection
        self.connection_string = f"postgresql+{_PG_DIALECT}://{cfg.db_user}:{cfg.db_password}@{cfg.db_host}:{cfg.db_port}/{cfg.db_name}"
        # Create a shared SQLAlchemy engine to avoid per-call engine creation downstream.
        self.engine: Engine = create_engine(self.connection_string, pool_pre_ping=True)

        self._init_llms_and_embeddings()
        self._init_db()

        # Create PGVector store
        from langchain_community.vectorstores.pgvector import DistanceStrategy
        # Convert distance_strategy string to enum
        distance_strategy_str = getattr(cfg, 'distance_strategy', 'COSINE').upper()
        distance_strategy_map = {
            'COSINE': DistanceStrategy.COSINE,
            'EUCLIDEAN': DistanceStrategy.EUCLIDEAN,
            'MAX_INNER_PRODUCT': DistanceStrategy.MAX_INNER_PRODUCT
        }
        distance_strategy = distance_strategy_map.get(distance_strategy_str, DistanceStrategy.COSINE)

        self.vector_store = PGVector(
            connection_string=self.connection_string,
            embedding_function=self.embeddings,
            collection_name=cfg.collection_name,
            distance_strategy=distance_strategy
        )

        # Use Japanese hybrid retriever for PGVector
        self.retriever = JapaneseHybridRetriever(
            vector_store=self.vector_store,
            connection_string=self.connection_string,
            engine=self.engine,
            config_params=cfg,
            text_processor=self.text_processor
        )

        # Initialize components for PGVector
        self.jargon_manager = JargonDictionaryManager(
            self.connection_string,
            cfg.jargon_table_name,
            engine=self.engine,
            collection_name=cfg.collection_name
        )
        self.ingestion_handler = IngestionHandler(
            cfg,
            self.vector_store,
            self.engine,
            self.text_processor
        )
        self.sql_handler = SQLHandler(cfg, self.connection_string, engine=self.engine)

        # Create the retrieval chain
        self.retrieval_chain = self._create_retrieval_chain()

        # Initialize evaluator
        self.evaluator = None  # Lazy initialization to avoid overhead when not needed

    def _init_llms_and_embeddings(self):
        """Initialize LLM and embedding models with Azure OpenAI."""
        import logging
        logger = logging.getLogger(__name__)
        logger.info("RAGSystem initialized with Azure OpenAI.")
        self.llm = AzureChatOpenAI(
            temperature=getattr(self.config, 'llm_temperature', 0.0),
            max_tokens=getattr(self.config, 'max_tokens', 4096),
            azure_deployment=self.config.azure_openai_chat_deployment_name,
            api_key=self.config.azure_openai_api_key,
            azure_endpoint=self.config.azure_openai_endpoint,
            api_version=self.config.azure_openai_api_version
        )
        self.embeddings = AzureOpenAIEmbeddings(
            azure_deployment=self.config.azure_openai_embedding_deployment_name,
            api_key=self.config.azure_openai_api_key,
            azure_endpoint=self.config.azure_openai_endpoint,
            api_version=self.config.azure_openai_api_version
        )

    def _init_db(self):
        """Initialize database tables and extensions."""
        with self.engine.connect() as conn, conn.begin():
            # Enable pgvector extension
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

            # Create table for keyword search
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id SERIAL PRIMARY KEY,
                    collection_name VARCHAR(255),
                    document_id VARCHAR(255),
                    chunk_id VARCHAR(255) UNIQUE NOT NULL,
                    content TEXT NOT NULL,
                    tokenized_content TEXT,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_chunk_collection ON document_chunks(collection_name)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_chunk_document ON document_chunks(document_id)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_chunk_id ON document_chunks(chunk_id)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_tokenized_gin ON document_chunks USING gin(to_tsvector('simple', tokenized_content))"))

            # Create parent-child chunk table
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS parent_child_chunks (
                    id SERIAL PRIMARY KEY,
                    collection_name VARCHAR(255),
                    parent_chunk_id VARCHAR(255) UNIQUE NOT NULL,
                    parent_content TEXT NOT NULL,
                    child_chunk_ids TEXT[] NOT NULL,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_parent_chunk_collection ON parent_child_chunks(collection_name)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_parent_chunk_id ON parent_child_chunks(parent_chunk_id)"))

            # スキーマ検証とマイグレーション
            self._validate_and_migrate_schema(conn)

    def _validate_and_migrate_schema(self, conn):
        """データベーススキーマを検証し、必要に応じてマイグレーション"""
        import logging
        logger = logging.getLogger(__name__)

        for table_name, expected_columns in EXPECTED_SCHEMAS.items():
            # 現在のカラムを取得
            result = conn.execute(text("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = :table_name
            """), {"table_name": table_name}).fetchall()

            existing_columns = {row[0] for row in result}
            expected_column_names = {col[0] for col in expected_columns}

            # 不要なカラムを削除
            columns_to_drop = existing_columns - expected_column_names
            for col in columns_to_drop:
                logger.info(f"Dropping obsolete column: {col} from {table_name}")
                conn.execute(text(f"ALTER TABLE {table_name} DROP COLUMN IF EXISTS {col}"))

            # 不足しているカラムを追加
            columns_to_add = expected_column_names - existing_columns
            for col_name in columns_to_add:
                # 型情報を取得
                col_type = next((col[1] for col in expected_columns if col[0] == col_name), 'TEXT')
                # PRIMARY KEYやUNIQUE等の制約は除外（ALTER TABLEでは追加できない）
                col_type_clean = col_type.replace('PRIMARY KEY', '').replace('UNIQUE', '').replace('NOT NULL', '').strip()
                if col_type_clean:  # 空文字列でない場合のみ追加
                    logger.info(f"Adding missing column: {col_name} {col_type_clean} to {table_name}")
                    try:
                        conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS {col_name} {col_type_clean}"))
                    except Exception as e:
                        logger.warning(f"Failed to add column {col_name} to {table_name}: {e}")

    def _create_retrieval_chain(self) -> Runnable:
        """
        Create a modular retrieval chain with optional features:
        - Query expansion
        - RAG fusion (multi-query + RRF)
        - Jargon augmentation
        - LLM reranking
        """
        import logging
        logger = logging.getLogger(__name__)

        # Step 1: Extract jargon terms from query (if enabled)
        def extract_jargon(inputs: Dict) -> Dict:
            """Extract technical terms from the query"""
            question = inputs.get("question", "")
            if not question or not inputs.get("use_jargon_augmentation", False):
                inputs["jargon_terms"] = []
                return inputs

            try:
                jargon_prompt = get_jargon_extraction_prompt()
                response = self.llm.invoke(jargon_prompt.format(question=question))

                # Parse terms from response
                import re
                terms = re.findall(r'[ぁ-んァ-ヴー一-龠a-zA-Z0-9]+', response.content if hasattr(response, 'content') else str(response))
                inputs["jargon_terms"] = terms[:5]  # Limit to top 5
            except Exception as e:
                logger.warning(f"Jargon extraction failed: {e}")
                inputs["jargon_terms"] = []

            return inputs

        # Step 2: Augment query with jargon definitions
        def augment_with_jargon(inputs: Dict) -> Dict:
            """Augment query with technical term definitions"""
            if not inputs.get("use_jargon_augmentation", False) or not inputs.get("jargon_terms"):
                inputs["augmented_query"] = inputs.get("question", "")
                inputs["jargon_augmentation"] = {}
                return inputs

            try:
                # Lookup all terms at once
                jargon_terms = inputs.get("jargon_terms", [])
                matched_terms = self.jargon_manager.lookup_terms(jargon_terms) if jargon_terms else {}

                # Augment query with definitions, aliases, and related terms using LLM
                from langchain_core.output_parsers import StrOutputParser

                question = inputs.get("question", "")
                if matched_terms:
                    # 1. Format jargon information
                    augmentation_parts = []
                    for term, info in matched_terms.items():
                        # 定義
                        parts = [f"{term}: {info['definition']}"]

                        # 類義語
                        if info.get('aliases'):
                            aliases_str = ", ".join(info['aliases'])
                            parts.append(f"  類義語: {aliases_str}")

                        # 関連語
                        if info.get('related_terms'):
                            related_str = ", ".join(info['related_terms'])
                            parts.append(f"  関連: {related_str}")

                        augmentation_parts.append("\n".join(parts))

                    jargon_definitions = "\n\n".join(augmentation_parts)

                    # 2. Optimize query using LLM with LCEL chain
                    try:
                        logger.info(f"Optimizing query with LLM: '{question}'")
                        logger.debug(f"Jargon definitions:\n{jargon_definitions}")

                        augmentation_chain = (
                            get_query_augmentation_prompt()
                            | self.llm
                            | StrOutputParser()
                        )

                        optimized_query = augmentation_chain.invoke({
                            "original_question": question,
                            "jargon_definitions": jargon_definitions
                        })

                        inputs["augmented_query"] = optimized_query.strip()
                        inputs["raw_augmented_query"] = f"{question}\n\n関連用語:\n{jargon_definitions}"  # For debugging

                        logger.info(f"✓ LLM optimization successful")
                        logger.info(f"  Original: {question}")
                        logger.info(f"  Optimized: {optimized_query.strip()}")

                    except Exception as llm_error:
                        logger.warning(f"✗ LLM query optimization failed: {llm_error}, using fallback")
                        logger.warning(f"  Error type: {type(llm_error).__name__}")
                        # Fallback: simple concatenation
                        inputs["augmented_query"] = f"{question}\n\n関連用語:\n{jargon_definitions}"
                        inputs["raw_augmented_query"] = inputs["augmented_query"]
                else:
                    inputs["augmented_query"] = question

                inputs["jargon_augmentation"] = {
                    "matched_terms": matched_terms,
                    "extracted_terms": inputs["jargon_terms"],
                    "augmented_query": inputs["augmented_query"]
                }
            except Exception as e:
                logger.warning(f"Jargon augmentation failed: {e}")
                inputs["augmented_query"] = inputs.get("question", "")
                inputs["jargon_augmentation"] = {}

            return inputs

        # Step 3: Query expansion
        def expand_query(inputs: Dict) -> Dict:
            """Expand query with related queries"""
            if not inputs.get("use_query_expansion", False) and not inputs.get("use_rag_fusion", False):
                base_query = inputs.get("augmented_query") or inputs.get("question", "")
                inputs["expanded_queries"] = [base_query] if base_query else []
                inputs["query_expansion"] = {}
                return inputs

            try:
                expansion_prompt = get_query_expansion_prompt()
                base_query = inputs.get("augmented_query") or inputs.get("question", "")
                response = self.llm.invoke(expansion_prompt.format(original_query=base_query))

                # Parse expanded queries
                import re
                expanded = re.findall(r'\d+\.\s*(.+)', response.content if hasattr(response, 'content') else str(response))
                inputs["expanded_queries"] = [base_query] + expanded[:4]  # Original + 4 variations
                inputs["query_expansion"] = {
                    "original_query": base_query,
                    "expanded_queries": expanded[:4]
                }
            except Exception as e:
                logger.warning(f"Query expansion failed: {e}")
                fallback_query = inputs.get("augmented_query") or inputs.get("question", "")
                inputs["expanded_queries"] = [fallback_query] if fallback_query else []
                inputs["query_expansion"] = {}

            return inputs

        # Step 4: Retrieve documents (with optional RAG fusion)
        def retrieve_documents(inputs: Dict) -> Dict:
            """Retrieve documents using configured search type"""
            search_type = inputs.get("search_type", "hybrid")
            default_query = inputs.get("question", "")
            queries = inputs.get("expanded_queries", [default_query] if default_query else [])

            # Set retriever search type
            original_search_type = getattr(self.retriever, 'search_type', 'hybrid')
            self.retriever.search_type = search_type

            try:
                if inputs.get("use_rag_fusion", False) and len(queries) > 1:
                    # Multi-query retrieval + RRF
                    all_results = []
                    for query in queries:
                        docs = self.retriever.get_relevant_documents(query)
                        all_results.append(docs)

                    # Apply RRF
                    inputs["documents"] = _reciprocal_rank_fusion(all_results)[:self.config.final_k]
                    inputs["golden_retriever"] = {
                        "enabled": True,
                        "num_queries": len(queries),
                        "fusion_method": "RRF"
                    }
                else:
                    # Single query retrieval
                    query = queries[0] if queries else inputs.get("question", "")
                    inputs["documents"] = self.retriever.get_relevant_documents(query)[:self.config.final_k] if query else []
                    inputs["golden_retriever"] = {"enabled": False}

            except Exception as e:
                logger.error(f"Document retrieval failed: {e}")
                inputs["documents"] = []
                inputs["golden_retriever"] = {"enabled": False, "error": str(e)}
            finally:
                # Restore original search type
                self.retriever.search_type = original_search_type

            return inputs

        # Step 5: Rerank documents (optional)
        def rerank_documents(inputs: Dict) -> Dict:
            """Rerank documents using LLM"""
            if not inputs.get("use_reranking", False) or not inputs.get("documents"):
                inputs["reranking"] = {"enabled": False}
                return inputs

            try:
                original_count = len(inputs["documents"])
                reranked_docs = _rerank_documents_with_llm(
                    inputs["documents"],
                    inputs.get("question", ""),
                    self.llm,
                    top_k=self.config.final_k
                )
                inputs["documents"] = reranked_docs
                inputs["reranking"] = {
                    "enabled": True,
                    "original_count": original_count,
                    "reranked_count": len(reranked_docs)
                }
            except Exception as e:
                logger.warning(f"Reranking failed: {e}")
                inputs["reranking"] = {"enabled": False, "error": str(e)}

            return inputs

        # Step 6: Format retrieval query for logging
        def add_retrieval_query(inputs: Dict) -> Dict:
            """Add the final retrieval query used"""
            inputs["retrieval_query"] = inputs.get("augmented_query") or inputs.get("question", "")
            return inputs

        # Build the chain using RunnableLambda for each step
        retrieval_chain = (
            RunnableLambda(extract_jargon)
            | RunnableLambda(augment_with_jargon)
            | RunnableLambda(expand_query)
            | RunnableLambda(retrieve_documents)
            | RunnableLambda(rerank_documents)
            | RunnableLambda(add_retrieval_query)
        )

        return retrieval_chain

    def delete_jargon_terms(self, terms: List[str]) -> tuple[int, int]:
        if not self.jargon_manager:
            return 0, 0
        return self.jargon_manager.delete_terms(terms)

    def delete_document_by_id(self, doc_id: str) -> tuple[bool, str]:
        """Delete a document by its ID from the current collection.

        Args:
            doc_id: Document ID to delete

        Returns:
            Tuple of (success: bool, message: str)
        """
        if not hasattr(self, 'ingestion_handler') or self.ingestion_handler is None:
            return False, "Ingestion handler not available"

        return self.ingestion_handler.delete_document_by_id(doc_id, self.config.collection_name)

    def ingest_documents(self, paths: List[str]) -> None:
        """Ingest documents into the vector store.

        Args:
            paths: List of file paths to ingest
        """
        if not hasattr(self, 'ingestion_handler') or self.ingestion_handler is None:
            raise ValueError("Ingestion handler not available")

        self.ingestion_handler.ingest_documents(paths)

    def get_chunks_by_document_id(self, doc_id: str):
        """Get chunks for a specific document ID.

        Args:
            doc_id: Document ID to get chunks for

        Returns:
            DataFrame with chunk information
        """
        if not hasattr(self, 'sql_handler') or self.sql_handler is None:
            raise ValueError("SQL handler not available")

        return self.sql_handler.get_chunks_by_document_id(doc_id)

    # --- Core Query Logic ---
    def query(self, question: str, search_type: str = None) -> Dict[str, Any]:
        """Standard query operation using RAG chain."""
        with get_openai_callback() as cb:
            result = self.retrieval_chain.invoke({
                "question": question,
                "use_jargon_augmentation": self.config.enable_jargon_augmentation
            })

            # Calculate confidence scores for retrieved docs
            if "context" in result and hasattr(result["context"], "__len__"):
                result["confidence_scores"] = [0.85] * len(result["context"])

            # Add total tokens
            result["total_tokens"] = cb.total_tokens if hasattr(cb, "total_tokens") else 0

            return result

    def rag_search(self, question: str, search_type: str = "hybrid") -> Dict[str, Any]:
        """Perform RAG search with specified search type.

        Args:
            question: The query to search for
            search_type: Type of search - "hybrid", "vector", or "keyword"

        Returns:
            Dictionary containing search results and metadata
        """
        # Set the search type on the retriever
        if hasattr(self.retriever, 'search_type'):
            original_search_type = self.retriever.search_type
            self.retriever.search_type = search_type

        try:
            # Create prompt for the query
            if self.config.enable_jargon_augmentation:
                rag_prompt = ChatPromptTemplate.from_messages([
                    ("system", """あなたはAzure RAGシステムのアシスタントです。
与えられたコンテキストに基づいて、ユーザーの質問に正確に答えてください。
コンテキストに情報がない場合は、「情報が見つかりません」と回答してください。

専門用語の定義が提供されている場合は、それを考慮して回答してください。"""),
                    ("human", """コンテキスト: {context}

専門用語定義（もしあれば）: {jargon_definitions}

質問: {question}

答え:""")
                ])
            else:
                rag_prompt = ChatPromptTemplate.from_messages([
                    ("system", """あなたはAzure RAGシステムのアシスタントです。
与えられたコンテキストに基づいて、ユーザーの質問に正確に答えてください。
コンテキストに情報がない場合は、「情報が見つかりません」と回答してください。"""),
                    ("human", """コンテキスト: {context}

質問: {question}

答え:""")
                ])

            # Execute retrieval and generation
            with get_openai_callback() as cb:
                rag_results = self.retrieval_chain.invoke({
                    "question": question,
                    "use_jargon_augmentation": self.config.enable_jargon_augmentation
                })

                # Process jargon if enabled
                jargon_definitions = ""
                if self.config.enable_jargon_augmentation and "jargon_augmentation" in rag_results:
                    jargon_info = rag_results["jargon_augmentation"]
                    if jargon_info.get("matched_terms"):
                        jargon_definitions = "\n".join([f"- {term}: {info['definition']}" for term, info in jargon_info["matched_terms"].items()])

                # Format context
                # Note: retrieval_chain returns "documents" key, not "context"
                docs = rag_results.get("documents") or rag_results.get("context", [])
                context = format_docs(docs)

                # Use augmented query if available, otherwise use original question
                final_question = rag_results.get("retrieval_query") or question

                # Generate answer
                chain = rag_prompt | self.llm | StrOutputParser()
                answer = chain.invoke({
                    "context": context,
                    "question": final_question,
                    "jargon_definitions": jargon_definitions
                })

                rag_results["answer"] = answer
                rag_results["total_tokens"] = cb.total_tokens if hasattr(cb, "total_tokens") else 0

            return rag_results

        finally:
            # Restore original search type
            if hasattr(self.retriever, 'search_type'):
                self.retriever.search_type = original_search_type

    def synthesize(self, question: str) -> Dict[str, Any]:
        """Synthesize answer using both RAG and SQL results."""
        rag_results = self.query(question, search_type="hybrid")

        # Note: retrieval_chain returns "documents" key, not "context"
        docs = rag_results.get("documents") or rag_results.get("context", [])
        return {
            "answer": rag_results.get("answer", ""),
            "source": "rag",
            "context": docs,
            "confidence_scores": rag_results.get("confidence_scores", []),
            "jargon_augmentation": rag_results.get("jargon_augmentation", {}),
            "query_expansion": rag_results.get("query_expansion", []),
            "retrieval_query": rag_results.get("retrieval_query", question),
            "golden_retriever": {}
        }

    def query_unified(
        self,
        question: str,
        use_query_expansion: bool = False,
        use_rag_fusion: bool = False,
        use_jargon_augmentation: bool = False,
        use_reranking: bool = False,
        search_type: str = None,  # None uses config.default_search_type
        config: Any = None
    ) -> Dict[str, Any]:
        """Unified query method supporting all advanced RAG features.

        Args:
            question: User query
            use_query_expansion: Enable query expansion
            use_rag_fusion: Enable RAG fusion (query expansion + RRF)
            use_jargon_augmentation: Enable jargon augmentation
            use_reranking: Enable LLM reranking
            search_type: Search type ("ハイブリッド検索", "ベクトル検索", "キーワード検索")
            config: Optional RunnableConfig for tracing

        Returns:
            Dictionary with answer, sources, and metadata
        """
        from langchain_community.callbacks import get_openai_callback
        from langchain_core.output_parsers import StrOutputParser
        from src.rag.prompts import get_answer_generation_prompt

        # Use config default if search_type not specified
        if search_type is None:
            search_type = self.config.default_search_type

        # Map Japanese search type to English (for backward compatibility)
        search_type_map = {
            "ハイブリッド検索": "hybrid",
            "ベクトル検索": "vector",
            "キーワード検索": "keyword"
        }
        search_type_eng = search_type_map.get(search_type, search_type)  # Pass through if already English

        with get_openai_callback() as cb:
            # Prepare input for retrieval chain
            retrieval_input = {
                "question": question,
                "use_query_expansion": use_query_expansion,
                "use_rag_fusion": use_rag_fusion,
                "use_jargon_augmentation": use_jargon_augmentation,
                "use_reranking": use_reranking,
                "search_type": search_type_eng,
                "config": config
            }

            # Execute retrieval chain (includes all advanced features)
            retrieval_result = self.retrieval_chain.invoke(retrieval_input, config=config)

            # Extract documents
            documents = retrieval_result.get("documents", [])

            # Format context for answer generation
            def format_docs(docs):
                return "\n\n---\n\n".join([doc.page_content for doc in docs])

            context = format_docs(documents)

            # Get jargon definitions and augmented query if augmentation was used
            jargon_definitions = ""
            query_for_answer = question  # Default to original question

            if use_jargon_augmentation and "jargon_augmentation" in retrieval_result:
                jargon_info = retrieval_result["jargon_augmentation"]
                if jargon_info.get("matched_terms"):
                    jargon_definitions = "\n".join([
                        f"- {term}: {info['definition']}"
                        for term, info in jargon_info["matched_terms"].items()
                    ])
                # Use augmented query for answer generation if available
                if jargon_info.get("augmented_query"):
                    query_for_answer = jargon_info["augmented_query"]

            # Generate answer using LLM
            answer_prompt = get_answer_generation_prompt()
            answer_chain = answer_prompt | self.llm | StrOutputParser()

            answer = answer_chain.invoke({
                "context": context,
                "question": query_for_answer,  # Use augmented query if available
                "jargon_definitions": jargon_definitions
            }, config=config)

            # Build sources from documents
            sources = []
            for doc in documents:
                if hasattr(doc, 'metadata'):
                    sources.append({
                        "content": doc.page_content if hasattr(doc, 'page_content') else str(doc),
                        "metadata": doc.metadata
                    })

            # Build result
            result = {
                "answer": answer,
                "sources": sources,
                "context": documents,
                "total_tokens": cb.total_tokens if hasattr(cb, "total_tokens") else 0,
                "retrieval_query": retrieval_result.get("retrieval_query", question),
                "query_expansion": retrieval_result.get("query_expansion", {}),
                "golden_retriever": retrieval_result.get("golden_retriever", {}),
                "jargon_augmentation": retrieval_result.get("jargon_augmentation", {}),
                "reranking": retrieval_result.get("reranking", {})
            }

            return result

    async def extract_terms(self, input_dir: str | Path, output_json: str | Path) -> None:
        """Extract terms from documents.

        Note: Only the first sample of each stage will be traced to LangSmith
        to avoid excessive requests while still allowing validation of the pipeline.
        """
        from src.rag.term_extraction import run_extraction_pipeline

        # Use connection_string only if using PGVector, otherwise pass None
        pg_url = self.connection_string
        await run_extraction_pipeline(
            Path(input_dir), Path(output_json),
            self.config, self.llm, self.embeddings,
            self.vector_store, pg_url, self.config.jargon_table_name,
            jargon_manager=None,
            collection_name=self.config.collection_name
        )
        print(f"[TermExtractor] Extraction complete -> {output_json}")

    # --- Evaluation Methods ---
    def initialize_evaluator(self,
                           k_values: List[int] = [1, 3, 5, 10],
                           similarity_method: str = "azure_embedding",
                           similarity_threshold: float = None) -> RAGEvaluator:
        """Initialize the evaluation system"""
        if self.evaluator is None:
            # Use config.confidence_threshold if similarity_threshold not provided
            threshold = similarity_threshold if similarity_threshold is not None else self.config.confidence_threshold
            self.evaluator = RAGEvaluator(
                config=self.config,
                k_values=k_values,
                similarity_method=similarity_method,
                similarity_threshold=threshold
            )
        return self.evaluator

    async def evaluate_system(self,
                             test_questions: List[Dict[str, Any]],
                             similarity_method: str = "azure_embedding",
                             export_path: Optional[str] = None) -> EvaluationMetrics:
        """Evaluate the RAG system with test questions"""
        if self.evaluator is None:
            self.initialize_evaluator(similarity_method=similarity_method)

        # Use the existing retrieval chain for evaluation
        eval_chain = self.retrieval_chain

        # Run evaluation
        results = await self.evaluator.evaluate_retrieval(
            test_questions=test_questions,
            retrieval_chain=eval_chain,
            llm=self.llm
        )

        # Export results if path provided
        if export_path:
            self.evaluator.export_results(results, export_path)

        return results.metrics

    def get_evaluation_results(self) -> Optional[EvaluationResults]:
        """Get the latest evaluation results"""
        if self.evaluator is None:
            return None
        return self.evaluator.last_results

    def export_evaluation_results(self, export_path: str) -> bool:
        """Export evaluation results to file"""
        if self.evaluator is None or self.evaluator.last_results is None:
            return False
        return self.evaluator.export_results(self.evaluator.last_results, export_path)