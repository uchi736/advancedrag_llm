"""
term_extraction.py
==================
LLMベースのシンプルな専門用語抽出モジュール
統計的手法を使用せず、LLMに直接テキストを渡して専門用語を抽出

LangGraphによるワークフロー型抽出:
- Stage 1: 候補抽出
- Stage 2: 初期選別
- Stage 2.5: 自己反省ループ（再帰的精緻化）
- Stage 3: 定義生成
- Stage 4: 類義語検出
"""

from __future__ import annotations

import json
import logging
import operator
from pathlib import Path
from typing import Dict, List, Optional, Any, Annotated, Literal, TypedDict
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers.fix import OutputFixingParser
from langchain_core.exceptions import OutputParserException
from langchain_core.documents import Document
from pydantic import BaseModel, Field
import asyncio
from tqdm import tqdm

# LangGraph imports
from langgraph.graph import StateGraph, END

# Domain classification (optional)
try:
    from src.rag.term_clustering import TermClusteringAnalyzer, CLUSTERING_AVAILABLE
except ImportError:
    CLUSTERING_AVAILABLE = False
    TermClusteringAnalyzer = None

from .retriever import similarity_search_without_jargon

logger = logging.getLogger(__name__)


# ========== Pydantic Models ==========
class ExtractedTerm(BaseModel):
    """抽出された専門用語候補"""
    term: str = Field(description="専門用語")
    brief_definition: Optional[str] = Field(default=None, description="周辺テキストから抽出した簡易的な定義・説明（1-2文）")
    synonyms: List[str] = Field(default_factory=list, description="類義語・別名のリスト")
    confidence: float = Field(default=1.0, description="信頼度スコア")
    domain: Optional[str] = Field(default=None, description="分野・ドメイン（例: 医療、工学、法律など）")
    rejection_reason: Optional[str] = Field(default=None, description="除外理由（除外された場合のみ）")
    confidence_level: Optional[str] = Field(default="middle", description="確信度レベル（high/middle/low）")


class ExtractedTermList(BaseModel):
    """専門用語リスト"""
    terms: List[ExtractedTerm] = Field(description="抽出された専門用語のリスト")


class FilteredTermResult(BaseModel):
    """Stage 2の選別結果（選ばれた用語と除外された用語）"""
    selected: List[ExtractedTerm] = Field(description="専門用語として選ばれた用語")
    rejected: List[ExtractedTerm] = Field(description="除外された用語（rejection_reason付き）")


class MissingTerm(BaseModel):
    """Stage 2.5: 漏れている用語候補"""
    term: str = Field(description="漏れている用語")
    evidence: Optional[str] = Field(default=None, description="必要と判断した文脈・根拠")
    suggested_domain: Optional[str] = Field(default=None, description="推定分野")


class ReflectionResult(BaseModel):
    """Stage 2.5: 自己反省の結果"""
    issues_found: List[str] = Field(description="発見された問題点")
    confidence_in_current_list: float = Field(description="現在のリストへの信頼度 (0-1)")
    suggested_actions: List[str] = Field(description="推奨される改善アクション")
    should_continue: bool = Field(description="反復を続けるべきか")
    reasoning: str = Field(description="判断の理由")
    missing_terms: List[MissingTerm] = Field(default_factory=list, description="候補から追加すべき漏れ用語")


# 類義語グループ用（Stage4）
class SynonymGroup(BaseModel):
    headword: str
    synonyms: List[str] = Field(default_factory=list)
    definition: Optional[str] = ""
    domain: Optional[str] = "一般"


class SynonymMap(BaseModel):
    term: str
    synonyms: List[str] = Field(default_factory=list)
    domain: Optional[str] = None


class SynonymGroupResponse(BaseModel):
    groups: List[SynonymGroup] = Field(default_factory=list, description="類義語グループ一覧")


class SynonymMapResponse(BaseModel):
    items: List[SynonymMap] = Field(default_factory=list, description="代表語→類義語の対応一覧")


# ========== LangGraph State ==========

def _overwrite_list(existing, new):
    """リデューサー: 新しいリストで上書き（operator.addの代替）"""
    return new


class TermExtractionState(TypedDict):
    """LangGraph用の状態定義"""
    # 入力
    text_chunks: List[str]

    # 抽出結果
    candidates: List[Dict]  # Stage 1の候補リスト
    technical_terms: List[Dict]  # 現在の専門用語リスト
    rejected_terms: Annotated[List[Dict], _overwrite_list]  # 除外された用語（上書き）

    # Stage 2.5 ループ用
    refinement_iteration: int  # 現在の反復回数
    max_refinement_iterations: int  # 最大反復回数
    refinement_converged: bool  # 収束フラグ
    reflection_history: Annotated[List[Dict], _overwrite_list]  # 反省履歴（上書き）
    previous_list_hash: Optional[int]  # 前回のリストハッシュ（変更検知用）
    last_added_count: int  # 前回の追加数（収束判定用）
    last_removed_count: int  # 前回の削除数（収束判定用）
    investigate_terms: Annotated[List[Dict], _overwrite_list]  # investigate対象用語（上書き）

    # 出力ディレクトリ
    output_dir: Optional[Path]


# ========== JargonDictionaryManager (互換性のため残す) ==========
class JargonDictionaryManager:
    """専門用語辞書の管理クラス（互換性用）"""

    def __init__(self, connection_string: str, table_name: str = "jargon_dictionary",
                 engine: Optional[Engine] = None, collection_name: str = "documents"):
        self.connection_string = connection_string
        self.table_name = table_name
        self.collection_name = collection_name
        self.engine: Engine = engine or create_engine(connection_string)
        self._init_jargon_table()

    def _init_jargon_table(self):
        """専門用語辞書テーブルの初期化"""
        with self.engine.connect() as conn:
            # Create table with collection_name column
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id SERIAL PRIMARY KEY,
                    term TEXT NOT NULL,
                    definition TEXT NOT NULL,
                    domain TEXT,
                    aliases TEXT[],
                    related_terms TEXT[],
                    collection_name TEXT NOT NULL DEFAULT 'documents',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(term, collection_name)
                )
            """))

            # Add collection_name column if it doesn't exist (for existing tables)
            conn.execute(text(f"""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = '{self.table_name}' AND column_name = 'collection_name'
                    ) THEN
                        ALTER TABLE {self.table_name} ADD COLUMN collection_name TEXT NOT NULL DEFAULT 'documents';
                    END IF;
                END $$;
            """))

            # Update UNIQUE constraint from (term) to (term, collection_name)
            conn.execute(text(f"""
                DO $$
                BEGIN
                    -- Drop old UNIQUE constraint on (term) if it exists
                    IF EXISTS (
                        SELECT 1 FROM pg_constraint
                        WHERE conname = '{self.table_name}_term_key'
                    ) THEN
                        ALTER TABLE {self.table_name} DROP CONSTRAINT {self.table_name}_term_key;
                    END IF;

                    -- Add new UNIQUE constraint on (term, collection_name) if it doesn't exist
                    IF NOT EXISTS (
                        SELECT 1 FROM pg_constraint
                        WHERE conname = '{self.table_name}_term_collection_name_key'
                    ) THEN
                        ALTER TABLE {self.table_name} ADD CONSTRAINT {self.table_name}_term_collection_name_key UNIQUE (term, collection_name);
                    END IF;
                END $$;
            """))

            conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_jargon_term ON {self.table_name} (LOWER(term), collection_name)"))
            conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_jargon_collection ON {self.table_name} (collection_name)"))
            conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_jargon_aliases ON {self.table_name} USING GIN(aliases)"))
            conn.commit()

    def add_term(self, term: str, definition: str, domain: Optional[str] = None,
                 aliases: Optional[List[str]] = None, related_terms: Optional[List[str]] = None) -> bool:
        """用語を辞書に追加または更新（現在のコレクション内）"""
        try:
            with self.engine.connect() as conn:
                conn.execute(text(f"""
                    INSERT INTO {self.table_name}
                    (term, definition, domain, aliases, related_terms, collection_name)
                    VALUES (:term, :definition, :domain, :aliases, :related_terms, :collection_name)
                    ON CONFLICT (term, collection_name) DO UPDATE SET
                        definition = EXCLUDED.definition,
                        domain = EXCLUDED.domain,
                        aliases = EXCLUDED.aliases,
                        related_terms = EXCLUDED.related_terms,
                        updated_at = CURRENT_TIMESTAMP
                """), {
                    "term": term, "definition": definition, "domain": domain,
                    "aliases": aliases or [], "related_terms": related_terms or [],
                    "collection_name": self.collection_name
                })
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error adding term: {e}")
            return False

    def lookup_terms(self, terms: List[str]) -> Dict[str, Dict[str, Any]]:
        """複数の用語を辞書から検索（現在のコレクション内）"""
        if not terms:
            return {}

        results = {}
        try:
            with self.engine.connect() as conn:
                placeholders = ', '.join([f':term_{i}' for i in range(len(terms))])
                query = text(f"""
                    SELECT term, definition, domain, aliases, related_terms
                    FROM {self.table_name}
                    WHERE collection_name = :collection_name
                      AND (LOWER(term) IN ({placeholders}) OR term = ANY(:aliases_check))
                """)
                params = {f"term_{i}": term.lower() for i, term in enumerate(terms)}
                params["aliases_check"] = terms
                params["collection_name"] = self.collection_name

                rows = conn.execute(query, params).fetchall()
                for row in rows:
                    results[row.term] = {
                        "definition": row.definition, "domain": row.domain,
                        "aliases": row.aliases or [], "related_terms": row.related_terms or []
                    }
        except Exception as e:
            logger.error(f"Error looking up terms: {e}")
        return results

    def delete_terms(self, terms: List[str]) -> tuple[int, int]:
        """複数の用語を一括削除（現在のコレクション内のみ）"""
        if not terms:
            return 0, 0

        deleted = 0
        errors = 0
        try:
            with self.engine.connect() as conn, conn.begin():
                for term in terms:
                    if not term:
                        errors += 1
                        continue
                    result = conn.execute(
                        text(f"DELETE FROM {self.table_name} WHERE term = :term AND collection_name = :collection_name"),
                        {"term": term, "collection_name": self.collection_name}
                    )
                    deleted += result.rowcount or 0
        except Exception as e:
            logger.error(f"Bulk delete error: {e}")
            return deleted, len(terms) - deleted
        return deleted, errors

    def sync_to_vector_store(self, vector_store, embeddings):
        """
        専門用語の定義文をベクトルストアに同期

        Args:
            vector_store: PGVector store instance
            embeddings: Embeddings instance
        """
        from langchain_core.documents import Document

        all_terms = self.get_all_terms()

        if not all_terms:
            logger.info("No jargon terms to sync to vector store")
            return

        documents = []
        for term_data in all_terms:
            term = term_data.get('term')
            definition = term_data.get('definition', '')
            aliases = term_data.get('aliases', [])

            if not term or not definition:
                continue

            # 用語名 + 類義語 + 定義文の形式で保存
            if aliases:
                aliases_str = ', '.join(aliases)
                content = f"{term} ({aliases_str}): {definition}"
            else:
                content = f"{term}: {definition}"

            doc = Document(
                page_content=content,
                metadata={
                    'type': 'jargon_term',  # ドキュメントと区別するための重要なフィールド
                    'term': term,
                    'aliases': aliases,
                    'collection_name': self.collection_name
                }
            )
            documents.append(doc)

        if documents:
            try:
                vector_store.add_documents(documents)
                logger.info(f"Synced {len(documents)} jargon terms to vector store for collection '{self.collection_name}'")
            except Exception as e:
                logger.error(f"Failed to sync jargon terms to vector store: {e}")
        else:
            logger.warning("No valid jargon term documents to sync")

    def get_all_terms(self) -> List[Dict[str, Any]]:
        """全ての用語を取得（現在のコレクション内のみ）"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(
                    text(f"SELECT * FROM {self.table_name} WHERE collection_name = :collection_name ORDER BY term"),
                    {"collection_name": self.collection_name}
                ).fetchall()
                return [dict(row._mapping) for row in result]
        except Exception as e:
            logger.error(f"Error getting all terms: {e}")
            return []


# ========== TermExtractor ==========
class TermExtractor:
    """LLMベースのシンプルな専門用語抽出クラス"""

    def __init__(self, config, llm, embeddings, vector_store, pg_url, jargon_table_name, collection_name="documents"):
        self.config = config
        self.llm = llm
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.pg_url = pg_url
        self.jargon_table_name = jargon_table_name
        self.collection_name = collection_name

        # データベース接続（接続プール設定を最適化）
        if pg_url:
            self.engine = create_engine(
                pg_url,
                pool_size=20,           # 基本プールサイズを20に増加
                max_overflow=30,        # 追加で30接続（合計50接続まで）
                pool_timeout=60,        # タイムアウトを60秒に延長
                pool_pre_ping=True,     # 接続の有効性を事前確認
                pool_recycle=3600       # 1時間で接続をリサイクル
            )
            logger.info(f"Database engine created with pool_size=20, max_overflow=30")

            # テーブルスキーマの初期化（collection_name列の追加）
            self._init_jargon_table()
        else:
            self.engine = None

        # ハイブリッドRetrieverの初期化（ベクトル + キーワード検索）
        self.hybrid_retriever = None
        if vector_store and pg_url and self.engine:
            try:
                from src.rag.retriever import JapaneseHybridRetriever
                from src.rag.text_processor import JapaneseTextProcessor

                self.hybrid_retriever = JapaneseHybridRetriever(
                    vector_store=vector_store,
                    connection_string=pg_url,
                    config_params=config,
                    text_processor=JapaneseTextProcessor(),
                    search_type="hybrid",
                    engine=self.engine  # 最適化されたengineを共有
                )
                logger.info("Hybrid retriever initialized for term extraction")
            except Exception as e:
                logger.warning(f"Hybrid retriever initialization failed, falling back to vector-only search: {e}")

        # プロンプトとパーサーの初期化
        self._init_prompts()
        # JSONパース補正付きパーサー（LLMが崩れたJSONを返す場合の保険）
        self.json_parser = self._build_fixing_parser(ExtractedTermList)
        self.term_group_parser = self._build_fixing_parser(SynonymGroupResponse)
        self.synonym_map_parser = self._build_fixing_parser(SynonymMapResponse)
        # LangChainの出力がAIMessageになる場合に備えて、contentを取り出すユーティリティ
        from langchain_core.runnables import RunnableLambda
        self._to_text = RunnableLambda(lambda x: x.content if hasattr(x, "content") else x)
        # 構造化出力対応LLM（response_format=json_schema）が利用可能なら差し替える
        self._init_structured_llms()

        # PDF プロセッサー（必要に応じて）
        self.pdf_processor = None
        try:
            from src.rag.pdf_processors import AzureDocumentIntelligenceProcessor
            self.pdf_processor = AzureDocumentIntelligenceProcessor(config)
        except Exception as e:
            logger.warning(f"PDF processor initialization failed: {e}")

    def _init_jargon_table(self):
        """専門用語辞書テーブルの初期化（collection_name列とUNIQUE制約のマイグレーション）"""
        if not self.engine:
            return

        try:
            with self.engine.connect() as conn:
                # Add collection_name column if it doesn't exist
                conn.execute(text(f"""
                    DO $$
                    BEGIN
                        IF NOT EXISTS (
                            SELECT 1 FROM information_schema.columns
                            WHERE table_name = '{self.jargon_table_name}' AND column_name = 'collection_name'
                        ) THEN
                            ALTER TABLE {self.jargon_table_name} ADD COLUMN collection_name TEXT NOT NULL DEFAULT 'documents';
                        END IF;
                    END $$;
                """))

                # Update UNIQUE constraint from (term) to (term, collection_name)
                conn.execute(text(f"""
                    DO $$
                    BEGIN
                        -- Drop old UNIQUE constraint on (term) if it exists
                        IF EXISTS (
                            SELECT 1 FROM pg_constraint
                            WHERE conname = '{self.jargon_table_name}_term_key'
                        ) THEN
                            ALTER TABLE {self.jargon_table_name} DROP CONSTRAINT {self.jargon_table_name}_term_key;
                        END IF;

                        -- Add new UNIQUE constraint on (term, collection_name) if it doesn't exist
                        IF NOT EXISTS (
                            SELECT 1 FROM pg_constraint
                            WHERE conname = '{self.jargon_table_name}_term_collection_name_key'
                        ) THEN
                            ALTER TABLE {self.jargon_table_name} ADD CONSTRAINT {self.jargon_table_name}_term_collection_name_key UNIQUE (term, collection_name);
                        END IF;
                    END $$;
                """))

                conn.commit()
                logger.info(f"TermExtractor: jargon table schema initialized with collection_name support and updated UNIQUE constraint")
        except Exception as e:
            logger.warning(f"TermExtractor: Failed to initialize jargon table schema: {e}")

    def _init_prompts(self):
        """プロンプトの初期化"""
        from .prompts import (
            get_candidate_extraction_prompt,
            get_technical_term_filter_prompt,
            get_term_synonym_grouping_prompt,
            get_synonym_detection_prompt
        )

        # プロンプトを prompts.py から取得
        self.candidate_extraction_prompt = get_candidate_extraction_prompt()
        self.technical_term_filter_prompt = get_technical_term_filter_prompt()
        self.term_synonym_grouping_prompt = get_term_synonym_grouping_prompt()
        self.synonym_prompt = get_synonym_detection_prompt()

    def _build_fixing_parser(self, pydantic_obj):
        """LLMのJSON崩れを補正するパーサーを生成"""
        base = JsonOutputParser(pydantic_object=pydantic_obj)
        return OutputFixingParser.from_llm(
            llm=self.llm,
            parser=base,
            max_retries=2
        )

    def _init_structured_llms(self):
        """LLMがjson_schemaレスポンスをサポートしていれば、構造化版を用意"""
        self._structured_llms = {}

        def _try_bind(name: str, model) -> None:
            try:
                schema = model.model_json_schema()
                bound = self.llm.bind(
                    response_format={
                        "type": "json_schema",
                        "json_schema": {"name": name, "schema": schema}
                    }
                )
                self._structured_llms[name] = bound
                logger.info(f"Structured output bound for {name}")
            except Exception as e:
                logger.debug(f"Structured output not available for {name}: {e}")

        _try_bind("candidate", ExtractedTermList)
        _try_bind("filter", FilteredTermResult)
        _try_bind("term_group", SynonymGroupResponse)
        _try_bind("synonym_map", SynonymMapResponse)

    async def extract_from_documents(self, file_paths: List[Path], output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """複数の文書から専門用語を抽出"""
        all_text = []

        for file_path in file_paths:
            try:
                text = self._load_file(file_path)
                if text:
                    all_text.append(text)
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")

        if not all_text:
            logger.error("No text extracted from files")
            return {"terms": [], "candidates": []}

        # 全テキストを結合
        combined_text = "\n\n".join(all_text)

        # 用語抽出（2段階処理）
        terms = await self._extract_terms(combined_text, output_dir=output_dir)

        # 候補プールを抽出（all_candidatesフィールドから）
        all_candidates = []
        if terms and len(terms) > 0 and "all_candidates" in terms[0]:
            # 最初の要素から候補リストを取得（すべて同じ）
            all_candidates = [{"term": c} for c in terms[0]["all_candidates"]]

            # クリーンアップ（all_candidatesフィールドを削除）
            for term in terms:
                term.pop("all_candidates", None)

        # 辞書形式で返す
        return {
            "terms": terms,           # 専門用語
            "candidates": all_candidates  # 全候補（類義語検出に使用済み）
        }

    def _load_file(self, file_path: Path) -> str:
        """ファイルを読み込んでテキストを返す"""
        suffix = file_path.suffix.lower()

        if suffix == '.pdf' and self.pdf_processor:
            # PDFはAzure Document Intelligenceで処理
            try:
                parsed = self.pdf_processor.parse_pdf(str(file_path))
                texts = []
                for text, _ in parsed.get("texts", []):
                    if text and text.strip():
                        texts.append(text)
                return "\n".join(texts)
            except Exception as e:
                logger.error(f"PDF processing failed for {file_path}: {e}")
                return ""
        elif suffix in ['.txt', '.md']:
            # テキストファイル
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            logger.warning(f"Unsupported file type: {suffix}")
            return ""

    async def _extract_terms(self, text: str, output_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
        """テキストから専門用語を抽出（LangGraph版を使用）"""
        # 設定で新実装が有効な場合はLangGraph版を使用
        use_langgraph = getattr(self.config, 'use_langgraph_extraction', True)

        if use_langgraph:
            return await self._extract_terms_with_langgraph(text, output_dir)
        else:
            return await self._extract_terms_legacy(text, output_dir)

    async def _extract_terms_legacy(self, text: str, output_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
        """テキストから専門用語を抽出（従来の2段階処理）"""
        import asyncio
        from tqdm import tqdm

        chunk_size = getattr(self.config, 'llm_extraction_chunk_size', 3000)

        # テキストをチャンクに分割
        chunks = self._split_text(text, chunk_size)

        # ========== 第1段階: 緩い候補抽出（並列処理） ==========
        logger.info(f"Stage 1: Extracting candidates from {len(chunks)} chunks (parallel)")
        print(f"\n[Stage 1] チャンクから候補抽出中 ({len(chunks)}個のチャンク)...")

        all_candidates = []
        chain = self.candidate_extraction_prompt | self.llm | self._to_text | self.json_parser

        # Windows select()の制限回避: 同時実行数を制限（512ファイルディスクリプタ上限）
        max_concurrent = getattr(self.config, 'max_concurrent_llm_calls', 20)
        semaphore = asyncio.Semaphore(max_concurrent)

        async def extract_from_chunk(i: int, chunk: str):
            """1つのチャンクから候補を抽出"""
            async with semaphore:  # 同時実行数を制限
                try:
                    # Trace only the first chunk to avoid excessive LangSmith requests
                    from langchain_core.runnables import RunnableConfig
                    config = RunnableConfig(
                        run_name=f"[TermExtraction] Stage1-CandidateExtraction",
                        tags=["term_extraction", "stage1", "sample"],
                        metadata={
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                            "is_sample": True
                        }
                    ) if i == 0 else None

                    result = await chain.ainvoke({
                        "text": chunk,
                        "format_instructions": self.json_parser.get_format_instructions()
                    }, config=config)

                    # 結果の処理
                    if hasattr(result, "terms"):
                        candidates = result.terms
                    elif isinstance(result, dict):
                        candidates = result.get("terms", [])
                    else:
                        candidates = []

                    # 辞書形式に変換
                    chunk_candidates = []
                    for candidate in candidates:
                        if hasattr(candidate, "dict"):
                            cand_dict = candidate.dict()
                        elif isinstance(candidate, dict):
                            cand_dict = candidate
                        else:
                            continue

                        # headwordフィールドに変換
                        if "term" in cand_dict:
                            cand_dict["headword"] = cand_dict.pop("term")

                        chunk_candidates.append(cand_dict)

                    return chunk_candidates

                except Exception as e:
                    logger.error(f"Error in candidate extraction for chunk {i+1}: {e}")
                    return []

        # 並列実行（プログレスバー付き）
        tasks = [extract_from_chunk(i, chunk) for i, chunk in enumerate(chunks)]

        with tqdm(total=len(tasks), desc="候補抽出", unit="chunk", ncols=80) as pbar:
            for coro in asyncio.as_completed(tasks):
                chunk_candidates = await coro
                all_candidates.extend(chunk_candidates)
                pbar.update(1)

        # 候補の重複除去
        unique_candidates = self._merge_duplicates(all_candidates)
        logger.info(f"Stage 1 complete: {len(unique_candidates)} unique candidates extracted")
        print(f"[OK] {len(unique_candidates)}個の候補を抽出しました\n")

        # Stage 1結果をファイルに出力
        if output_dir:
            stage1_file = output_dir / "stage1_candidates.json"
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(stage1_file, "w", encoding="utf-8") as f:
                json.dump({"candidates": unique_candidates}, f, ensure_ascii=False, indent=2)
            logger.info(f"Stage 1 results saved to {stage1_file}")
            print(f"[OK] Stage 1 結果を保存: {stage1_file}\n")

        # ========== 第2段階: 専門用語の選別 ==========
        technical_terms = []

        if unique_candidates:
            logger.info("Stage 2: Filtering technical terms from candidates")
            print(f"[Stage 2] 専門用語を選別中...")
            technical_terms = await self._filter_technical_terms(unique_candidates, output_dir=output_dir)
            logger.info(f"Stage 2 complete: {len(technical_terms)} technical terms selected")
            print(f"[OK] {len(technical_terms)}個の専門用語を選別しました\n")

            # Stage 2結果をファイルに出力
            if output_dir:
                stage2_file = output_dir / "stage2_technical_terms.json"
                with open(stage2_file, "w", encoding="utf-8") as f:
                    json.dump({"technical_terms": technical_terms}, f, ensure_ascii=False, indent=2)
                logger.info(f"Stage 2 results saved to {stage2_file}")
                print(f"[OK] Stage 2 結果を保存: {stage2_file}\n")

        # ========== 第3段階: RAGベースの定義生成（専門用語のみ） ==========
        if technical_terms and self.vector_store:
            logger.info("Stage 3: Generating definitions using RAG for technical terms")
            print(f"[Stage 3] 定義を生成中 ({len(technical_terms)}個の専門用語)...")
            technical_terms = await self._generate_definitions_with_rag(technical_terms)
            print(f"[OK] 定義生成が完了しました\n")
        else:
            logger.info("Stage 3: Skipping definition generation (no vector store)")
            print(f"[Stage 3] ベクトルストアがないため定義生成をスキップします\n")

        # ========== 第4段階: 類義語検出（2段階処理） ==========
        if technical_terms:
            # Stage 4a: 専門用語間の類義語判定 + 代表語への集約
            logger.info("Stage 4a: Detecting synonyms among technical terms and merging")
            representative_terms = await self._detect_and_merge_term_synonyms(technical_terms)

            # Stage 4b: 代表語と一般語候補の類義語判定
            if representative_terms and unique_candidates:
                logger.info("Stage 4b: Detecting synonyms between representative terms and general candidates")
                representative_terms = await self._detect_synonyms_with_candidates(
                    representative_terms, unique_candidates
                )

            technical_terms = representative_terms

        # 候補プールも返却用データに含める
        for term in technical_terms:
            term["all_candidates"] = [c["headword"] for c in unique_candidates]

        return technical_terms

    def _split_text(self, text: str, chunk_size: int) -> List[str]:
        """テキストをチャンクに分割"""
        chunks = []

        # 改行で分割してから再結合
        lines = text.split('\n')
        current_chunk = []
        current_size = 0

        for line in lines:
            line_size = len(line)

            if current_size + line_size > chunk_size and current_chunk:
                # 現在のチャンクを保存
                chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
                current_size = line_size
            else:
                current_chunk.append(line)
                current_size += line_size + 1  # +1 for newline

        # 最後のチャンクを追加
        if current_chunk:
            chunks.append('\n'.join(current_chunk))

        return chunks

    def _merge_duplicates(self, terms: List[Dict]) -> List[Dict]:
        """重複する用語をマージ"""
        merged = {}

        for term in terms:
            headword = term.get("headword", "").strip()
            if not headword:
                continue

            # 正規化（小文字化）
            key = headword.lower()

            if key in merged:
                # 既存の用語とマージ
                existing = merged[key]

                # 類義語をマージ
                existing_synonyms = set(existing.get("synonyms", []))
                new_synonyms = set(term.get("synonyms", []))
                existing["synonyms"] = list(existing_synonyms | new_synonyms)

                # 信頼度は平均を取る
                existing["confidence"] = (existing.get("confidence", 1.0) + term.get("confidence", 1.0)) / 2

                # brief_definitionを連結
                existing_def = existing.get("brief_definition", "")
                new_def = term.get("brief_definition", "")
                if existing_def and new_def and new_def not in existing_def:
                    existing["brief_definition"] = f"{existing_def} {new_def}"
                elif not existing_def and new_def:
                    existing["brief_definition"] = new_def
            else:
                # 新規追加
                merged[key] = term.copy()
                merged[key]["headword"] = headword  # 元のケースを保持

        return list(merged.values())

    def _clean_synonyms(self, term: Dict) -> Dict:
        """類義語リストからheadword自身や空文字を除去し、重複を排除"""
        head = term.get("headword", "")
        head_lower = head.lower().strip() if head else ""
        cleaned = []
        seen = set()
        for syn in term.get("synonyms", []):
            if not syn:
                continue
            syn_str = str(syn).strip()
            if not syn_str:
                continue
            syn_lower = syn_str.lower()
            if syn_lower == head_lower:
                continue
            if syn_lower in seen:
                continue
            seen.add(syn_lower)
            cleaned.append(syn_str)
        term["synonyms"] = cleaned
        return term

    async def _filter_technical_terms(self, candidates: List[Dict], output_dir: Optional[Path] = None, return_rejected: bool = False) -> List[Dict]:
        """候補から専門用語を選別（第2段階）- バッチ処理対応

        return_rejected=Trueの場合、(selected, rejected)のタプルを返す
        """
        if not candidates:
            return ([], []) if return_rejected else []

        try:
            # バッチサイズを取得
            batch_size = getattr(self.config, 'stage2_batch_size', 50)

            # FilteredTermResult用のパーサーを使用（補正付き）
            filter_parser = self._build_fixing_parser(FilteredTermResult)
            llm_for_stage2 = self._structured_llms.get("filter", self.llm)

            # 候補をバッチに分割
            batches = [candidates[i:i + batch_size] for i in range(0, len(candidates), batch_size)]
            total_batches = len(batches)

            logger.info(f"Processing {len(candidates)} candidates in {total_batches} batches (batch size: {batch_size})")

            # 各バッチを処理
            all_selected = []
            all_rejected = []

            async def process_batch(batch_idx: int, batch: List[Dict]) -> tuple:
                """単一バッチを処理"""
                candidates_json = json.dumps(batch, ensure_ascii=False, indent=2)

                # Trace only the first batch to avoid excessive LangSmith requests
                from langchain_core.runnables import RunnableConfig
                config = RunnableConfig(
                    run_name=f"[TermExtraction] Stage2-TechnicalTermFilter",
                    tags=["term_extraction", "stage2", "sample"],
                    metadata={
                        "batch_index": batch_idx,
                        "total_batches": total_batches,
                        "batch_size": len(batch),
                        "is_sample": True
                    }
                ) if batch_idx == 0 else None

                try:
                    chain = self.technical_term_filter_prompt | llm_for_stage2 | self._to_text | filter_parser
                    result = await chain.ainvoke({
                        "candidates_json": candidates_json,
                        "format_instructions": filter_parser.get_format_instructions()
                    }, config=config)

                    # 結果の処理
                    selected_terms = []
                    rejected_terms = []

                    if hasattr(result, "selected"):
                        selected_terms = result.selected
                        rejected_terms = result.rejected
                    elif isinstance(result, dict):
                        selected_terms = result.get("selected", [])
                        rejected_terms = result.get("rejected", [])

                    return (selected_terms, rejected_terms)

                except Exception as e:
                    logger.error(f"Stage2 batch {batch_idx} parsing failed: {e}")
                    print(f"[WARN] Stage2 バッチ {batch_idx+1} のパースに失敗。バッチ全体を 'selected' として扱います。")
                    # フォールバック: 元バッチをそのまま選定扱いで返す
                    return (batch, [])

            # バッチを並列処理（プログレスバー付き）
            tasks = [process_batch(i, batch) for i, batch in enumerate(batches)]

            with tqdm(total=len(tasks), desc="専門用語選別", unit="batch", ncols=80) as pbar:
                for coro in asyncio.as_completed(tasks):
                    selected, rejected = await coro
                    all_selected.extend(selected)
                    all_rejected.extend(rejected)
                    pbar.update(1)

            # 辞書形式に変換（selected）
            output = []
            for term in all_selected:
                if hasattr(term, "dict"):
                    term_dict = term.dict()
                elif isinstance(term, dict):
                    term_dict = term
                else:
                    continue

                # headwordフィールドに変換
                if "term" in term_dict:
                    term_dict["headword"] = term_dict.pop("term")

                output.append(term_dict)

            # 辞書形式に変換（rejected）
            rejected_output = []
            for term in all_rejected:
                if hasattr(term, "dict"):
                    term_dict = term.dict()
                elif isinstance(term, dict):
                    term_dict = term
                else:
                    continue

                # headwordフィールドに変換
                if "term" in term_dict:
                    term_dict["headword"] = term_dict.pop("term")

                rejected_output.append(term_dict)

            # Stage 2除外用語をファイルに出力
            if output_dir and rejected_output:
                stage2_rejected_file = output_dir / "stage2_rejected.json"
                with open(stage2_rejected_file, "w", encoding="utf-8") as f:
                    json.dump({"rejected_terms": rejected_output}, f, ensure_ascii=False, indent=2)
                logger.info(f"Stage 2 rejected terms saved to {stage2_rejected_file}")
                print(f"[OK] Stage 2 除外用語を保存: {stage2_rejected_file} ({len(rejected_output)}個)\n")

            # 検証: selected + rejected = total candidates
            processed_count = len(output) + len(rejected_output)
            if processed_count != len(candidates):
                logger.warning(f"Validation failed: {processed_count} processed (selected: {len(output)}, rejected: {len(rejected_output)}) != {len(candidates)} input candidates. Missing: {len(candidates) - processed_count}")
                print(f"[WARN] 警告: 処理済み用語数 ({processed_count}) が入力候補数 ({len(candidates)}) と一致しません。未処理: {len(candidates) - processed_count}個\n")
                # フォールバック: 未処理候補を選定扱いで追加
                missing = len(candidates) - processed_count
                if missing > 0:
                    # 既に処理済みの headword を控える
                    done = {t.get("headword") for t in output} | {t.get("headword") for t in rejected_output}
                    for cand in candidates:
                        hw = cand.get("headword")
                        if hw not in done:
                            output.append(cand)
                            done.add(hw)
                            if len(output) + len(rejected_output) == len(candidates):
                                break
                    print(f"[OK] 未処理 {missing} 件をフォールバックで追加しました\n")

            logger.info(f"Technical term filtering: {len(output)} selected, {len(rejected_output)} rejected from {len(candidates)} candidates")
            if return_rejected:
                return output, rejected_output
            return output

        except Exception as e:
            logger.error(f"Technical term filtering failed: {e}")
            if return_rejected:
                return candidates, []
            return candidates  # エラー時は全候補を返す

    async def _generate_definitions_with_rag(self, technical_terms: List[Dict]) -> List[Dict]:
        """RAGを使用して専門用語の定義を生成（並列処理 + ハイブリッド検索）"""
        import asyncio
        from tqdm import tqdm

        if not self.llm:
            logger.warning("LLM not available for definition generation")
            return technical_terms

        # ハイブリッド検索が使用可能かチェック
        use_hybrid = self.hybrid_retriever is not None
        if not use_hybrid and not self.vector_store:
            logger.warning("Neither hybrid retriever nor vector store available for definition generation")
            return technical_terms

        # プロンプトを取得
        from .prompts import get_definition_generation_prompt
        from langchain_core.output_parsers import StrOutputParser

        prompt = get_definition_generation_prompt()
        chain = prompt | self.llm | self._to_text | StrOutputParser()

        # セマフォで並列数を制限（DB接続負荷対策）
        semaphore = asyncio.Semaphore(10)

        async def generate_definition(term: Dict) -> Dict:
            """1つの用語の定義を生成（リトライ機構付き）"""
            async with semaphore:  # 同時実行数を最大10に制限
                headword = term.get("headword", "")
                if not headword:
                    return term

                # リトライ設定
                max_retries = 3
                retry_delay = 2  # 秒

                for attempt in range(max_retries):
                    try:
                        docs = []
                        if self.engine and self.embeddings:
                            docs = similarity_search_without_jargon(
                                engine=self.engine, embeddings=self.embeddings,
                                query=headword, collection_name=self.collection_name, k=5,
                            )
                            logger.info(f"[Stage3 RAG] '{headword}': {len(docs)} document chunks")
                        else:
                            logger.warning(f"[Stage3 RAG] '{headword}': engine/embeddings not available")

                        if docs:
                            # コンテキストを結合（最大3000文字）
                            context = "\n\n".join([doc.page_content for doc in docs])[:3000]

                            # 定義生成
                            definition = await chain.ainvoke({
                                "term": headword,
                                "context": context
                            })

                            term["definition"] = definition.strip()
                        else:
                            # コンテキストが見つからない場合は簡易定義
                            term["definition"] = f"{headword}（関連文書が見つかりません）"
                            logger.warning(f"No context found for: {headword}")

                        # 成功したらループを抜ける
                        break

                    except Exception as e:
                        if attempt < max_retries - 1:
                            # リトライ可能な場合
                            logger.warning(f"Retry {attempt + 1}/{max_retries} for '{headword}': {e}")
                            await asyncio.sleep(retry_delay)
                        else:
                            # 最終試行でも失敗
                            logger.error(f"Failed to generate definition for '{headword}' after {max_retries} attempts: {e}")
                            term["definition"] = f"{headword}（定義生成エラー）"

                return term

        # 並列実行（プログレスバー付き）
        # Wrap only the first term with RunnableConfig for tracing
        async def generate_with_tracing(idx: int, term: Dict) -> Dict:
            """定義生成をトレース付きでラップ（最初の1つのみ）"""
            if idx == 0:
                # Trace only the first term to avoid excessive LangSmith requests
                from langchain_core.runnables import RunnableConfig
                config = RunnableConfig(
                    run_name=f"[TermExtraction] Stage3-DefinitionGeneration",
                    tags=["term_extraction", "stage3", "sample"],
                    metadata={
                        "term_index": idx,
                        "total_terms": len(technical_terms),
                        "headword": term.get("headword", ""),
                        "is_sample": True
                    }
                )
                # Override chain.ainvoke to use config for this call only
                headword = term.get("headword", "")
                if headword:
                    docs = []
                    if self.engine and self.embeddings:
                        docs = similarity_search_without_jargon(
                            engine=self.engine, embeddings=self.embeddings,
                            query=headword, collection_name=self.collection_name, k=5,
                        )

                    if docs:
                        context = "\n\n".join([doc.page_content for doc in docs])[:3000]
                        definition = await chain.ainvoke({
                            "term": headword,
                            "context": context
                        }, config=config)
                        term["definition"] = definition.strip()
                    else:
                        term["definition"] = f"{headword}（関連文書が見つかりません）"
                return term
            else:
                return await generate_definition(term)

        tasks = [generate_with_tracing(i, term) for i, term in enumerate(technical_terms)]

        results = []
        with tqdm(total=len(tasks), desc="定義生成", unit="用語", ncols=80) as pbar:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                results.append(result)
                pbar.update(1)

        # 元の順序を保持するため、headwordでソート
        headword_to_result = {r.get("headword"): r for r in results}
        ordered_results = [headword_to_result[t.get("headword")] for t in technical_terms if t.get("headword") in headword_to_result]

        return ordered_results

    async def _detect_and_merge_term_synonyms(self, technical_terms: List[Dict]) -> List[Dict]:
        """専門用語間の類義語を判定し、代表語に集約（Stage 4a）"""
        if not technical_terms:
            return technical_terms

        try:
            logger.info(f"Detecting synonyms among {len(technical_terms)} technical terms")
            print(f"[Stage 4a] 専門用語間の類義語を判定中 ({len(technical_terms)}個)...")

            # 用語数が多い場合はバッチ処理
            batch_size = 50  # 50個ずつ処理
            all_representative_terms = []

            if len(technical_terms) > batch_size:
                # バッチ処理
                print(f"   用語数が多いため、{batch_size}個ずつバッチ処理します...")
                batches = [technical_terms[i:i + batch_size] for i in range(0, len(technical_terms), batch_size)]

                for batch_idx, batch in enumerate(batches, 1):
                    print(f"   バッチ {batch_idx}/{len(batches)} を処理中...")
                    batch_result = await self._group_terms_batch(batch)
                    all_representative_terms.extend(batch_result)

                logger.info(f"Grouped {len(technical_terms)} terms into {len(all_representative_terms)} representative terms (batched)")
                print(f"[OK] {len(technical_terms)}個の専門用語を{len(all_representative_terms)}個の代表語に集約しました\n")
                return all_representative_terms

            # 50個以下の場合は一括処理
            return await self._group_terms_batch(technical_terms)

        except Exception as e:
            logger.error(f"Term synonym detection failed: {e}")
            print(f"[WARN] 専門用語間の類義語判定でエラーが発生しました: {e}\n")
            return technical_terms

    async def _group_terms_batch(self, terms: List[Dict]) -> List[Dict]:
        """用語のバッチをグループ化"""
        try:
            # 専門用語をJSON化
            technical_terms_json = json.dumps(
                [{"term": t.get("headword"), "definition": t.get("definition", "")} for t in terms],
                ensure_ascii=False,
                indent=2
            )

            # LLMで専門用語のグループ化
            # Always trace Stage4 since it's only called once
            from langchain_core.runnables import RunnableConfig
            config = RunnableConfig(
                run_name=f"[TermExtraction] Stage4-SynonymGrouping",
                tags=["term_extraction", "stage4"],
                metadata={
                    "total_terms": len(terms),
                    "description": "Groups technical terms into synonym clusters"
                }
            )

            llm_for_stage4a = self._structured_llms.get("term_group", self.llm)
            chain = self.term_synonym_grouping_prompt | llm_for_stage4a | self._to_text
            result = await chain.ainvoke({
                "technical_terms_json": technical_terms_json
            }, config=config)

            # 結果のパース
            if hasattr(result, "content"):
                result_text = result.content
            else:
                result_text = str(result)

            try:
                parsed = self.term_group_parser.parse(result_text)
                groups = getattr(parsed, "groups", []) or []

                representative_terms = []
                for group in groups:
                    g = group if isinstance(group, dict) else group.dict()
                    term_dict = {
                        "headword": g.get("headword", ""),
                        "synonyms": g.get("synonyms", []),
                        "definition": g.get("definition", ""),
                        "domain": g.get("domain", "一般")
                    }
                    representative_terms.append(term_dict)

                logger.info(f"Grouped {len(terms)} terms into {len(representative_terms)} representative terms")
                representative_terms = [self._clean_synonyms(t) for t in representative_terms]
                if not representative_terms:
                    logger.warning("Term grouping returned empty list; keeping original terms")
                    return terms
                return representative_terms

            except Exception as e:
                logger.error(f"Failed to parse term grouping result: {e}")
                logger.error(f"Result text (first 1000 chars): {result_text[:1000]}")
                logger.error(f"Result text (last 500 chars): {result_text[-500:]}")
                print(f"[WARN] バッチのグループ化に失敗しました。\n")
                print(f"   JSONパースエラー: {e}\n")
                print(f"   LLM出力の最初: {result_text[:200]}...\n")
                return terms

        except Exception as e:
            logger.error(f"Batch grouping failed: {e}")
            return terms

    async def _detect_and_merge_term_synonyms_hdbscan(self, technical_terms: List[Dict]) -> List[Dict]:
        """HDBSCAN + UMAP によるドメイン分類ベースの類義語グループ化（Stage 4a代替）"""
        if not technical_terms:
            return technical_terms

        # クラスタリングライブラリの可用性チェック
        if not CLUSTERING_AVAILABLE or TermClusteringAnalyzer is None:
            logger.warning("Clustering libraries not available, falling back to LLM method")
            print("[WARN] クラスタリングライブラリが利用できません。LLM方式にフォールバックします。\n")
            return await self._detect_and_merge_term_synonyms(technical_terms)

        min_terms = getattr(self.config, 'domain_min_terms_for_clustering', 10)

        # 用語数が少なければLLM方式にフォールバック
        if len(technical_terms) < min_terms:
            logger.info(f"Too few terms ({len(technical_terms)} < {min_terms}), falling back to LLM method")
            print(f"[INFO] 用語数が少ないため ({len(technical_terms)}個 < {min_terms}個)、LLM方式を使用します。\n")
            return await self._detect_and_merge_term_synonyms(technical_terms)

        try:
            logger.info(f"Using HDBSCAN clustering for {len(technical_terms)} terms")
            print(f"[Stage 4a] HDBSCAN + UMAPによるドメイン分類を実行中 ({len(technical_terms)}個)...")

            # TermClusteringAnalyzer を初期化
            analyzer = TermClusteringAnalyzer(
                connection_string=self.pg_url,
                min_terms=min_terms,
                embeddings=self.embeddings,
                llm=self.llm,
                config=self.config
            )

            # 専門用語を形式変換
            specialized_terms = [
                {
                    "term": t.get("headword"),
                    "definition": t.get("definition", ""),
                    "related_terms": t.get("synonyms", []),
                    "text": f"{t.get('headword')}: {t.get('definition', '')}"
                }
                for t in technical_terms
            ]

            # クラスタリング実行
            use_llm_naming = getattr(self.config, 'enable_domain_cluster_naming', True)
            result = await analyzer.extract_semantic_synonyms_hybrid(
                specialized_terms=specialized_terms,
                candidate_terms=[],  # Stage 4bで処理
                max_synonyms=5,
                use_llm_naming=use_llm_naming
            )

            # 結果をterm_extraction形式に変換
            cluster_names = result.get('cluster_names', {})
            cluster_mapping = result.get('clusters', {})
            synonyms_dict = result.get('synonyms', {})

            # 元のtechnical_termsにドメイン情報と類義語を追加
            for term in technical_terms:
                headword = term.get("headword")
                cluster_id = cluster_mapping.get(headword, -1)

                # ドメイン名設定
                if cluster_id >= 0 and cluster_id in cluster_names:
                    term["domain"] = cluster_names[cluster_id]
                else:
                    term["domain"] = "未分類"

                # 類義語追加
                if headword in synonyms_dict:
                    existing = set(term.get("synonyms", []))
                    for syn in synonyms_dict[headword]:
                        existing.add(syn["term"])
                    term["synonyms"] = list(existing)

            n_clusters = len(set(cluster_names.values())) if cluster_names else 0
            logger.info(f"HDBSCAN clustering complete: {n_clusters} clusters detected")
            print(f"[OK] HDBSCANクラスタリング完了: {n_clusters}個のクラスタを検出\n")

            return technical_terms

        except Exception as e:
            logger.error(f"HDBSCAN clustering failed: {e}")
            print(f"[WARN] HDBSCANクラスタリングでエラーが発生しました: {e}\n")
            print("[INFO] LLM方式にフォールバックします。\n")
            return await self._detect_and_merge_term_synonyms(technical_terms)

    async def _detect_and_merge_term_synonyms_hybrid(self, technical_terms: List[Dict]) -> List[Dict]:
        """ハイブリッド方式: 用語数に応じてLLMまたはHDBSCANを自動選択（Stage 4a代替）"""
        if not technical_terms:
            return technical_terms

        threshold = getattr(self.config, 'domain_min_terms_for_clustering', 10)

        if len(technical_terms) < threshold:
            logger.info(f"Using LLM method (terms: {len(technical_terms)} < threshold: {threshold})")
            print(f"[Stage 4a] LLM方式を使用（用語数: {len(technical_terms)} < 閾値: {threshold}）")
            return await self._detect_and_merge_term_synonyms(technical_terms)
        else:
            logger.info(f"Using HDBSCAN method (terms: {len(technical_terms)} >= threshold: {threshold})")
            print(f"[Stage 4a] HDBSCAN方式を使用（用語数: {len(technical_terms)} >= 閾値: {threshold}）")
            return await self._detect_and_merge_term_synonyms_hdbscan(technical_terms)

    async def _detect_synonyms_with_candidates(self, representative_terms: List[Dict], all_candidates: List[Dict]) -> List[Dict]:
        """類義語の検出（代表語と一般語候補、Stage 4b）"""
        if not representative_terms or not all_candidates:
            return representative_terms

        try:
            logger.info(f"Detecting synonyms between {len(representative_terms)} representative terms and candidates")
            print(f"[Stage 4b] 代表語と一般語の類義語を判定中...")

            # 代表語のheadwordリストを作成（専門用語として既に選ばれたものを除外するため）
            representative_headwords = {t.get("headword") for t in representative_terms}

            # 一般語候補のみを抽出（専門用語を除外）
            general_candidates = [c for c in all_candidates if c.get("headword") not in representative_headwords]

            if not general_candidates:
                logger.info("No general candidates found, skipping candidate synonym detection")
                print(f"[OK] 一般語候補がないため、この段階をスキップします\n")
                return representative_terms

            # 代表語と候補をJSON化
            representative_terms_json = json.dumps(
                [{"term": t.get("headword"), "definition": t.get("definition", "")} for t in representative_terms],
                ensure_ascii=False,
                indent=2
            )
            candidates_json = json.dumps(
                [{"term": c.get("headword"), "definition": c.get("definition", "")} for c in general_candidates],
                ensure_ascii=False,
                indent=2
            )

            # 類義語検出
            llm_for_stage4b = self._structured_llms.get("synonym_map", self.llm)
            chain = self.synonym_prompt | llm_for_stage4b | self._to_text
            result = await chain.ainvoke({
                "representative_terms_json": representative_terms_json,
                "candidates_json": candidates_json
            })

            # 結果のパース
            if hasattr(result, "content"):
                result_text = result.content
            else:
                result_text = str(result)

            try:
                parsed = self.synonym_map_parser.parse(result_text)
                if isinstance(parsed, dict):
                    syn_items = parsed.get("items", []) or []
                else:
                    syn_items = getattr(parsed, "items", []) or []
                if callable(syn_items):
                    syn_items = []

                for term in representative_terms:
                    headword = term.get("headword", "")
                    for syn_item in syn_items:
                        item_dict = syn_item if isinstance(syn_item, dict) else syn_item.dict()
                        if item_dict.get("term") == headword:
                            new_synonyms = item_dict.get("synonyms", [])
                            existing_synonyms = set(term.get("synonyms", []))
                            term["synonyms"] = list(existing_synonyms | set(new_synonyms))
                            self._clean_synonyms(term)
                            break

                logger.info(f"Added general term synonyms to {len(representative_terms)} representative terms")
                print(f"[OK] 代表語に一般語の類義語を追加しました\n")

            except Exception as e:
                logger.error(f"Failed to parse synonym detection result: {e}")
                logger.error(f"Result text (first 1000 chars): {result_text[:1000]}")
                logger.error(f"Result text (last 500 chars): {result_text[-500:]}")
                print(f"[WARN] 類義語判定結果のパースに失敗しました\n")
                print(f"   JSONパースエラー: {e}\n")
                print(f"   LLM出力の最初: {result_text[:200]}...\n")

            return representative_terms

        except Exception as e:
            logger.error(f"Synonym detection with candidates failed: {e}")
            print(f"[WARN] 一般語との類義語判定でエラーが発生しました: {e}\n")
            return representative_terms

    def save_to_database(self, terms: List[Dict]) -> int:
        """用語をデータベースに保存"""
        if not self.engine:
            logger.warning("No database connection")
            return 0

        saved_count = 0

        with self.engine.begin() as conn:
            for term in terms:
                try:
                    domain = term.get("domain")
                    related_terms = term.get("related_terms", [])
                    conn.execute(
                        text(f"""
                            INSERT INTO {self.jargon_table_name} (term, definition, domain, aliases, related_terms, collection_name)
                            VALUES (:term, :definition, :domain, :aliases, :related_terms, :collection_name)
                            ON CONFLICT (term, collection_name) DO UPDATE
                            SET definition = EXCLUDED.definition,
                                domain = EXCLUDED.domain,
                                aliases = EXCLUDED.aliases,
                                related_terms = EXCLUDED.related_terms,
                                updated_at = CURRENT_TIMESTAMP
                        """),
                        {
                            "term": term.get("headword"),
                            "definition": term.get("definition", ""),
                            "domain": domain,
                            "aliases": term.get("synonyms", []),
                            "related_terms": related_terms,
                            "collection_name": self.collection_name
                        }
                    )
                    saved_count += 1
                except Exception as e:
                    logger.error(f"Error saving term {term.get('headword')}: {e}")

        logger.info(f"Saved {saved_count} terms to database")
        return saved_count

    def _is_duplicate(self, headword: str, state: TermExtractionState,
                      skip_rejected_headwords: set = None) -> bool:
        """既存候補/専門用語/除外済み用語との重複チェック

        Args:
            skip_rejected_headwords: rejected_termsチェックから除外する用語名のセット
                （同一反復内でRemoveされた用語をMissing処理で再追加可能にするため）
        """
        if not headword:
            return True

        key = headword.lower().strip()

        # 専門用語リストとの重複チェック
        existing_terms = {t.get("headword", "").lower().strip() for t in state["technical_terms"]}
        if key in existing_terms:
            return True

        # 候補リストとの重複チェック
        existing_candidates = {c.get("headword", "").lower().strip() for c in state["candidates"]}
        if key in existing_candidates:
            return True

        # rejected_termsとの重複チェック（Stage2.5で一度却下された用語を再度拾わない）
        rejected = state.get("rejected_terms", [])
        rejected_keys = {r.get("headword", "").lower().strip() for r in rejected}
        if skip_rejected_headwords:
            rejected_keys -= {s.lower().strip() for s in skip_rejected_headwords}
        if key in rejected_keys:
            return True

        return False

    # ========== LangGraph Workflow Nodes ==========

    async def _build_workflow(self) -> StateGraph:
        """LangGraphワークフローを構築"""
        from .prompts import get_self_reflection_prompt, get_term_refinement_prompt

        # ワークフロー定義
        workflow = StateGraph(TermExtractionState)

        # Stage 2.5 ON/OFF設定
        enable_stage25 = getattr(self.config, 'enable_stage25_refinement', True)

        # ノード追加
        workflow.add_node("stage1_extract_candidates", self._node_stage1_extract_candidates)
        workflow.add_node("stage2_initial_filter", self._node_stage2_initial_filter)
        workflow.add_node("stage3_generate_definitions", self._node_stage3_generate_definitions)
        workflow.add_node("stage4_detect_synonyms", self._node_stage4_detect_synonyms)

        # エッジ定義
        workflow.set_entry_point("stage1_extract_candidates")
        workflow.add_edge("stage1_extract_candidates", "stage2_initial_filter")

        if enable_stage25:
            # Stage 2.5 有効時: Stage2 → Stage2.5 → Stage3
            workflow.add_node("stage25_self_reflection", self._node_stage25_self_reflection)
            workflow.add_node("stage25_refine_terms", self._node_stage25_refine_terms)
            workflow.add_edge("stage2_initial_filter", "stage25_self_reflection")
            workflow.add_edge("stage25_self_reflection", "stage25_refine_terms")

            # Stage 2.5 ループの条件分岐
            workflow.add_conditional_edges(
                "stage25_refine_terms",
                self._should_continue_refinement,
                {
                    "continue": "stage25_self_reflection",  # ループバック
                    "finish": "stage3_generate_definitions"  # 次のステージへ
                }
            )
            logger.info("[Workflow] Stage 2.5 Self-Reflection enabled")
        else:
            # Stage 2.5 無効時: Stage2 → Stage3 直接
            workflow.add_edge("stage2_initial_filter", "stage3_generate_definitions")
            logger.info("[Workflow] Stage 2.5 Self-Reflection disabled (skipped)")

        # 固定フロー
        workflow.add_edge("stage3_generate_definitions", "stage4_detect_synonyms")
        workflow.add_edge("stage4_detect_synonyms", END)

        return workflow.compile()

    async def _node_stage1_extract_candidates(self, state: TermExtractionState) -> TermExtractionState:
        """Stage 1: 候補抽出ノード"""
        logger.info("[Stage 1] Extracting candidates...")
        print(f"\n[Stage 1] チャンクから候補抽出中 ({len(state['text_chunks'])}個のチャンク)...")

        all_candidates = []
        llm_for_stage1 = self._structured_llms.get("candidate", self.llm)
        chain = self.candidate_extraction_prompt | llm_for_stage1 | self._to_text | self.json_parser

        # 並列処理（パース失敗時はスキップ）
        async def _run_chunk(i: int, chunk: str):
            try:
                return await chain.ainvoke({
                    "text": chunk,
                    "format_instructions": self.json_parser.get_format_instructions()
                })
            except Exception as e:
                logger.error(f"Stage1 candidate parsing failed for chunk {i}: {e}")
                print(f"[WARN] Stage1 chunk {i+1} のJSONパースに失敗しました。スキップします。")
                return None

        tasks = [_run_chunk(i, chunk) for i, chunk in enumerate(state["text_chunks"])]

        # プログレスバー付き実行
        results = []
        with tqdm(total=len(tasks), desc="候補抽出", unit="chunk", ncols=80) as pbar:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                results.append(result)
                pbar.update(1)

        # 結果を集約
        for result in results:
            if result is None:
                continue
            if hasattr(result, "terms"):
                candidates = result.terms
            elif isinstance(result, dict):
                candidates = result.get("terms", [])
            else:
                continue

            for candidate in candidates:
                if hasattr(candidate, "dict"):
                    cand_dict = candidate.dict()
                elif isinstance(candidate, dict):
                    cand_dict = candidate
                else:
                    continue

                if "term" in cand_dict:
                    cand_dict["headword"] = cand_dict.pop("term")

                all_candidates.append(cand_dict)

        # 重複除去
        state["candidates"] = self._merge_duplicates(all_candidates)
        logger.info(f"Stage 1 complete: {len(state['candidates'])} candidates")
        print(f"[OK] {len(state['candidates'])}個の候補を抽出しました\n")

        # Stage 1結果を保存
        if state.get("output_dir"):
            stage1_file = state["output_dir"] / "stage1_candidates.json"
            state["output_dir"].mkdir(parents=True, exist_ok=True)
            with open(stage1_file, "w", encoding="utf-8") as f:
                json.dump({"candidates": state["candidates"]}, f, ensure_ascii=False, indent=2)
            print(f"[OK] Stage 1 結果を保存: {stage1_file}\n")

        return state

    async def _node_stage2_initial_filter(self, state: TermExtractionState) -> TermExtractionState:
        """Stage 2: 初期選別ノード（確信度ベーストリアージ付き）"""
        logger.info("[Stage 2] Initial filtering...")
        print(f"[Stage 2] 専門用語を選別中...")

        # バッチ処理で選別（rejected も取得）
        selected, rejected = await self._filter_technical_terms(
            state["candidates"], output_dir=state.get("output_dir"), return_rejected=True
        )

        # 確信度で3分割: high→スルーパス, middle→Stage2.5対象, low→即却下
        high_terms = []
        middle_terms = []
        low_terms = []

        for term in selected:
            cl = term.get("confidence_level", "middle")
            if cl == "high":
                high_terms.append(term)
            elif cl == "low":
                low_terms.append(term)
            else:  # middle or unknown
                middle_terms.append(term)

        # low確信度はrejected_termsへ
        if "rejected_terms" not in state:
            state["rejected_terms"] = []
        for term in low_terms:
            state["rejected_terms"].append({
                **term,
                "rejection_reason": "低確信度により自動除外（confidence_level=low）",
            })

        # Stage2で明示的にrejectedされたものも追加
        for term in rejected:
            state["rejected_terms"].append(term)

        # technical_terms = high + middle（highはStage2.5スキップ対象としてマーク）
        for term in high_terms:
            term["skip_reflection"] = True
        state["technical_terms"] = high_terms + middle_terms

        state["refinement_iteration"] = 0
        state["refinement_converged"] = False
        state["previous_list_hash"] = None

        logger.info(f"Stage 2 complete: {len(high_terms)} high, {len(middle_terms)} middle, {len(low_terms)} low (rejected)")
        print(f"[OK] {len(selected)}個選別 → high: {len(high_terms)}個（スルーパス）, middle: {len(middle_terms)}個（反省対象）, low: {len(low_terms)}個（自動除外）\n")

        return state

    async def _node_stage25_self_reflection(self, state: TermExtractionState) -> TermExtractionState:
        """Stage 2.5a: 自己反省ノード（バッチ処理版）"""
        from .prompts import get_self_reflection_prompt
        from langchain_core.output_parsers import JsonOutputParser

        refinement_iteration = state.get('refinement_iteration', 0)
        max_iterations = state.get('max_refinement_iterations', 3)
        logger.info(f"[Stage 2.5] Self-reflection (iteration {refinement_iteration})...")
        print(f"[Stage 2.5] 自己反省中 (反復 {refinement_iteration + 1}/{max_iterations})...")

        # バッチサイズ設定（configから取得、デフォルト100）
        batch_size_terms = getattr(self.config, 'reflection_batch_size_terms', 100)
        batch_size_candidates = getattr(self.config, 'reflection_batch_size_candidates', 50)

        all_technical_terms = state["technical_terms"]
        # high確信度はStage2.5の反省対象から除外
        technical_terms = [t for t in all_technical_terms if not t.get("skip_reflection")]
        high_terms = [t for t in all_technical_terms if t.get("skip_reflection")]
        candidates = state["candidates"]

        reflection_history = state.get("reflection_history", [])

        logger.info(f"[Stage 2.5] Reflecting on {len(technical_terms)} middle terms (skipping {len(high_terms)} high-confidence terms)")
        if high_terms:
            print(f"  （high確信度 {len(high_terms)}個はスキップ、{len(technical_terms)}個を反省対象）")

        # middle対象が0件ならスキップ
        if len(technical_terms) == 0:
            logger.info("[Stage 2.5] No middle-confidence terms to reflect on, skipping")
            print("  反省対象なし、Stage 2.5をスキップ")
            return {
                "refinement_confidence": 1.0,
                "reflection_history": reflection_history + [{"issues_found": [], "suggested_actions": [], "skipped": True}],
            }

        # 前回の反省
        previous_reflection = "初回の反省"
        if reflection_history:
            prev = reflection_history[-1]
            previous_reflection = f"前回の問題点: {', '.join(prev.get('issues_found', []))}\n前回のアクション: {', '.join(prev.get('suggested_actions', []))}"

        # バッチ分割
        term_batches = [technical_terms[i:i + batch_size_terms]
                       for i in range(0, len(technical_terms), batch_size_terms)] or [[]]
        candidate_batches = [candidates[i:i + batch_size_candidates]
                           for i in range(0, len(candidates), batch_size_candidates)] or [[]]

        num_batches = max(len(term_batches), len(candidate_batches))
        logger.info(f"Processing {len(technical_terms)} terms and {len(candidates)} candidates in {num_batches} batch(es)")
        if num_batches > 1:
            print(f"  （{num_batches}バッチに分割して処理）")

        # JsonOutputParser使用（補正付き）
        reflection_parser = self._build_fixing_parser(ReflectionResult)
        prompt = get_self_reflection_prompt()
        chain = prompt | self.llm | self._to_text | reflection_parser

        # バッチ処理結果を集約
        all_issues: List[str] = []
        all_actions: List[str] = []
        all_missing: List[Dict] = []
        confidence_scores: List[float] = []
        all_reasonings: List[str] = []

        for batch_idx in range(num_batches):
            # 各バッチのデータを取得（インデックスがはみ出る場合は空リスト）
            term_batch = term_batches[batch_idx] if batch_idx < len(term_batches) else []
            candidate_batch = candidate_batches[batch_idx] if batch_idx < len(candidate_batches) else []

            if not term_batch and not candidate_batch:
                continue

            terms_json = json.dumps(term_batch, ensure_ascii=False, indent=2)
            candidates_sample = json.dumps(candidate_batch, ensure_ascii=False, indent=2)

            # 却下済み用語リストを作成（LLMが同じ用語を繰り返し指摘しないように）
            rejected_terms = state.get("rejected_terms", [])
            rejected_headwords = [r.get("headword", "") for r in rejected_terms if r.get("headword")]
            rejected_terms_list = ", ".join(rejected_headwords[:50]) if rejected_headwords else "（なし）"
            if rejected_headwords:
                logger.debug(f"Passing {len(rejected_headwords)} rejected terms to LLM: {rejected_headwords[:5]}...")

            try:
                batch_reflection = await chain.ainvoke({
                    "num_terms": len(technical_terms),
                    "terms_json": terms_json,
                    "num_candidates": len(candidates),
                    "candidates_sample": candidates_sample,
                    "rejected_terms_list": rejected_terms_list,
                    "previous_reflection": previous_reflection
                })

                # 結果を集約
                all_issues.extend(batch_reflection.get("issues_found", []))
                all_actions.extend(batch_reflection.get("suggested_actions", []))
                all_missing.extend(batch_reflection.get("missing_terms", []))
                confidence_scores.append(batch_reflection.get("confidence_in_current_list", 0.5))
                all_reasonings.append(batch_reflection.get("reasoning", ""))

                logger.debug(f"Batch {batch_idx + 1}/{num_batches}: "
                           f"issues={len(batch_reflection.get('issues_found', []))}, "
                           f"missing={len(batch_reflection.get('missing_terms', []))}")

            except Exception as e:
                logger.error(f"Reflection parsing failed for batch {batch_idx + 1}: {e}")
                continue

        # 結果を統合（重複排除）
        unique_issues = list(dict.fromkeys(all_issues))  # 順序保持で重複排除
        unique_actions = list(dict.fromkeys(all_actions))
        # missing_termsはtermで重複排除
        seen_terms = set()
        unique_missing = []
        for m in all_missing:
            term = m.get("term", "")
            if term and term not in seen_terms:
                seen_terms.add(term)
                unique_missing.append(m)

        # 最終的なreflection結果
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5
        should_continue = avg_confidence < 0.9 or len(unique_issues) > 0 or len(unique_missing) > 0

        reflection = {
            "issues_found": unique_issues,
            "confidence_in_current_list": avg_confidence,
            "suggested_actions": unique_actions,
            "should_continue": should_continue,
            "reasoning": " | ".join(all_reasonings) if all_reasonings else "バッチ処理完了",
            "missing_terms": unique_missing  # 制限なし: LLMが検出した全ての漏れ用語を処理
        }

        logger.info(f"Aggregated reflection: issues={len(unique_issues)}, "
                   f"actions={len(unique_actions)}, missing={len(unique_missing)}, "
                   f"avg_confidence={avg_confidence:.2f}")

        # デバッグ: パース後のreflection構造を確認（特にmissing_terms）
        if isinstance(reflection, dict) and reflection.get("missing_terms"):
            logger.debug(f"Parsed reflection contains {len(reflection['missing_terms'])} missing terms")
            for idx, mt in enumerate(reflection["missing_terms"][:3]):  # 最初の3個だけログ
                logger.debug(f"  Missing term {idx}: {mt}")

        # 反省履歴に追加
        if "reflection_history" not in state:
            state["reflection_history"] = []
        state["reflection_history"].append(reflection)

        # 収束判定
        if reflection["confidence_in_current_list"] >= 0.9 and not reflection["should_continue"]:
            state["refinement_converged"] = True
            logger.info("[OK] Refinement converged (high confidence)")
            print(f"[OK] 収束しました（信頼度: {reflection['confidence_in_current_list']:.2f}）\n")
        else:
            logger.info(f"Confidence: {reflection['confidence_in_current_list']:.2f}, Issues: {len(reflection['issues_found'])}, Missing: {len(reflection.get('missing_terms', []))}")
            print(f"  信頼度: {reflection['confidence_in_current_list']:.2f}")
            if reflection["issues_found"]:
                print(f"  問題点: {len(reflection['issues_found'])}個検出")
            if reflection.get("missing_terms"):
                print(f"  漏れ用語: {len(reflection['missing_terms'])}個発見")
                # 最初の3つの漏れ用語を表示
                for mt in reflection["missing_terms"][:3]:
                    print(f"    - {mt.get('term', 'unknown')}")

        # UIコールバック呼び出し
        if hasattr(self, 'ui_callback') and self.ui_callback:
            try:
                self.ui_callback("stage25_reflection", {
                    "iteration": refinement_iteration + 1,
                    "max_iterations": max_iterations,
                    "confidence": reflection.get("confidence_in_current_list", 0.0),
                    "issues": reflection.get("issues_found", []),
                    "missing": [mt.get("term", "") for mt in reflection.get("missing_terms", [])],
                    "reasoning": reflection.get("reasoning", "")
                })
            except Exception as e:
                logger.warning(f"UI callback failed: {e}")

        # 反省結果スナップショット（デバッグ用）
        if state.get("output_dir"):
            try:
                iter_idx = refinement_iteration + 1
                snapshot = {
                    "iteration": iter_idx,
                    "max_iterations": max_iterations,
                    "confidence": reflection.get("confidence_in_current_list"),
                    "issues": reflection.get("issues_found", []),
                    "suggested_actions": reflection.get("suggested_actions", []),
                    "missing_terms": reflection.get("missing_terms", []),
                    "terms_count": len(technical_terms),  # 全件数
                    "candidates_count": len(candidates),  # 全件数
                    "num_batches": num_batches
                }
                snap_file = state["output_dir"] / f"stage25_reflection_iter{iter_idx}.json"
                with open(snap_file, "w", encoding="utf-8") as f:
                    json.dump(snapshot, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.warning(f"Failed to save reflection snapshot: {e}")

        return state

    async def _node_stage25_refine_terms(self, state: TermExtractionState) -> TermExtractionState:
        """Stage 2.5b: 改善実行ノード"""
        from .prompts import get_term_refinement_prompt

        reflection_history = state.get("reflection_history", [])
        if not reflection_history:
            logger.warning("[Stage 2.5] No reflection history found, skipping refinement")
            return {"refinement_iteration": state.get("refinement_iteration", 0) + 1}
        reflection = reflection_history[-1]
        refinement_iteration = state.get("refinement_iteration", 0)
        before_counts = {
            "terms": len(state.get("technical_terms", [])),
            "rejected": len(state.get("rejected_terms", []))
        }

        # 各処理の結果を追跡
        direct_removed_count = 0  # Remove処理での削除数
        added_count = 0           # Missing処理での追加数
        missing_rejected_count = 0  # Missing処理での却下数
        investigated_kept = 0     # Investigate処理での保持数
        investigated_removed = 0  # Investigate処理での除外数
        just_removed_headwords = set()  # 同一反復内でRemoveされた用語名（Missing処理の重複チェック除外用）

        # 1. Remove処理（既存）
        if reflection.get("suggested_actions") and not state.get("refinement_converged", False):
            logger.info("[Stage 2.5] Refining terms...")
            print(f"[Stage 2.5] 改善アクション実行中...")

            # 用語リストをJSON化
            terms_json = json.dumps(state["technical_terms"], ensure_ascii=False, indent=2)

            # LLM実行
            prompt = get_term_refinement_prompt()
            chain = prompt | self.llm | self._to_text

            result = await chain.ainvoke({
                "issues": "\n".join(reflection["issues_found"]),
                "actions": "\n".join(reflection["suggested_actions"]),
                "terms_json": terms_json
            })

            # アクションパース
            import re
            # AIMessageのcontent属性に対応
            content = getattr(result, "content", None)
            if content is None:
                if hasattr(result, "strip"):
                    content = result.strip()
                else:
                    content = str(result)

            json_match = re.search(r'\[.*\]', content, re.DOTALL)

            if json_match:
                try:
                    actions = json.loads(json_match.group(0))

                    # Remove上限: middle用語の半分（最低3個）で全滅を防止
                    middle_count = len([t for t in state["technical_terms"] if not t.get("skip_reflection")])
                    remove_limit = max(3, middle_count // 2)

                    for action in actions:
                        if action.get("action") == "remove":
                            if direct_removed_count >= remove_limit:
                                logger.warning(f"Remove limit reached ({remove_limit}/{middle_count}), deferring remaining to next iteration")
                                print(f"  [WARN] 削除上限に到達（{remove_limit}個）、残りは次回判定")
                                break
                            term_name = action.get("term")
                            if not term_name:
                                continue
                            term_obj = next((t for t in state["technical_terms"]
                                           if t["headword"] == term_name), None)
                            if term_obj and term_obj.get("skip_reflection"):
                                logger.debug(f"Skipping removal of high-confidence term: {term_name}")
                                continue
                            if term_obj:
                                state["technical_terms"] = [t for t in state["technical_terms"] if t["headword"] != term_name]
                                if "rejected_terms" not in state:
                                    state["rejected_terms"] = []
                                state["rejected_terms"].append({
                                    **term_obj,
                                    "rejection_reason": action.get("reason", "不明"),
                                    "rejected_at_stage25": state.get("refinement_iteration", 0)
                                })
                                direct_removed_count += 1
                                just_removed_headwords.add(term_name)

                        elif action.get("action") == "investigate":
                            # investigate対象を収集（後でまとめてRAG再判定）
                            term_name = action.get("term")
                            if not term_name:
                                continue
                            term_obj = next((t for t in state["technical_terms"]
                                           if t["headword"] == term_name), None)
                            if term_obj and term_obj.get("skip_reflection"):
                                logger.debug(f"Skipping investigation of high-confidence term: {term_name}")
                                continue
                            if term_obj:
                                if "investigate_terms" not in state:
                                    state["investigate_terms"] = []
                                state["investigate_terms"].append({
                                    **term_obj,
                                    "investigate_reason": action.get("reason", "専門性が不明")
                                })

                    logger.info(f"Refinement: removed {direct_removed_count} terms")
                    if direct_removed_count > 0:
                        print(f"  Remove: {direct_removed_count}個除外")

                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse refinement actions: {e}")
                    print(f"[WARN] 改善アクションのパースに失敗")

        # 2. Missing追加処理（RAG定義生成付き）
        missing_terms = reflection.get("missing_terms", [])
        if missing_terms:
            logger.info(f"[Stage 2.5] Processing {len(missing_terms)} missing terms with RAG...")
            print(f"[Stage 2.5] 漏れ用語を処理中 ({len(missing_terms)}個)...")

            # デバッグ: missing_termsの構造を確認
            logger.debug(f"missing_terms structure: {type(missing_terms)}, first item: {missing_terms[0] if missing_terms else 'empty'}")

            # ハイブリッド検索が使用可能かチェック
            use_hybrid = self.hybrid_retriever is not None
            if not use_hybrid and not self.vector_store:
                logger.warning("Neither hybrid retriever nor vector store available for missing terms definition generation")

            # missing → candidates形式に変換 + RAG定義生成
            new_candidates = []
            duplicate_count = 0  # 重複スキップ数をカウント
            for m in missing_terms:
                # JsonOutputParserは常にdictを返すため、dict前提で処理
                term_str = m.get("term", "")
                hw = term_str.strip()
                if not hw:
                    continue

                # 重複チェック（同一反復でRemoveされた用語は除外しない→再追加を許可）
                if self._is_duplicate(hw, state, skip_rejected_headwords=just_removed_headwords):
                    logger.debug(f"Skipping duplicate missing term: {hw}")
                    duplicate_count += 1
                    continue

                definition = m.get("evidence", "")
                try:
                    docs = []
                    if self.engine and self.embeddings:
                        docs = similarity_search_without_jargon(
                            engine=self.engine, embeddings=self.embeddings,
                            query=hw, collection_name=self.collection_name, k=3,
                        )

                    if docs:
                        # 上位3件の文書から定義生成
                        context = "\n".join([d.page_content[:500] for d in docs[:3]])
                        definition = await self._generate_definition_for_missing_term(hw, context)
                        logger.debug(f"Generated RAG definition for '{hw}': {definition[:50]}...")
                    else:
                        logger.debug(f"No RAG docs found for '{hw}', using evidence as definition")
                except Exception as e:
                    logger.warning(f"RAG definition generation failed for '{hw}': {e}")
                    # エラー時はevidenceをフォールバック

                # 候補形式に変換
                new_candidates.append({
                    "headword": hw,
                    "brief_definition": definition,  # ★ RAG定義またはevidence
                    "domain": m.get("suggested_domain"),
                    "confidence": 0.65,  # RAG定義付きなので少し高め
                    "source": "stage25_missing_with_rag"
                })
                logger.debug(f"Added missing term candidate: {hw} (definition: {definition[:50]}...)")

            if new_candidates:
                logger.info(f"Filtered {len(new_candidates)} unique missing terms for Stage2 incremental filtering (skipped {duplicate_count} duplicates)")
                print(f"  → 重複スキップ: {duplicate_count}個、Stage2判定: {len(new_candidates)}個")

                # デバッグ: Stage2に渡す候補のbrief_definitionを確認
                for c in new_candidates:
                    logger.info(f"[Stage2.5 Input] {c.get('headword')}: brief_definition='{c.get('brief_definition', '')[:80]}...'")
                    print(f"    判定対象: {c.get('headword')} | 定義: {c.get('brief_definition', '')[:60]}...")

                # 増分Stage2実行（全候補ではなく新規のみ）
                selected = await self._filter_technical_terms(new_candidates)
                if not selected:
                    logger.warning("Stage2 incremental filtering returned 0 terms (all rejected or parse failure).")

                # 選択されたものをtechnical_termsにマージ
                selected_headwords = {t.get("headword") for t in selected}
                for term in selected:
                    state["technical_terms"].append(term)
                    logger.debug(f"Added term to technical_terms: {term.get('headword')}")

                # 却下されたものをrejected_termsに追加（次の反復で再度拾わないように）
                for cand in new_candidates:
                    if cand["headword"] not in selected_headwords:
                        if "rejected_terms" not in state:
                            state["rejected_terms"] = []
                        state["rejected_terms"].append({
                            **cand,
                            "rejection_reason": "Stage2.5 missing_terms増分フィルタで除外",
                            "rejected_at_stage25": refinement_iteration
                        })
                        logger.debug(f"Rejected missing term: {cand['headword']}")

                added_count = len(selected)
                missing_rejected_count = len(new_candidates) - added_count
                logger.info(f"Added {added_count} terms from missing candidates (rejected {missing_rejected_count})")
                print(f"  → 結果: 追加{added_count}個、Stage2却下{missing_rejected_count}個")
            else:
                # 全て重複でスキップされた場合
                logger.info(f"All {duplicate_count} missing terms were duplicates, nothing to add")
                print(f"  → 全て重複: {duplicate_count}個スキップ（既に処理済み）")

        # 3. Investigate処理（RAG再定義 → 増分Stage2フィルタで再判定）
        investigate_terms = state.get("investigate_terms", [])
        if investigate_terms:
            logger.info(f"[Stage 2.5] Re-evaluating {len(investigate_terms)} investigate terms with RAG...")
            print(f"[Stage 2.5] 疑わしい用語を再評価中 ({len(investigate_terms)}個、RAG再定義)...")

            # ハイブリッド検索が使用可能かチェック
            use_hybrid = self.hybrid_retriever is not None

            # investigate → 一旦technical_termsから除外 → RAG定義生成 → 増分フィルタ
            investigate_candidates = []
            for inv_term in investigate_terms:
                hw = inv_term.get("headword", "").strip()
                if not hw:
                    continue

                # technical_termsから一旦除外
                term_obj = next((t for t in state["technical_terms"]
                               if t["headword"] == hw), None)
                if term_obj:
                    state["technical_terms"] = [t for t in state["technical_terms"] if t["headword"] != hw]

                definition = inv_term.get("brief_definition", "")
                try:
                    docs = []
                    if self.engine and self.embeddings:
                        docs = similarity_search_without_jargon(
                            engine=self.engine, embeddings=self.embeddings,
                            query=hw, collection_name=self.collection_name, k=3,
                        )

                    if docs:
                        context = "\n".join([d.page_content[:500] for d in docs[:3]])
                        definition = await self._generate_definition_for_missing_term(hw, context)
                        logger.debug(f"Re-generated RAG definition for investigate term '{hw}': {definition[:50]}...")
                except Exception as e:
                    logger.warning(f"RAG definition re-generation failed for '{hw}': {e}")

                investigate_candidates.append({
                    "headword": hw,
                    "brief_definition": definition,
                    "domain": inv_term.get("domain"),
                    "confidence": 0.5,  # 再評価なので低めに設定
                    "source": "stage25_investigate_reevaluated",
                    "original_reason": inv_term.get("investigate_reason", "")
                })

            if investigate_candidates:
                # 増分Stage2フィルタで再判定
                revalidated = await self._filter_technical_terms(investigate_candidates)

                # 通ったものはtechnical_termsに戻す
                revalidated_headwords = {t.get("headword") for t in revalidated}
                for term in revalidated:
                    state["technical_terms"].append(term)
                    investigated_kept += 1
                    logger.debug(f"Investigate term kept: {term.get('headword')}")

                # 通らなかったものはrejected_termsへ
                for inv_cand in investigate_candidates:
                    if inv_cand["headword"] not in revalidated_headwords:
                        if "rejected_terms" not in state:
                            state["rejected_terms"] = []
                        state["rejected_terms"].append({
                            **inv_cand,
                            "rejection_reason": f"Stage2.5 investigate再評価で除外: {inv_cand.get('original_reason', '')}",
                            "rejected_at_stage25": refinement_iteration
                        })
                        investigated_removed += 1
                        logger.debug(f"Investigate term rejected: {inv_cand['headword']}")

                logger.info(f"Investigate re-evaluation: {investigated_kept} kept, {investigated_removed} removed")
                print(f"  Investigate: {investigated_kept}個保持、{investigated_removed}個除外")

            # investigate_termsをクリア
            state["investigate_terms"] = []

        # 変化のサマリー
        total_removed = direct_removed_count + investigated_removed
        total_added = added_count
        if total_removed == 0 and total_added == 0:
            print(f"[Stage 2.5] 変化なし\n")
        else:
            print(f"[Stage 2.5] → 合計: 削除{total_removed}個、追加{total_added}個\n")

        # UIコールバック呼び出し
        if hasattr(self, 'ui_callback') and self.ui_callback:
            try:
                self.ui_callback("stage25_action", {
                    "removed": total_removed,
                    "added": total_added
                })
            except Exception as e:
                logger.warning(f"UI callback failed: {e}")

        # 収束判定用にカウントを記録
        state["last_removed_count"] = total_removed
        state["last_added_count"] = total_added

        if state.get("output_dir"):
            try:
                iter_idx = refinement_iteration + 1
                after_counts = {
                    "terms": len(state.get("technical_terms", [])),
                    "rejected": len(state.get("rejected_terms", []))
                }
                snapshot = {
                    "iteration": iter_idx,
                    "direct_removed_count": direct_removed_count,
                    "missing_added_count": added_count,
                    "missing_rejected_count": missing_rejected_count,
                    "investigated_kept": investigated_kept,
                    "investigated_removed": investigated_removed,
                    "total_removed": total_removed,
                    "total_added": total_added,
                    "suggested_actions": reflection.get("suggested_actions", []),
                    "missing_terms": missing_terms,
                    "before_counts": before_counts,
                    "after_counts": after_counts
                }
                snap_file = state["output_dir"] / f"stage25_action_iter{iter_idx}.json"
                with open(snap_file, "w", encoding="utf-8") as f:
                    json.dump(snapshot, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.warning(f"Failed to save Stage2.5 action snapshot: {e}")

        state["refinement_iteration"] = refinement_iteration + 1
        return state

    def _should_continue_refinement(self, state: TermExtractionState) -> Literal["continue", "finish"]:
        """Stage 2.5のループ制御"""

        # 収束フラグチェック
        if state.get("refinement_converged", False):
            return "finish"

        # 最大反復回数チェック
        refinement_iteration = state.get("refinement_iteration", 0)
        max_iterations = state.get("max_refinement_iterations", 3)
        if refinement_iteration >= max_iterations:
            logger.info("Max refinement iterations reached")
            print(f"[Stage 2.5] 最大反復回数に達しました\n")
            return "finish"

        # ガード1: リストハッシュが前回と同じなら強制終了
        current_hash = hash(frozenset(t["headword"] for t in state["technical_terms"]))
        if state.get("previous_list_hash") is not None:
            if state["previous_list_hash"] == current_hash:
                logger.warning("Term list unchanged (hash match) - forcing convergence")
                print(f"[Stage 2.5] 用語リストに変化なし - 収束と判定\n")
                return "finish"
        state["previous_list_hash"] = current_hash

        # ガード2: 前回と同じ問題指摘なら強制終了
        reflection_history = state.get("reflection_history", [])
        if len(reflection_history) >= 2:
            last_issues = set(reflection_history[-1].get("issues_found", []))
            prev_issues = set(reflection_history[-2].get("issues_found", []))
            if last_issues and prev_issues:
                overlap = len(last_issues & prev_issues) / max(len(last_issues), 1)
                if overlap > 0.8:  # 80%重複
                    logger.warning(f"Issue overlap {overlap:.1%} - forcing convergence")
                    print(f"[Stage 2.5] 問題指摘が重複（{overlap:.0%}） - 収束と判定\n")
                    return "finish"

        # ガード3: 追加も削除もなし → 収束（NEW）
        if state.get("refinement_iteration", 0) > 0:
            last_added = state.get("last_added_count", 0)
            last_removed = state.get("last_removed_count", 0)

            if last_added == 0 and last_removed == 0:
                logger.info("No changes in last iteration - converged")
                print(f"[Stage 2.5] 前回の反復で変化なし - 収束と判定\n")
                return "finish"

        # ガード4: missing_termsが空 かつ confidence高い → 収束（NEW）
        if reflection_history:
            latest = reflection_history[-1]
            missing = latest.get("missing_terms", [])

            if not missing and latest["confidence_in_current_list"] >= 0.85:
                logger.info("No missing terms and high confidence - converged")
                print(f"[Stage 2.5] 漏れなし & 高信頼度 - 収束と判定\n")
                return "finish"

            # 従来のconfidenceチェック
            if latest["confidence_in_current_list"] >= 0.9:
                return "finish"

        return "continue"

    async def _generate_definition_for_missing_term(self, term: str, context: str) -> str:
        """Stage 2.5 の missing_terms 用の定義生成（簡易版）

        Args:
            term: 用語名
            context: RAG検索で取得した文脈

        Returns:
            生成された定義（1-2文）
        """
        prompt_text = f"""以下の文脈から、専門用語「{term}」の簡潔な定義（1-2文、50文字以内）を生成してください。

文脈:
{context[:1000]}

定義:"""

        try:
            result = await self.llm.ainvoke(prompt_text)
            content = getattr(result, "content", None)
            if content is None:
                if hasattr(result, "strip"):
                    content = result.strip()
                else:
                    content = str(result)
            return content.strip()
        except Exception as e:
            logger.warning(f"Definition generation failed for '{term}': {e}")
            return f"{term}に関する専門用語（定義生成エラー）"

    async def _node_stage3_generate_definitions(self, state: TermExtractionState) -> TermExtractionState:
        """Stage 3: 定義生成ノード"""
        logger.info("[Stage 3] Generating definitions...")
        print(f"[Stage 3] 定義を生成中 ({len(state['technical_terms'])}個の専門用語)...")

        state["technical_terms"] = await self._generate_definitions_with_rag(state["technical_terms"])

        logger.info(f"Stage 3 complete: {len(state['technical_terms'])} definitions generated")
        print(f"[OK] 定義生成が完了しました\n")

        return state

    async def _node_stage4_detect_synonyms(self, state: TermExtractionState) -> TermExtractionState:
        """Stage 4: 類義語検出ノード（分野分類方式選択可能）"""
        logger.info("[Stage 4] Detecting synonyms...")
        print(f"[Stage 4] 類義語を検出中...")

        # 分野分類方式を取得
        domain_method = getattr(self.config, 'stage4_domain_method', 'llm')
        logger.info(f"Domain classification method: {domain_method}")

        # Stage 4a: 専門用語間の類義語（方式に応じて分岐）
        if domain_method == 'hdbscan':
            representative_terms = await self._detect_and_merge_term_synonyms_hdbscan(state["technical_terms"])
        elif domain_method == 'hybrid':
            representative_terms = await self._detect_and_merge_term_synonyms_hybrid(state["technical_terms"])
        else:  # デフォルト: llm
            representative_terms = await self._detect_and_merge_term_synonyms(state["technical_terms"])

        # Stage 4b: 代表語と候補の類義語（共通処理）
        if representative_terms and state["candidates"]:
            representative_terms = await self._detect_synonyms_with_candidates(
                representative_terms,
                state["candidates"]
            )

        state["technical_terms"] = representative_terms

        logger.info(f"Stage 4 complete: {len(state['technical_terms'])} representative terms")
        print(f"[OK] 類義語検出が完了しました\n")

        return state

    async def _extract_terms_with_langgraph(self, text: str, output_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
        """LangGraphを使用した専門用語抽出（新実装）"""
        logger.info("Starting LangGraph-based term extraction workflow...")

        # テキストをチャンクに分割
        chunk_size = getattr(self.config, 'llm_extraction_chunk_size', 3000)
        chunks = self._split_text(text, chunk_size)

        # 初期状態
        initial_state: TermExtractionState = {
            "text_chunks": chunks,
            "candidates": [],
            "technical_terms": [],
            "rejected_terms": [],
            "refinement_iteration": 0,
            "max_refinement_iterations": getattr(self.config, 'max_refinement_iterations', 3),
            "refinement_converged": False,
            "reflection_history": [],
            "previous_list_hash": None,
            "last_added_count": 0,
            "last_removed_count": 0,
            "investigate_terms": [],
            "output_dir": output_dir
        }

        # ワークフロー構築
        workflow = await self._build_workflow()

        # グラフ可視化を保存
        if output_dir:
            try:
                graph_image = workflow.get_graph().draw_mermaid_png()
                graph_file = output_dir / "langgraph_workflow.png"
                output_dir.mkdir(parents=True, exist_ok=True)
                with open(graph_file, "wb") as f:
                    f.write(graph_image)
                logger.info(f"LangGraph workflow visualization saved to {graph_file}")
                print(f"[OK] ワークフローグラフを保存: {graph_file}\n")
            except Exception as e:
                logger.warning(f"Failed to save workflow graph: {e}")

        # LangGraphワークフロー全体のトレース設定（1回のみ、無料枠節約）
        from langchain_core.runnables import RunnableConfig
        workflow_config = RunnableConfig(
            run_name="[TermExtraction] LangGraph-Workflow",
            tags=["term_extraction", "langgraph", "workflow"],
            metadata={
                "num_chunks": len(chunks),
                "max_refinement_iterations": initial_state["max_refinement_iterations"],
                "workflow_version": "v3_with_missing_addition_loop"
            }
        )

        # 実行
        final_state = await workflow.ainvoke(initial_state, config=workflow_config)

        # 反省ログを保存
        if output_dir:
            reflection_log = output_dir / "reflection_log.json"
            with open(reflection_log, "w", encoding="utf-8") as f:
                json.dump({
                    "reflections": final_state.get("reflection_history", []),
                    "iterations": final_state.get("refinement_iteration", 0),
                    "converged": final_state.get("refinement_converged", False),
                    "final_term_count": len(final_state.get("technical_terms", [])),
                    "rejected_count": len(final_state.get("rejected_terms", [])),
                    "rejected_terms_detail": final_state.get("rejected_terms", []),
                    "last_added_count": final_state.get("last_added_count", 0),
                    "last_removed_count": final_state.get("last_removed_count", 0)
                }, f, ensure_ascii=False, indent=2)
            logger.info(f"Reflection log saved to {reflection_log}")
            print(f"[OK] 反省ログを保存: {reflection_log}\n")

        return final_state.get("technical_terms", [])


# ========== Utility Functions ==========
async def run_extraction_pipeline(input_dir: Path, output_json: Path, config, llm, embeddings, vector_store, pg_url, jargon_table_name, jargon_manager=None, collection_name="documents", ui_callback=None):
    """専門用語抽出パイプラインの実行

    Args:
        ui_callback: Optional callback function(event_type: str, data: dict) for UI updates
    """
    extractor = TermExtractor(config, llm, embeddings, vector_store, pg_url, jargon_table_name, collection_name=collection_name)
    extractor.ui_callback = ui_callback  # コールバックを設定

    # ファイルの検索
    supported_exts = ['.txt', '.md', '.pdf']
    files = [p for ext in supported_exts for p in input_dir.glob(f"**/*{ext}")] if input_dir else []

    output_dir = output_json.parent

    if files:
        logger.info(f"Found {len(files)} files to process")
        result = await extractor.extract_from_documents(files, output_dir=output_dir)
        terms = result.get("terms", [])
    elif pg_url and collection_name:
        # DBフォールバック: document_chunksからテキスト取得
        logger.info(f"No files found. Loading text from document_chunks (collection: {collection_name})...")
        print(f"[INFO] DBからチャンクを取得中 (collection: {collection_name})...")
        from sqlalchemy import create_engine, text as sa_text
        engine = create_engine(pg_url)
        with engine.connect() as conn:
            rows = conn.execute(sa_text(
                "SELECT content FROM document_chunks WHERE collection_name = :coll ORDER BY chunk_id"
            ), {"coll": collection_name}).fetchall()
        if not rows:
            logger.error(f"No document chunks found for collection: {collection_name}")
            print(f"[ERROR] コレクション '{collection_name}' にチャンクがありません")
            return
        combined_text = "\n\n".join(r.content for r in rows)
        logger.info(f"Loaded {len(rows)} chunks ({len(combined_text)} chars) from DB")
        print(f"[OK] {len(rows)}チャンク取得 ({len(combined_text)}文字)")
        terms = await extractor._extract_terms(combined_text, output_dir=output_dir)
    else:
        logger.error(f"No supported files found in {input_dir}")
        return

    # JSONファイルに保存
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump({"terms": terms}, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved {len(terms)} terms to {output_json}")

    # データベースに保存
    if terms and pg_url:
        extractor.save_to_database(terms)


__all__ = [
    "JargonDictionaryManager",
    "TermExtractor",
    "run_extraction_pipeline"
]


# ========== LangGraph CLI Export ==========
# LangSmith Studio でデバッグするためのグラフエクスポート
# 使い方: langgraph dev を実行してから
# https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024 にアクセス

async def _get_graph_for_cli():
    """LangGraph CLI用のグラフ取得関数（遅延import対応）"""
    import os
    import sys
    # 絶対インポートを使用
    try:
        from src.rag.config import Config
    except ImportError:
        # パスに追加してインポート
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        from src.rag.config import Config

    # 基本的なimportのみ（OpenAI/Azureは大抵入ってる）
    from langchain_openai import ChatOpenAI, AzureChatOpenAI

    # 設定を取得
    config = Config()
    provider = (getattr(config, "llm_provider", "azure") or "azure").lower()
    temperature = getattr(config, "llm_temperature", 0.0)
    max_tokens = getattr(config, "max_tokens", None)

    def _build_openai_llm():
        model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
        api_key = config.openai_api_key or os.getenv("OPENAI_API_KEY") or "DUMMY"
        return ChatOpenAI(model=model, temperature=temperature, api_key=api_key, max_tokens=max_tokens)

    def _build_azure_llm():
        api_key = getattr(config, 'azure_openai_api_key', None) or os.getenv("AZURE_OPENAI_API_KEY") or "DUMMY"
        endpoint = getattr(config, 'azure_openai_endpoint', None) or os.getenv("AZURE_OPENAI_ENDPOINT") or "https://dummy"
        deployment = (
            getattr(config, 'azure_openai_chat_deployment_name', None)
            or os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
            or "dummy"
        )
        api_version = getattr(config, 'azure_openai_api_version', '2024-02-15-preview')
        return AzureChatOpenAI(
            azure_deployment=deployment,
            temperature=temperature,
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
            max_tokens=max_tokens
        )

    def _build_gemini_llm():
        # Gemini使用時のみimport
        from langchain_google_genai import ChatGoogleGenerativeAI
        model = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-pro")
        api_key = os.getenv("GEMINI_API_KEY") or getattr(config, "gemini_api_key", None) or "DUMMY"
        return ChatGoogleGenerativeAI(model=model, temperature=temperature, google_api_key=api_key)

    def _build_hf_llm():
        # HuggingFace使用時のみimport
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        from langchain_huggingface import HuggingFacePipeline
        model_id = getattr(config, "hf_model_id", None) or os.getenv("HF_MODEL_ID")
        if not model_id:
            raise ValueError("Hugging Face model ID is required for huggingface_local provider")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
        return HuggingFacePipeline(pipeline=pipe)

    def _build_vllm_llm():
        # VLLM使用時のみimport (OpenAI-compatible API)
        from langchain_openai import ChatOpenAI
        endpoint = getattr(config, "vllm_endpoint", None) or os.getenv("VLLM_ENDPOINT")
        if not endpoint:
            raise ValueError("VLLM endpoint is required for vllm provider")

        # Ensure endpoint ends with /v1
        if not endpoint.endswith("/v1"):
            endpoint = endpoint.rstrip("/") + "/v1"

        vllm_model = getattr(config, "vllm_model", "") or os.getenv("VLLM_MODEL", "")
        vllm_api_key = getattr(config, "vllm_api_key", "EMPTY") or os.getenv("VLLM_API_KEY", "EMPTY")
        vllm_temperature = getattr(config, "vllm_temperature", temperature)
        vllm_max_tokens = getattr(config, "vllm_max_tokens", max_tokens) or 4096

        return ChatOpenAI(
            base_url=endpoint,
            api_key=vllm_api_key,
            model=vllm_model,
            temperature=vllm_temperature,
            max_tokens=vllm_max_tokens,
        )

    try:
        if provider == "openai":
            llm = _build_openai_llm()
        elif provider == "azure":
            llm = _build_azure_llm()
        elif provider == "gemini":
            llm = _build_gemini_llm()
        elif provider in {"huggingface_local", "huggingface"}:
            llm = _build_hf_llm()
        elif provider == "vllm":
            llm = _build_vllm_llm()
        else:
            llm = _build_openai_llm()
    except Exception as e:
        logger.warning(f"LLM initialisation failed for provider '{provider}': {e} - falling back to OpenAI default")
        llm = _build_openai_llm()

    # TermExtractorインスタンスを作成
    extractor = TermExtractor(
        config=config,
        llm=llm,
        embeddings=None,
        vector_store=None,
        pg_url=None,
        jargon_table_name="jargon_dictionary"
    )

    # 本物のワークフローを返す
    return await extractor._build_workflow()


# ===== LangGraph CLI export (本番用グラフ) =====
import asyncio
from concurrent.futures import ThreadPoolExecutor

# スレッドプールエグゼキューターを作成（グローバル）
_executor = ThreadPoolExecutor(max_workers=1)

def make_graph(config=None):
    """LangGraph CLIが呼ぶエントリーポイント

    Args:
        config: RunnableConfig（現時点では使用しない）

    Returns:
        CompiledStateGraph: コンパイル済みのワークフロー
    """
    # ASGIサーバー内でblocking callを避けるため、別スレッドで実行
    future = _executor.submit(_run_in_thread)
    return future.result()

def _run_in_thread():
    """別スレッドで非同期関数を実行"""
    # 新しいイベントループを作成して実行
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(_get_graph_for_cli())
    finally:
        loop.close()

# Deprecated: グローバル変数でのエクスポートは削除
# LangGraph CLIは make_graph 関数を直接呼ぶようになりました
