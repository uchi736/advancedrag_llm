"""
term_extraction.py
==================
LLMベースのシンプルな専門用語抽出モジュール
統計的手法を使用せず、LLMに直接テキストを渡して専門用語を抽出
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.documents import Document
from pydantic import BaseModel, Field
import asyncio
from tqdm import tqdm

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


class ExtractedTermList(BaseModel):
    """専門用語リスト"""
    terms: List[ExtractedTerm] = Field(description="抽出された専門用語のリスト")


class FilteredTermResult(BaseModel):
    """Stage 2の選別結果（選ばれた用語と除外された用語）"""
    selected: List[ExtractedTerm] = Field(description="専門用語として選ばれた用語")
    rejected: List[ExtractedTerm] = Field(description="除外された用語（rejection_reason付き）")


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

    def __init__(self, config, llm, embeddings, vector_store, pg_url, jargon_table_name):
        self.config = config
        self.llm = llm
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.pg_url = pg_url
        self.jargon_table_name = jargon_table_name

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
        self.json_parser = JsonOutputParser(pydantic_object=ExtractedTermList)

        # PDF プロセッサー（必要に応じて）
        self.pdf_processor = None
        try:
            from src.rag.pdf_processors import AzureDocumentIntelligenceProcessor
            self.pdf_processor = AzureDocumentIntelligenceProcessor(config)
        except Exception as e:
            logger.warning(f"PDF processor initialization failed: {e}")

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
        """テキストから専門用語を抽出（2段階処理）"""
        import asyncio
        from tqdm import tqdm

        chunk_size = getattr(self.config, 'llm_extraction_chunk_size', 3000)

        # テキストをチャンクに分割
        chunks = self._split_text(text, chunk_size)

        # ========== 第1段階: 緩い候補抽出（並列処理） ==========
        logger.info(f"Stage 1: Extracting candidates from {len(chunks)} chunks (parallel)")
        print(f"\n[Stage 1] チャンクから候補抽出中 ({len(chunks)}個のチャンク)...")

        all_candidates = []
        chain = self.candidate_extraction_prompt | self.llm | self.json_parser

        async def extract_from_chunk(i: int, chunk: str):
            """1つのチャンクから候補を抽出"""
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
        print(f"✓ {len(unique_candidates)}個の候補を抽出しました\n")

        # Stage 1結果をファイルに出力
        if output_dir:
            stage1_file = output_dir / "stage1_candidates.json"
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(stage1_file, "w", encoding="utf-8") as f:
                json.dump({"candidates": unique_candidates}, f, ensure_ascii=False, indent=2)
            logger.info(f"Stage 1 results saved to {stage1_file}")
            print(f"✓ Stage 1 結果を保存: {stage1_file}\n")

        # ========== 第2段階: 専門用語の選別 ==========
        technical_terms = []

        if unique_candidates:
            logger.info("Stage 2: Filtering technical terms from candidates")
            print(f"[Stage 2] 専門用語を選別中...")
            technical_terms = await self._filter_technical_terms(unique_candidates, output_dir=output_dir)
            logger.info(f"Stage 2 complete: {len(technical_terms)} technical terms selected")
            print(f"✓ {len(technical_terms)}個の専門用語を選別しました\n")

            # Stage 2結果をファイルに出力
            if output_dir:
                stage2_file = output_dir / "stage2_technical_terms.json"
                with open(stage2_file, "w", encoding="utf-8") as f:
                    json.dump({"technical_terms": technical_terms}, f, ensure_ascii=False, indent=2)
                logger.info(f"Stage 2 results saved to {stage2_file}")
                print(f"✓ Stage 2 結果を保存: {stage2_file}\n")

        # ========== 第3段階: RAGベースの定義生成（専門用語のみ） ==========
        if technical_terms and self.vector_store:
            logger.info("Stage 3: Generating definitions using RAG for technical terms")
            print(f"[Stage 3] 定義を生成中 ({len(technical_terms)}個の専門用語)...")
            technical_terms = await self._generate_definitions_with_rag(technical_terms)
            print(f"✓ 定義生成が完了しました\n")
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

    async def _filter_technical_terms(self, candidates: List[Dict], output_dir: Optional[Path] = None) -> List[Dict]:
        """候補から専門用語を選別（第2段階）- バッチ処理対応"""
        if not candidates:
            return []

        try:
            # バッチサイズを取得
            batch_size = getattr(self.config, 'stage2_batch_size', 50)

            # FilteredTermResult用のパーサーを使用
            from langchain_core.output_parsers import JsonOutputParser
            filter_parser = JsonOutputParser(pydantic_object=FilteredTermResult)

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

                chain = self.technical_term_filter_prompt | self.llm | filter_parser
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
                print(f"✓ Stage 2 除外用語を保存: {stage2_rejected_file} ({len(rejected_output)}個)\n")

            # 検証: selected + rejected = total candidates
            processed_count = len(output) + len(rejected_output)
            if processed_count != len(candidates):
                logger.warning(f"Validation failed: {processed_count} processed (selected: {len(output)}, rejected: {len(rejected_output)}) != {len(candidates)} input candidates. Missing: {len(candidates) - processed_count}")
                print(f"⚠️  警告: 処理済み用語数 ({processed_count}) が入力候補数 ({len(candidates)}) と一致しません。未処理: {len(candidates) - processed_count}個\n")

            logger.info(f"Technical term filtering: {len(output)} selected, {len(rejected_output)} rejected from {len(candidates)} candidates")
            return output

        except Exception as e:
            logger.error(f"Technical term filtering failed: {e}")
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
        chain = prompt | self.llm | StrOutputParser()

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
                        # ハイブリッド検索（ベクトル + キーワード）または ベクトルのみ
                        if use_hybrid:
                            # ハイブリッド検索: ベクトル + FTS + RRF融合
                            docs = await self.hybrid_retriever.aget_relevant_documents(headword)
                        else:
                            # フォールバック: ベクトル検索のみ
                            docs = self.vector_store.similarity_search(headword, k=5)

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
                    if use_hybrid:
                        docs = await self.hybrid_retriever.aget_relevant_documents(headword, config=config)
                    else:
                        docs = self.vector_store.similarity_search(headword, k=5)

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
                print(f"✓ {len(technical_terms)}個の専門用語を{len(all_representative_terms)}個の代表語に集約しました\n")
                return all_representative_terms

            # 50個以下の場合は一括処理
            return await self._group_terms_batch(technical_terms)

        except Exception as e:
            logger.error(f"Term synonym detection failed: {e}")
            print(f"⚠️  専門用語間の類義語判定でエラーが発生しました: {e}\n")
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

            chain = self.term_synonym_grouping_prompt | self.llm
            result = await chain.ainvoke({
                "technical_terms_json": technical_terms_json
            }, config=config)

            # 結果のパース
            if hasattr(result, "content"):
                result_text = result.content
            else:
                result_text = str(result)

            # JSONとして解析
            try:
                # JSONブロックを抽出（```json ... ``` の場合に対応）
                import re
                json_match = re.search(r'```json\s*(\[.*?\])\s*```', result_text, re.DOTALL)
                if json_match:
                    result_text = json_match.group(1)
                else:
                    # ```なしの場合、配列部分のみ抽出
                    array_match = re.search(r'\[.*\]', result_text, re.DOTALL)
                    if array_match:
                        result_text = array_match.group(0)

                # JSONとして不正な文字を修正（LLMがコメント付きで出力する場合）
                # // コメントを削除
                result_text = re.sub(r'//.*', '', result_text)

                grouped_terms = json.loads(result_text)

                # 辞書形式に変換
                representative_terms = []
                for group in grouped_terms:
                    if isinstance(group, dict):
                        term_dict = {
                            "headword": group.get("headword", ""),
                            "synonyms": group.get("synonyms", []),
                            "definition": group.get("definition", ""),
                            "domain": group.get("domain", "一般")
                        }
                        representative_terms.append(term_dict)

                logger.info(f"Grouped {len(terms)} terms into {len(representative_terms)} representative terms")

                return representative_terms

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse term grouping result: {e}")
                logger.error(f"Result text (first 1000 chars): {result_text[:1000]}")
                logger.error(f"Result text (last 500 chars): {result_text[-500:]}")
                # エラー時は元の用語リストをそのまま返す
                print(f"⚠️  バッチのグループ化に失敗しました。\n")
                print(f"   JSONパースエラー: {e}\n")
                print(f"   LLM出力の最初: {result_text[:200]}...\n")
                return terms

        except Exception as e:
            logger.error(f"Batch grouping failed: {e}")
            return terms

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
                print(f"✓ 一般語候補がないため、この段階をスキップします\n")
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
            chain = self.synonym_prompt | self.llm
            result = await chain.ainvoke({
                "representative_terms_json": representative_terms_json,
                "candidates_json": candidates_json
            })

            # 結果のパース
            if hasattr(result, "content"):
                result_text = result.content
            else:
                result_text = str(result)

            # JSONとして解析
            try:
                # JSONブロックを抽出（```json ... ``` の場合に対応）
                import re
                json_match = re.search(r'```json\s*(\[.*?\])\s*```', result_text, re.DOTALL)
                if json_match:
                    result_text = json_match.group(1)
                else:
                    # ```なしの場合、配列部分のみ抽出
                    array_match = re.search(r'\[.*\]', result_text, re.DOTALL)
                    if array_match:
                        result_text = array_match.group(0)

                # JSONとして不正な文字を修正（LLMがコメント付きで出力する場合）
                # // コメントを削除
                result_text = re.sub(r'//.*', '', result_text)

                synonym_data = json.loads(result_text)

                # 代表語リストに一般語候補からの類義語を追加
                for term in representative_terms:
                    headword = term.get("headword", "")

                    # synonym_dataから該当する用語を探す
                    for syn_item in synonym_data:
                        if isinstance(syn_item, dict) and syn_item.get("term") == headword:
                            new_synonyms = syn_item.get("synonyms", [])
                            existing_synonyms = set(term.get("synonyms", []))
                            # 一般語候補からの類義語も含める
                            term["synonyms"] = list(existing_synonyms | set(new_synonyms))
                            break

                logger.info(f"Added general term synonyms to {len(representative_terms)} representative terms")
                print(f"✓ 代表語に一般語の類義語を追加しました\n")

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse synonym detection result: {e}")
                logger.error(f"Result text (first 1000 chars): {result_text[:1000]}")
                logger.error(f"Result text (last 500 chars): {result_text[-500:]}")
                print(f"⚠️  類義語判定結果のパースに失敗しました\n")
                print(f"   JSONパースエラー: {e}\n")
                print(f"   LLM出力の最初: {result_text[:200]}...\n")

            return representative_terms

        except Exception as e:
            logger.error(f"Synonym detection with candidates failed: {e}")
            print(f"⚠️  一般語との類義語判定でエラーが発生しました: {e}\n")
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
                    conn.execute(
                        text(f"""
                            INSERT INTO {self.jargon_table_name} (term, definition, aliases)
                            VALUES (:term, :definition, :aliases)
                            ON CONFLICT (term) DO UPDATE
                            SET definition = EXCLUDED.definition,
                                aliases = EXCLUDED.aliases,
                                updated_at = CURRENT_TIMESTAMP
                        """),
                        {
                            "term": term.get("headword"),
                            "definition": term.get("definition", ""),
                            "aliases": term.get("synonyms", [])
                        }
                    )
                    saved_count += 1
                except Exception as e:
                    logger.error(f"Error saving term {term.get('headword')}: {e}")

        logger.info(f"Saved {saved_count} terms to database")
        return saved_count


# ========== Utility Functions ==========
async def run_extraction_pipeline(input_dir: Path, output_json: Path, config, llm, embeddings, vector_store, pg_url, jargon_table_name, jargon_manager=None):
    """専門用語抽出パイプラインの実行"""
    extractor = TermExtractor(config, llm, embeddings, vector_store, pg_url, jargon_table_name)

    # ファイルの検索
    supported_exts = ['.txt', '.md', '.pdf']
    files = [p for ext in supported_exts for p in input_dir.glob(f"**/*{ext}")]

    if not files:
        logger.error(f"No supported files found in {input_dir}")
        return

    logger.info(f"Found {len(files)} files to process")

    # 用語抽出（output_dirを渡す）
    output_dir = output_json.parent
    result = await extractor.extract_from_documents(files, output_dir=output_dir)
    terms = result.get("terms", [])

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