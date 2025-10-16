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

logger = logging.getLogger(__name__)


# ========== Pydantic Models ==========
class ExtractedTerm(BaseModel):
    """抽出された専門用語候補（定義なし）"""
    term: str = Field(description="専門用語")
    synonyms: List[str] = Field(default_factory=list, description="類義語・別名のリスト")
    confidence: float = Field(default=1.0, description="信頼度スコア")


class ExtractedTermList(BaseModel):
    """専門用語リスト"""
    terms: List[ExtractedTerm] = Field(description="抽出された専門用語のリスト")


# ========== JargonDictionaryManager (互換性のため残す) ==========
class JargonDictionaryManager:
    """専門用語辞書の管理クラス（互換性用）"""

    def __init__(self, connection_string: str, table_name: str = "jargon_dictionary", engine: Optional[Engine] = None):
        self.connection_string = connection_string
        self.table_name = table_name
        self.engine: Engine = engine or create_engine(connection_string)
        self._init_jargon_table()

    def _init_jargon_table(self):
        """専門用語辞書テーブルの初期化"""
        with self.engine.connect() as conn:
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id SERIAL PRIMARY KEY,
                    term TEXT UNIQUE NOT NULL,
                    definition TEXT NOT NULL,
                    domain TEXT,
                    aliases TEXT[],
                    related_terms TEXT[],
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_jargon_term ON {self.table_name} (LOWER(term))"))
            conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_jargon_aliases ON {self.table_name} USING GIN(aliases)"))
            conn.commit()

    def add_term(self, term: str, definition: str, domain: Optional[str] = None,
                 aliases: Optional[List[str]] = None, related_terms: Optional[List[str]] = None) -> bool:
        """用語を辞書に追加または更新"""
        try:
            with self.engine.connect() as conn:
                conn.execute(text(f"""
                    INSERT INTO {self.table_name}
                    (term, definition, domain, aliases, related_terms)
                    VALUES (:term, :definition, :domain, :aliases, :related_terms)
                    ON CONFLICT (term) DO UPDATE SET
                        definition = EXCLUDED.definition,
                        domain = EXCLUDED.domain,
                        aliases = EXCLUDED.aliases,
                        related_terms = EXCLUDED.related_terms,
                        updated_at = CURRENT_TIMESTAMP
                """), {
                    "term": term, "definition": definition, "domain": domain,
                    "aliases": aliases or [], "related_terms": related_terms or []
                })
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error adding term: {e}")
            return False

    def lookup_terms(self, terms: List[str]) -> Dict[str, Dict[str, Any]]:
        """複数の用語を辞書から検索"""
        if not terms:
            return {}

        results = {}
        try:
            with self.engine.connect() as conn:
                placeholders = ', '.join([f':term_{i}' for i in range(len(terms))])
                query = text(f"""
                    SELECT term, definition, domain, aliases, related_terms
                    FROM {self.table_name}
                    WHERE LOWER(term) IN ({placeholders})
                    OR term = ANY(:aliases_check)
                """)
                params = {f"term_{i}": term.lower() for i, term in enumerate(terms)}
                params["aliases_check"] = terms

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
        """複数の用語を一括削除"""
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
                        text(f"DELETE FROM {self.table_name} WHERE term = :term"),
                        {"term": term}
                    )
                    deleted += result.rowcount or 0
        except Exception as e:
            logger.error(f"Bulk delete error: {e}")
            return deleted, len(terms) - deleted
        return deleted, errors

    def get_all_terms(self) -> List[Dict[str, Any]]:
        """全ての用語を取得"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(f"SELECT * FROM {self.table_name} ORDER BY term")).fetchall()
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

        # データベース接続
        self.engine = create_engine(pg_url) if pg_url else None

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
        # 第1段階：緩い候補抽出プロンプト（定義なし）
        self.candidate_extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", """あなたは専門用語候補の抽出エキスパートです。
与えられたテキストから、専門用語の可能性がある単語・フレーズを幅広く抽出してください。

抽出基準（緩めに判定）:
- 技術的・専門的な概念を表す可能性がある語句
- 業界用語、学術用語、技術用語の可能性があるもの
- 略語・頭字語
- カタカナ語、英語
- 複合語、専門的な動詞・名詞
- 文脈で特別な意味を持ちそうな一般語も含める

出力形式:
{format_instructions}

注意事項:
- この段階では緩めに判定し、可能性があるものは幅広く含める
- 「学習」「処理」「システム」「実装」なども文脈次第で含める
- 信頼度は専門用語らしさを0.0-1.0で表現
- この段階では定義は不要（用語名のみ）"""),
            ("user", "以下のテキストから専門用語候補を抽出してください:\n\n{text}")
        ])

        # 第2段階：専門用語選別プロンプト
        self.technical_term_filter_prompt = ChatPromptTemplate.from_messages([
            ("system", """抽出された候補から、本当に専門用語として適切なものだけを選別してください。

専門用語として残すべきもの:
- 明確に技術的・専門的な用語
- 業界特有の確立された用語
- 略語・頭字語
- 複合語の専門用語

一般語として除外すべきもの（ただし類義語候補としては有用）:
- 汎用的すぎる単語（データ、システム、処理など）
- 一般的な動詞・形容詞
- 広すぎる概念

出力形式:
{format_instructions}

注意：除外した語も後で類義語検出に使うため、関連性が高いものは記憶しておいてください。"""),
            ("user", "以下の候補から専門用語を選別してください:\n{candidates_json}")
        ])

        # 類義語抽出プロンプト（専門用語＋候補プール全体を使用）
        self.synonym_prompt = ChatPromptTemplate.from_messages([
            ("system", """専門用語と候補語の間の類義語・関連語を検出してください。

⚠️ 重要な制約:
- 候補プールに存在しない語句は絶対に追加しないでください
- LLMの一般知識から類義語を生成せず、必ず候補プール内の語句のみを使用してください
- 文書に実際に出現した語句のみを類義語として検出してください

入力:
- technical_terms: 確定した専門用語
- candidates: すべての候補語（専門用語以外も含む）

類義語の種類（候補プール内に存在する場合のみ）:
1. 完全な同義語（同じ概念を指す）
2. 略語と正式名称の関係
3. 英語と日本語の対訳
4. 専門用語と一般的な表現の関係（例：「機械学習」と「学習」「訓練」）
5. 表記ゆれ・別名

出力形式:
- JSON形式で、各専門用語に対して候補プールから関連する語をリストアップ
- 候補プールに存在しない語句は絶対に含めないこと

例（誤った例）:
専門用語「電磁石」に対して「エレクトロマグネット」を追加 → ❌ 候補プールにない場合は追加禁止

正しい例:
専門用語「電磁石」に対して候補プール内の「磁石」「電磁」などのみを追加 → ✅"""),
            ("user", "専門用語:\n{technical_terms_json}\n\n候補プール:\n{candidates_json}\n\n上記から類義語関係を検出してください。候補プールに存在しない語句は絶対に追加しないでください。")
        ])

    async def extract_from_documents(self, file_paths: List[Path]) -> Dict[str, Any]:
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
        terms = await self._extract_terms(combined_text)

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

    async def _extract_terms(self, text: str) -> List[Dict[str, Any]]:
        """テキストから専門用語を抽出（2段階処理）"""
        chunk_size = getattr(self.config, 'llm_extraction_chunk_size', 3000)

        # テキストをチャンクに分割
        chunks = self._split_text(text, chunk_size)

        # ========== 第1段階: 緩い候補抽出 ==========
        all_candidates = []

        for i, chunk in enumerate(chunks, 1):
            logger.info(f"Stage 1: Extracting candidates from chunk {i}/{len(chunks)}")
            try:
                # 候補抽出プロンプトの実行
                chain = self.candidate_extraction_prompt | self.llm | self.json_parser
                result = await chain.ainvoke({
                    "text": chunk,
                    "format_instructions": self.json_parser.get_format_instructions()
                })

                # 結果の処理
                if hasattr(result, "terms"):
                    candidates = result.terms
                elif isinstance(result, dict):
                    candidates = result.get("terms", [])
                else:
                    candidates = []

                # 辞書形式に変換
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

                    all_candidates.append(cand_dict)

            except Exception as e:
                logger.error(f"Error in candidate extraction for chunk {i}: {e}")

        # 候補の重複除去
        unique_candidates = self._merge_duplicates(all_candidates)
        logger.info(f"Stage 1 complete: {len(unique_candidates)} unique candidates extracted")

        # ========== 第2段階: 専門用語の選別 ==========
        technical_terms = []

        if unique_candidates:
            logger.info("Stage 2: Filtering technical terms from candidates")
            technical_terms = await self._filter_technical_terms(unique_candidates)
            logger.info(f"Stage 2 complete: {len(technical_terms)} technical terms selected")

        # ========== 第3段階: RAGベースの定義生成（専門用語のみ） ==========
        if technical_terms and self.vector_store:
            logger.info("Stage 3: Generating definitions using RAG for technical terms")
            technical_terms = await self._generate_definitions_with_rag(technical_terms)
        else:
            logger.info("Stage 3: Skipping definition generation (no vector store)")

        # ========== 第4段階: 類義語検出（候補プール全体を使用） ==========
        if technical_terms and unique_candidates:
            logger.info("Stage 4: Detecting synonyms using full candidate pool")
            technical_terms = await self._detect_synonyms_with_candidates(
                technical_terms, unique_candidates
            )

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
            else:
                # 新規追加
                merged[key] = term.copy()
                merged[key]["headword"] = headword  # 元のケースを保持

        return list(merged.values())

    async def _filter_technical_terms(self, candidates: List[Dict]) -> List[Dict]:
        """候補から専門用語を選別（第2段階）"""
        if not candidates:
            return []

        try:
            # JSON文字列に変換
            candidates_json = json.dumps(candidates, ensure_ascii=False, indent=2)

            # 専門用語選別実行
            chain = self.technical_term_filter_prompt | self.llm | self.json_parser
            result = await chain.ainvoke({
                "candidates_json": candidates_json,
                "format_instructions": self.json_parser.get_format_instructions()
            })

            # 結果の処理
            if hasattr(result, "terms"):
                technical_terms = result.terms
            elif isinstance(result, dict):
                technical_terms = result.get("terms", [])
            else:
                technical_terms = []

            # 辞書形式に変換
            output = []
            for term in technical_terms:
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

            logger.info(f"Technical term filtering: {len(output)}/{len(candidates)} selected as technical terms")
            return output

        except Exception as e:
            logger.error(f"Technical term filtering failed: {e}")
            return candidates  # エラー時は全候補を返す

    async def _generate_definitions_with_rag(self, technical_terms: List[Dict]) -> List[Dict]:
        """RAGを使用して専門用語の定義を生成"""
        if not self.vector_store or not self.llm:
            logger.warning("Vector store or LLM not available for definition generation")
            return technical_terms

        # プロンプトを取得
        from .prompts import get_definition_generation_prompt
        from langchain_core.output_parsers import StrOutputParser

        prompt = get_definition_generation_prompt()
        chain = prompt | self.llm | StrOutputParser()

        for i, term in enumerate(technical_terms, 1):
            try:
                headword = term.get("headword", "")
                if not headword:
                    continue

                # ベクトルストアから関連文書を検索
                logger.debug(f"Searching context for: {headword}")
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
                    logger.info(f"[{i}/{len(technical_terms)}] Generated definition for: {headword}")
                else:
                    # コンテキストが見つからない場合は簡易定義
                    term["definition"] = f"{headword}（関連文書が見つかりません）"
                    logger.warning(f"[{i}/{len(technical_terms)}] No context found for: {headword}")

            except Exception as e:
                logger.error(f"Failed to generate definition for '{term.get('headword')}': {e}")
                term["definition"] = ""

        return technical_terms

    async def _detect_synonyms_with_candidates(self, technical_terms: List[Dict], all_candidates: List[Dict]) -> List[Dict]:
        """類義語の検出（候補プール全体を使用）"""
        if not technical_terms or not all_candidates:
            return technical_terms

        try:
            # 専門用語と候補をJSON化
            technical_terms_json = json.dumps(
                [{"term": t.get("headword"), "definition": t.get("definition", "")} for t in technical_terms],
                ensure_ascii=False,
                indent=2
            )
            candidates_json = json.dumps(
                [{"term": c.get("headword"), "definition": c.get("definition", "")} for c in all_candidates],
                ensure_ascii=False,
                indent=2
            )

            # 類義語検出
            chain = self.synonym_prompt | self.llm
            result = await chain.ainvoke({
                "technical_terms_json": technical_terms_json,
                "candidates_json": candidates_json
            })

            # 結果のパース
            if hasattr(result, "content"):
                result_text = result.content
            else:
                result_text = str(result)

            # JSONとして解析
            try:
                synonym_data = json.loads(result_text)

                # 元の専門用語リストに類義語を追加
                for term in technical_terms:
                    headword = term.get("headword", "")

                    # synonym_dataから該当する用語を探す
                    for syn_item in synonym_data:
                        if isinstance(syn_item, dict) and syn_item.get("term") == headword:
                            new_synonyms = syn_item.get("synonyms", [])
                            existing_synonyms = set(term.get("synonyms", []))
                            # 候補プールからの類義語も含める
                            term["synonyms"] = list(existing_synonyms | set(new_synonyms))
                            break

            except json.JSONDecodeError:
                logger.warning("Failed to parse synonym detection result")

            return technical_terms

        except Exception as e:
            logger.error(f"Synonym detection with candidates failed: {e}")
            return technical_terms

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

    # 用語抽出
    result = await extractor.extract_from_documents(files)
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