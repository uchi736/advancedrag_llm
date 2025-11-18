"""
RAG System Prompt Templates
This module contains all prompt templates used in the RAG system.
"""

from langchain_core.prompts import ChatPromptTemplate

# RAG-related prompts
JARGON_EXTRACTION = """あなたは専門用語抽出の専門家です。以下の質問を分析し、専門用語や技術用語を抽出してください。

【抽出対象】
- 業界固有の専門用語
- 技術的な専門用語
- 略語（アルファベット3文字以上）
- 製品名・サービス名
- 重要なキーワード

【除外対象】
- 一般的な日常用語
- 助詞・助動詞・接続詞
- 数字のみ

【出力形式】
抽出した専門用語を改行区切りで出力してください（最大{{max_terms}}個）。

質問: {question}

抽出された専門用語:"""

QUERY_AUGMENTATION = """あなたは専門技術文書検索のための質問最適化の専門家です。
元の質問と専門用語情報（定義・類義語・関連語）を活用して、RAG検索エンジンで最大限の関連情報を取得できる質問に再構築してください。

【改良方針】
1. **質問意図の保持**: 元の質問の核心的な意図を変えない
2. **専門用語情報の活用**: 提供された定義・類義語・関連語を質問文中に自然に組み込む（定義文をそのまま貼り付けない）
3. **類義語による検索語拡張**: 類義語を追加して検索範囲を拡大
4. **関連語による文脈補完**: 関連技術要素を含めることで関連情報の取得率を向上
5. **簡潔さと明確さ**: 50-100文字程度、冗長な説明的語句や重複表現は避ける

【入力情報】
▼ 元の質問:
{original_question}

▼ 専門用語情報（定義・類義語・関連語）:
{jargon_definitions}

【出力形式】
改良後の質問のみを出力（前置きや説明は不要）

改良後の質問:"""

QUERY_EXPANSION = """あなたは検索クエリ拡張の専門家です。元の質問を分析し、検索精度を向上させる複数の関連クエリを生成してください。

【拡張手法】
1. 同義語・類義語による言い換え
2. 具体例の追加
3. 上位概念・下位概念の追加
4. 関連する文脈の追加
5. 異なる視点からのアプローチ

【元の質問】
{original_query}

【出力指示】
- 3-5個のクエリを生成（番号付き）
- 各クエリは「1. クエリ内容」の形式で出力
- 元の質問の意図を保持
- 検索で異なる観点の情報が得られるよう多様化

拡張されたクエリ:"""

RERANKING = """あなたは文書関連性評価の専門家です。質問に対する各ドキュメントの関連性を詳細に分析し、最適な順序で並び替えてください。

【評価基準】
1. **直接的関連性** (40点): 質問に直接答える内容が含まれているか
2. **情報の質** (30点): 正確で詳細な情報が提供されているか
3. **文脈の一致** (20点): 質問の文脈・意図と合致しているか
4. **情報の新しさ** (10点): 最新の情報が含まれているか

【質問】
{question}

【ドキュメント一覧】
{documents}

【出力指示】
各ドキュメントの関連性スコア（0-100）を順番に数字のみで出力してください。
例: 85,72,95,63,78

関連性スコア:"""

ANSWER_GENERATION = """以下のコンテキスト情報を基に、質問に対して具体的で分かりやすい回答を作成してください。

コンテキスト:
{context}

専門用語定義（もしあれば）:
{jargon_definitions}

質問: {question}

回答:"""

# Convenience functions to get ChatPromptTemplate objects
def get_jargon_extraction_prompt(max_terms=5):
    template = JARGON_EXTRACTION.replace("{{max_terms}}", str(max_terms))
    return ChatPromptTemplate.from_template(template)

def get_query_augmentation_prompt():
    return ChatPromptTemplate.from_template(QUERY_AUGMENTATION)

def get_query_expansion_prompt():
    return ChatPromptTemplate.from_template(QUERY_EXPANSION)

def get_reranking_prompt():
    return ChatPromptTemplate.from_template(RERANKING)

def get_answer_generation_prompt():
    return ChatPromptTemplate.from_template(ANSWER_GENERATION)

# Term extraction prompts
DEFINITION_GENERATION_SYSTEM_PROMPT = """あなたは専門用語の定義作成の専門家です。

**定義作成の原則:**
1. **簡潔性**: 1〜3文で定義を完結させる
2. **正確性**: 技術的に正確な情報のみを使用
3. **明確性**: 専門家でない読者にも理解できる表現
4. **コンテキスト**: 提供された文脈を活用
5. **構造化**: 必要に応じて箇条書きや段落分け

**出力形式:**
- 定義本文のみを出力
- 余計な前置きや締めくくりは不要
"""

DEFINITION_GENERATION_USER_PROMPT_SIMPLE = """以下の専門用語の定義を作成してください。

**専門用語:** {term}

**関連コンテキスト:**
{context}

上記の情報を基に、正確で理解しやすい定義を作成してください。"""

def get_definition_generation_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", DEFINITION_GENERATION_SYSTEM_PROMPT),
        ("human", DEFINITION_GENERATION_USER_PROMPT_SIMPLE)
    ])