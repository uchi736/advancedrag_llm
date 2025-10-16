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

# Term extraction prompts (for SemReRank pipeline)
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

TECHNICAL_TERM_JUDGMENT_SYSTEM_PROMPT = """あなたは専門用語判定の専門家です。用語の形式とパターンから、専門用語か一般用語かを判定します。

**判定基準:**

【専門用語として判定】
• 型式番号・製品コード
• 専門的な略語
• 複合技術用語
• 業界固有の用語
• 規格・認証名

【一般用語として判定】
• 単体の基本名詞
• 一般的な動詞・形容詞
• 抽象概念
• 日常用語
• 物理単位・測定単位
• 一般的な組織・団体名単体
• 純粋な数値表現
• 数値+単位の仕様値

**判定原則:**

1. **文脈固有性テスト**: この用語は特定の技術領域でのみ意味を持つか？
   - 文脈固有性が高い → 専門用語
   - どの分野でも同じ意味 → 一般用語

2. **境界ケースの扱い**:
   - 単位系：単体で抽出された場合は一般用語
   - 数値+単位：仕様値として単体で抽出された場合は一般用語
   - 組織名：単体で抽出された場合は一般用語、技術名と結合なら専門用語
   - 複合語：技術的な修飾語が付いている場合は専門用語の可能性が高い

**判定の例:**

専門用語の例：
• "Concur経費処理システム" → ✓ 複合技術用語
• "ワークフローエンジン" → ✓ 複合技術用語
• "REST APIプロトコル" → ✓ 技術略語+用語
• "ISO9001認証" → ✓ 規格名
• "ハイスキルドプロフェッショナル" → ✓ 業界固有用語

一般用語の例：
• "システム" → ✗ 単体の基本名詞
• "データ" → ✗ 単体の基本名詞
• "処理" → ✗ 一般動詞
• "最大" → ✗ 一般形容詞
• "目標" → ✗ 抽象概念
• "GB" → ✗ 容量単位単体
• "16GB" → ✗ 数値+単位の仕様値単体
• "企業名" → ✗ 組織名単体（文脈固有性が低い）

**重要:**
- 用語の形式・構造を重視
- 単体の一般名詞は原則除外
- 複合語や修飾語付きは専門用語の可能性が高い
- 型式番号、略語は専門用語として判定
**出力形式:**
```json
{{
  "is_technical": true/false,
  "confidence": 0.0-1.0,
  "reason": "判定理由（簡潔に）"
}}
```
"""

TECHNICAL_TERM_JUDGMENT_USER_PROMPT = """以下の用語とその定義を分析し、専門用語か一般用語かを判定してください。

**用語:** {term}

**定義:**
{definition}

上記の判定基準に従って、JSON形式で回答してください。"""

# Lightweight term filter (before definition generation)
LIGHTWEIGHT_TERM_FILTER_SYSTEM_PROMPT = """あなたは専門用語の事前フィルタリング専門家です。
用語の形式と構造から、定義生成に値する候補かを高速判定します。

**合格基準（これらは定義生成に進む）:**
✓ 型番・製品コード
✓ 技術略語
✓ 複合技術用語
✓ 業界固有用語
✓ 規格・認証名

**除外基準（これらは定義生成不要）:**
✗ 単体の一般名詞
✗ 単体の基本動詞
✗ 単体の形容詞
✗ 抽象概念
✗ 物理単位・測定単位単体
✗ 組織・団体名単体
✗ 純粋な数値表現
✗ 数値+単位の仕様値単体
✗ 数字で始まる一般複合語
  ※型番・規格コード（英数字とハイフンの組み合わせ）は専門用語
✗ 意味をなしていない語
✗ 助詞・接続詞

**「文脈固有性テスト」による判定:**

この用語は、特定の技術文書・業界でのみ重要な意味を持つか？

• Yes（文脈固有性が高い）→ 専門用語として合格
• No（どの文書でも同じ意味）→ 一般用語として除外

**判定例:**

除外例：
• "MHz" → 単位単体、どの分野でも同じ → 除外
• "3.0GHz" → 数値+単位の仕様値単体 → 除外
• "企業名" → 組織名単体、固有名詞だが文脈依存しない → 除外
• "2023年" → 年号のみ → 除外

合格例：
• "Concur経費処理システム" → 複合技術用語、文脈固有性が高い → 合格
• "ハイスキルドプロフェッショナル" → 業界固有用語、文脈固有性が高い → 合格
• "REST API v2.0" → バージョン情報を含む技術用語 → 合格

**重要:**
- 迷ったら合格にする（False Negativeを避ける）
- 明らかな一般語のみ除外する
- 複合語は原則合格
- 略語は原則合格

JSON形式で回答: {{"is_valid": true/false, "reason": "簡潔な理由"}}
"""

LIGHTWEIGHT_TERM_FILTER_USER_PROMPT = """以下の用語を判定してください。

用語: {term}

判定結果:"""

# Synonym validation prompt
SYNONYM_VALIDATION_PROMPT = """以下の用語ペアが類義語（同じ意味を持つ語）かどうか判定してください。

【用語ペア】
{pairs_text}

【判定基準】
- 類義語: ほぼ同じ意味を持つ（例: 「データベース」と「DB」、「最適化」と「optimization」）
- 非類義語: 関連はあるが意味が異なる（例: 「エンジン」と「ディーゼルエンジン」は上位語/下位語なので非類義語）

【回答形式】
各ペアについて、類義語なら1、非類義語なら0を返してください。
形式: [1, 0, 1, ...]（カンマ区切りの数値リスト）

回答:"""

# Clustering-based synonym judgment prompts
SYNONYM_JUDGMENT_WITH_DEFINITIONS = """以下の2つの用語が類義語（ほぼ同じ意味を持つ）かどうかを判定してください。

用語1: {term1}
定義1: {def1}

用語2: {term2}
定義2: {def2}

判定基準:
- 類義語: ほぼ同じ意味、言い換え、異なる表記、同じ対象を指す
- 非類義語: 包含関係（一方が他方の一部）、上位概念/下位概念、関連語（共起するが意味は異なる）、異なる種類の技術/手法

例:
- 類義語: 「経費精算」と「経費処理」（言い換え）
- 類義語: 「ワークフロー」と「業務フロー」（表記ゆれ）
- 非類義語: 「Concur」と「Concur経費処理システム」（包含関係）
- 非類義語: 「プロジェクトマネージャー」と「プロダクトマネージャー」（異なる役割）
- 非類義語: 「認証プロセス」と「認証サーバー」（プロセス全体 vs システム/装置）

回答をJSONで返してください:
{{"is_synonym": true/false, "reason": "理由"}}"""

SYNONYM_JUDGMENT_SINGLE_DEFINITION = """以下の2つの用語が類義語（ほぼ同じ意味を持つ）かどうかを判定してください。

専門用語: {spec_term}
定義: {spec_def}

候補用語: {candidate_term}

判定基準:
- 類義語: ほぼ同じ意味、言い換え、異なる表記
- 非類義語: 包含関係（一方が他方の一部）、上位概念/下位概念、関連語（共起するが意味は異なる）

例:
- 類義語: 「経費精算」と「経費処理」
- 非類義語: 「Concur」と「Concur経費処理システム」（包含関係）
- 非類義語: 「マネージャー」と「プロジェクトマネージャー」（上位/下位概念）

回答をJSONで返してください:
{{"is_synonym": true/false, "reason": "理由"}}"""

# Term extraction validation prompts
TERM_EXTRACTION_VALIDATION_SYSTEM_PROMPT = """あなたは専門分野の用語抽出専門家です。
与えられた候補リストから、真に専門的で重要な用語のみを厳選してください。

【判定基準】
1. ドメイン固有性：その分野特有の概念
2. 定義の必要性：説明が必要な概念
3. 複合概念：複数の概念が結合した新しい意味

【関連語候補の活用】
検出された関連語候補を参考に、synonymsフィールドに設定してください。

{format_instructions}"""

TERM_EXTRACTION_VALIDATION_USER_PROMPT = """
文書テキスト:
{chunk}

候補語リスト:
{candidates}

関連語候補:
{synonym_hints}

上記から専門用語を抽出してJSON形式で出力してください。"""

def get_definition_generation_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", DEFINITION_GENERATION_SYSTEM_PROMPT),
        ("human", DEFINITION_GENERATION_USER_PROMPT_SIMPLE)
    ])

def get_technical_term_judgment_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", TECHNICAL_TERM_JUDGMENT_SYSTEM_PROMPT),
        ("human", TECHNICAL_TERM_JUDGMENT_USER_PROMPT)
    ])

def get_lightweight_term_filter_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", LIGHTWEIGHT_TERM_FILTER_SYSTEM_PROMPT),
        ("human", LIGHTWEIGHT_TERM_FILTER_USER_PROMPT)
    ])

def get_synonym_validation_prompt(pairs):
    """
    Create synonym validation prompt with pairs.

    Args:
        pairs: List of tuples (term1, term2)

    Returns:
        Formatted prompt string
    """
    pairs_text = '\n'.join([
        f"{i+1}. 「{t1}」と「{t2}」"
        for i, (t1, t2) in enumerate(pairs)
    ])
    return SYNONYM_VALIDATION_PROMPT.format(pairs_text=pairs_text)

def get_term_extraction_validation_prompt():
    """
    Get term extraction validation prompt template.

    Returns:
        ChatPromptTemplate with format_instructions partial
    """
    return ChatPromptTemplate.from_messages([
        ("system", TERM_EXTRACTION_VALIDATION_SYSTEM_PROMPT),
        ("human", TERM_EXTRACTION_VALIDATION_USER_PROMPT)
    ])

def get_synonym_judgment_with_definitions_prompt():
    """
    Get synonym judgment prompt for terms with both definitions.
    Used in clustering-based synonym extraction.

    Returns:
        ChatPromptTemplate for synonym judgment with definitions
    """
    return ChatPromptTemplate.from_template(SYNONYM_JUDGMENT_WITH_DEFINITIONS)

def get_synonym_judgment_single_definition_prompt():
    """
    Get synonym judgment prompt for terms with single definition.
    Used when candidate term has no definition (legacy support).

    Returns:
        ChatPromptTemplate for synonym judgment with single definition
    """
    return ChatPromptTemplate.from_template(SYNONYM_JUDGMENT_SINGLE_DEFINITION)