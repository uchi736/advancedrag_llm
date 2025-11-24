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

REVERSE_LOOKUP_QUERY_EXPANSION = """あなたは検索クエリ最適化の専門家です。
ユーザーの質問と、逆引き検索で特定された専門用語リストを使って、より正確で自然な検索クエリに再構築してください。

【改良方針】
1. 元の質問の意図を保持
2. 専門用語を自然に組み込む（単純な羅列ではなく文章として統合）
3. 簡潔で明確（80-120文字程度）
4. 検索エンジンが関連文書を見つけやすい表現

【入力情報】
元の質問: {original_query}

逆引きで特定された専門用語: {identified_terms}

【出力形式】
改良後の検索クエリのみを1行で出力してください。
「改良後の検索クエリ:」「クエリ:」などのラベルや前置きは一切含めないでください。
"""

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

def get_reverse_lookup_query_expansion_prompt():
    return ChatPromptTemplate.from_template(REVERSE_LOOKUP_QUERY_EXPANSION)

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

# Term extraction: Stage 1 - Candidate extraction
CANDIDATE_EXTRACTION_SYSTEM_PROMPT = """あなたは専門用語候補の抽出エキスパートです。
与えられたテキストから、専門用語の可能性がある単語・フレーズを幅広く抽出してください。

抽出基準（緩めに判定）:
- 技術的・専門的な概念を表す可能性がある語句
- 業界用語、学術用語、技術用語の可能性があるもの
- 略語・頭字語
- カタカナ語、英語
- 複合語、専門的な動詞・名詞
- 文脈で特別な意味を持ちそうな一般語も含める

**重要**: 各用語について、**周辺テキストから簡易的な定義・説明**を抽出してください。
- brief_definition: その用語が文中でどう使われているか、何を意味しているかを1-2文で記述
- 文脈から推測できる範囲で構いません
- 定義が見つからない場合は、用語の使用例や関連する説明を記述

出力形式:
{format_instructions}

注意事項:
- この段階では緩めに判定し、可能性があるものは幅広く含める
- 「学習」「処理」「システム」「実装」なども文脈次第で含める
- 信頼度は専門用語らしさを0.0-1.0で表現
- **brief_definition は必須**: 周辺テキストから文脈的な説明を抽出すること"""

CANDIDATE_EXTRACTION_USER_PROMPT = "以下のテキストから専門用語候補を抽出してください:\n\n{text}"

# Term extraction: Stage 2 - Technical term filtering
TECHNICAL_TERM_FILTER_SYSTEM_PROMPT = """抽出された候補から、本当に専門用語として適切なものだけを選別してください。

**判定の材料**:
各候補には `brief_definition` フィールドがあります。これは周辺テキストから抽出された簡易的な定義・説明です。
この定義を**必ず参照**して、その用語が専門用語かどうかを判断してください。

専門用語として残すべきもの:
- 明確に技術的・専門的な用語（brief_definitionから専門性が読み取れる）
- 業界特有の確立された用語（brief_definitionから特定分野での使用が明確）
- 略語・頭字語（brief_definitionから専門的な意味が確認できる）
- 複合語の専門用語（brief_definitionから専門的な概念が読み取れる）

一般語として除外すべきもの（ただし類義語候補としては有用）:
- 汎用的すぎる単語（brief_definitionから一般的な使用が読み取れる）
- 一般的な動詞・形容詞（brief_definitionから専門性が感じられない）
- 広すぎる概念（brief_definitionから特定の専門分野に絞られない）

**重要**: brief_definitionの内容を考慮せずに用語名だけで判断しないでください！
例えば「学習」は一般語に見えますが、brief_definitionが「機械学習アルゴリズムの訓練プロセス」であれば専門用語として扱うべきです。

出力形式:
{format_instructions}

**重要**: 選別結果には以下の2つのリストを含めてください:
1. **selected**: 専門用語として選ばれた用語のリスト
2. **rejected**: 除外された用語のリスト（各用語に除外理由 rejection_reason を含める）

除外理由の例:
- "brief_definitionから汎用的な使用が確認され、専門性が低い"
- "一般的な動詞で、brief_definitionにも専門的な意味が見られない"
- "概念が広すぎ、brief_definitionから特定分野に絞られない"
- "日常的な表現で、brief_definitionにも専門的文脈が不足"

注意：除外した語も後で類義語検出に使うため、関連性が高いものは記憶しておいてください。"""

TECHNICAL_TERM_FILTER_USER_PROMPT = "以下の候補から専門用語を選別してください:\n{candidates_json}"

# Term extraction: Stage 4a - Term synonym grouping
TERM_SYNONYM_GROUPING_SYSTEM_PROMPT = """専門用語リストから、同義語・類義語のグループを作成し、代表語に集約してください。

目的:
- 同じ概念を指す専門用語を1つの代表語にまとめる
- 表記ゆれや言語違い（日本語/英語）を統一
- 重複を削減して効率化
- 各用語の分野（ドメイン）を自動分類

グループ化の基準:
1. 完全な同義語（同じ概念）: 「機械学習」と「Machine Learning」
2. 略語と正式名称: 「ML」と「機械学習」
3. 英語と日本語: 「Deep Learning」と「深層学習」
4. 表記ゆれ: 「ヒートエクスチェンジャー」と「熱交換器」

代表語の選択基準（優先順位）:
1. 日本語 > 英語 > 略語
2. 定義が充実している方
3. より正式な名称（フルスペル）

分野（domain）の分類基準:
- 用語が属する主要な学問・産業分野を1つ選択
- 例: 医療、工学、法律、経済、情報技術、製造、エネルギー、環境、建築、化学など
- 複数の分野にまたがる場合は最も関連が強い分野を選択
- 一般的すぎる用語は「一般」とする

出力形式（JSON配列）:
[
  {{
    "headword": "機械学習",           // 代表語（日本語優先）
    "synonyms": ["Machine Learning", "ML"],  // 同義語リスト
    "definition": "...",               // 定義（代表語の定義を採用）
    "domain": "情報技術"              // 分野（必須）
  }},
  {{
    "headword": "深層学習",
    "synonyms": ["Deep Learning", "DL"],
    "definition": "...",
    "domain": "情報技術"
  }},
  {{
    "headword": "熱交換器",
    "synonyms": ["ヒートエクスチェンジャー"],
    "definition": "...",
    "domain": "工学"
  }}
]

注意:
- JSONは完全な形式で出力すること
- 定義は簡潔に（1-2文、100文字程度）まとめてください
- 文字列の途中で改行しないこと

注意事項:
- 同義語でないものは別グループにする
- 関連語（例: 機械学習と深層学習）は別グループ
- 各用語は必ずいずれかのグループに含める
- 代表語に選ばれなかった用語も必ずsynonymsに含める
- domainは必ず指定すること"""

TERM_SYNONYM_GROUPING_USER_PROMPT = "以下の専門用語リストをグループ化してください:\n\n{technical_terms_json}"

# Term extraction: Stage 4b - Synonym detection with candidates
SYNONYM_DETECTION_SYSTEM_PROMPT = """専門用語（代表語）と一般語候補の間の類義語・関連語を検出してください。

⚠️ 重要な制約:
- 候補プールに存在しない語句は絶対に追加しないでください
- LLMの一般知識から類義語を生成せず、必ず候補プール内の語句のみを使用してください
- 文書に実際に出現した語句のみを類義語として検出してください

入力:
- representative_terms: 代表語（専門用語間で既に集約済み）
- candidates: 一般語候補（専門用語として選ばれなかった語）

類義語の種類（候補プール内に存在する場合のみ）:
1. 専門用語と一般的な表現の関係（例：「機械学習」と「学習」「訓練」）
2. 略語・別名
3. 英語と日本語の対訳（候補に含まれる場合）

出力形式（JSON配列）:
[
  {{
    "term": "機械学習",
    "synonyms": ["学習", "訓練"]
  }},
  {{
    "term": "深層学習",
    "synonyms": []
  }},
  {{
    "term": "ニューラルネットワーク",
    "synonyms": ["ネットワーク"]
  }}
]

注意事項:
- JSONは完全な形式で出力すること
- 候補プールに存在しない語句は絶対に含めないこと
- 類義語が見つからない場合は空配列 [] を指定
- 文字列の途中で改行しないこと
- コメントは含めないこと"""

SYNONYM_DETECTION_USER_PROMPT = "代表語:\n{representative_terms_json}\n\n一般語候補:\n{candidates_json}\n\n上記から類義語関係を検出してください。候補プールに存在しない語句は絶対に追加しないでください。"

def get_candidate_extraction_prompt():
    """Stage 1: 候補抽出プロンプト"""
    return ChatPromptTemplate.from_messages([
        ("system", CANDIDATE_EXTRACTION_SYSTEM_PROMPT),
        ("user", CANDIDATE_EXTRACTION_USER_PROMPT)
    ])

def get_technical_term_filter_prompt():
    """Stage 2: 専門用語選別プロンプト"""
    return ChatPromptTemplate.from_messages([
        ("system", TECHNICAL_TERM_FILTER_SYSTEM_PROMPT),
        ("user", TECHNICAL_TERM_FILTER_USER_PROMPT)
    ])

def get_term_synonym_grouping_prompt():
    """Stage 4a: 専門用語間の類義語判定プロンプト"""
    return ChatPromptTemplate.from_messages([
        ("system", TERM_SYNONYM_GROUPING_SYSTEM_PROMPT),
        ("user", TERM_SYNONYM_GROUPING_USER_PROMPT)
    ])

def get_synonym_detection_prompt():
    """Stage 4b: 代表語と一般語の類義語判定プロンプト"""
    return ChatPromptTemplate.from_messages([
        ("system", SYNONYM_DETECTION_SYSTEM_PROMPT),
        ("user", SYNONYM_DETECTION_USER_PROMPT)
    ])