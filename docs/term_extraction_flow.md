# 用語抽出フロー

## シーケンス図

```mermaid
sequenceDiagram
    participant User as ユーザー
    participant UI as Streamlit UI
    participant TE as TermExtraction
    participant SE as StatisticalExtractor
    participant SR as SemReRank
    participant VS as VectorStore
    participant LLM as Azure OpenAI (LLM)
    participant ESS as ExtractSemanticSynonyms
    participant TCA as TermClusteringAnalyzer
    participant DB as PostgreSQL

    User->>UI: PDFアップロード
    UI->>TE: extract_terms_from_documents()

    Note over TE: Phase 1: 候補抽出
    loop 各ドキュメント
        TE->>SE: extract_candidates(text)
        SE->>SE: Sudachi形態素解析 (C mode)
        SE->>SE: 複合語抽出 (N-gram, 名詞連続)
        SE-->>TE: 候補用語 + 頻度
    end

    Note over TE: Phase 2: 統計スコア計算
    TE->>SE: calculate_tfidf(documents, candidates)
    SE-->>TE: TF-IDFスコア
    TE->>SE: calculate_cvalue(candidates, full_text)
    SE-->>TE: C-valueスコア
    TE->>SE: calculate_combined_scores (Stage A: seed)
    SE-->>TE: シードスコア (C-value重視)
    TE->>SE: calculate_combined_scores (Stage B: final)
    SE-->>TE: 基底スコア (TF-IDF重視)

    Note over TE: Phase 3: 略語ボーナス
    TE->>TE: 略語パターン判定 (^[A-Z]{2,5}$)
    TE->>TE: 略語スコア × 1.3

    Note over TE: Phase 4: SemReRank
    TE->>SR: enhance_scores(candidates, base_scores, seed_scores)
    SR->>SR: グラフ構築 (用語間類似度)
    SR->>SR: PageRank実行
    SR->>SR: 最終スコア = base × (1 + α × pagerank)
    SR-->>TE: 強化スコア

    Note over TE: Phase 5: 表記ゆれ・関連語検出
    TE->>SE: detect_variants(candidates)
    SE->>SE: Levenshtein距離 + カタカナ正規化
    SE-->>TE: 類義語マップ (表記ゆれ)

    TE->>SE: detect_related_terms(candidates, full_text)
    SE->>SE: 包含関係検出 (部分文字列)
    SE->>SE: PMI共起分析 (window=10)
    SE-->>TE: 関連語マップ (包含・共起)

    Note over TE: Phase 6: 軽量LLMフィルタ (略語以外)
    TE->>TE: 上位N%選択 + 全略語
    TE->>LLM: lightweight_llm_filter(candidates)
    LLM-->>TE: フィルタ通過用語

    Note over TE: Phase 7: RAG定義生成
    loop フィルタ通過用語
        TE->>VS: similarity_search(term, k=5)
        VS-->>TE: 関連文書チャンク
        TE->>LLM: generate_definition(term, context)
        LLM-->>TE: 定義文
    end

    Note over TE: Phase 8: 重量LLMフィルタ
    TE->>LLM: technical_term_judgment(term, definition)
    LLM-->>TE: {is_technical: true/false, confidence: 0.9}

    Note over TE: Phase 9: DB保存
    TE->>DB: INSERT INTO jargon_dictionary
    Note right of DB: term, definition,<br/>aliases (表記ゆれ),<br/>related_terms (包含・共起)

    Note over TE: Phase 10: 意味ベース類義語抽出
    TE->>ESS: extract_and_save_semantic_synonyms()
    ESS->>DB: SELECT term, definition, related_terms
    DB-->>ESS: 専門用語リスト
    ESS->>ESS: load_candidate_terms_from_extraction()
    Note right of ESS: term_extraction_debug.json<br/>から候補用語読み込み

    Note over ESS: LLM定義生成 (Option B)
    loop 候補用語
        ESS->>LLM: generate_definition(term)
        LLM-->>ESS: 40-50文字の定義
    end

    ESS->>TCA: extract_semantic_synonyms_hybrid()

    Note over TCA: Embedding生成
    TCA->>LLM: embed_documents(specialized: term + definition)
    LLM-->>TCA: 1536次元ベクトル
    TCA->>LLM: embed_documents(candidates: term + LLM定義)
    LLM-->>TCA: 1536次元ベクトル

    Note over TCA: 次元圧縮・クラスタリング
    TCA->>TCA: UMAP (1536→20次元, cosine)
    TCA->>TCA: HDBSCAN (min_cluster_size=20%, epsilon=0.5)

    Note over TCA: 類義語抽出
    loop 各専門用語
        TCA->>TCA: 同一クラスタ内の用語を検索
        TCA->>TCA: コサイン類似度計算 (threshold=0.50)
        TCA->>TCA: 自分自身を除外
        TCA->>TCA: related_termsに含まれる用語を除外
        TCA->>TCA: 上位10個を類義語として保存
    end

    Note over TCA: LLMクラスタ命名
    TCA->>LLM: name_cluster(terms_in_cluster)
    LLM-->>TCA: クラスタ名 (例: "軸受技術")

    TCA->>DB: UPDATE jargon_dictionary SET aliases, domain
    Note right of DB: aliases: 意味ベース類義語<br/>domain: クラスタ名

    TCA-->>ESS: {synonyms, clusters, cluster_names}
    ESS-->>TE: 類義語辞書
    TE-->>UI: 抽出完了
    UI-->>User: 専門用語一覧表示
```

## 主要フェーズ詳細

### Phase 1: 候補抽出
- **技術**: Sudachi形態素解析 (ハイブリッドアプローチ)
  - **Mode A (短単位)**: N-gram生成、品詞判定、形態素数計算
  - **Mode C (長単位)**: 自然な複合語抽出 (例: "舶用ディーゼルエンジン")
- **手法**:
  1. Mode C複合語抽出
  2. Mode A + N-gram (名詞連続パターン)
- **出力**: 候補用語 + 頻度

### Phase 2: 統計スコア計算
- **TF-IDF**: 文書全体での重要度
- **C-value**: 複合語としての専門性
- **2段階スコアリング**:
  - Stage A (seed): C-value重視 → SemReRankのシード選定用
  - Stage B (final): TF-IDF重視 → 最終スコア計算用

### Phase 3: 略語ボーナス
- **判定パターン**: `^[A-Z]{2,5}$`
- **ボーナス倍率**: 1.3倍
- **目的**: 技術文書で重要な略語を優先

### Phase 4: SemReRank
- **グラフ構築**: 用語間の意味的類似度
- **アルゴリズム**: PageRank
- **最終スコア**: `base_score × (1 + α × pagerank_score)`

### Phase 5: 表記ゆれ・関連語検出
#### 表記ゆれ (variants)
- Levenshtein距離
- カタカナ正規化
- 例: "コンピュータ" ↔ "コンピューター"

#### 関連語 (related_terms)
- **包含関係**: 部分文字列検出
  - 例: "ILIPS" ⊂ "ILIPS環境価値管理"
- **PMI共起分析**:
  - ウィンドウサイズ: 10単語
  - 閾値: PMI ≥ 2.0, 共起回数 ≥ 3

### Phase 6: 軽量LLMフィルタ
- **対象**: 略語以外の上位N%
- **目的**: 定義生成コストを削減
- **略語**: 無条件で次フェーズへ

### Phase 7: RAG定義生成
- **検索**: ベクトル類似度検索 (k=5)
- **略語対応**: クエリ拡張 (例: "ETC 略語")
- **LLM**: Azure OpenAI GPT-4
- **出力**: 専門用語の定義文

### Phase 8: 重量LLMフィルタ
- **判定**: 専門用語 vs 一般用語
- **出力**: `{is_technical: bool, confidence: float}`
- **バッチ処理**: 10件/バッチ

### Phase 9: DB保存
```sql
INSERT INTO jargon_dictionary (
  term,
  definition,
  aliases,          -- 表記ゆれ (Phase 5)
  related_terms     -- 包含・共起関係 (Phase 5)
)
```

### Phase 10: 意味ベース類義語抽出
#### Step 1: データ読み込み
- **専門用語**: DB (`term`, `definition`, `related_terms`)
- **候補用語**: `term_extraction_debug.json`

#### Step 2: LLM定義生成 (Option B)
- **対象**: 候補用語 (定義なし)
- **長さ**: 40-50文字
- **目的**: 候補用語の意味情報を充実化
- **効果**: F1スコア 83.3%, Recall 93.8%

#### Step 3: Embedding生成
- **専門用語**: `"{term}: {definition}"`
- **候補用語**: `"{term}: {LLM定義}"`
- **モデル**: text-embedding-3-small
- **次元**: 1536

#### Step 4: 次元圧縮・クラスタリング
- **UMAP**: 1536次元 → 20次元 (cosine距離)
- **HDBSCAN**:
  - `min_cluster_size`: データ数の20%
  - `cluster_selection_epsilon`: 0.5

#### Step 5: 類義語抽出
- **同一クラスタ内**で類似度計算
- **コサイン類似度**: threshold = 0.50
- **除外ルール**:
  1. 自分自身
  2. `related_terms`に含まれる用語 (包含・共起関係)
- **最大数**: 10件/用語

#### Step 6: LLMクラスタ命名
- **入力**: クラスタ内の用語リスト
- **出力**: クラスタ名 (例: "軸受技術", "環境・持続可能技術")

#### Step 7: DB更新
```sql
UPDATE jargon_dictionary SET
  aliases = [...],    -- 意味ベース類義語で上書き
  domain = '...'      -- クラスタ名
```

## データフロー

```
PDFファイル
  ↓
候補用語 (Phase 1-4)
  ↓
表記ゆれ検出 (Phase 5) → aliases (一時)
  ↓
関連語検出 (Phase 5) → related_terms
  ↓
LLMフィルタ + 定義生成 (Phase 6-8)
  ↓
DB保存 (Phase 9)
  term, definition, aliases (表記ゆれ), related_terms
  ↓
意味ベース類義語抽出 (Phase 10)
  ↓
DB更新
  aliases ← 意味ベース類義語 (上書き)
  domain ← クラスタ名
```

## 重要な注意点

### aliases フィールドの2段階更新
1. **Phase 9**: 表記ゆれ (Levenshtein距離ベース)
2. **Phase 10**: 意味ベース類義語 (HDBSCAN + LLM) で**上書き**

### related_terms vs aliases の違い
- **related_terms** (包含・共起):
  - 包含関係: "ILIPS" ⊂ "ILIPS環境価値管理"
  - PMI共起: 同じ文脈で頻繁に出現
  - **Phase 10で除外**: 類義語に含めない

- **aliases** (意味ベース類義語):
  - 同一クラスタ内の意味的に近い用語
  - コサイン類似度 ≥ 0.50
  - `related_terms`に含まれない用語のみ

### 最適化済みパラメータ
- **類似度閾値**: 0.50 (F1=83.3%, Recall=93.8%, Precision=75.0%)
- **HDBSCAN epsilon**: 0.5
- **最大類義語数**: 10件/用語

## ファイル構成

- **src/rag/term_extraction.py**: Phase 1-9のメインロジック
- **src/rag/advanced_term_extraction.py**: StatisticalExtractor実装
- **src/scripts/extract_semantic_synonyms.py**: Phase 10のエントリポイント
- **src/scripts/term_clustering_analyzer.py**: HDBSCAN + UMAP実装
- **output/term_extraction_debug.json**: 候補用語の中間ファイル
