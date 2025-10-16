# 用語抽出フロー 詳細仕様書

## 目次
1. [概要](#概要)
2. [アーキテクチャ](#アーキテクチャ)
3. [Phase 1: 候補抽出](#phase-1-候補抽出)
4. [Phase 2: 統計スコア計算](#phase-2-統計スコア計算)
5. [Phase 3: 略語ボーナス](#phase-3-略語ボーナス)
6. [Phase 4: SemReRank](#phase-4-semrerank)
7. [Phase 5: 表記ゆれ・関連語検出](#phase-5-表記ゆれ関連語検出)
8. [Phase 6: 軽量LLMフィルタ](#phase-6-軽量llmフィルタ)
9. [Phase 7: RAG定義生成](#phase-7-rag定義生成)
10. [Phase 8: 重量LLMフィルタ](#phase-8-重量llmフィルタ)
11. [Phase 9: DB保存](#phase-9-db保存)
12. [Phase 10: 意味ベース類義語抽出](#phase-10-意味ベース類義語抽出)
13. [パラメータ最適化結果](#パラメータ最適化結果)
14. [データフロー図](#データフロー図)

---

## 概要

本システムは、技術文書（PDF）から専門用語を自動抽出し、定義・類義語・関連語を付与してデータベースに保存するパイプラインです。

### 主要技術スタック
- **形態素解析**: Sudachi (Mode A + Mode C ハイブリッド)
- **統計手法**: TF-IDF, C-value
- **グラフアルゴリズム**: SemReRank (PageRank)
- **クラスタリング**: HDBSCAN + UMAP
- **LLM**: Azure OpenAI (GPT-4, text-embedding-3-small)
- **データベース**: PostgreSQL + pgvector

### 処理フロー概要
```
PDFアップロード
  ↓
Phase 1-4: 候補抽出 + スコアリング
  ↓
Phase 5: 表記ゆれ・関連語検出
  ↓
Phase 6-8: LLMフィルタ + 定義生成
  ↓
Phase 9: DB保存 (term, definition, aliases, related_terms)
  ↓
Phase 10: 意味ベース類義語抽出 (HDBSCAN) → aliases更新, domain設定
```

---

## アーキテクチャ

### クラス構成

#### `TermExtraction` (src/rag/term_extraction.py)
- **役割**: 全体のオーケストレーション
- **主要メソッド**:
  - `extract_terms_from_documents()`: メインエントリポイント
  - `_process_per_document_with_global_scoring()`: Phase 1-9の実行

#### `StatisticalExtractor` (src/rag/advanced_term_extraction.py)
- **役割**: 統計ベースの候補抽出・スコアリング
- **主要メソッド**:
  - `extract_candidates()`: 候補用語抽出
  - `calculate_tfidf()`: TF-IDFスコア計算
  - `calculate_cvalue()`: C-valueスコア計算
  - `detect_variants()`: 表記ゆれ検出
  - `detect_related_terms()`: 関連語検出 (包含・共起)

#### `SemReRank` (src/rag/semrerank.py)
- **役割**: グラフベースのスコア強化
- **主要メソッド**:
  - `enhance_scores()`: PageRankによるスコア強化

#### `TermClusteringAnalyzer` (src/scripts/term_clustering_analyzer.py)
- **役割**: 意味ベース類義語抽出
- **主要メソッド**:
  - `extract_semantic_synonyms_hybrid()`: HDBSCAN + UMAP
  - `name_clusters_with_llm()`: LLMによるクラスタ命名

---

## Phase 1: 候補抽出

### 目的
技術文書から専門用語候補を網羅的に抽出する。

### 技術: Sudachiハイブリッドアプローチ

#### Mode A (短単位)
- **用途**:
  - N-gram生成
  - 品詞判定
  - 形態素数計算
- **例**: "舶用ディーゼルエンジン" → ["舶用", "ディーゼル", "エンジン"]

#### Mode C (長単位)
- **用途**: 自然な複合語抽出
- **例**: "舶用ディーゼルエンジン" → ["舶用ディーゼルエンジン"]

### 抽出手法

#### 1. 正規表現パターン抽出
**対象**:
- 括弧内の略語: `（BMS）`, `(AVR)`
- 型式番号: `6DE-18`, `L28ADF`, `12V170`
- 化学式: `CO2`, `NOx`, `PM2.5`
- シンプルな略語: `BMS`, `AVR`, `EMS` (2-5文字の大文字)
- 略語+用語: `MPPT制御`, `BMS装置`
- 数値+単位: `1,400°C`, `20万回転`

**実装**: [advanced_term_extraction.py:145-175](../src/rag/advanced_term_extraction.py#L145-L175)

```python
patterns = [
    r'[（(][A-Z]{2,5}[）)]',              # 括弧内略語
    r'\b[0-9]+[A-Z]+[-_][0-9]+[A-Z]*\b', # 型式番号
    r'\b[A-Z]{2,5}\b',                    # 略語
    r'\b[A-Z]{2,5}(制御|装置|システム)\b', # 略語+用語
    # ... 他多数
]
```

#### 2. Mode C複合語抽出
**対象**: 名詞系の複合語（2文字以上）

**フィルタ条件**:
- 品詞: `名詞`
- 最小文字数: 2
- `_is_valid_term()` 通過

**実装**: [advanced_term_extraction.py:253-281](../src/rag/advanced_term_extraction.py#L253-L281)

```python
def _extract_with_mode_c_with_spans(self, text: str):
    tokens = self._safe_tokenize(text, self.sudachi_mode_c)
    for token in tokens:
        term = token.surface()
        pos = token.part_of_speech()[0]
        if pos == '名詞' and len(term) >= 2:
            if self._is_valid_term(term):
                # スパン位置を記録
                mode_c_terms[term].add((start, end))
```

#### 3. Mode A + N-gram抽出
**対象**: 名詞・接頭辞の連続パターン

**処理フロー**:
1. 文単位で分割（句点・改行で区切り）
2. Mode Aでトークン化
3. 名詞・接頭辞の連続を検出
4. N-gram生成 (n=2〜5)

**実装**: [advanced_term_extraction.py:288-350](../src/rag/advanced_term_extraction.py#L288-L350)

```python
def _extract_ngrams_with_spans(self, text: str):
    sentences = self._split_into_sentences(text)
    for sentence in sentences:
        tokens = self._safe_tokenize(sentence, self.sudachi_mode_a)

        # 名詞・接頭辞の連続を検出
        current_sequence = []
        for token in tokens:
            pos = token.part_of_speech()[0]
            if pos in ['名詞', '接頭辞']:
                current_sequence.append(token.surface())
            else:
                # N-gram生成
                for n in range(min_term_length, max_term_length + 1):
                    for i in range(len(current_sequence) - n + 1):
                        ngram = ''.join(current_sequence[i:i+n])
                        if self._is_valid_term(ngram):
                            ngrams[ngram].add(span)
```

#### 4. 複合名詞抽出
**対象**: 名詞・接頭辞・接尾辞の組み合わせ

**品詞パターン**:
- 名詞 + 名詞
- 接頭辞 + 名詞
- 名詞 + 接尾辞

**実装**: [advanced_term_extraction.py:383-444](../src/rag/advanced_term_extraction.py#L383-L444)

### スパンベース重複排除

**問題**: 異なる手法で同じ位置の用語が重複抽出される

**解決策**: スパン（開始位置, 終了位置）で重複を排除

**実装**: [advanced_term_extraction.py:213-233](../src/rag/advanced_term_extraction.py#L213-L233)

```python
def merge_candidates_by_spans(self, candidates_with_spans):
    # スパン集合のサイズ = ユニークな出現回数
    return {term: len(spans) for term, spans in candidates_with_spans.items()}
```

### 用語妥当性チェック

**除外条件**:
- ひらがなのみ
- 数字のみ
- 記号のみ
- ストップワード（"について", "こと", "もの" など）
- 1文字の用語（略語を除く）

**実装**: [advanced_term_extraction.py:560-610](../src/rag/advanced_term_extraction.py#L560-L610)

```python
def _is_valid_term(self, term: str) -> bool:
    # ひらがなのみ除外
    if re.match(r'^[\u3040-\u309F]+$', term):
        return False

    # ストップワード除外
    if term in self.stop_words:
        return False

    # 品詞チェック
    tokens = self._safe_tokenize(term, self.sudachi_mode_a)
    # 名詞・接頭辞・接尾辞のみ許可
    for token in tokens:
        pos = token.part_of_speech()[0]
        if pos not in ['名詞', '接頭辞', '接尾辞', '記号']:
            return False

    return True
```

### 最小頻度フィルタ

**設定**: `min_frequency = 1` (デフォルト)

全ドキュメントで1回以上出現した用語のみを候補とする。

### Phase 1 出力例

```python
{
    "電動ターボ機械": 15,
    "ガス軸受": 42,
    "BMS": 8,
    "舶用ディーゼルエンジン": 23,
    "CO2": 31,
    # ...
}
```

---

## Phase 2: 統計スコア計算

### 目的
候補用語の専門性・重要度を数値化する。

### TF-IDF (Term Frequency-Inverse Document Frequency)

**目的**: 文書全体での用語の重要度を評価

**計算式**:
```
TF(t, d) = (用語tの文書d内での出現回数) / (文書d内の総単語数)
IDF(t) = log(総文書数 / (用語tを含む文書数))
TF-IDF(t, d) = TF(t, d) × IDF(t)
```

**文書単位**: 文（句点で区切り）

**実装**: [advanced_term_extraction.py:780-853](../src/rag/advanced_term_extraction.py#L780-L853)

```python
def calculate_tfidf(self, documents: List[str], vocabulary: Dict[str, int]):
    # 各用語をMode Aでトークン化
    term_token_map = {}
    for term in vocabulary:
        term_tokens = self._safe_tokenize(term, self.sudachi_mode_a)
        term_token_map[term] = tuple([t.surface() for t in term_tokens])

    # 文書頻度カウント
    df = defaultdict(int)
    for doc in documents:
        tokens = self._safe_tokenize(doc, self.sudachi_mode_a)
        token_surfaces = tuple([t.surface() for t in tokens])

        for term, term_tokens in term_token_map.items():
            # N-gramで用語マッチング
            if self._contains_subsequence(token_surfaces, term_tokens):
                df[term] += 1

    # TF-IDF計算
    idf = {term: math.log(len(documents) / (df[term] + 1))
           for term in vocabulary}
    # ...
```

### C-value

**目的**: 複合語としての専門性を評価

**計算式**:
```
C-value(t) = log2(|t|) × freq(t)    (tが他の用語に含まれない場合)
C-value(t) = log2(|t|) × (freq(t) - Σ freq(s) / |S|)  (tが他の用語sに含まれる場合)
```

- `|t|`: 用語tの形態素数
- `freq(t)`: 用語tの出現頻度
- `S`: 用語tを含むより長い用語の集合

**実装**: [advanced_term_extraction.py:903-1018](../src/rag/advanced_term_extraction.py#L903-L1018)

```python
def calculate_cvalue(self, candidates: Dict[str, int], full_text: str):
    # 形態素数を計算
    term_lengths = {}
    for term in candidates:
        tokens = self._safe_tokenize(term, self.sudachi_mode_a)
        term_lengths[term] = len(tokens)

    # 包含関係を検出
    nested_terms = defaultdict(list)
    for term1 in candidates:
        for term2 in candidates:
            if term1 != term2 and term1 in term2:
                nested_terms[term1].append(term2)

    # C-value計算
    cvalue_scores = {}
    for term, freq in candidates.items():
        length = term_lengths[term]

        if term in nested_terms:
            # ネストされている場合
            parent_freq = sum(candidates[p] for p in nested_terms[term])
            adjusted_freq = freq - (parent_freq / len(nested_terms[term]))
            cvalue = math.log2(length + 1) * max(adjusted_freq, 0.1)
        else:
            cvalue = math.log2(length + 1) * freq

        cvalue_scores[term] = cvalue
```

### 2段階スコアリング

#### Stage A: Seed (シード選定用)
**重み**: C-value重視

**計算式**:
```
seed_score = 0.3 × TF-IDF + 0.7 × C-value
```

**用途**: SemReRankのシード選定（高品質な専門用語を優先）

#### Stage B: Final (最終スコア用)
**重み**: TF-IDF重視

**計算式**:
```
final_score = 0.7 × TF-IDF + 0.3 × C-value
```

**用途**: 最終的な用語ランキング

**実装**: [advanced_term_extraction.py:1020-1078](../src/rag/advanced_term_extraction.py#L1020-L1078)

```python
def calculate_combined_scores(self, tfidf_scores, cvalue_scores, stage="final"):
    if stage == "seed":
        # C-value重視（専門性優先）
        tfidf_weight = 0.3
        cvalue_weight = 0.7
    else:  # final
        # TF-IDF重視（重要度優先）
        tfidf_weight = 0.7
        cvalue_weight = 0.3

    combined = {}
    for term in tfidf_scores:
        # 正規化
        tfidf_norm = tfidf_scores[term] / max_tfidf
        cvalue_norm = cvalue_scores[term] / max_cvalue

        # 複合語ボーナス（形態素数に応じて最大1.5倍）
        morpheme_count = len(self._safe_tokenize(term, self.sudachi_mode_a))
        compound_bonus = min(1.0 + (morpheme_count - 1) * 0.25, 1.5)

        combined[term] = (tfidf_norm * tfidf_weight +
                          cvalue_norm * cvalue_weight) * compound_bonus

    return combined
```

### Phase 2 出力例

```python
seed_scores = {
    "舶用ディーゼルエンジン": 0.85,  # C-value高（複合語）
    "ガス軸受": 0.72,
    "BMS": 0.45,
    # ...
}

base_scores = {
    "CO2": 0.89,  # TF-IDF高（頻出）
    "ガス軸受": 0.78,
    "舶用ディーゼルエンジン": 0.73,
    # ...
}
```

---

## Phase 3: 略語ボーナス

### 目的
技術文書で重要な略語を優先的に抽出する。

### 略語判定パターン

**正規表現**: `^[A-Z]{2,5}$`

**条件**:
- 2〜5文字の大文字のみ
- 例: `BMS`, `AVR`, `EMS`, `MPPT`, `SFOC`

### ボーナス倍率

**設定**: `1.3倍`

```python
seed_scores[term] *= 1.3
base_scores[term] *= 1.3
```

**実装**: [term_extraction.py:512-527](../src/rag/term_extraction.py#L512-L527)

```python
abbreviation_pattern = re.compile(r'^[A-Z]{2,5}$')
for term in candidates:
    if abbreviation_pattern.match(term):
        if term in base_scores:
            base_scores[term] *= 1.3
        if term in seed_scores:
            seed_scores[term] *= 1.3
        logger.info(f"  [BONUS] {term}: abbreviation bonus applied (×1.3)")
```

### Phase 3 出力例

```python
base_scores = {
    "BMS": 0.585,  # 0.45 × 1.3
    "MPPT": 0.715,  # 0.55 × 1.3
    "CO2": 0.89,    # そのまま（略語パターン外）
    # ...
}
```

---

## Phase 4: SemReRank

### 目的
用語間の意味的関係を考慮してスコアを強化する。

### アルゴリズム: PageRank

**基本アイデア**: 他の重要な用語と類似している用語は重要

### 処理フロー

#### 1. シード選定
**基準**: `seed_scores`の上位N%

**パラメータ**: `seed_percentile` (デフォルト: 30%)

```python
n_seeds = max(10, int(len(candidates) * 0.3))
seed_terms = sorted(candidates, key=lambda t: seed_scores[t], reverse=True)[:n_seeds]
```

#### 2. グラフ構築
**ノード**: 全候補用語
**エッジ**: 意味的類似度 ≥ 閾値

**類似度計算**: Embedding + コサイン類似度

```python
# Embedding生成
embeddings = azure_openai_embeddings.embed_documents(candidates)

# 類似度行列
similarity_matrix = cosine_similarity(embeddings)

# グラフ構築（閾値: 0.5）
graph = {}
for i, term1 in enumerate(candidates):
    graph[term1] = {}
    for j, term2 in enumerate(candidates):
        if similarity_matrix[i][j] >= 0.5:
            graph[term1][term2] = similarity_matrix[i][j]
```

#### 3. PageRank実行
**初期値**: シード用語に高い初期スコア

**反復計算**:
```
PR(t) = (1 - d) + d × Σ (PR(u) × w(u, t) / Σ w(u, v))
```

- `d`: ダンピング係数 (0.85)
- `w(u, t)`: エッジの重み（類似度）

**収束条件**: 変化量 < 0.001 または 最大100回反復

#### 4. 最終スコア計算
```
enhanced_score = base_score × (1 + α × pagerank_score)
```

- `α`: PageRankの影響度 (デフォルト: 0.3)

**実装**: [semrerank.py](../src/rag/semrerank.py)

### Phase 4 出力例

```python
enhanced_scores = {
    "ガス軸受": 0.936,      # 0.78 × (1 + 0.3 × 0.65) = 0.936
    "電動ターボ機械": 0.842,  # "ガス軸受"と高類似度
    "BMS": 0.623,           # 孤立用語（類似度低）
    # ...
}
```

---

## Phase 5: 表記ゆれ・関連語検出

### 5-1. 表記ゆれ検出 (variants)

**目的**: 同じ意味の異なる表記を検出

#### 検出手法

##### 1. Levenshtein距離
**閾値**: 編集距離 ≤ 2

```python
import Levenshtein

for term1 in candidates:
    for term2 in candidates:
        if Levenshtein.distance(term1, term2) <= 2:
            variants[term1].add(term2)
```

##### 2. カタカナ正規化
**パターン**:
- 長音符の有無: "コンピュータ" ↔ "コンピューター"
- 小文字の統一: "ユーザ" ↔ "ユーザー"

```python
def normalize_katakana(term):
    # 長音符を統一
    term = term.replace('ー', '')
    # 小文字を統一
    term = term.translate(KATAKANA_SMALL_TO_NORMAL)
    return term
```

#### 実装
[advanced_term_extraction.py:612-642](../src/rag/advanced_term_extraction.py#L612-L642)

```python
def detect_variants(self, candidates: List[str]):
    variants = defaultdict(set)

    for i, cand1 in enumerate(candidates):
        for cand2 in candidates[i+1:]:
            if cand1 != cand2:
                # Levenshtein距離
                if Levenshtein.distance(cand1, cand2) <= 2:
                    variants[cand1].add(cand2)
                    variants[cand2].add(cand1)

                # カタカナ正規化
                norm1 = self.normalize_katakana(cand1)
                norm2 = self.normalize_katakana(cand2)
                if norm1 == norm2:
                    variants[cand1].add(cand2)
                    variants[cand2].add(cand1)

    return {k: list(v) for k, v in variants.items() if v}
```

### 5-2. 関連語検出 (related_terms)

**目的**: 包含関係・共起関係の用語を検出

#### 検出手法

##### 1. 包含関係
**条件**: 短い用語が長い用語に含まれる

**例**:
- "ILIPS" ⊂ "ILIPS環境価値管理プラットフォーム"
- "エンジン" ⊂ "舶用ディーゼルエンジン"

```python
for cand1 in candidates:
    for cand2 in candidates:
        if cand1 != cand2:
            # 包含チェック
            if cand1 in cand2 and len(cand2) > len(cand1) + 1:
                related[cand1].add(cand2)
                related[cand2].add(cand1)
```

##### 2. PMI共起分析 (Pointwise Mutual Information)

**ウィンドウサイズ**: 10単語

**計算式**:
```
PMI(x, y) = log2(P(x, y) / (P(x) × P(y)))
```

**閾値**:
- PMI ≥ 2.0
- 共起回数 ≥ 3

```python
# 共起カウント（ウィンドウ: 10単語）
for i, word in enumerate(words):
    if word in candidates:
        window_start = max(0, i - 10)
        window_end = min(len(words), i + 11)

        for j in range(window_start, window_end):
            if i != j and words[j] in candidates:
                cooccurrence_map[word][words[j]] += 1

# PMI計算
for cand1, related_counts in cooccurrence_map.items():
    for cand2, cooccur_count in related_counts.items():
        if cooccur_count >= 3:
            p_xy = cooccur_count / total_words
            p_x = word_freq[cand1] / total_words
            p_y = word_freq[cand2] / total_words

            pmi = math.log2(p_xy / (p_x * p_y))

            if pmi >= 2.0:
                related[cand1].add(cand2)
```

#### 実装
[advanced_term_extraction.py:646-733](../src/rag/advanced_term_extraction.py#L646-L733)

### Phase 5 出力例

```python
synonym_map = {
    "コンピュータ": ["コンピューター"],
    "ユーザ": ["ユーザー"],
    # ...
}

related_map = {
    "ILIPS": ["ILIPS環境価値管理", "ILIPS環境価値管理プラットフォーム"],
    "エンジン": ["ディーゼルエンジン", "舶用ディーゼルエンジン"],
    "ガス軸受": ["電動ターボ機械"],  # PMI共起
    # ...
}
```

---

## Phase 6: 軽量LLMフィルタ

### 目的
定義生成前に低品質な候補を除外し、コストを削減する。

### 対象選定

#### 略語
**処理**: 無条件で次フェーズへ（定義生成）

**理由**: 技術文書では略語が重要

#### 非略語
**処理**: 上位N%を選択 → 軽量フィルタ

**パラメータ**: `definition_generation_percentile` (デフォルト: 50%)

```python
abbreviations = [t for t in terms if self._is_abbreviation(t.term)]
non_abbreviations = [t for t in terms if not self._is_abbreviation(t.term)]

n_candidates = int(len(non_abbreviations) * 0.5)
candidate_terms = non_abbreviations[:n_candidates]
```

### フィルタロジック

**判定**: LLMに用語のみを提示し、専門用語かどうかを判定

**プロンプト**:
```
以下の用語は技術文書の専門用語として妥当ですか？
用語: {term}

回答をJSONで返してください:
{
  "is_technical": true/false,
  "reason": "理由"
}
```

**実装**: [term_extraction.py:800-870](../src/rag/term_extraction.py#L800-L870)

```python
async def _lightweight_llm_filter(self, terms: List[ExtractedTerm]):
    filtered = []

    for term in terms:
        try:
            prompt = f"以下の用語は技術文書の専門用語として妥当ですか？\n用語: {term.term}"
            response = await self.llm.ainvoke(prompt)
            result = self._parse_llm_json(response.content)

            if result.get("is_technical", False):
                filtered.append(term)
        except Exception as e:
            logger.error(f"Lightweight filter failed for '{term.term}': {e}")
            # エラー時は通過させる（保守的）
            filtered.append(term)

    return filtered
```

### 有効化/無効化

**設定**: `Config.enable_lightweight_filter` (デフォルト: True)

無効化すると、上位N%をそのまま次フェーズへ。

### Phase 6 出力例

**入力**: 100個の非略語候補（上位50%）+ 10個の略語
**出力**: 35個のフィルタ通過用語 + 10個の略語 = 45個

---

## Phase 7: RAG定義生成

### 目的
ベクトル検索でPDF内の関連文脈を取得し、LLMで定義を生成する。

### 処理フロー

#### 1. ベクトル検索
**クエリ**: 用語名（略語の場合は拡張クエリ）

**拡張クエリ例**:
- 通常: `"ガス軸受"`
- 略語: `"BMS 略語"`

**検索パラメータ**:
- `k=5`: 上位5件の関連チャンクを取得
- 最大コンテキスト長: 3000文字

```python
is_abbr = self._is_abbreviation(term.term)
search_query = f"{term.term} 略語" if is_abbr else term.term

docs = self.vector_store.similarity_search(search_query, k=5)
context = "\n\n".join([doc.page_content for doc in docs])[:3000]
```

#### 2. LLM定義生成
**モデル**: Azure OpenAI GPT-4

**プロンプト**:
```
以下の文脈から、専門用語の定義を生成してください。

用語: {term}
文脈:
{context}

定義は2-3文（100-150文字）で、技術的な正確性を重視してください。
```

**実装**: [term_extraction.py:616-653](../src/rag/term_extraction.py#L616-L653)

```python
from .prompts import get_definition_generation_prompt
from langchain_core.output_parsers import StrOutputParser

prompt = get_definition_generation_prompt()
chain = prompt | self.llm | StrOutputParser()

for term in terms_for_definition:
    try:
        docs = self.vector_store.similarity_search(search_query, k=5)
        if docs:
            context = "\n\n".join([doc.page_content for doc in docs])[:3000]
            definition = await chain.ainvoke({"term": term.term, "context": context})
            term.definition = definition.strip()
        else:
            # コンテキストが見つからない場合
            if is_abbr:
                term.definition = f"{term.term}（専門用語の略語）"
    except Exception as e:
        logger.error(f"Failed to generate definition for '{term.term}': {e}")
        if is_abbr:
            term.definition = f"{term.term}（専門用語の略語）"
```

#### 3. フォールバック処理

**略語の特別扱い**:
- コンテキストが見つからない場合: `"{term}（専門用語の略語）"`
- 定義生成失敗の場合: 同上

**非略語**:
- コンテキストが見つからない場合: `definition = ""`（次フェーズで除外）
- 定義生成失敗の場合: 同上

### Phase 7 出力例

```python
terms = [
    ExtractedTerm(
        term="ガス軸受",
        definition="ガス軸受は、回転軸と軸受の間に空気などの気体の薄い膜を形成し、非接触で軸を支持する軸受の一種です。潤滑油を使用せずに高速回転や高負荷に耐えられ、航空機や燃料電池自動車の電動ターボ機械などの軽量化と高効率化に貢献します。",
        score=0.936,
        # ...
    ),
    ExtractedTerm(
        term="BMS",
        definition="BMS（専門用語の略語）",
        score=0.623,
        # ...
    ),
    # ...
]
```

---

## Phase 8: 重量LLMフィルタ

### 目的
定義がある用語に対して、専門用語か一般用語かを厳密に判定する。

### 判定ロジック

**入力**: 用語 + 定義

**プロンプト**:
```
以下の用語と定義を見て、これが技術文書の専門用語かどうかを判定してください。

用語: {term}
定義: {definition}

以下の基準で判定してください:
- 専門性: 特定分野の専門知識が必要か
- 技術性: 技術的な概念や製品を指すか
- 一般性: 日常会話で使われる一般用語ではないか

回答をJSONで返してください:
{
  "is_technical": true/false,
  "confidence": 0.0-1.0,
  "reason": "理由"
}
```

**実装**: [term_extraction.py:656-694](../src/rag/term_extraction.py#L656-L694)

```python
from .prompts import get_technical_term_judgment_prompt
from langchain_core.output_parsers import StrOutputParser

terms_with_def = [t for t in terms if t.definition]

prompt = get_technical_term_judgment_prompt()
chain = prompt | self.llm | StrOutputParser()

batch_size = 10
technical_terms = []

for i in range(0, len(terms_with_def), batch_size):
    batch = terms_with_def[i:i+batch_size]
    batch_inputs = [{"term": t.term, "definition": t.definition} for t in batch]

    result_texts = await chain.abatch(batch_inputs)
    for term, result_text in zip(batch, result_texts):
        result = self._parse_llm_json(result_text)
        if result and result.get("is_technical", False):
            term.metadata["confidence"] = result.get("confidence", 0.0)
            term.metadata["reason"] = result.get("reason", "")
            technical_terms.append(term)
            logger.info(f"  [OK] {term.term}: 専門用語")
        else:
            logger.info(f"  [NG] {term.term}: 一般用語")

terms = technical_terms
```

### バッチ処理

**バッチサイズ**: 10件

**理由**: LLM APIの並列処理効率化

### Phase 8 出力例

**フィルタ前**: 45個の用語
**フィルタ後**: 28個の専門用語

除外例:
- "システム" (一般的すぎる)
- "技術" (抽象的)
- "装置" (非特定的)

---

## Phase 9: DB保存

### 目的
抽出した専門用語をデータベースに保存する。

### テーブルスキーマ

```sql
CREATE TABLE jargon_dictionary (
    id SERIAL PRIMARY KEY,
    term TEXT UNIQUE NOT NULL,
    definition TEXT NOT NULL,
    domain TEXT,
    aliases TEXT[],
    related_terms TEXT[],
    confidence_score FLOAT DEFAULT 1.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 保存データ

**フィールド**:
- `term`: 専門用語
- `definition`: LLM生成の定義
- `domain`: クラスタ名（Phase 10で更新）
- `aliases`: 表記ゆれ（Phase 5）→ Phase 10で意味ベース類義語に上書き
- `related_terms`: 包含・共起関係（Phase 5）
- `confidence_score`: LLMの信頼度（Phase 8）

### INSERT/UPDATE処理

**ON CONFLICT**: 既存用語は更新

```sql
INSERT INTO jargon_dictionary
    (term, definition, aliases, related_terms, confidence_score)
VALUES (:term, :definition, :aliases, :related_terms, :confidence_score)
ON CONFLICT (term) DO UPDATE
SET definition = EXCLUDED.definition,
    aliases = EXCLUDED.aliases,
    related_terms = EXCLUDED.related_terms,
    confidence_score = EXCLUDED.confidence_score,
    updated_at = CURRENT_TIMESTAMP
```

**実装**: [term_extraction.py:932-950](../src/rag/term_extraction.py#L932-L950)

```python
for term in terms:
    conn.execute(
        text(f"""
            INSERT INTO {self.jargon_table_name}
                (term, definition, aliases, related_terms, confidence_score)
            VALUES (:term, :definition, :aliases, :related_terms, :confidence_score)
            ON CONFLICT (term) DO UPDATE
            SET definition = EXCLUDED.definition,
                aliases = EXCLUDED.aliases,
                related_terms = EXCLUDED.related_terms,
                confidence_score = EXCLUDED.confidence_score,
                updated_at = CURRENT_TIMESTAMP
        """),
        {
            "term": term["headword"],
            "definition": term["definition"],
            "aliases": term.get("synonyms", []),
            "related_terms": term.get("related_terms", []),
            "confidence_score": term.get("metadata", {}).get("confidence", 1.0)
        }
    )
```

### デバッグ出力

**ファイル**: `output/term_extraction_debug.json`

**内容**: Phase 10で使用する候補用語リスト

```json
{
  "candidates_for_semrerank": {
    "ガス軸受": 42,
    "電動ターボ機械": 15,
    "BMS": 8,
    ...
  }
}
```

### Phase 9 出力例

**保存件数**: 28件

**DB状態**:
```
| term        | definition           | aliases           | related_terms         | domain |
|-------------|----------------------|-------------------|-----------------------|--------|
| ガス軸受    | ガス軸受は、回転軸と… | ["ガスベアリング"] | ["電動ターボ機械"]    | NULL   |
| BMS         | BMS（専門用語の略語）| []                | []                    | NULL   |
```

**注**: `domain`はNULL（Phase 10で更新）、`aliases`は表記ゆれ（Phase 10で意味ベース類義語に上書き）

---

## Phase 10: 意味ベース類義語抽出

### 目的
HDBSCAN クラスタリングで意味的に類似した用語をグループ化し、類義語とクラスタ名を付与する。

### 処理フロー概要

```
1. 専門用語読み込み (DB)
2. 候補用語読み込み (term_extraction_debug.json)
3. LLM定義生成 (候補用語)
4. Embedding生成 (専門用語 + 候補用語)
5. UMAP次元圧縮
6. HDBSCANクラスタリング
7. 類義語抽出
8. LLMクラスタ命名
9. DB更新 (aliases, domain)
```

### 10-1. データ読み込み

#### 専門用語読み込み
**ソース**: PostgreSQL `jargon_dictionary`

**取得フィールド**:
```sql
SELECT term, definition, related_terms
FROM jargon_dictionary
WHERE definition IS NOT NULL AND definition != ''
```

**データ形式**:
```python
specialized_terms = [
    {
        "term": "ガス軸受",
        "definition": "ガス軸受は、回転軸と...",
        "related_terms": ["電動ターボ機械"],
        "text": "ガス軸受: ガス軸受は、回転軸と..."
    },
    # ...
]
```

**実装**: [extract_semantic_synonyms.py:35-65](../src/scripts/extract_semantic_synonyms.py#L35-L65)

#### 候補用語読み込み
**ソース**: `output/term_extraction_debug.json`

**取得キー**: `candidates_for_semrerank`

**データ形式**:
```python
candidate_terms = [
    {
        "term": "電動ターボ機械",
        "text": "電動ターボ機械"
    },
    # ...
]
```

**実装**: [extract_semantic_synonyms.py:68-110](../src/scripts/extract_semantic_synonyms.py#L68-L110)

### 10-2. LLM定義生成 (Option B)

**対象**: 候補用語（定義なし）

**目的**: 候補用語に意味情報を付与し、クラスタリング精度を向上

**プロンプト**:
```
以下の専門用語の定義を1-2文（40-50文字）で生成してください。
技術的文脈や関連概念を含めてください。

用語: {term}

定義のみを返してください:
```

**実装**: [extract_semantic_synonyms.py:117-158](../src/scripts/extract_semantic_synonyms.py#L117-L158)

```python
async def generate_definitions_for_candidates(candidate_terms, llm):
    enriched = []

    prompt_template = ChatPromptTemplate.from_template(
        "以下の専門用語の定義を1-2文（40-50文字）で生成してください。\n"
        "技術的文脈や関連概念を含めてください。\n\n"
        "用語: {term}\n\n"
        "定義のみを返してください:"
    )

    for cand in candidate_terms:
        term = cand['term']
        try:
            response = await llm.ainvoke(prompt_template.format(term=term))
            definition = response.content.strip()
            text = f"{term}: {definition}"
        except Exception as e:
            logger.warning(f"Failed to generate definition for '{term}': {e}")
            text = cand.get('text', term)

        enriched.append({"term": term, "text": text})

    return enriched
```

**効果**: F1スコア 83.3%, Recall 93.8%, Precision 75.0%

### 10-3. Embedding生成

#### 専門用語
**テキスト**: `"{term}: {definition}"`

**例**: `"ガス軸受: ガス軸受は、回転軸と軸受の間に空気などの気体の薄い膜を形成し..."`

#### 候補用語
**テキスト**: `"{term}: {LLM定義}"`

**例**: `"電動ターボ機械: 電動モータで駆動される高速回転機械で、燃料電池システムのターボチャージャーや空冷システムの高速ブロワなどに用いられる。"`

#### Embedding モデル
**モデル**: Azure OpenAI `text-embedding-3-small`
**次元**: 1536

**実装**: [term_clustering_analyzer.py:264-280](../src/scripts/term_clustering_analyzer.py#L264-L280)

```python
# 専門用語のEmbedding
spec_texts = [t['text'] for t in specialized_terms]
spec_embeddings_list = self.embeddings.embed_documents(spec_texts)
spec_embeddings = np.array(spec_embeddings_list)

# 候補用語のEmbedding
cand_texts = [t.get('text', t['term']) for t in candidate_terms]
cand_embeddings_list = self.embeddings.embed_documents(cand_texts)
cand_embeddings = np.array(cand_embeddings_list)

# 統合
all_embeddings = np.vstack([spec_embeddings, cand_embeddings])
```

### 10-4. UMAP次元圧縮

**目的**: 1536次元を低次元に圧縮し、HDBSCANの精度向上

**パラメータ**:
- `n_components`: 20 (データ数に応じて調整: min(20, データ数/2))
- `n_neighbors`: 15 (データ数に応じて調整: min(15, データ数/3))
- `min_dist`: 0.1
- `metric`: cosine
- `random_state`: 42

**実装**: [term_clustering_analyzer.py:286-301](../src/scripts/term_clustering_analyzer.py#L286-L301)

```python
n_samples = len(all_embeddings)
n_components = min(20, max(2, n_samples // 2))
n_neighbors = min(15, max(2, n_samples // 3))

umap_reducer = umap.UMAP(
    n_components=n_components,
    n_neighbors=n_neighbors,
    min_dist=0.1,
    metric='cosine',
    random_state=42
)
reduced_embeddings = umap_reducer.fit_transform(normalized_embeddings)
```

### 10-5. HDBSCANクラスタリング

**アルゴリズム**: 階層的密度ベース空間クラスタリング

**パラメータ**:
- `min_cluster_size`: データ数の20% (最低2)
- `min_samples`: 1
- `cluster_selection_epsilon`: 0.5
- `metric`: euclidean (UMAP後の空間)

**実装**: [term_clustering_analyzer.py:303-312](../src/scripts/term_clustering_analyzer.py#L303-L312)

```python
min_cluster_size = max(2, int(n_samples * 0.2))

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=min_cluster_size,
    min_samples=1,
    cluster_selection_epsilon=0.5,
    metric='euclidean'
)
clusters = clusterer.fit_predict(reduced_embeddings)
```

**クラスタラベル**:
- `0, 1, 2, ...`: 通常クラスタ
- `-1`: ノイズクラスタ（類義語なし）

### 10-6. 類義語抽出

**対象**: 専門用語のみ（候補用語は類義語候補として使用）

**処理フロー**:
1. 専門用語のクラスタIDを取得
2. 同一クラスタ内の他の用語を検索
3. コサイン類似度を計算
4. 除外ルール適用（自分自身、related_terms）
5. **LLM判定による最終フィルタ** ← 新機能
6. 上位N個を類義語として保存

#### 除外ルール（基本フィルタ）

##### 1. 自分自身
```python
if other_term_name == term_name:
    continue
```

##### 2. 包含・共起関係 (related_terms)
```python
related_terms = spec_term.get('related_terms', [])
if other_term_name in related_terms:
    logger.debug(f"Skipping '{other_term_name}': in related_terms")
    continue
```

#### 類似度計算

**手法**: コサイン類似度（正規化済みEmbeddingの内積）

**閾値**: 0.50

```python
term_embedding = normalized_embeddings[idx]
other_embedding = normalized_embeddings[other_idx]
similarity = float(np.dot(term_embedding, other_embedding))

if similarity >= 0.50:
    # 類義語候補に追加
```

#### LLM判定による最終フィルタ

**目的**: コサイン類似度だけでは判別できない包含関係・異なる種類の技術を除外

**対象**:
- ✅ **専門用語同士**: 両方の定義を使って判定
- ✅ **専門用語 ↔ 候補用語**: 候補用語のLLM生成定義を使って判定

**判定基準**:

**類義語 (Synonyms)**:
- ✅ ほぼ同じ意味
- ✅ 言い換え（例: 「ガス軸受」↔「気体軸受」）
- ✅ 異なる表記（例: 「コンピュータ」↔「コンピューター」）
- ✅ 同じ対象を指す

**非類義語 (Not Synonyms)**:
- ❌ 包含関係（一方が他方の一部）
  - 例: 「ILIPS」⊂「ILIPS環境価値管理プラットフォーム」
  - 例: 「ホットプレス」⊂「拡散接合プロセス」
- ❌ 上位概念/下位概念
  - 例: 「エンジン」↔「ディーゼルエンジン」
- ❌ 関連語（共起するが意味は異なる）
  - 例: 「発電特性評価」↔「海流発電実証試験」
- ❌ 異なる種類の技術/手法
  - 例: 「ガス軸受」↔「磁気軸受」（異なる種類の軸受）

**実装**: [term_clustering_analyzer.py:296-365](../src/scripts/term_clustering_analyzer.py#L296-L365)

```python
async def llm_judge_synonym_with_definitions(
    self,
    term1: str,
    def1: str,
    term2: str,
    def2: str
) -> bool:
    """
    LLMで2つの用語が類義語かどうかを判定（両方の定義あり）
    """
    prompt_template = ChatPromptTemplate.from_template(
        "以下の2つの用語が類義語（ほぼ同じ意味を持つ）かどうかを判定してください。\n\n"
        "用語1: {term1}\n"
        "定義1: {def1}\n\n"
        "用語2: {term2}\n"
        "定義2: {def2}\n\n"
        "判定基準:\n"
        "- 類義語: ほぼ同じ意味、言い換え、異なる表記、同じ対象を指す\n"
        "- 非類義語: 包含関係（一方が他方の一部）、上位概念/下位概念、"
        "関連語（共起するが意味は異なる）、異なる種類の技術/手法\n\n"
        "例:\n"
        "- 類義語: 「コンピュータ」と「コンピューター」（表記ゆれ）\n"
        "- 類義語: 「ガス軸受」と「気体軸受」（言い換え）\n"
        "- 非類義語: 「ILIPS」と「ILIPS環境価値管理プラットフォーム」（包含関係）\n"
        "- 非類義語: 「ガス軸受」と「磁気軸受」（異なる種類の軸受）\n"
        "- 非類義語: 「拡散接合プロセス」と「真空ホットプレス」（プロセス全体 vs 装置/手段）\n\n"
        "回答をJSONで返してください:\n"
        '{{"is_synonym": true/false, "reason": "理由"}}'
    )
    # ... LLM呼び出し、JSON解析
```

**処理フロー（LLM判定統合版）**:

```python
for idx in range(spec_count):
    spec_term = specialized_terms[idx]
    term_name = spec_term['term']
    cluster_id = clusters[idx]

    # ノイズクラスタはスキップ
    if cluster_id == -1:
        continue

    # 同一クラスタ内の他の用語を検索
    same_cluster_indices = [...]

    similarities = []
    for other_idx in same_cluster_indices:
        other_term_name = all_terms[other_idx]['term']

        # 基本除外ルール
        if other_term_name == term_name:
            continue
        if other_term_name in spec_term.get('related_terms', []):
            continue

        # コサイン類似度
        similarity = float(np.dot(
            normalized_embeddings[idx],
            normalized_embeddings[other_idx]
        ))

        if similarity >= 0.50:
            similarities.append({
                'term': other_term_name,
                'similarity': similarity,
                'is_specialized': other_idx < spec_count
            })

    # LLM判定による最終フィルタ
    if use_llm_for_candidates and similarities:
        llm_filtered = []
        spec_def = spec_term.get('definition', '')

        for sim_item in similarities:
            candidate_term = sim_item['term']
            is_specialized = sim_item['is_specialized']

            # 候補用語の定義を取得
            if is_specialized:
                # 専門用語: DBから定義取得
                candidate_def = next(
                    (t.get('definition', '') for t in specialized_terms
                     if t['term'] == candidate_term),
                    ''
                )
            else:
                # 候補用語: LLM生成定義を抽出
                candidate_obj = next(
                    (t for t in all_terms[spec_count:]
                     if t['term'] == candidate_term),
                    None
                )
                text = candidate_obj.get('text', '') if candidate_obj else ''
                candidate_def = text.split(':', 1)[1].strip() if ':' in text else ''

            # LLM判定
            is_synonym = await self.llm_judge_synonym_with_definitions(
                term_name, spec_def, candidate_term, candidate_def
            )

            if is_synonym:
                llm_filtered.append(sim_item)

        similarities = llm_filtered

    # 類似度順にソート、上位10個のみ保存
    similarities.sort(key=lambda x: x['similarity'], reverse=True)
    if similarities:
        synonyms_dict[term_name] = similarities[:10]
```

#### 最大類義語数

**設定**: 10件/用語

**理由**: 類似度順にソートし、上位10件のみ保存

#### 効果検証結果

**LLM判定なし**:
- 類義語を持つ専門用語: 5件
- 誤判定例: 「ILIPS環境価値管理プラットフォーム」→ aliases: [ILIPS, ILIPS環境, ...] （包含関係を誤って類義語判定）

**LLM判定あり**:
- 類義語を持つ専門用語: 10件
- 除外された候補用語: 7件（全て包含関係・異なる種類を正しく除外）
  - 「ILIPS」⊂「ILIPS環境価値管理プラットフォーム」
  - 「蓄熱」⊂「固定層蓄熱システム」
  - 「ホットプレス」⊂「拡散接合プロセス」
  - 「ガス軸受」↔「磁気軸受」（異なる種類）

**精度向上**: Precision大幅改善、Recall維持

### 10-7. LLMクラスタ命名

**対象**: 専門用語が含まれるクラスタのみ

**入力**: クラスタ内の用語リスト

**プロンプト**:
```
以下の専門用語グループに適切なクラスタ名（技術分野名）を付けてください。

用語:
- {term1}
- {term2}
- {term3}
...

クラスタ名は、これらの用語が属する技術分野や概念を端的に表す名前にしてください（例: "軸受技術", "環境・持続可能技術"）。

クラスタ名のみを返してください:
```

**実装**: [term_clustering_analyzer.py:433-483](../src/scripts/term_clustering_analyzer.py#L433-L483)

```python
async def name_clusters_with_llm(self, cluster_terms_map: Dict[int, List[str]]):
    cluster_names = {}

    prompt_template = ChatPromptTemplate.from_template(
        "以下の専門用語グループに適切なクラスタ名（技術分野名）を付けてください。\n\n"
        "用語:\n{terms}\n\n"
        "クラスタ名は、これらの用語が属する技術分野や概念を端的に表す名前にしてください"
        "（例: 「軸受技術」、「環境・持続可能技術」）。\n\n"
        "クラスタ名のみを返してください:"
    )

    for cluster_id, terms in cluster_terms_map.items():
        terms_text = "\n".join([f"- {term}" for term in terms])

        try:
            response = await self.llm.ainvoke(prompt_template.format(terms=terms_text))
            cluster_name = response.content.strip()
            cluster_names[cluster_id] = cluster_name
            logger.info(f"Cluster {cluster_id}: {cluster_name}")
        except Exception as e:
            logger.error(f"Failed to name cluster {cluster_id}: {e}")
            cluster_names[cluster_id] = f"クラスタ{cluster_id}"

    return cluster_names
```

### 10-8. DB更新

**更新対象**: `aliases`, `domain`

**SQL**:
```sql
UPDATE jargon_dictionary
SET aliases = :aliases,
    domain = :domain,
    updated_at = CURRENT_TIMESTAMP
WHERE term = :term
```

**実装**: [term_clustering_analyzer.py:485-528](../src/scripts/term_clustering_analyzer.py#L485-L528)

```python
def update_semantic_synonyms_to_db(self, synonyms_dict, cluster_mapping, cluster_names):
    engine = create_engine(self.connection_string)

    with engine.connect() as conn:
        for term, synonyms in synonyms_dict.items():
            # 類義語リスト
            aliases = [s['term'] for s in synonyms]

            # クラスタ名取得
            cluster_id = cluster_mapping.get(term, -1)
            domain = cluster_names.get(cluster_id, None) if cluster_id >= 0 else None

            conn.execute(
                text(f"""
                    UPDATE {self.jargon_table_name}
                    SET aliases = :aliases,
                        domain = :domain,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE term = :term
                """),
                {
                    "term": term,
                    "aliases": aliases,
                    "domain": domain
                }
            )

        conn.commit()
```

### Phase 10 出力例

**DB更新後**:
```
| term             | definition           | aliases                          | related_terms              | domain              |
|------------------|----------------------|----------------------------------|----------------------------|---------------------|
| ガス軸受         | ガス軸受は、回転軸と… | ["電動ターボ機械", "軸受評価装置"] | ["磁気軸受"]               | 軸受技術            |
| ILIPS環境価値... | ILIPS環境価値管理... | ["ILIPS環境価値管理", "ILIPS"]   | ["価値管理プラットフォーム"] | 環境・持続可能技術  |
| BMS              | BMS（専門用語の略語）| []                               | []                         | NULL                |
```

**注**:
- `aliases`: 意味ベース類義語（Phase 5の表記ゆれから上書き）
- `related_terms`: 包含・共起関係（Phase 5から保持）
- `domain`: LLMによるクラスタ名
- BMSはノイズクラスタ → `domain = NULL`, `aliases = []`

---

## パラメータ最適化結果

### 類似度閾値最適化 (threshold_optimization.py)

**テスト範囲**: 0.50 〜 0.85 (0.05刻み)

**結果**:
```
閾値 = 0.50: Recall: 93.8%, Precision: 75.0%, F1: 83.3%, 抽出ペア: 20
閾値 = 0.55: Recall: 75.0%, Precision: 70.6%, F1: 72.7%, 抽出ペア: 17
閾値 = 0.60: Recall: 68.8%, Precision: 68.8%, F1: 68.8%, 抽出ペア: 16
閾値 = 0.65: Recall: 56.2%, Precision: 100.0%, F1: 72.0%, 抽出ペア: 9

最適閾値: 0.50 - F1スコア 83.3%
```

**採用値**: `similarity_threshold = 0.50`

### HDBSCAN パラメータ

**`min_cluster_size`**: データ数の20% (動的調整)

**理由**: データ数に応じて適切なクラスタサイズを設定

**`cluster_selection_epsilon`**: 0.5

**最適化前**: 0.3 → 8クラスタ（細かすぎる）
**最適化後**: 0.5 → 4クラスタ（適度）

---

## データフロー図

```
┌─────────────────────┐
│  PDFファイル        │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────────────────────────┐
│ Phase 1-4: 候補抽出 + スコアリング          │
│ - Sudachi (Mode A + C)                      │
│ - TF-IDF + C-value                          │
│ - 略語ボーナス                              │
│ - SemReRank                                 │
└──────────┬──────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────┐
│ Phase 5: 表記ゆれ・関連語検出               │
│ - aliases (表記ゆれ) ← Levenshtein距離     │
│ - related_terms (包含・共起) ← PMI分析      │
└──────────┬──────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────┐
│ Phase 6-8: LLMフィルタ + 定義生成           │
│ - 軽量フィルタ (略語以外)                   │
│ - RAG定義生成 (全通過用語)                  │
│ - 重量フィルタ (専門用語判定)               │
└──────────┬──────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────┐
│ Phase 9: DB保存                             │
│ - term, definition                          │
│ - aliases (表記ゆれ)                        │
│ - related_terms (包含・共起)                │
└──────────┬──────────────────────────────────┘
           │
           │ term_extraction_debug.json
           │ (候補用語リスト)
           ▼
┌─────────────────────────────────────────────┐
│ Phase 10: 意味ベース類義語抽出              │
│ 1. 専門用語読み込み (DB)                    │
│ 2. 候補用語読み込み (JSON)                  │
│ 3. LLM定義生成 (候補用語)                   │
│ 4. Embedding生成 (1536次元)                 │
│ 5. UMAP次元圧縮 (1536→20)                   │
│ 6. HDBSCANクラスタリング                    │
│ 7. 類義語抽出 (threshold=0.50)              │
│    - 自分自身を除外                         │
│    - related_termsの用語を除外              │
│ 8. LLMクラスタ命名                          │
│ 9. DB更新                                   │
│    - aliases ← 意味ベース類義語 (上書き)    │
│    - domain ← クラスタ名                    │
└──────────┬──────────────────────────────────┘
           │
           ▼
┌─────────────────────┐
│ 最終データ (DB)     │
│ - term              │
│ - definition        │
│ - aliases (意味)    │
│ - related_terms     │
│ - domain            │
└─────────────────────┘
```

---

## ファイル構成

### メインロジック
- **src/rag/term_extraction.py**: Phase 1-9のオーケストレーション
- **src/rag/advanced_term_extraction.py**: StatisticalExtractor実装
- **src/rag/semrerank.py**: SemReRank (PageRank)
- **src/scripts/extract_semantic_synonyms.py**: Phase 10エントリポイント
- **src/scripts/term_clustering_analyzer.py**: HDBSCAN + UMAP実装

### 補助ファイル
- **src/rag/config.py**: 全パラメータ設定
- **src/rag/prompts.py**: LLMプロンプトテンプレート
- **output/term_extraction_debug.json**: 候補用語の中間ファイル

### テスト/最適化
- **threshold_optimization.py**: 類似度閾値最適化スクリプト

---

## 重要な注意点

### aliases フィールドの2段階更新

1. **Phase 9**: 表記ゆれ (Levenshtein距離ベース)
   - 例: ["コンピューター"]

2. **Phase 10**: 意味ベース類義語 (HDBSCAN + LLM) で**上書き**
   - 例: ["電動ターボ機械", "軸受評価装置"]

### related_terms vs aliases の違い

#### related_terms (包含・共起)
- **包含関係**: "ILIPS" ⊂ "ILIPS環境価値管理"
- **PMI共起**: 同じ文脈で頻繁に出現
- **Phase 10での扱い**: 類義語から除外

#### aliases (意味ベース類義語)
- **同一クラスタ**: 意味的に近い用語
- **コサイン類似度**: ≥ 0.50
- **除外ルール**: `related_terms`に含まれない用語のみ

### Sudachi の制限

**最大入力サイズ**: 49149バイト

**対策**: `_safe_tokenize()` で文単位に自動分割

**実装**: [advanced_term_extraction.py:82-143](../src/rag/advanced_term_extraction.py#L82-L143)

### LLM定義生成の効果 (Option B)

**実装前**: 候補用語は用語名のみ → 意味情報不足 → クラスタリング精度低下

**実装後**: 40-50文字の定義生成 → F1=83.3%, Recall=93.8%

---

## まとめ

本システムは、10段階のパイプラインで技術文書から専門用語を高精度に抽出し、定義・類義語・関連語を自動的に付与します。

**主要な特徴**:
- Sudachi Mode A + C のハイブリッドアプローチ
- TF-IDF + C-value の2段階スコアリング
- SemReRankによるグラフベース強化
- HDBSCAN + LLM定義生成による高精度類義語抽出 (F1=83.3%)
- 包含関係と意味的類義語の明確な分離

**最適化パラメータ**:
- 類似度閾値: 0.50
- HDBSCAN epsilon: 0.5
- 最大類義語数: 10件/用語
