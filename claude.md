# Claude Code プロジェクト概要

## プロジェクト概要

LLMベースの専門用語抽出機能を備えたRAGシステム。LangGraphワークフローによる4+1ステージの用語抽出プロセスを実装。

## 主要ファイル

| ファイル | 役割 |
|----------|------|
| `src/rag/term_extraction.py` | LangGraphワークフロー、用語抽出の全ステージ実装 |
| `src/rag/prompts.py` | LLMプロンプトテンプレート（Self-Reflection含む） |
| `src/rag/retriever.py` | ハイブリッド検索（ベクトル+FTS） |
| `src/rag/config.py` | 設定管理（バッチサイズ、並列数など） |
| `src/core/rag_system.py` | RAGシステムのメインロジック |

## 用語抽出ワークフロー

```
Stage 1: 候補抽出（チャンク並列処理）
    ↓
Stage 2: 初期フィルタリング（バッチ処理）
    ↓
Stage 2.5: Self-Reflection Loop ←── 現在の重点開発領域
    ├─ 2.5a: 自己反省（問題点・漏れ用語検出）
    ├─ 2.5b: 改善実行（remove/keep/investigate）
    └─ 収束判定（confidence, ハッシュ, 重複チェック）
    ↓
Stage 3: RAG定義生成
    ↓
Stage 4: 類義語検出
```

## 最近の課題と修正履歴

### 解決済み

#### 1. Windows "too many file descriptors" エラー（2e3ba14）
- **問題**: Stage 1で大量チャンクを並列処理すると`select()`の512制限に到達
- **修正**: `asyncio.Semaphore(20)`で同時実行数を制限
- **設定**: `config.max_concurrent_llm_calls = 20`

#### 2. Stage 2.5 重複missing_terms検出（5a687cf）
- **問題**: 同じ漏れ用語が反復ごとに繰り返し報告される
- **原因**: LLMが`rejected_terms`を知らないため
- **修正**:
  - `rejected_terms_list`をSelf-Reflectionプロンプトに追加
  - プロンプトで「既に却下済みの用語は指摘しないこと」を明示

#### 3. RAG検索で文書チャンクが取得できない（5a687cf）
- **問題**: メタデータフィルタ`{"type": {"$ne": "jargon_term"}}`がNULL値も除外
- **原因**: 文書チャンクは`type`フィールドがNULL
- **修正**: フィルタなしで取得 → 手動で`jargon_term`除外

#### 4. Stage 2.5 無限ループ（4da674e）
- **問題**: 同じreflectionが336回繰り返される
- **原因**:
  - `operator.add` reducerによるリスト重複
  - 固定サンプリング（同じ50件を毎回評価）
  - 弱い収束条件
- **修正**:
  - カスタムreducer `_overwrite_list`に変更
  - ランダムサンプリング導入
  - 収束ガード追加（ハッシュ比較、問題重複80%）

#### 5. HNSW インデックス作成エラー（3e43674）
- **問題**: 次元数が不明でインデックス作成失敗
- **修正**: 自動次元検出を追加

#### 6. ベクトル検索でjargon_termが文書チャンクを埋没させる（未コミット）
- **問題**: fetch-then-filterパターン（k件取得→Python除外）で、jargon_term(142件)がdocument(6件)を圧倒し、フィルタ後0件になる
- **原因**: `similarity_search(k=20)` → 上位20件が全てjargon_term → 除外後0件
- **修正**: `similarity_search_without_jargon()` ヘルパーを追加。SQLのWHERE句で`cmetadata->>'type' != 'jargon_term'`を事前除外してからベクトル検索
- **影響範囲**: retriever.py, term_extraction.py x4箇所, rag_system.py（計6箇所を置換）
- **対象外**: reverse_lookup.pyはjargon_term検索が目的なので変更なし

#### 7. Stage 2 確信度トリアージ（未コミット）
- **問題**: Stage 2.5で全用語を反省対象にするのは非効率
- **修正**: Stage 2でper-term確信度(high/middle/low)を出力
  - high → Stage 2.5スキップ、middle → 反省対象、low → 即却下
- **関連修正**:
  - `_filter_technical_terms`の空candidates時unpackエラー修正
  - Stage 2.5でmiddle=0の場合は即スキップ

#### 8. CLI DB直接取得パス（未コミット）
- **問題**: `extract_terms.py`で`--input`なし時に「No supported files found」
- **修正**: `run_extraction_pipeline`にDBフォールバック追加（`document_chunks`テーブルから取得）

### 残課題・検討事項

#### 1. missing_terms上限問題
- **現状**: `unique_missing[:10]`で10件に固定
- **課題**: 用語数に関係なく固定上限は不適切かもしれない
- **検討**: 動的制限の導入？（ただし現状の重複検出修正で問題解消の可能性）

#### 2. Stage 4a 類義語グループ化
- **現象**: 空のグループが返される場合がある
- **原因**: LLM出力パース失敗 or 類義語なしと判断
- **状態**: 要調査

## 設定パラメータ

```python
# config.py
llm_extraction_chunk_size = 3000      # チャンクサイズ
stage2_batch_size = 50                # Stage2バッチサイズ
max_concurrent_llm_calls = 20         # 同時LLM呼び出し数
reflection_batch_size_terms = 100     # Stage2.5 terms/バッチ
reflection_batch_size_candidates = 50 # Stage2.5 candidates/バッチ
max_refinement_iterations = 3         # Stage2.5 最大反復回数
```

## Stage 2.5 フロー詳細

### 2.5a Self-Reflection
1. technical_terms と candidates をバッチ分割
2. 各バッチでLLM呼び出し → issues, confidence, missing_terms
3. バッチ結果を統合（重複排除）
4. `rejected_terms`をLLMに渡す（重複報告防止）

### 2.5b Refine Terms
1. **Remove処理**: action="remove"の用語を`rejected_terms`へ
2. **Missing処理**:
   - `_is_duplicate()`でチェック（technical_terms, candidates, rejected_terms）
   - RAG定義生成 → 増分Stage2フィルタ
   - 通過 → technical_terms / 却下 → rejected_terms
3. **Investigate処理**:
   - 一旦technical_termsから除外
   - RAG再定義 → 増分Stage2フィルタ
   - 結果に応じてtechnical_termsまたはrejected_termsへ

### 収束条件
- `confidence >= 0.9`
- 最大反復回数到達（デフォルト3回）
- 用語リストハッシュ不変
- 問題指摘80%以上重複
- 追加0件 & 削除0件
- missing=0 & confidence>=0.85

## 技術スタック

- **LLM**: Azure OpenAI (GPT-4o-mini)
- **Embedding**: Azure OpenAI text-embedding-3-small
- **Vector Store**: PostgreSQL + pgvector
- **Workflow**: LangGraph
- **UI**: Streamlit
- **PDF処理**: Azure Document Intelligence

## 開発時の注意点

1. **ベクトル検索フィルタ**: `similarity_search_without_jargon()`でSQLレベル事前除外。LangChainのメタデータフィルタはNULL除外問題があるため使用しない
2. **並列処理**: Windowsでは`asyncio.Semaphore`で同時実行数を制限
3. **LLMプロンプト**: 明示的な指示（「〜しないこと」）が重要
4. **State管理**: LangGraphのreducerは`operator.add`だとリスト重複に注意
