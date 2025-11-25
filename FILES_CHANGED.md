# 変更ファイル一覧

**期間**: 2025/11/18 〜 2025/11/25 (UI高速化以降)
**総変更ファイル数**: 16件
**合計**: +1,976行追加 / -600行削除

---

## 📊 変更頻度ランキング

| 順位 | ファイル | 変更回数 | 追加 | 削除 | 主な変更内容 |
|------|----------|----------|------|------|--------------|
| 1 | `src/rag/term_extraction.py` | 12 | +883 | -0 | 用語抽出機能の大幅強化 |
| 2 | `src/ui/dictionary_tab.py` | 6 | +54 | -0 | 辞書管理UI改善 |
| 3 | `src/core/rag_system.py` | 6 | +240 | -0 | RAGコア機能拡張 |
| 4 | `src/ui/documents_tab.py` | 4 | +123 | -0 | ドキュメント管理UI |
| 5 | `src/rag/prompts.py` | 4 | +482 | -0 | プロンプト整理 |
| 6 | `src/ui/state.py` | 3 | +26 | -0 | 状態管理 |
| 7 | `src/ui/settings_tab.py` | 3 | +138 | -0 | 設定UI拡張 |
| 8 | `src/ui/chat_tab.py` | 3 | +90 | -0 | チャット機能拡張 |
| 9 | `src/rag/config.py` | 3 | +33 | -0 | 設定追加 |
| 10 | `src/rag/reverse_lookup.py` | 2 | +349 | -0 | 逆引き検索強化 |
| 11 | `.env.example` | 2 | +16 | -0 | 環境変数例 |
| 12 | `.claude/settings.local.json` | 2 | +4 | -0 | Claude設定 |
| 13 | `src/utils/helpers.py` | 1 | +76 | -0 | ヘルパー関数追加 |
| 14 | `src/rag/retriever.py` | 1 | +23 | -0 | 検索機能改善 |
| 15 | `src/rag/ingestion.py` | 1 | +29 | -0 | ドキュメント取り込み |
| 16 | `requirements.txt` | 1 | +10 | -0 | 依存関係追加 |

---

## 📁 ファイル別詳細

### 1. src/rag/term_extraction.py
**変更回数**: 12回 | **追加**: +883行 | **役割**: 専門用語抽出エンジン

#### 主な変更内容
- **並列処理実装** (aa2553d)
  - ThreadPoolExecutorによるStage 1並列化
  - プログレスバー追加

- **Stage 2精度向上** (cd1360d, 64b944b, b23ec1a)
  - brief_definition追加
  - 重複用語のbrief_definition統合
  - バッチ処理で全候補を処理

- **コレクション対応** (7b98980, ca19824)
  - collection_nameサポート追加
  - per-collectionの用語管理

- **ベクトルストア同期** (ba329f5)
  - `sync_to_vector_store()`実装
  - 専門用語をPGVectorに同期（`type='jargon_term'`）

- **バグ修正** (55fe6f0, 67b0b73, ef4e08e)
  - UNIQUE制約マイグレーション
  - テーブルスキーマ初期化
  - トレーシングのTypeError修正

#### コミット履歴
- aa2553d: 並列処理とプログレスバー
- 01c3408: ステージ出力ファイル、prompts.pyリファクタ
- cd1360d: brief_definition追加
- 64b944b: brief_definition統合
- b23ec1a: Stage 2バッチ処理修正
- ef4e08e: トレーシングTypeError修正
- ca19824: collection_name未使用の修正
- 7b98980: collection_name対応
- 8bb6091: コレクション管理改善
- 67b0b73: テーブルスキーマ初期化
- 55fe6f0: UNIQUE制約マイグレーション
- ba329f5: ベクトルストア同期

---

### 2. src/ui/dictionary_tab.py
**変更回数**: 6回 | **追加**: +54行 | **役割**: 専門用語辞書管理UI

#### 主な変更内容
- **UIパフォーマンス改善** (26d9122, a0dd7cb)
  - バルーンアニメーション削除
  - インデントエラー修正

- **コレクション対応** (ca19824, 07e16ed, d01deb1)
  - collection_name反映
  - 強制切り替え対応
  - 古いrag_system使用バグ修正

- **機能改善** (8bb6091)
  - 辞書管理機能強化

#### コミット履歴
- d01deb1: 古いrag_system使用バグ修正
- a0dd7cb: インデントエラー修正
- 26d9122: バルーンアニメーション削除
- ca19824: collection_name未使用の修正
- 07e16ed: 強制切り替え対応
- 8bb6091: コレクション管理改善

---

### 3. src/core/rag_system.py
**変更回数**: 6回 | **追加**: +240行 | **役割**: RAGシステムのコア

#### 主な変更内容
- **Hugging Face対応** (a10f482)
  - `_init_huggingface_models()`実装
  - LLMプロバイダー分岐初期化
  - デバイス自動検出（CUDA/MPS/CPU）
  - 量子化サポート（4-bit/8-bit）

- **逆引き検索統合** (ba329f5)
  - ReverseLookupEngine初期化
  - 専門用語ベクトル化

- **回答生成修正** (f8f0708)
  - augmented_query使用に変更

- **コレクション対応** (ca19824, 7b98980, 8bb6091)
  - collection_name反映
  - コレクション管理改善

#### コミット履歴
- a10f482: Hugging Face対応
- ba329f5: 逆引き検索統合
- ca19824: collection_name未使用の修正
- 7b98980: collection_name対応
- 8bb6091: コレクション管理改善
- f8f0708: augmented_query使用

---

### 4. src/ui/documents_tab.py
**変更回数**: 4回 | **追加**: +123行 | **役割**: ドキュメント管理UI

#### 主な変更内容
- **コレクション管理** (c95fb75, 988b816)
  - カテゴリベースコレクション管理UI
  - 作成後の即時更新

- **バルーンアニメーション削除** (26d9122)
  - パフォーマンス向上

- **機能改善** (8bb6091)
  - ドキュメント削除バグ修正

#### コミット履歴
- 988b816: コレクション切り替え即時更新
- 8bb6091: コレクション管理改善
- c95fb75: カテゴリベースコレクション管理
- 26d9122: バルーンアニメーション削除

---

### 5. src/rag/prompts.py
**変更回数**: 4回 | **追加**: +482行 | **役割**: プロンプトテンプレート管理

#### 主な変更内容
- **逆引きクエリ拡張** (a10f482)
  - `REVERSE_LOOKUP_QUERY_EXPANSION`追加
  - LLM出力クリーニング対応

- **brief_definition追加** (cd1360d)
  - Stage 2精度向上用プロンプト

- **リファクタリング** (01c3408, 35b5d9f)
  - 用語抽出プロンプト統合
  - 未使用テンプレート削除

#### コミット履歴
- a10f482: 逆引きクエリ拡張プロンプト
- cd1360d: brief_definitionプロンプト
- 01c3408: prompts.pyリファクタ
- 35b5d9f: 未使用テンプレート削除

---

### 6. src/ui/state.py
**変更回数**: 3回 | **追加**: +26行 | **役割**: UI状態管理

#### 主な変更内容
- **逆引き検索状態** (a10f482)
  - `use_reverse_lookup`, `last_reverse_lookup`追加

- **コレクション管理** (c95fb75, 988b816)
  - コレクション状態管理
  - 強制切り替えフラグ

#### コミット履歴
- a10f482: 逆引き検索状態追加
- 988b816: コレクション切り替え即時更新
- c95fb75: カテゴリベースコレクション管理

---

### 7. src/ui/settings_tab.py
**変更回数**: 3回 | **追加**: +138行 | **役割**: システム設定UI

#### 主な変更内容
- **Hugging Face設定UI** (a10f482)
  - LLMプロバイダー選択ラジオボタン
  - HF設定セクション（モデルID、デバイス、量子化等）
  - リアルタイム切り替え対応

- **バグ修正** (fb0a8c7, d29ab32)
  - "is not in list"エラー修正
  - initialize_rag_systemシグネチャ修正

#### コミット履歴
- d29ab32: initialize_rag_systemシグネチャ修正
- fb0a8c7: "is not in list"エラー修正
- a10f482: Hugging Face設定UI追加

---

### 8. src/ui/chat_tab.py
**変更回数**: 3回 | **追加**: +90行 | **役割**: チャットUI

#### 主な変更内容
- **逆引き検索UI** (a10f482)
  - 逆引き検索チェックボックス追加
  - クエリ処理詳細表示

- **コレクション対応** (07e16ed, 8bb6091)
  - 強制切り替え対応
  - コレクション管理改善

#### コミット履歴
- a10f482: 逆引き検索UI追加
- 07e16ed: 強制切り替え対応
- 8bb6091: コレクション管理改善

---

### 9. src/rag/config.py
**変更回数**: 3回 | **追加**: +33行 | **役割**: システム設定

#### 主な変更内容
- **Hugging Face設定** (a10f482)
  - `llm_provider`, `hf_model_id`, `hf_device`等追加
  - 量子化オプション

- **用語抽出設定** (01c3408, b23ec1a)
  - `llm_extraction_chunk_size`, `stage2_batch_size`追加

#### コミット履歴
- a10f482: Hugging Face設定追加
- b23ec1a: Stage 2バッチサイズ設定
- 01c3408: 用語抽出設定追加

---

### 10. src/rag/reverse_lookup.py
**変更回数**: 2回 | **追加**: +349行 | **役割**: 逆引き検索エンジン

#### 主な変更内容
- **ハイブリッド検索実装** (ba329f5)
  - `_keyword_search()`: 辞書ベースキーワード検索
  - `_vector_search()`: ベクトル類似度検索
  - `_reciprocal_rank_fusion()`: RRF統合
  - `_llm_rerank()`: LLMリランキング

- **検索ソース追跡** (a10f482)
  - 各用語がkeyword/vector/hybridのどれで発見されたか追跡

#### コミット履歴
- a10f482: 検索ソース追跡追加
- ba329f5: ハイブリッド検索実装

---

### 11-16. その他のファイル

#### .env.example
**変更回数**: 2回 | **追加**: +16行
- HF設定例追加 (a10f482)
- 用語抽出設定例追加 (01c3408)

#### .claude/settings.local.json
**変更回数**: 2回 | **追加**: +4行
- Claude Code設定更新 (ba329f5, 01c3408)

#### src/utils/helpers.py
**変更回数**: 1回 | **追加**: +76行
- ヘルパー関数追加 (8bb6091)

#### src/rag/retriever.py
**変更回数**: 1回 | **追加**: +23行
- 専門用語除外フィルタ追加 (ba329f5)

#### src/rag/ingestion.py
**変更回数**: 1回 | **追加**: +29行
- コレクション対応改善 (8bb6091)

#### requirements.txt
**変更回数**: 1回 | **追加**: +10行
- HF依存関係追加 (a10f482)
  - langchain-huggingface
  - transformers
  - torch
  - accelerate
  - sentence-transformers
  - bitsandbytes

---

## 🔍 機能別ファイル分類

### RAGコア機能
- `src/core/rag_system.py` - メインシステム
- `src/rag/config.py` - 設定管理
- `src/rag/prompts.py` - プロンプト管理
- `src/rag/retriever.py` - 検索エンジン
- `src/rag/ingestion.py` - ドキュメント取り込み

### 専門用語機能
- `src/rag/term_extraction.py` - 用語抽出エンジン
- `src/rag/reverse_lookup.py` - 逆引き検索

### UI
- `src/ui/chat_tab.py` - チャット画面
- `src/ui/dictionary_tab.py` - 辞書管理画面
- `src/ui/documents_tab.py` - ドキュメント管理画面
- `src/ui/settings_tab.py` - 設定画面
- `src/ui/state.py` - 状態管理

### その他
- `src/utils/helpers.py` - ヘルパー関数
- `requirements.txt` - 依存関係
- `.env.example` - 環境変数例
- `.claude/settings.local.json` - Claude設定

---

## 📈 変更の影響範囲

### 機能追加の影響（大）
- **term_extraction.py**: 用語抽出機能の大幅強化（並列処理、精度向上）
- **rag_system.py**: Hugging Face対応、逆引き検索統合
- **reverse_lookup.py**: ハイブリッド検索実装

### UI改善の影響（中）
- **settings_tab.py**: LLMプロバイダー選択UI
- **chat_tab.py**: 逆引き検索UI
- **documents_tab.py**: コレクション管理UI

### バグ修正の影響（小〜中）
- 各種エラー修正、動作安定化
