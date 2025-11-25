# 変更履歴

## UI高速化（2025/11/18）以降の変更

**期間**: 2025/11/18 〜 2025/11/25
**基準コミット**: 547073c (Optimize tab rendering to reduce WebSocket errors)
**コミット数**: 24件
**変更ファイル数**: 16件
**追加**: +1,976行 / **削除**: -600行

---

## 📊 変更統計

### 最も変更されたファイル
1. `src/rag/term_extraction.py` - 12回変更
2. `src/ui/dictionary_tab.py` - 6回変更
3. `src/core/rag_system.py` - 6回変更
4. `src/ui/documents_tab.py` - 4回変更
5. `src/rag/prompts.py` - 4回変更

### 変更カテゴリ別
- **機能追加**: 8件
- **バグ修正**: 14件
- **UI改善**: 2件

---

## 🚀 主要機能追加

### 1. Hugging Face ローカルLLM対応 (a10f482)
**日付**: 2025/11/25

**概要**:
- Azure OpenAIに加えて、Hugging FaceのローカルLLMをサポート
- デフォルトモデル: `tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3`

**主な機能**:
- 4-bit/8-bit量子化サポート（メモリ効率化）
- デバイス自動検出（CUDA/MPS/CPU）
- UI設定でAzure/Hugging Face切り替え可能
- ローカル埋め込みモデル対応（`intfloat/multilingual-e5-large`）

**変更ファイル**:
- `requirements.txt` - langchain-huggingface, transformers, torch等追加
- `src/rag/config.py` - HF設定フィールド追加
- `src/core/rag_system.py` - プロバイダー分岐初期化
- `src/ui/settings_tab.py` - LLMプロバイダー選択UI追加
- `.env.example` - HF設定例追加

### 2. ハイブリッド逆引き検索 (ba329f5)
**日付**: 2025/11/22

**概要**:
- 説明文から専門用語を特定する逆引き検索を大幅強化
- 3段階ハイブリッドアプローチ

**アーキテクチャ**:
1. **Phase 1**: Hybrid Search（キーワード検索 + ベクトル検索）をRRFで統合
2. **Phase 2**: 信頼度ベースフィルタリング（高: >0.9, 中: 0.6-0.9, 低: <0.6）
3. **Phase 3**: LLMリランキング（曖昧なケースのみ、約30%のクエリ）

**主な改善**:
- 専門用語辞書をPGVectorに同期（`type='jargon_term'`メタデータで識別）
- ドキュメント検索で専門用語を除外するフィルタ追加
- RRF式をHybridRetrieverと統一
- 軽量なLLMプロンプト（約170トークン）でコスト削減

**変更ファイル**:
- `src/rag/reverse_lookup.py` - 完全オーバーホール
- `src/rag/term_extraction.py` - `sync_to_vector_store()`追加
- `src/rag/retriever.py` - ベクトル検索に専門用語除外フィルタ
- `src/core/rag_system.py` - 初期化時に専門用語ベクトル化

### 3. カテゴリベースのコレクション管理 (c95fb75)
**日付**: 2025/11/19

**概要**:
- ドキュメントをカテゴリごとに整理するコレクション管理機能

**主な機能**:
- コレクション作成・削除・切り替え
- コレクションごとの専門用語辞書管理
- UI上でコレクション選択可能

**変更ファイル**:
- `src/ui/documents_tab.py` - コレクション管理UI
- `src/ui/state.py` - コレクション状態管理

### 4. 用語抽出の並列処理とプログレスバー (aa2553d)
**日付**: 2025/11/18

**概要**:
- `concurrent.futures.ThreadPoolExecutor`で並列処理
- Streamlitプログレスバーで進捗表示

**主な改善**:
- Stage 1（LLM抽出）の並列化
- Stage 2（LLMフィルタリング）のバッチ処理
- チャンクごとの進捗表示

**変更ファイル**:
- `src/rag/term_extraction.py`

---

## 🐛 バグ修正（時系列）

### 2025/11/25
- **d29ab32**: `initialize_rag_system`呼び出しシグネチャ修正（collection_name引数欠落）
- **fb0a8c7**: 無効なllm_provider値での"is not in list"エラー修正
- **a10f482**: 逆引きクエリ拡張のLLM出力クリーニング（tsquery構文エラー防止）

### 2025/11/22
- **55fe6f0**: jargon_dictionaryのUNIQUE制約マイグレーション修正

### 2025/11/21
- **67b0b73**: TermExtractorのテーブルスキーマ初期化追加
- **d01deb1**: コレクション切り替え後の古いrag_system使用バグ修正
- **a0dd7cb**: バルーン削除後のインデントエラー修正

### 2025/11/20
- **ef4e08e**: 用語抽出トレーシングのTypeError修正
- **ca19824**: 用語抽出でcollection_name未使用の修正

### 2025/11/19
- **07e16ed**: チャット・辞書タブにforce_collection_switch修正適用
- **988b816**: コレクション作成後の即時更新修正
- **8bb6091**: コレクション管理とドキュメント削除のバグ修正
- **f8f0708**: 回答生成で元質問の代わりに拡張クエリ使用

### 2025/11/18
- **b23ec1a**: Stage 2フィルタリングのバッチ処理修正
- **64b944b**: 複数チャンクからの用語重複時にbrief_definition統合
- **cd1360d**: brief_definition追加（Stage 2精度向上）

---

## 🎨 UI改善

### 2025/11/21
- **26d9122**: バルーンアニメーション削除（パフォーマンス向上）

### 2025/11/18
- **547073c**: タブレンダリング最適化（WebSocketエラー削減）
  - chat_tab.py: プログレスバー更新頻度削減（5件ごと）
  - dictionary_tab.py: MD5ハッシュで安定したボタンキー
  - documents_tab.py: ループをst.data_editorに置き換え

---

## 🔧 リファクタリング

### 2025/11/18
- **35b5d9f**: 未使用プロンプトテンプレート削除
- **01c3408**: ステージ出力ファイル追加、prompts.pyへリファクタ

### 2025/11/20
- **7b98980**: jargon_dictionaryにcollection_name対応（per-collectionの用語管理）

---

## 📝 詳細な変更ログ

### [d29ab32] Fix initialize_rag_system call signature
**変更ファイル**: `src/ui/settings_tab.py`

**問題**: 設定適用時に`TypeError: initialize_rag_system() missing 1 required positional argument: 'config_obj'`

**修正**: `initialize_rag_system(collection_name, config)`の2引数形式に統一

### [fb0a8c7] Fix 'is not in list' error
**変更ファイル**: `src/ui/settings_tab.py`

**問題**: llm_provider値が無効な場合に`ValueError: 'xxx' is not in list`

**修正**: ラジオボタンindex計算前に値の妥当性チェック追加

### [a10f482] Add Hugging Face local LLM support
**変更ファイル**:
- `.env.example`
- `requirements.txt`
- `src/core/rag_system.py`
- `src/rag/config.py`
- `src/rag/prompts.py`
- `src/rag/reverse_lookup.py`
- `src/ui/chat_tab.py`
- `src/ui/settings_tab.py`
- `src/ui/state.py`

**追加機能**:
1. LLMプロバイダー選択（Azure/Hugging Face）
2. HuggingFacePipeline + ChatHuggingFace統合
3. HuggingFaceEmbeddings対応
4. 量子化オプション（4-bit/8-bit）
5. デバイス自動検出（CUDA/MPS/CPU）

**設定例**:
```env
LLM_PROVIDER=huggingface
HF_MODEL_ID=tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3
HF_EMBEDDING_MODEL_ID=intfloat/multilingual-e5-large
HF_DEVICE=cuda
HF_LOAD_IN_4BIT=true
```

### [ba329f5] Implement hybrid reverse lookup
**変更ファイル**:
- `.claude/settings.local.json`
- `src/core/rag_system.py`
- `src/rag/retriever.py`
- `src/rag/reverse_lookup.py`
- `src/rag/term_extraction.py`

**主要変更**:
1. `JargonDictionaryManager.sync_to_vector_store()` - 専門用語のベクトル化
2. `HybridRetriever._vector_search()` - 専門用語除外フィルタ追加
3. `ReverseLookupEngine` - 完全オーバーホール
   - `_keyword_search()` - 辞書ベースキーワード検索
   - `_vector_search()` - ベクトル類似度検索
   - `_reciprocal_rank_fusion()` - RRF統合
   - `_llm_rerank()` - LLMリランキング

**設計判断**:
- 同じPGVectorテーブルでドキュメントと専門用語を管理（`type`メタデータで識別）
- RRF式をHybridRetrieverと統一（一貫性）
- 軽量プロンプト（約170トークン）でコスト最小化
- 専門用語がドキュメント検索に混入しないよう明示的フィルタ

### [55fe6f0] Fix UNIQUE constraint migration
**変更ファイル**: `src/rag/term_extraction.py`

**問題**: jargon_dictionaryテーブルのマイグレーション失敗

**修正**: UNIQUE制約の適切な処理

### [67b0b73] Add table schema initialization
**変更ファイル**: `src/rag/term_extraction.py`

**追加**: TermExtractor初期化時にテーブルスキーマ作成

### [d01deb1] Fix term extraction with stale rag_system
**変更ファイル**: `src/ui/dictionary_tab.py`

**問題**: コレクション切り替え後に古いrag_systemを使用

**修正**: 最新のrag_systemを取得してから用語抽出実行

### [a0dd7cb] Fix indentation error
**変更ファイル**: `src/ui/dictionary_tab.py`

**問題**: バルーン削除後のインデントエラー

**修正**: インデント修正

### [26d9122] Remove balloon animations
**変更ファイル**:
- `src/ui/dictionary_tab.py`
- `src/ui/documents_tab.py`

**理由**: パフォーマンス向上（不要なアニメーション削除）

### [ca19824] Fix collection_name not being used
**変更ファイル**:
- `src/core/rag_system.py`
- `src/rag/term_extraction.py`
- `src/ui/dictionary_tab.py`

**問題**: 用語抽出でcollection_nameが反映されない

**修正**: collection_nameを正しく渡すように修正

### [ef4e08e] Fix TypeError in term extraction tracing
**変更ファイル**: `src/rag/term_extraction.py`

**問題**: トレーシングコードでTypeError

**修正**: 型チェック追加

### [7b98980] Add collection_name support to jargon_dictionary
**変更ファイル**:
- `src/core/rag_system.py`
- `src/rag/term_extraction.py`

**追加**: コレクションごとの専門用語管理機能

### [07e16ed] Apply force_collection_switch fix
**変更ファイル**:
- `src/ui/chat_tab.py`
- `src/ui/dictionary_tab.py`

**修正**: チャット・辞書タブでのコレクション切り替え即時反映

### [988b816] Fix collection switching
**変更ファイル**:
- `src/ui/documents_tab.py`
- `src/ui/state.py`

**問題**: コレクション作成後に即時更新されない

**修正**: 作成直後に強制更新

### [8bb6091] Improve collection management
**変更ファイル**:
- `src/core/rag_system.py`
- `src/rag/ingestion.py`
- `src/rag/term_extraction.py`
- `src/ui/chat_tab.py`
- `src/ui/dictionary_tab.py`
- `src/ui/documents_tab.py`
- `src/utils/helpers.py`

**改善**:
- コレクション管理機能強化
- ドキュメント削除バグ修正
- ヘルパー関数追加

### [c95fb75] Add category-based collection management
**変更ファイル**:
- `src/ui/documents_tab.py`
- `src/ui/state.py`

**追加**: カテゴリベースのコレクション管理UI

### [f8f0708] Fix answer generation query
**変更ファイル**: `src/core/rag_system.py`

**問題**: 回答生成時に元質問を使用（拡張クエリを活用していない）

**修正**: augmented_queryを使用するよう変更

### [b23ec1a] Fix Stage 2 batch processing
**変更ファイル**:
- `src/rag/config.py`
- `src/rag/term_extraction.py`

**問題**: Stage 2フィルタリングが全候補を処理していない

**修正**: バッチ処理で全候補を処理

### [64b944b] Merge brief_definition
**変更ファイル**: `src/rag/term_extraction.py`

**追加**: 複数チャンクからの用語重複時にbrief_definitionを統合

### [cd1360d] Add brief_definition
**変更ファイル**:
- `src/rag/prompts.py`
- `src/rag/term_extraction.py`

**追加**: Stage 2精度向上のためbrief_definition追加

### [01c3408] Add stage output files and refactor
**変更ファイル**:
- `.claude/settings.local.json`
- `.env.example`
- `src/rag/config.py`
- `src/rag/prompts.py`
- `src/rag/term_extraction.py`

**リファクタ**:
- ステージ出力ファイル追加（デバッグ用）
- プロンプトをprompts.pyに統合

### [aa2553d] Add parallel processing and progress bars
**変更ファイル**: `src/rag/term_extraction.py`

**追加**:
- ThreadPoolExecutorで並列処理
- プログレスバー表示

### [35b5d9f] Remove unused prompt templates
**変更ファイル**: `src/rag/prompts.py`

**削除**: 未使用のプロンプトテンプレート

---

## 🔗 関連リンク

- **GitHubリポジトリ**: https://github.com/uchi736/advancedrag_llm
- **基準コミット**: 547073c (Optimize tab rendering to reduce WebSocket errors)
