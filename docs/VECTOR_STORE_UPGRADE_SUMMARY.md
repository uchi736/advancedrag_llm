# Vector Store Upgrade Summary

## 概要
RAGシステムにChromaDBサポートを追加し、PostgreSQL + pgvectorが使用できない環境でも動作可能にしました。

## 実装内容

### 1. アダプターパターンの実装
- **ファイル**: `src/rag/vector_store_adapter.py`
- **内容**:
  - `VectorStoreAdapter`: 抽象基底クラス
  - `PGVectorAdapter`: PGVector実装
  - `ChromaDBAdapter`: ChromaDB実装
  - `VectorStoreFactory`: ファクトリークラス

### 2. 設定の拡張
- **ファイル**: `src/rag/config.py`
- **追加設定**:
  - `vector_store_type`: ベクトルストアの選択
  - `chroma_persist_directory`: ChromaDBのローカルストレージパス
  - `chroma_server_host/port`: ChromaDBサーバー接続設定

### 3. 既存コードの更新
- **RAGSystem** (`src/core/rag_system.py`):
  - VectorStoreFactoryを使用してベクトルストアを作成
  - 既存コードとの互換性を保持

- **JapaneseHybridRetriever** (`src/rag/retriever.py`):
  - アダプターとネイティブストアの両方に対応

- **IngestionHandler** (`src/rag/ingestion.py`):
  - アダプターインターフェースに対応

### 4. ドキュメント
- **設定ガイド**: `docs/vector-store-configuration.md`
- **サンプル環境変数**: `.env.example`
- **README更新**: 新機能の説明を追加

### 5. ツールとテスト
- **移行スクリプト**: `scripts/vector_store_migration.py`
- **テストスクリプト**: `tests/test_vector_stores.py`
- **使用例**: `examples/chromadb_usage.py`

### 6. 依存関係
- **requirements.txt**: ChromaDBを追加 (`chromadb==0.5.0`)

## 使用方法

### ChromaDBへの切り替え

1. **環境変数の設定**:
```env
VECTOR_STORE_TYPE=chromadb
CHROMA_PERSIST_DIRECTORY=./chroma_db
```

2. **自動切り替え**:
```bash
python scripts/vector_store_migration.py chromadb
```

3. **アプリケーションの起動**:
```bash
streamlit run app.py
```

## 互換性

- **後方互換性**: 既存のPGVectorユーザーは変更なしで動作
- **データ移行**: 自動移行はサポートされていないため、再インデックスが必要
- **API互換性**: 両実装は同じインターフェースを共有

## メリット

1. **柔軟性**: PostgreSQLが使えない環境でも動作
2. **開発効率**: ローカル開発時にPostgreSQLのセットアップ不要
3. **選択肢**: 用途に応じて最適なベクトルストアを選択可能
4. **拡張性**: 新しいベクトルストアの追加が容易

## 注意事項

1. **データの非互換性**: PGVectorとChromaDB間でデータ形式が異なる
2. **機能制限**: ChromaDB使用時はText-to-SQL機能のみ利用不可（SQL分析が必要な場合はPGVector推奨）
3. **パフォーマンス**: 用途により最適な選択が異なる

## 機能比較表（更新版）

| 機能 | PGVector | ChromaDB |
|------|----------|----------|
| ベクトル検索 | ✅ | ✅ |
| キーワード検索 | ✅（ネイティブ） | ✅（BM25） |
| ハイブリッド検索 | ✅ | ✅ |
| Text-to-SQL | ✅ | ❌ |
| 専門用語辞書 | ✅（PostgreSQL） | ✅（ChromaDB） |
| ドキュメント取り込み | ✅ | ✅ |
| クエリ拡張 | ✅ | ✅ |
| RAG Fusion | ✅ | ✅ |

## 新機能の実装

### ChromaDBでのキーワード検索
- **BM25 Retriever**: LangChain標準のBM25実装を使用
- **ハイブリッド検索**: ベクトル検索とBM25の結果をRRFで統合

### ChromaDBでの専門用語辞書
- 別のChromaDBコレクションとして実装
- PostgreSQL版と同じインターフェースを提供

## テスト方法

```bash
# 両方のベクトルストアをテスト
python tests/test_vector_stores.py

# ChromaDBのみテスト
VECTOR_STORE_TYPE=chromadb python tests/test_vector_stores.py
```

## トラブルシューティング

### ChromaDB接続エラー
- ChromaDBがインストールされているか確認: `pip install chromadb`
- ローカルストレージの書き込み権限を確認
- サーバーモードの場合、ChromaDBサーバーが起動しているか確認

### PGVector接続エラー
- PostgreSQLが起動しているか確認
- pgvector拡張がインストールされているか確認
- 接続情報が正しいか確認

## 今後の拡張可能性

- 他のベクトルストア（Pinecone, Weaviate, Qdrant等）の追加
- データ移行ツールの実装
- ベクトルストア間のベンチマーク機能