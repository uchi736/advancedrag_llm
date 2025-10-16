# 高度なRAGシステム - ドキュメント

## 📚 ドキュメント構成

### 🏗️ [architecture/](./architecture/) - システム設計
システムアーキテクチャと基本設計に関するドキュメント。

- **[rag-fundamentals.md](./architecture/rag-fundamentals.md)** - RAGシステムの基礎概念
- **[hybrid-search.md](./architecture/hybrid-search.md)** - ハイブリッド検索の実装設計

### ⚙️ [features/](./features/) - 機能別ドキュメント
各機能の詳細仕様と使用方法。

#### 📝 [term-extraction/](./features/term-extraction/) - 専門用語抽出
- **[extraction-logic.md](./features/term-extraction/extraction-logic.md)** - 抽出ロジック詳細（シーケンス図付き）
- **[synonym-detection.md](./features/term-extraction/synonym-detection.md)** - 類義語検出アルゴリズム

#### 🕸️ [knowledge-graph/](./features/knowledge-graph/) - ナレッジグラフ
- **[planning.md](./features/knowledge-graph/planning.md)** - 実装計画とシステム設計

#### 📊 [evaluation/](./features/evaluation/) - 評価システム
- **[csv-format.md](./features/evaluation/csv-format.md)** - 評価データCSVフォーマット
- **[ui-guide.md](./features/evaluation/ui-guide.md)** - 評価UI操作ガイド

### 📖 [guides/](./guides/) - 実装ガイド
技術実装のベストプラクティスとガイドライン。

- **[azure-openai.md](./guides/azure-openai.md)** - Azure OpenAI統合ガイド
- **[japanese-nlp.md](./guides/japanese-nlp.md)** - 日本語NLPの課題と対策
- **[vector-search.md](./guides/vector-search.md)** - ベクトル検索実装ガイド
- **[reranking.md](./guides/reranking.md)** - リランキング技術

### 🔬 [research/](./research/) - 研究・実験
研究成果と実験レポート。

- **[research-plan.md](./research/research-plan.md)** - 研究計画
- **[lexical-mismatch.md](./research/lexical-mismatch.md)** - 語彙ミスマッチの検証

## 🚀 クイックリンク

### よく参照されるドキュメント
1. [専門用語抽出ロジック（シーケンス図付き）](./features/term-extraction/extraction-logic.md)
2. [評価システムUI操作ガイド](./features/evaluation/ui-guide.md)
3. [ハイブリッド検索実装](./architecture/hybrid-search.md)

### 開発者向け
1. [Azure OpenAI統合](./guides/azure-openai.md)
2. [日本語NLP実装](./guides/japanese-nlp.md)
3. [ベクトル検索ガイド](./guides/vector-search.md)

## 📝 ドキュメント更新履歴

- **2025-01-22**: ドキュメント構成を整理、シーケンス図を追加
- **2025-01-22**: 類義語検出ロジックを実装に合わせて更新
- **2025-01-20**: ナレッジグラフ計画書を作成