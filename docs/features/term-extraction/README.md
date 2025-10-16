# 専門用語抽出機能

## 概要
技術文書から専門用語とその類義語を自動抽出し、RAGシステムで活用可能な辞書を構築する機能です。

## ドキュメント一覧

### [extraction-logic.md](./extraction-logic.md)
専門用語抽出の詳細ロジックとアルゴリズムの完全な仕様書。
- SudachiPyによる形態素解析
- C値・NC値アルゴリズム
- 6つの類義語検出手法
- LLMによる検証と定義生成
- 処理フローのシーケンス図

### [synonym-detection.md](./synonym-detection.md)
類義語・関連語検出の詳細仕様書（独立した完全版）。
- 6つの検出アルゴリズムの詳細実装
- スコアリングと信頼度レベル（最大20点評価）
- パフォーマンス最適化（キャッシュ・並列処理）
- 評価指標（Precision/Recall/F1スコア）
- トラブルシューティングガイド

## 関連ファイル

### 実装
- `/src/scripts/term_extractor_embeding.py` - メイン実装
- `/src/scripts/term_extractor_with_c_value.py` - C値/NC値実装
- `/src/scripts/term_clustering_analyzer.py` - クラスタリング分析

### 出力
- `/output/terms.json` - 抽出された専門用語辞書

## 使用方法
```bash
python src/scripts/term_extractor_embeding.py
```