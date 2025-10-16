# 意味ベース類義語抽出機能（HDBSCAN）

## 概要

HDBSCANクラスタリングを使用して、専門用語の意味的に類似した用語を自動抽出し、類義語として登録する機能です。

## 特徴

### 2段階エンベディングアプローチ

#### ①専門用語（定義あり）
**エンベディング対象**: 用語 + 定義文
```
"ETC: 電動ターボコンプレッサは、燃料電池システムにおいて..."
```
→ 定義の意味も含めた類似度判定（高精度）

#### ②候補用語（LLM簡易フィルタ前）
**エンベディング対象**: 用語のみ
```
"過給機"
"コンプレッサ"
```
→ 用語名の意味だけで類似度判定（軽量）

### 検出できる類義語の例

- **専門用語同士**: ETC ↔ 電動ターボチャージャ ↔ 電動ターボコンプレッサ
- **専門用語↔一般語**: ETC ↔ 過給機 ↔ コンプレッサ
- **専門用語↔略語**: 燃料電池スタック ↔ FCスタック ↔ スタック

## データベーススキーマ

```sql
ALTER TABLE jargon_dictionary
ADD COLUMN IF NOT EXISTS semantic_synonyms TEXT[];

COMMENT ON COLUMN jargon_dictionary.synonyms
    IS '編集距離ベースの表記ゆれ（Levenshtein距離）';

COMMENT ON COLUMN jargon_dictionary.semantic_synonyms
    IS 'HDBSCANによる意味ベースの類義語（エンベディング空間での近傍）';
```

### 類義語の種類

| カラム名 | 検出方法 | 用途 | 例 |
|---------|---------|------|-----|
| `synonyms` | 編集距離 | 表記ゆれ | 電動ターボチャージャ ↔ 電動ターボチャージャー |
| `semantic_synonyms` | HDBSCAN | 意味的類義語 | ETC ↔ 電動ターボチャージャ ↔ 過給機 |
| `related_terms` | 包含・PMI共起 | 関連語 | コンプレッサシステム ⊃ コンプレッサ |

## 使い方

### 1. マイグレーション実行

```bash
psql -U <user> -d <database> -f migrations/add_semantic_synonyms.sql
```

### 2. 類義語抽出実行

```bash
cd src/scripts
python extract_semantic_synonyms.py
```

### 実行フロー

1. **専門用語読み込み**: DBから定義付き専門用語を読み込み
2. **候補用語読み込み**: `output/term_extraction_debug.json`からLLM簡易フィルタ前の候補用語を読み込み
3. **エンベディング生成**:
   - 専門用語: 用語+定義
   - 候補用語: 用語のみ
4. **UMAP次元圧縮**: 1536次元 → 20次元
5. **HDBSCANクラスタリング**: 密度ベースクラスタリング
6. **類義語抽出**: 同一クラスタ内でコサイン類似度計算（閾値0.85以上）
7. **DB更新**: `semantic_synonyms`列に保存

### 処理時間・コスト

| 項目 | 専門用語100個 + 候補用語900個 |
|------|------------------------------|
| 処理時間 | 約16-30秒 |
| Azure OpenAI APIコスト | 約$0.001 |

## パラメータ調整

`extract_semantic_synonyms.py`内で調整可能：

```python
synonyms_dict = analyzer.extract_semantic_synonyms_hybrid(
    specialized_terms=specialized_terms,
    candidate_terms=candidate_terms,
    similarity_threshold=0.85,  # コサイン類似度の閾値（0.0-1.0）
    max_synonyms=5              # 各用語の最大類義語数
)
```

### 推奨値

- **similarity_threshold**:
  - 0.9以上: 非常に類似（ほぼ同義）
  - 0.85-0.9: 類似（推奨）
  - 0.7-0.85: やや類似

- **max_synonyms**:
  - 3-5個: 高品質な類義語のみ（推奨）
  - 10個以上: ノイズが増える可能性

## トラブルシューティング

### 候補用語が読み込めない

`output/term_extraction_debug.json`が存在しない場合、専門用語のみでクラスタリングされます。

**解決策**:
1. 用語抽出を実行して中間ファイルを生成
2. または `load_candidate_terms_from_extraction()` を修正して別のデータソースを指定

### 用語数が少ない

専門用語が3個未満の場合、クラスタリングはスキップされます。

**解決策**:
1. まず用語抽出を実行してDBに専門用語を登録
2. 最低3個以上の専門用語が必要

### API

エラー

Azure OpenAI APIの呼び出しに失敗する場合があります。

**解決策**:
1. `.env`ファイルでAPIキー・エンドポイントを確認
2. レート制限に注意（大量の用語がある場合）

## 今後の拡張

- [ ] UIからの実行（設定タブにボタン追加）
- [ ] 類義語の承認/却下インターフェース
- [ ] 定期実行（スケジューラ）
- [ ] 類義語の品質評価メトリクス
