# ベクトル検索の完全ガイド

## ベクトル検索とは

ベクトル検索は、文書や文章を高次元ベクトル空間における点として表現し、ベクトル間の距離や角度を用いて意味的類似性を計算する検索手法です。従来のキーワード検索とは異なり、単語の出現頻度ではなく文の意味や文脈を考慮した検索が可能で、同義語や関連概念も効果的に検索できます。

## ベクトル検索の仕組み

### 1. 埋め込み（Embedding）の生成

テキストは埋め込みモデル（Embedding Model）によって高次元の密ベクトルに変換されます。

**主要な埋め込みモデル：**
- **OpenAI text-embedding-ada-002**: 1536次元、多言語対応
- **OpenAI text-embedding-3-small**: 1536次元、高性能・軽量
- **OpenAI text-embedding-3-large**: 3072次元、最高性能
- **Sentence-BERT**: 768次元、多言語BERT基盤
- **multilingual-E5**: 1024次元、多言語特化

### 2. ベクトル間の類似度計算

生成されたベクトル間の類似性を以下の指標で計算します：

#### コサイン類似度（推奨）
```
similarity = (A · B) / (||A|| × ||B||)
```
- 範囲: -1 ～ 1（1が最も類似）
- ベクトルの方向の類似性を測定
- テキストの意味的類似性に適している

#### ユークリッド距離
```
distance = sqrt(Σ(Ai - Bi)²)
```
- 値が小さいほど類似
- ベクトル間の絶対的な距離を測定
- 次元の呪いの影響を受けやすい

#### 内積（Dot Product）
```
similarity = A · B
```
- ベクトルの方向と大きさを考慮
- 正規化されていないベクトルで使用
- 高速計算が可能

## ベクトル検索の利点

### 1. 意味的類似性の理解
- 「車」「自動車」「クルマ」などの同義語を同一視
- 「犬」「ペット」「動物」などの階層関係を理解
- 文脈による意味の変化を考慮

### 2. 多言語対応
- 言語を跨いだ意味的検索が可能
- 翻訳なしでの多言語文書検索
- クロスリンガル情報検索の実現

### 3. ノイズに対する耐性
- タイポや表記揺れに頑健
- 文法的な違いを吸収
- 省略語や略語にも対応

### 4. 関連概念の発見
- 直接的な一致がない場合も関連情報を検索
- 概念の類推による検索拡張
- 創発的な関連性の発見

## ベクトル検索の実装

### データベース選択

#### 1. Pinecone
```python
import pinecone

# 初期化
pinecone.init(api_key="your-api-key")

# インデックス作成
index = pinecone.Index("your-index")

# ベクトル挿入
index.upsert(vectors=[
    ("doc1", [0.1, 0.2, 0.3, ...], {"text": "document content"})
])

# 検索実行
results = index.query(
    vector=[0.1, 0.2, 0.3, ...],
    top_k=5,
    include_metadata=True
)
```

#### 2. Weaviate
```python
import weaviate

client = weaviate.Client("http://localhost:8080")

# スキーマ定義
schema = {
    "class": "Document",
    "vectorizer": "text2vec-openai",
    "properties": [
        {"name": "content", "dataType": ["text"]}
    ]
}

# データ追加
client.data_object.create({
    "content": "document text"
}, class_name="Document")

# 検索実行
result = client.query.get("Document", ["content"]).with_near_text({
    "concepts": ["search query"]
}).with_limit(5).do()
```

#### 3. PostgreSQL + pgvector
```sql
-- 拡張機能有効化
CREATE EXTENSION vector;

-- テーブル作成
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding VECTOR(1536)
);

-- インデックス作成
CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops);

-- 検索実行
SELECT content, embedding <=> $1 as distance 
FROM documents 
ORDER BY embedding <=> $1 
LIMIT 5;
```

### パフォーマンス最適化

#### 1. インデックス構造の選択

**HNSW（Hierarchical Navigable Small World）**
- 特徴：階層構造によるグラフベース検索
- 利点：高速検索、高い精度
- 欠点：メモリ使用量が多い
- 適用場面：リアルタイム検索が重要な場合

**IVF（Inverted File）**
- 特徴：クラスタリングベースの検索
- 利点：メモリ効率が良い
- 欠点：精度がやや劣る場合がある
- 適用場面：大規模データセットでメモリが制約

**LSH（Locality Sensitive Hashing）**
- 特徴：ハッシュベースの近似検索
- 利点：非常に高速
- 欠点：精度の保証が困難
- 適用場面：速度重視、大規模検索

#### 2. ベクトル圧縮

**Product Quantization (PQ)**
```python
import faiss

# PQインデックスの作成
index = faiss.IndexPQ(dimension, M=8, nbits=8)
index.train(training_vectors)
index.add(vectors)

# 検索実行
distances, indices = index.search(query_vector, k=5)
```

**Scalar Quantization**
- ベクトル値の量子化による圧縮
- メモリ使用量を1/4～1/8に削減
- 検索速度の向上

#### 3. 並列処理の実装

```python
import concurrent.futures
import numpy as np
from sentence_transformers import SentenceTransformer

def encode_batch(texts, model):
    return model.encode(texts)

def parallel_encoding(texts, model, batch_size=32, max_workers=4):
    batches = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(encode_batch, batch, model) for batch in batches]
        results = [future.result() for future in futures]
    
    return np.vstack(results)
```

## ベクトル検索の評価

### 評価指標

1. **Recall@K**
   - 上位K件に含まれる関連文書の割合
   - 検索の網羅性を測定

2. **Precision@K**
   - 上位K件における関連文書の精度
   - 検索結果の質を測定

3. **nDCG@K**
   - 順位を考慮した評価指標
   - 上位の関連文書により高い重みを付与

4. **レスポンス時間**
   - クエリから結果取得までの時間
   - システムの実用性を評価

### 評価用データセット

```python
def create_evaluation_dataset():
    return [
        {
            "query": "機械学習の基礎概念",
            "relevant_docs": [
                "機械学習は人工知能の一分野で...",
                "教師あり学習、教師なし学習、強化学習...",
                "アルゴリズムの種類と特徴について..."
            ]
        },
        # 他の評価例...
    ]

def evaluate_vector_search(search_function, dataset):
    recall_scores = []
    precision_scores = []
    
    for item in dataset:
        results = search_function(item["query"], k=10)
        relevant_found = 0
        
        for result in results:
            if result["content"] in item["relevant_docs"]:
                relevant_found += 1
        
        recall = relevant_found / len(item["relevant_docs"])
        precision = relevant_found / len(results)
        
        recall_scores.append(recall)
        precision_scores.append(precision)
    
    return {
        "avg_recall": np.mean(recall_scores),
        "avg_precision": np.mean(precision_scores)
    }
```

## ベクトル検索の課題と対策

### 主要な課題

1. **次元の呪い**
   - 高次元空間での距離の意味の曖昧化
   - 全ての点が等距離に見える現象

2. **計算コスト**
   - 大規模データでの検索時間の増加
   - メモリ使用量の増大

3. **コールドスタート問題**
   - 新しい文書の適切な表現の困難
   - ドメイン固有表現への対応

### 対策とベストプラクティス

1. **適切な前処理**
```python
def preprocess_text(text):
    # 正規化
    text = text.lower().strip()
    
    # 特殊文字の除去
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # 過度な空白の統合
    text = re.sub(r'\s+', ' ', text)
    
    # 最大長制限
    text = text[:512] if len(text) > 512 else text
    
    return text
```

2. **ハイブリッド検索の実装**
```python
def hybrid_search(query, vector_weight=0.7, keyword_weight=0.3):
    # ベクトル検索
    vector_results = vector_search(query)
    
    # キーワード検索
    keyword_results = keyword_search(query)
    
    # スコア正規化と融合
    combined_scores = {}
    for doc_id, score in vector_results.items():
        combined_scores[doc_id] = vector_weight * score
    
    for doc_id, score in keyword_results.items():
        if doc_id in combined_scores:
            combined_scores[doc_id] += keyword_weight * score
        else:
            combined_scores[doc_id] = keyword_weight * score
    
    return sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
```

3. **段階的検索戦略**
```python
def tiered_search(query, initial_k=1000, final_k=10):
    # 第1段階：高速な粗検索
    candidates = fast_vector_search(query, k=initial_k)
    
    # 第2段階：詳細な再ランキング
    reranked = detailed_rerank(query, candidates)
    
    return reranked[:final_k]
```

## 最新動向と今後の発展

### 新しいアーキテクチャ

1. **Dense Passage Retrieval (DPR)**
   - BERTベースのデュアルエンコーダ
   - クエリと文書の独立エンコーディング
   - 高い検索精度の実現

2. **ColBERT**
   - トークンレベルでの相互作用
   - 遅延相互作用による効率化
   - 高精度と高速性の両立

3. **SPLADE**
   - スパース表現とデンス表現の融合
   - 解釈可能性の向上
   - キーワード検索との親和性

### 新興技術

1. **Neural Information Retrieval**
   - ニューラルネットワークベースの統合システム
   - 学習可能な検索パラメータ
   - エンドツーエンドの最適化

2. **Multimodal Vector Search**
   - テキスト、画像、音声の統合検索
   - クロスモーダル類似性計算
   - マルチメディア情報検索

3. **Federated Vector Search**
   - 分散環境での協調検索
   - プライバシー保護検索
   - 複数データソースの統合

## まとめ

ベクトル検索は現代の情報検索システムにおいて不可欠な技術となっており、特に大規模言語モデルとの組み合わせにより、従来では不可能だった高度な意味的検索が実現されています。適切な実装とチューニングにより、ユーザーのニーズにより正確に応答できるインテリジェントな検索システムの構築が可能です。