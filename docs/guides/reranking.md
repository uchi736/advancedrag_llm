# リランキング技術の完全ガイド

## リランキングとは

リランキング（Re-ranking）は、初期検索で得られた候補文書を再評価し、クエリとの関連性に基づいてより正確な順序に並び替える後処理技術です。初期検索段階では計算効率を重視した簡易的な手法を使用し、その後により精密で計算コストの高い手法を用いて上位候補の順位を調整することで、検索の精度と効率のバランスを実現します。

## リランキングの必要性と効果

### 初期検索の限界

**1. 計算効率の制約**
- 大規模データセットでは高速な検索が必要
- 精密な類似度計算は計算コストが高い
- インデックス構造による近似的な検索

**2. 文脈の理解不足**
- 単純な語彙一致やベクトル類似度
- クエリと文書の相互作用を十分考慮できない
- 長いテキストの意味的関係の把握が困難

### リランキングによる改善効果

**1. 精度の向上**
- より適切な関連文書を上位に配置
- 検索結果の質的向上
- ユーザー満足度の改善

**2. 計算効率の最適化**
- 候補数を絞ってから精密計算
- 全体的な処理時間の短縮
- リソース使用量の最適化

## Cross-Encoderによるリランキング

### Cross-Encoderの仕組み

Cross-Encoderは、クエリと文書を同時に処理し、両者の相互作用を考慮してより精密な関連性スコアを計算するモデルです。

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class CrossEncoderReranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
    
    def rerank(self, query, candidates, top_k=10):
        """
        候補文書を再ランキング
        
        Args:
            query: 検索クエリ
            candidates: 候補文書のリスト
            top_k: 返す上位文書数
        
        Returns:
            再ランキングされた文書リスト
        """
        scores = []
        
        for candidate in candidates:
            # クエリと候補文書をペアにして入力
            inputs = self.tokenizer(
                query, candidate['content'],
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # 関連性スコアを取得
                score = torch.softmax(outputs.logits, dim=1)[0][1].item()
                scores.append({
                    'document': candidate,
                    'score': score
                })
        
        # スコア順にソート
        ranked_results = sorted(scores, key=lambda x: x['score'], reverse=True)
        return ranked_results[:top_k]

# 使用例
reranker = CrossEncoderReranker()
query = "機械学習の基本概念について"
candidates = [
    {'content': '機械学習は人工知能の一分野で、データから自動的にパターンを学習する技術です'},
    {'content': '深層学習はニューラルネットワークを用いた機械学習手法の一種です'},
    {'content': '今日の天気は晴れで、気温は25度です'}
]

reranked = reranker.rerank(query, candidates)
for i, result in enumerate(reranked, 1):
    print(f"{i}. Score: {result['score']:.4f}")
    print(f"   Content: {result['document']['content']}")
```

### 高効率なバッチ処理

```python
class BatchCrossEncoderReranker(CrossEncoderReranker):
    def rerank_batch(self, query, candidates, batch_size=32, top_k=10):
        """バッチ処理による高速リランキング"""
        all_scores = []
        
        for i in range(0, len(candidates), batch_size):
            batch_candidates = candidates[i:i+batch_size]
            
            # バッチ用の入力準備
            texts = [(query, cand['content']) for cand in batch_candidates]
            
            # トークナイゼーション
            inputs = self.tokenizer(
                texts,
                truncation=True,
                max_length=512,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_scores = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
            
            # 結果を蓄積
            for j, score in enumerate(batch_scores):
                all_scores.append({
                    'document': batch_candidates[j],
                    'score': float(score)
                })
        
        # ソートして上位k件を返す
        ranked_results = sorted(all_scores, key=lambda x: x['score'], reverse=True)
        return ranked_results[:top_k]
```

## 様々なリランキング手法

### 1. BERTベースリランキング

```python
from transformers import AutoModel
import torch.nn.functional as F

class BERTReranker:
    def __init__(self, model_name="cl-tohoku/bert-base-japanese-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
    
    def get_embeddings(self, text):
        """テキストのBERT埋め込みを取得"""
        inputs = self.tokenizer(
            text, truncation=True, max_length=512, return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # [CLS]トークンの表現を使用
            embeddings = outputs.last_hidden_state[:, 0, :].cpu()
        
        return embeddings
    
    def rerank(self, query, candidates, top_k=10):
        """コサイン類似度によるリランキング"""
        query_embedding = self.get_embeddings(query)
        
        scores = []
        for candidate in candidates:
            doc_embedding = self.get_embeddings(candidate['content'])
            
            # コサイン類似度計算
            similarity = F.cosine_similarity(
                query_embedding, doc_embedding, dim=1
            ).item()
            
            scores.append({
                'document': candidate,
                'score': similarity
            })
        
        return sorted(scores, key=lambda x: x['score'], reverse=True)[:top_k]
```

### 2. TF-IDFベースリランキング

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class TFIDFReranker:
    def __init__(self, max_features=10000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',  # 日本語の場合は適切なストップワードを設定
            ngram_range=(1, 2)
        )
        self.is_fitted = False
    
    def fit(self, corpus):
        """コーパスでTF-IDFベクトライザーを学習"""
        self.vectorizer.fit(corpus)
        self.is_fitted = True
    
    def rerank(self, query, candidates, top_k=10):
        if not self.is_fitted:
            # 候補文書でオンザフライ学習
            corpus = [query] + [cand['content'] for cand in candidates]
            self.vectorizer.fit(corpus)
        
        # TF-IDFベクトル化
        query_vector = self.vectorizer.transform([query])
        candidate_vectors = self.vectorizer.transform([cand['content'] for cand in candidates])
        
        # コサイン類似度計算
        similarities = cosine_similarity(query_vector, candidate_vectors)[0]
        
        scores = []
        for i, candidate in enumerate(candidates):
            scores.append({
                'document': candidate,
                'score': similarities[i]
            })
        
        return sorted(scores, key=lambda x: x['score'], reverse=True)[:top_k]
```

### 3. 学習可能なリランキング

```python
import torch.nn as nn
import torch.optim as optim

class LearnableReranker(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256):
        super(LearnableReranker, self).__init__()
        
        # 特徴量抽出レイヤー
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),  # クエリと文書の埋め込みを結合
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, query_embedding, doc_embedding):
        """クエリと文書の埋め込みから関連性スコアを計算"""
        # 埋め込みを結合
        combined = torch.cat([query_embedding, doc_embedding], dim=-1)
        
        # スコア計算
        score = self.feature_extractor(combined)
        return score
    
    def train_model(self, training_data, epochs=100, lr=0.001):
        """モデルの学習"""
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        
        for epoch in range(epochs):
            total_loss = 0
            
            for query_emb, doc_emb, label in training_data:
                optimizer.zero_grad()
                
                # 予測
                pred = self.forward(query_emb, doc_emb)
                
                # 損失計算
                loss = criterion(pred, label.float())
                
                # 勾配計算と更新
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss / len(training_data):.4f}")
```

## ハイブリッドリランキング戦略

### 複数手法の組み合わせ

```python
class HybridReranker:
    def __init__(self):
        self.cross_encoder = CrossEncoderReranker()
        self.bert_reranker = BERTReranker()
        self.tfidf_reranker = TFIDFReranker()
    
    def rerank(self, query, candidates, method="ensemble", top_k=10):
        """複数の手法を組み合わせたリランキング"""
        
        if method == "cascade":
            return self._cascade_rerank(query, candidates, top_k)
        elif method == "ensemble":
            return self._ensemble_rerank(query, candidates, top_k)
        elif method == "selective":
            return self._selective_rerank(query, candidates, top_k)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _cascade_rerank(self, query, candidates, top_k):
        """段階的リランキング"""
        # Stage 1: TF-IDFで候補を半分に絞る
        stage1_results = self.tfidf_reranker.rerank(
            query, candidates, top_k=len(candidates)//2
        )
        
        # Stage 2: BERTでさらに絞る
        stage1_candidates = [r['document'] for r in stage1_results]
        stage2_results = self.bert_reranker.rerank(
            query, stage1_candidates, top_k=min(20, len(stage1_candidates))
        )
        
        # Stage 3: Cross-Encoderで最終ランキング
        stage2_candidates = [r['document'] for r in stage2_results]
        final_results = self.cross_encoder.rerank(
            query, stage2_candidates, top_k=top_k
        )
        
        return final_results
    
    def _ensemble_rerank(self, query, candidates, top_k, weights=None):
        """アンサンブルリランキング"""
        if weights is None:
            weights = {"cross_encoder": 0.5, "bert": 0.3, "tfidf": 0.2}
        
        # 各手法でスコア計算
        ce_results = self.cross_encoder.rerank(query, candidates, top_k=len(candidates))
        bert_results = self.bert_reranker.rerank(query, candidates, top_k=len(candidates))
        tfidf_results = self.tfidf_reranker.rerank(query, candidates, top_k=len(candidates))
        
        # スコアの正規化と統合
        combined_scores = {}
        
        for results, weight in [(ce_results, weights["cross_encoder"]),
                               (bert_results, weights["bert"]),
                               (tfidf_results, weights["tfidf"])]:
            
            # スコアを0-1に正規化
            scores = [r['score'] for r in results]
            if max(scores) > min(scores):
                normalized_scores = [
                    (s - min(scores)) / (max(scores) - min(scores)) 
                    for s in scores
                ]
            else:
                normalized_scores = [1.0] * len(scores)
            
            # 重み付きスコアを加算
            for i, result in enumerate(results):
                doc_id = id(result['document'])
                if doc_id not in combined_scores:
                    combined_scores[doc_id] = {
                        'document': result['document'],
                        'score': 0
                    }
                combined_scores[doc_id]['score'] += weight * normalized_scores[i]
        
        # ソートして返す
        final_results = sorted(
            combined_scores.values(),
            key=lambda x: x['score'],
            reverse=True
        )
        
        return final_results[:top_k]
    
    def _selective_rerank(self, query, candidates, top_k):
        """クエリタイプに基づく選択的リランキング"""
        query_type = self._classify_query(query)
        
        if query_type == "factual":
            # 事実確認的クエリはTF-IDFを重視
            return self.tfidf_reranker.rerank(query, candidates, top_k)
        elif query_type == "semantic":
            # 意味的クエリはBERTを重視
            return self.bert_reranker.rerank(query, candidates, top_k)
        else:
            # その他はCross-Encoderを使用
            return self.cross_encoder.rerank(query, candidates, top_k)
    
    def _classify_query(self, query):
        """クエリタイプの分類（簡易版）"""
        factual_indicators = ['いつ', 'どこで', '誰が', '何を', 'when', 'where', 'who', 'what']
        semantic_indicators = ['なぜ', 'どのように', '説明', '理由', 'why', 'how', 'explain']
        
        query_lower = query.lower()
        
        if any(indicator in query_lower for indicator in factual_indicators):
            return "factual"
        elif any(indicator in query_lower for indicator in semantic_indicators):
            return "semantic"
        else:
            return "general"
```

## 性能最適化とベストプラクティス

### 1. キャッシュシステム

```python
import hashlib
from functools import lru_cache

class CachedReranker:
    def __init__(self, base_reranker, cache_size=1000):
        self.base_reranker = base_reranker
        self.cache = {}
        self.cache_size = cache_size
    
    def _get_cache_key(self, query, candidates):
        """キャッシュキーの生成"""
        candidate_hashes = [
            hashlib.md5(cand['content'].encode()).hexdigest()[:8]
            for cand in candidates
        ]
        key_data = f"{query}:{':'.join(sorted(candidate_hashes))}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def rerank(self, query, candidates, top_k=10):
        cache_key = self._get_cache_key(query, candidates)
        
        if cache_key in self.cache:
            cached_result = self.cache[cache_key]
            return cached_result[:top_k]
        
        # キャッシュにない場合は計算
        result = self.base_reranker.rerank(query, candidates, top_k=len(candidates))
        
        # キャッシュサイズ制限
        if len(self.cache) >= self.cache_size:
            # 最も古いエントリを削除（簡易LRU）
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[cache_key] = result
        return result[:top_k]
```

### 2. 並列処理

```python
import concurrent.futures
from typing import List, Dict, Any

class ParallelReranker:
    def __init__(self, rerankers: Dict[str, Any], max_workers=4):
        self.rerankers = rerankers
        self.max_workers = max_workers
    
    def parallel_rerank(self, query: str, candidates: List[Dict], top_k=10):
        """複数のリランカーを並列実行"""
        
        def run_reranker(name_and_reranker):
            name, reranker = name_and_reranker
            try:
                result = reranker.rerank(query, candidates, top_k=len(candidates))
                return name, result
            except Exception as e:
                print(f"Error in {name}: {e}")
                return name, []
        
        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(run_reranker, item): item[0] 
                for item in self.rerankers.items()
            }
            
            for future in concurrent.futures.as_completed(futures):
                name, result = future.result()
                results[name] = result
        
        return results
```

### 3. A/Bテスト対応評価

```python
class RerankingEvaluator:
    def __init__(self):
        self.evaluation_metrics = ['ndcg', 'map', 'precision', 'recall']
    
    def compare_rerankers(self, rerankers: Dict, test_queries: List, ground_truth: Dict):
        """複数のリランカーを比較評価"""
        
        results = {}
        
        for reranker_name, reranker in rerankers.items():
            print(f"\n評価中: {reranker_name}")
            metrics = self.evaluate_single_reranker(
                reranker, test_queries, ground_truth
            )
            results[reranker_name] = metrics
            
            # 結果表示
            for metric, score in metrics.items():
                print(f"  {metric}: {score:.4f}")
        
        return results
    
    def evaluate_single_reranker(self, reranker, test_queries, ground_truth):
        """単一リランカーの評価"""
        
        all_scores = {metric: [] for metric in self.evaluation_metrics}
        
        for query_data in test_queries:
            query = query_data['query']
            candidates = query_data['candidates']
            relevant_docs = ground_truth.get(query, [])
            
            # リランキング実行
            reranked = reranker.rerank(query, candidates, top_k=10)
            
            # メトリクス計算
            for metric in self.evaluation_metrics:
                score = self.calculate_metric(metric, reranked, relevant_docs)
                all_scores[metric].append(score)
        
        # 平均スコア計算
        avg_scores = {
            metric: np.mean(scores) 
            for metric, scores in all_scores.items()
        }
        
        return avg_scores
    
    def calculate_metric(self, metric, ranked_results, relevant_docs):
        """評価指標の計算"""
        if metric == 'ndcg':
            return self.calculate_ndcg(ranked_results, relevant_docs)
        elif metric == 'map':
            return self.calculate_map(ranked_results, relevant_docs)
        elif metric == 'precision':
            return self.calculate_precision_at_k(ranked_results, relevant_docs, k=10)
        elif metric == 'recall':
            return self.calculate_recall_at_k(ranked_results, relevant_docs, k=10)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def calculate_ndcg(self, ranked_results, relevant_docs, k=10):
        """nDCG@k計算"""
        dcg = 0
        for i, result in enumerate(ranked_results[:k]):
            doc_id = result['document'].get('id', result['document']['content'])
            if doc_id in relevant_docs:
                dcg += 1 / np.log2(i + 2)
        
        # Ideal DCG
        idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant_docs), k)))
        
        return dcg / idcg if idcg > 0 else 0
```

## まとめ

リランキング技術は、現代の情報検索システムにおいて検索精度を向上させる重要な技術です。Cross-Encoderによる高精度なリランキングから、効率的なハイブリッド手法まで、用途に応じて適切な手法を選択し、継続的な評価と改善を通じて最適化することで、ユーザーにより良い検索体験を提供できます。