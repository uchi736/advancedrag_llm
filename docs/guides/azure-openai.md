# Azure OpenAI統合ガイド

## Azure OpenAI Serviceの概要

Azure OpenAI Serviceは、OpenAIの大規模言語モデル（GPT、Embedding、DALL-E等）をMicrosoftのクラウドプラットフォームで提供するエンタープライズグレードのAIサービスです。高い可用性、セキュリティ、コンプライアンスを備えており、企業での本格的なAI活用に適したソリューションとなっています。

## Azure OpenAIの料金体系

### 基本料金構造

Azure OpenAIは使用したトークン数に基づく従量課金制を採用しており、モデルと処理タイプによって異なる料金設定があります。

#### GPT-4oモデル料金（2024年基準）

```python
# 料金計算の例
class AzureOpenAICostCalculator:
    def __init__(self):
        self.pricing = {
            "gpt-4o": {
                "input": 0.0025,   # $0.0025 per 1K input tokens
                "output": 0.01     # $0.01 per 1K output tokens
            },
            "gpt-4": {
                "input": 0.03,     # $0.03 per 1K input tokens
                "output": 0.06     # $0.06 per 1K output tokens
            },
            "gpt-35-turbo": {
                "input": 0.0015,   # $0.0015 per 1K input tokens
                "output": 0.002    # $0.002 per 1K output tokens
            },
            "text-embedding-ada-002": {
                "usage": 0.0001    # $0.0001 per 1K tokens
            },
            "text-embedding-3-small": {
                "usage": 0.00002   # $0.00002 per 1K tokens
            },
            "text-embedding-3-large": {
                "usage": 0.00013   # $0.00013 per 1K tokens
            }
        }
    
    def calculate_chat_cost(self, model, input_tokens, output_tokens):
        """チャット完了の料金計算"""
        if model not in self.pricing:
            raise ValueError(f"Model {model} not found in pricing")
        
        input_cost = (input_tokens / 1000) * self.pricing[model]["input"]
        output_cost = (output_tokens / 1000) * self.pricing[model]["output"]
        
        return {
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": input_cost + output_cost,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }
    
    def calculate_embedding_cost(self, model, tokens):
        """埋め込みの料金計算"""
        if model not in self.pricing:
            raise ValueError(f"Model {model} not found in pricing")
        
        cost = (tokens / 1000) * self.pricing[model]["usage"]
        return {
            "cost": cost,
            "tokens": tokens,
            "cost_per_1k": self.pricing[model]["usage"]
        }

# 使用例
calculator = AzureOpenAICostCalculator()

# RAGクエリの典型的なコスト計算
input_tokens = 1500  # クエリ + コンテキスト
output_tokens = 300  # 生成された回答

cost = calculator.calculate_chat_cost("gpt-4o", input_tokens, output_tokens)
print(f"総コスト: ${cost['total_cost']:.4f}")
print(f"入力コスト: ${cost['input_cost']:.4f}")
print(f"出力コスト: ${cost['output_cost']:.4f}")

# 埋め込み生成のコスト
embedding_cost = calculator.calculate_embedding_cost("text-embedding-3-small", 1000)
print(f"埋め込みコスト: ${embedding_cost['cost']:.6f}")
```

### エンタープライズプランと割引

```python
class EnterpriseAzurePricing:
    def __init__(self):
        # PTU (Provisioned Throughput Units) 料金
        self.ptu_pricing = {
            "gpt-4o": 168.00,      # $168 per PTU per month
            "gpt-35-turbo": 50.40, # $50.40 per PTU per month
        }
        
        # コミット割引（年間契約）
        self.commit_discounts = {
            "1_year": 0.15,    # 15% discount
            "3_year": 0.25     # 25% discount
        }
    
    def calculate_ptu_cost(self, model, ptu_count, months=1, commitment=None):
        """PTU料金計算"""
        base_cost = self.ptu_pricing[model] * ptu_count * months
        
        if commitment and commitment in self.commit_discounts:
            discount = self.commit_discounts[commitment]
            base_cost *= (1 - discount)
        
        return {
            "base_cost": base_cost,
            "ptu_count": ptu_count,
            "months": months,
            "commitment": commitment,
            "monthly_cost": base_cost / months if months > 1 else base_cost
        }
```

## Azure OpenAIクライアントの実装

### 基本的な統合

```python
import os
from openai import AzureOpenAI
from typing import List, Optional, Dict, Any
import asyncio
import time
from tenacity import retry, stop_after_attempt, wait_exponential

class EnhancedAzureOpenAIClient:
    def __init__(self, 
                 endpoint: str = None,
                 api_key: str = None,
                 api_version: str = "2024-02-01",
                 chat_deployment: str = None,
                 embedding_deployment: str = None):
        
        # 環境変数から設定を取得
        self.endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.api_version = api_version
        self.chat_deployment = chat_deployment or os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
        self.embedding_deployment = embedding_deployment or os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
        
        # クライアント初期化
        self.client = AzureOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version=self.api_version
        )
        
        # 使用量追跡
        self.usage_tracker = {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_embedding_tokens": 0,
            "request_count": 0,
            "error_count": 0
        }
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def chat_completion(self, 
                       messages: List[Dict[str, str]], 
                       temperature: float = 0.7,
                       max_tokens: int = 1000,
                       **kwargs) -> Dict[str, Any]:
        """チャット完了のリクエスト"""
        
        try:
            start_time = time.time()
            
            response = self.client.chat.completions.create(
                model=self.chat_deployment,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            end_time = time.time()
            
            # 使用量追跡
            usage = response.usage
            self.usage_tracker["total_input_tokens"] += usage.prompt_tokens
            self.usage_tracker["total_output_tokens"] += usage.completion_tokens
            self.usage_tracker["request_count"] += 1
            
            return {
                "content": response.choices[0].message.content,
                "usage": {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens
                },
                "response_time": end_time - start_time,
                "model": response.model
            }
            
        except Exception as e:
            self.usage_tracker["error_count"] += 1
            print(f"Chat completion error: {e}")
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def create_embeddings(self, 
                         texts: List[str], 
                         batch_size: int = 100) -> List[List[float]]:
        """テキストの埋め込み生成（バッチ処理対応）"""
        
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            try:
                start_time = time.time()
                
                response = self.client.embeddings.create(
                    model=self.embedding_deployment,
                    input=batch_texts
                )
                
                end_time = time.time()
                
                # 埋め込み結果の抽出
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)
                
                # 使用量追跡
                self.usage_tracker["total_embedding_tokens"] += response.usage.total_tokens
                self.usage_tracker["request_count"] += 1
                
                print(f"Processed batch {i//batch_size + 1}: {len(batch_texts)} texts, "
                      f"Time: {end_time - start_time:.2f}s")
                
                # レート制限対策
                time.sleep(0.1)
                
            except Exception as e:
                self.usage_tracker["error_count"] += 1
                print(f"Embedding error for batch {i//batch_size + 1}: {e}")
                raise
        
        return all_embeddings
    
    async def async_chat_completion(self, 
                                   messages: List[Dict[str, str]], 
                                   **kwargs) -> Dict[str, Any]:
        """非同期チャット完了"""
        # 同期版をasyncioで実行
        return await asyncio.to_thread(self.chat_completion, messages, **kwargs)
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """使用量サマリーの取得"""
        calculator = AzureOpenAICostCalculator()
        
        # コスト計算
        chat_cost = calculator.calculate_chat_cost(
            "gpt-4o",  # デフォルトモデル
            self.usage_tracker["total_input_tokens"],
            self.usage_tracker["total_output_tokens"]
        )
        
        embedding_cost = calculator.calculate_embedding_cost(
            "text-embedding-3-small",  # デフォルトモデル
            self.usage_tracker["total_embedding_tokens"]
        )
        
        return {
            "tokens": self.usage_tracker,
            "estimated_costs": {
                "chat_cost": chat_cost["total_cost"],
                "embedding_cost": embedding_cost["cost"],
                "total_cost": chat_cost["total_cost"] + embedding_cost["cost"]
            },
            "requests": {
                "successful": self.usage_tracker["request_count"],
                "failed": self.usage_tracker["error_count"],
                "success_rate": (
                    self.usage_tracker["request_count"] / 
                    (self.usage_tracker["request_count"] + self.usage_tracker["error_count"])
                ) if (self.usage_tracker["request_count"] + self.usage_tracker["error_count"]) > 0 else 0
            }
        }
```

### RAGシステムとの統合

```python
class AzureOpenAIRAGIntegration:
    def __init__(self, azure_client: EnhancedAzureOpenAIClient):
        self.azure_client = azure_client
        self.system_prompt = """あなたは親切で知識豊富なAIアシスタントです。
提供されたコンテキスト情報を基に、正確で有用な回答を提供してください。
コンテキストに含まれていない情報については、そのことを明確に示してください。"""
    
    def generate_rag_response(self, 
                             query: str, 
                             context_documents: List[str],
                             max_context_length: int = 4000) -> Dict[str, Any]:
        """RAG応答の生成"""
        
        # コンテキストの構築
        context = self._build_context(context_documents, max_context_length)
        
        # メッセージの構築
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"コンテキスト:\n{context}\n\n質問: {query}"}
        ]
        
        # 応答生成
        response = self.azure_client.chat_completion(
            messages=messages,
            temperature=0.3,  # RAGでは一貫性を重視
            max_tokens=1000
        )
        
        return {
            "answer": response["content"],
            "context_used": context,
            "usage": response["usage"],
            "response_time": response["response_time"]
        }
    
    def _build_context(self, documents: List[str], max_length: int) -> str:
        """コンテキストの構築（長さ制限付き）"""
        context = ""
        
        for i, doc in enumerate(documents):
            doc_text = f"[ソース {i+1}]\n{doc}\n\n"
            
            if len(context) + len(doc_text) > max_length:
                break
            
            context += doc_text
        
        return context.strip()
    
    async def batch_rag_processing(self, 
                                  queries: List[Dict[str, Any]],
                                  max_concurrent: int = 5) -> List[Dict[str, Any]]:
        """複数クエリの並列処理"""
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single_query(query_data):
            async with semaphore:
                try:
                    response = await asyncio.to_thread(
                        self.generate_rag_response,
                        query_data["query"],
                        query_data["documents"]
                    )
                    return {
                        "query_id": query_data.get("id"),
                        "success": True,
                        "response": response
                    }
                except Exception as e:
                    return {
                        "query_id": query_data.get("id"),
                        "success": False,
                        "error": str(e)
                    }
        
        # 並列実行
        tasks = [process_single_query(query) for query in queries]
        results = await asyncio.gather(*tasks)
        
        return results
```

## 高度なAzure OpenAI活用

### コンテンツフィルタリング

```python
class AzureContentFilter:
    def __init__(self, azure_client):
        self.azure_client = azure_client
        self.content_filter_settings = {
            "hate": {"filtered": True, "severity": "medium"},
            "sexual": {"filtered": True, "severity": "medium"},
            "violence": {"filtered": True, "severity": "medium"},
            "self_harm": {"filtered": True, "severity": "medium"}
        }
    
    def safe_chat_completion(self, messages: List[Dict[str, str]], **kwargs):
        """コンテンツフィルタリング付きチャット完了"""
        
        # リクエストにコンテンツフィルター設定を追加
        filter_kwargs = kwargs.copy()
        filter_kwargs.update(self.content_filter_settings)
        
        try:
            response = self.azure_client.chat_completion(messages, **filter_kwargs)
            
            # レスポンスのフィルタリング結果をチェック
            if hasattr(response, 'choices') and response.choices:
                finish_reason = response.choices[0].finish_reason
                if finish_reason == "content_filter":
                    return {
                        "filtered": True,
                        "reason": "Content was filtered by Azure OpenAI",
                        "content": None
                    }
            
            return {
                "filtered": False,
                "content": response["content"],
                "usage": response.get("usage")
            }
            
        except Exception as e:
            if "content_filter" in str(e).lower():
                return {
                    "filtered": True,
                    "reason": f"Content filtered: {str(e)}",
                    "content": None
                }
            raise
```

### パフォーマンス監視

```python
import logging
from datetime import datetime, timedelta
from collections import defaultdict

class AzureOpenAIMonitor:
    def __init__(self):
        self.metrics = defaultdict(list)
        self.error_log = []
        
        # ログ設定
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("azure_openai_monitor")
    
    def log_request(self, 
                   model: str, 
                   tokens_used: int, 
                   response_time: float,
                   success: bool = True,
                   error: str = None):
        """リクエストのログ記録"""
        
        timestamp = datetime.now()
        
        request_data = {
            "timestamp": timestamp,
            "model": model,
            "tokens_used": tokens_used,
            "response_time": response_time,
            "success": success
        }
        
        self.metrics["requests"].append(request_data)
        
        if not success and error:
            self.error_log.append({
                "timestamp": timestamp,
                "model": model,
                "error": error
            })
            self.logger.error(f"Azure OpenAI error: {error}")
        
        # メトリクスの記録
        self.logger.info(
            f"Request: {model}, Tokens: {tokens_used}, "
            f"Response Time: {response_time:.2f}s, Success: {success}"
        )
    
    def get_performance_report(self, hours: int = 24) -> Dict[str, Any]:
        """パフォーマンスレポートの生成"""
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # 期間内のリクエストをフィルタリング
        recent_requests = [
            req for req in self.metrics["requests"] 
            if req["timestamp"] > cutoff_time
        ]
        
        if not recent_requests:
            return {"message": "No data available for the specified period"}
        
        # 統計計算
        successful_requests = [req for req in recent_requests if req["success"]]
        failed_requests = [req for req in recent_requests if not req["success"]]
        
        total_tokens = sum(req["tokens_used"] for req in successful_requests)
        avg_response_time = sum(req["response_time"] for req in successful_requests) / len(successful_requests)
        
        # モデル別統計
        model_stats = defaultdict(lambda: {"requests": 0, "tokens": 0, "avg_time": 0})
        for req in successful_requests:
            model = req["model"]
            model_stats[model]["requests"] += 1
            model_stats[model]["tokens"] += req["tokens_used"]
        
        for model in model_stats:
            model_requests = [req for req in successful_requests if req["model"] == model]
            if model_requests:
                model_stats[model]["avg_time"] = sum(
                    req["response_time"] for req in model_requests
                ) / len(model_requests)
        
        return {
            "period_hours": hours,
            "total_requests": len(recent_requests),
            "successful_requests": len(successful_requests),
            "failed_requests": len(failed_requests),
            "success_rate": len(successful_requests) / len(recent_requests),
            "total_tokens_used": total_tokens,
            "average_response_time": avg_response_time,
            "model_statistics": dict(model_stats),
            "recent_errors": self.error_log[-10:]  # 最新の10件のエラー
        }
```

### コスト最適化戦略

```python
class CostOptimizer:
    def __init__(self, azure_client):
        self.azure_client = azure_client
        self.cost_thresholds = {
            "daily": 100.0,    # $100 per day
            "monthly": 2000.0  # $2000 per month
        }
    
    def optimize_request(self, 
                        messages: List[Dict[str, str]], 
                        target_model: str = "gpt-4o") -> Dict[str, Any]:
        """コスト最適化されたリクエスト"""
        
        # コンテキスト長の推定
        total_chars = sum(len(msg["content"]) for msg in messages)
        estimated_tokens = total_chars // 4  # 概算
        
        # コストが高すぎる場合は軽量モデルを選択
        if estimated_tokens > 8000 and target_model == "gpt-4o":
            optimized_model = "gpt-35-turbo"
            print(f"High token count detected. Switching to {optimized_model}")
        else:
            optimized_model = target_model
        
        # コンテキストの圧縮
        if estimated_tokens > 12000:
            messages = self._compress_context(messages, max_tokens=10000)
        
        return {
            "optimized_model": optimized_model,
            "optimized_messages": messages,
            "estimated_savings": self._calculate_savings(target_model, optimized_model, estimated_tokens)
        }
    
    def _compress_context(self, messages: List[Dict[str, str]], max_tokens: int) -> List[Dict[str, str]]:
        """コンテキストの圧縮"""
        compressed_messages = []
        total_chars = 0
        max_chars = max_tokens * 4  # 概算
        
        # システムメッセージは保持
        for msg in messages:
            if msg["role"] == "system":
                compressed_messages.append(msg)
                total_chars += len(msg["content"])
        
        # ユーザーメッセージを後ろから追加（最新を優先）
        user_messages = [msg for msg in messages if msg["role"] != "system"]
        user_messages.reverse()
        
        for msg in user_messages:
            if total_chars + len(msg["content"]) > max_chars:
                # 切り詰め
                remaining_chars = max_chars - total_chars
                if remaining_chars > 100:  # 最低限の長さは確保
                    truncated_msg = {
                        "role": msg["role"],
                        "content": msg["content"][:remaining_chars] + "..."
                    }
                    compressed_messages.append(truncated_msg)
                break
            else:
                compressed_messages.append(msg)
                total_chars += len(msg["content"])
        
        return compressed_messages
    
    def _calculate_savings(self, original_model: str, optimized_model: str, tokens: int) -> Dict[str, float]:
        """節約額の計算"""
        calculator = AzureOpenAICostCalculator()
        
        if original_model == optimized_model:
            return {"savings": 0.0, "original_cost": 0.0, "optimized_cost": 0.0}
        
        # 概算コスト計算（入力のみ）
        original_cost = calculator.calculate_chat_cost(original_model, tokens, 0)["input_cost"]
        optimized_cost = calculator.calculate_chat_cost(optimized_model, tokens, 0)["input_cost"]
        
        return {
            "savings": original_cost - optimized_cost,
            "original_cost": original_cost,
            "optimized_cost": optimized_cost,
            "savings_percentage": ((original_cost - optimized_cost) / original_cost * 100) if original_cost > 0 else 0
        }
```

## まとめ

Azure OpenAI Serviceは、エンタープライズ向けの包括的なAIソリューションとして、高い性能とセキュリティを提供します。適切な実装とコスト最適化により、大規模なRAGシステムでも効率的で経済的な運用が可能です。継続的な監視と最適化を通じて、最高のユーザーエクスペリエンスとROIを実現できます。