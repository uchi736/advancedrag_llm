"""
RAG Evaluation Module
====================
This module provides evaluation functionality for RAG systems,
including recall, precision, MRR, nDCG, and hit rate metrics.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import math
import asyncio
import os
import re
from datetime import datetime
from dotenv import load_dotenv
from openai import AzureOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.documents import Document

# Import from utils
from src.utils.helpers import cosine_similarity

@dataclass
class EvaluationResults:
    """評価結果を格納するデータクラス"""
    question: str
    recall_at_k: Dict[int, float]
    precision_at_k: Dict[int, float]
    mrr: float
    ndcg_at_k: Dict[int, float]
    hit_rate_at_k: Dict[int, float]
    retrieved_docs: Optional[List[Document]] = None
    expected_sources: Optional[List[str]] = None
    relevance_scores: Optional[List[float]] = None

@dataclass
class EvaluationMetrics:
    """集約された評価メトリクスを格納するデータクラス"""
    mrr: float
    recall_at_k: Dict[int, float]
    precision_at_k: Dict[int, float]
    ndcg_at_k: Dict[int, float]
    hit_rate_at_k: Dict[int, float]
    num_questions: int
    method: str
    timestamp: str

class RAGEvaluator:
    """RAG評価システム（Azure LLM対応版）"""
    
    def __init__(self, 
                 config: Any,
                 k_values: List[int] = [1, 3, 5, 10],
                 similarity_method: str = "azure_embedding",
                 similarity_threshold: float = 0.7,
                 api_delay: float = 0.1):
        """
        Initialize the RAG evaluator
        
        Args:
            config: Configuration object from rag.config.Config
            k_values: List of k values for evaluation metrics
            similarity_method: Method for similarity calculation ("azure_embedding", "azure_llm", "hybrid", "text_overlap")
            similarity_threshold: Threshold for relevance determination
            api_delay: Delay between API calls to avoid rate limiting
        """
        self.config = config
        self.k_values = k_values
        self.similarity_method = similarity_method
        self.similarity_threshold = similarity_threshold
        self.api_delay = api_delay
        
        load_dotenv()
        
        # Azure OpenAI Client for LLM-based similarity
        self.azure_client = AzureOpenAI(
            azure_endpoint=config.azure_openai_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=config.azure_openai_api_key or os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=config.azure_openai_api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15")
        )
        
        # LangChain Azure Embeddings for embedding-based similarity
        self.embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=config.azure_openai_endpoint,
            api_key=config.azure_openai_api_key,
            api_version=config.azure_openai_api_version,
            azure_deployment=config.azure_openai_embedding_deployment_name
        )
        
        # Model names from config (support multiple providers)
        provider = getattr(config, 'llm_provider', 'azure') or 'azure'

        if provider == "azure":
            self.embedding_model = config.azure_openai_embedding_deployment_name or "text-embedding-3-small"
            self.llm_model = config.azure_openai_chat_deployment_name or "gpt-4o-mini"
        elif provider == "vllm":
            # VLLM uses Azure embeddings, but VLLM for generation
            self.embedding_model = config.azure_openai_embedding_deployment_name or "text-embedding-3-small"
            self.llm_model = "vllm"
        elif provider in ("huggingface", "huggingface_local"):
            self.embedding_model = config.hf_embedding_model_id or "intfloat/multilingual-e5-large"
            self.llm_model = config.hf_model_id or "huggingface-model"
        elif provider == "openai":
            self.embedding_model = "text-embedding-3-small"
            self.llm_model = "gpt-4o-mini"
        elif provider == "gemini":
            # Gemini uses Azure embeddings typically
            self.embedding_model = config.azure_openai_embedding_deployment_name or "text-embedding-3-small"
            self.llm_model = "gemini-1.5-pro"
        else:
            # Fallback to Azure defaults
            self.embedding_model = getattr(config, 'azure_openai_embedding_deployment_name', 'text-embedding-3-small')
            self.llm_model = getattr(config, 'azure_openai_chat_deployment_name', 'gpt-4o-mini')

    def load_csv_data(self, csv_path: str) -> pd.DataFrame:
        """Load evaluation data from CSV file"""
        try:
            return pd.read_csv(csv_path, encoding='utf-8')
        except Exception as e:
            print(f"CSV読み込みエラー: {e}")
            return pd.DataFrame()

    def extract_chunk_content(self, chunk_text: str) -> str:
        """Extract content from chunk text"""
        if pd.isna(chunk_text) or not isinstance(chunk_text, str): 
            return ""
        return chunk_text.split("---", 1)[1].strip() if "---" in chunk_text else chunk_text.strip()

    async def get_azure_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get text embedding using Azure OpenAI"""
        if not text: 
            return None
        try:
            await asyncio.sleep(self.api_delay)
            response = await asyncio.to_thread(
                self.azure_client.embeddings.create, 
                input=[text], 
                model=self.embedding_model
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            print(f"Azure埋め込み取得エラー: {e}")
            return None

    async def calculate_azure_llm_similarity(self, question: str, expected_source: str, chunk_content: str) -> float:
        """Calculate similarity using Azure LLM"""
        if not chunk_content: 
            return 0.0
        
        prompt = f"""
以下の質問と期待される引用元に対し、実際のチャンク内容がどの程度関連しているか0.0から1.0の数値で評価してください。

質問: {question}
期待される引用元: {expected_source}
実際のチャンク内容: {chunk_content[:2000]}

評価基準:
- 1.0: 完全に一致または非常に高い関連性
- 0.7-0.9: 高い関連性
- 0.4-0.6: 中程度の関連性
- 0.1-0.3: 低い関連性
- 0.0: 関連性なし

評価スコア（数値のみを出力）:"""
        
        try:
            await asyncio.sleep(self.api_delay)
            response = await asyncio.to_thread(
                self.azure_client.chat.completions.create,
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0
            )
            numbers = re.findall(r'[0-9]*\.?[0-9]+', response.choices[0].message.content)
            return float(numbers[0]) if numbers else 0.0
        except Exception as e:
            print(f"Azure LLM類似度計算エラー: {e}")
            return 0.0

    def calculate_text_overlap(self, text1: str, text2: str) -> float:
        """Calculate text overlap using Jaccard similarity"""
        if not text1 or not text2: 
            return 0.0
        words1 = set(re.findall(r'\w+', text1.lower()))
        words2 = set(re.findall(r'\w+', text2.lower()))
        if not words1 or not words2: 
            return 0.0
        return len(words1.intersection(words2)) / len(words1.union(words2))

    async def evaluate_retrieval_quality(self, 
                                        question: str,
                                        retrieved_docs: List[Document],
                                        expected_sources: List[str]) -> EvaluationResults:
        """
        Evaluate retrieval quality for a single question
        
        Args:
            question: The input question
            retrieved_docs: List of retrieved documents from RAG system
            expected_sources: List of expected source contents or descriptions
        
        Returns:
            EvaluationResults object containing all metrics
        """
        if not expected_sources or not retrieved_docs:
            return EvaluationResults(
                question=question,
                recall_at_k={},
                precision_at_k={},
                mrr=0.0,
                ndcg_at_k={},
                hit_rate_at_k={},
                retrieved_docs=retrieved_docs,
                expected_sources=expected_sources,
                relevance_scores=[]
            )

        print(f"  評価中: {question} (期待{len(expected_sources)}件, 取得{len(retrieved_docs)}件)")

        # Calculate relevance matrix
        relevance_matrix = []
        for doc in retrieved_docs:
            chunk_content = doc.page_content
            chunk_relevances = []
            
            for expected_source in expected_sources:
                score = 0.0
                
                if self.similarity_method == 'azure_embedding':
                    embedding1 = await self.get_azure_embedding(chunk_content)
                    embedding2 = await self.get_azure_embedding(expected_source)
                    if embedding1 is not None and embedding2 is not None:
                        score = cosine_similarity(
                            embedding1.reshape(1, -1), 
                            embedding2.reshape(1, -1)
                        )[0][0]
                        
                elif self.similarity_method == 'azure_llm':
                    score = await self.calculate_azure_llm_similarity(
                        question, expected_source, chunk_content
                    )
                    
                elif self.similarity_method == 'text_overlap':
                    score = self.calculate_text_overlap(expected_source, chunk_content)
                    
                elif self.similarity_method == 'hybrid':
                    # Combine embedding and text overlap
                    embedding1 = await self.get_azure_embedding(chunk_content)
                    embedding2 = await self.get_azure_embedding(expected_source)
                    embed_score = 0.0
                    if embedding1 is not None and embedding2 is not None:
                        embed_score = cosine_similarity(
                            embedding1.reshape(1, -1), 
                            embedding2.reshape(1, -1)
                        )[0][0]
                    overlap_score = self.calculate_text_overlap(expected_source, chunk_content)
                    score = 0.7 * embed_score + 0.3 * overlap_score
                
                is_relevant = score >= self.similarity_threshold
                chunk_relevances.append((is_relevant, score))
            
            relevance_matrix.append(chunk_relevances)
        
        # Calculate metrics
        recall_at_k, precision_at_k, ndcg_at_k, hit_rate_at_k = {}, {}, {}, {}
        
        for k in self.k_values:
            k_chunks = min(k, len(retrieved_docs))
            k_relevance_matrix = relevance_matrix[:k_chunks]
            
            # Recall@K: Fraction of relevant sources found in top k results
            found_sources = sum(
                1 for i in range(len(expected_sources)) 
                if any(k_relevance_matrix[j][i][0] for j in range(k_chunks))
            )
            recall_at_k[k] = found_sources / len(expected_sources) if expected_sources else 0.0
            
            # Precision@K: Fraction of retrieved chunks that are relevant
            relevant_chunks = sum(
                1 for i in range(k_chunks) 
                if any(is_rel for is_rel, _ in k_relevance_matrix[i])
            )
            precision_at_k[k] = relevant_chunks / k_chunks if k_chunks > 0 else 0.0
            
            # Hit Rate@K: Binary metric - 1 if any relevant in top k, 0 otherwise
            hit_rate_at_k[k] = 1.0 if relevant_chunks > 0 else 0.0
            
            # nDCG@K: Normalized Discounted Cumulative Gain
            dcg = sum(
                max(s for _, s in r) / math.log2(i + 2) 
                for i, r in enumerate(k_relevance_matrix)
            )
            ideal_scores = sorted(
                [max(s for _, s in r) for r in relevance_matrix], 
                reverse=True
            )
            idcg = sum(
                s / math.log2(i + 2) 
                for i, s in enumerate(ideal_scores[:k_chunks])
            )
            ndcg_at_k[k] = dcg / idcg if idcg > 0 else 0.0
        
        # MRR: Mean Reciprocal Rank
        mrr = next(
            (1.0 / (i + 1) for i, r in enumerate(relevance_matrix) 
             if any(is_rel for is_rel, _ in r)), 
            0.0
        )
        
        # Extract relevance scores for analysis
        relevance_scores = [
            max(score for _, score in chunk_relevances) 
            for chunk_relevances in relevance_matrix
        ]
        
        return EvaluationResults(
            question=question,
            recall_at_k=recall_at_k,
            precision_at_k=precision_at_k,
            mrr=mrr,
            ndcg_at_k=ndcg_at_k,
            hit_rate_at_k=hit_rate_at_k,
            retrieved_docs=retrieved_docs,
            expected_sources=expected_sources,
            relevance_scores=relevance_scores
        )

    async def evaluate_single_row(self, row: pd.Series, rag_system=None) -> EvaluationResults:
        """Evaluate a single row from CSV data"""
        question = row['質問']
        expected_sources = [
            str(s) for i in range(1, 6) 
            if pd.notna(s := row.get(f'想定の引用元{i}')) and str(s).strip()
        ]
        retrieved_contents = [
            content for i in range(1, 6) 
            if pd.notna(chunk := row.get(f'チャンク{i}')) and 
            (content := self.extract_chunk_content(chunk))
        ]
        
        # If no chunks in CSV, use RAG system to retrieve documents
        if not retrieved_contents and rag_system is not None:
            rag_result = rag_system.query(question)
            retrieved_docs = rag_result.get('retrieved_docs', [])
        else:
            # Convert retrieved contents to Document objects
            retrieved_docs = [
                Document(page_content=content, metadata={"chunk_id": f"csv_chunk_{i}"})
                for i, content in enumerate(retrieved_contents)
            ]
        
        return await self.evaluate_retrieval_quality(question, retrieved_docs, expected_sources)

    async def evaluate_csv(self, csv_path: str, rag_system=None) -> List[EvaluationResults]:
        """Evaluate all questions in a CSV file"""
        df = self.load_csv_data(csv_path)
        if df.empty: 
            return []
        
        print(f"\n=== RAG評価開始 ({self.similarity_method}) ===")
        print(f"処理データ数: {len(df)}件")
        print(f"評価モデル: {self.embedding_model if 'embed' in self.similarity_method else self.llm_model}")
        
        results = []
        for _, row in df.iterrows():
            result = await self.evaluate_single_row(row, rag_system=rag_system)
            results.append(result)
        
        return results

    async def evaluate_rag_system(self, 
                                 rag_system: Any,
                                 test_questions: List[Dict[str, Any]]) -> List[EvaluationResults]:
        """
        Evaluate a RAG system directly
        
        Args:
            rag_system: The RAG system instance with a retrieval_chain
            test_questions: List of dictionaries with 'question' and 'expected_sources' keys
        
        Returns:
            List of EvaluationResults
        """
        results = []
        
        print(f"\n=== RAGシステム評価開始 ({self.similarity_method}) ===")
        print(f"質問数: {len(test_questions)}件\n")
        
        for test_case in test_questions:
            question = test_case['question']
            expected_sources = test_case.get('expected_sources', [])
            
            try:
                # Retrieve documents using the RAG system
                retrieval_result = await rag_system.retrieval_chain.ainvoke({
                    "question": question,
                    "use_jargon_augmentation": rag_system.config.enable_jargon_augmentation
                })
                
                # Extract retrieved documents
                retrieved_docs = retrieval_result.get('context', [])
                if isinstance(retrieved_docs, str):
                    # If context is a formatted string, parse it back to documents
                    retrieved_docs = []
                
                # Evaluate retrieval quality
                result = await self.evaluate_retrieval_quality(
                    question, retrieved_docs, expected_sources
                )
                results.append(result)
                
            except Exception as e:
                print(f"評価エラー (質問: {question}): {e}")
                # Add empty result for failed evaluation
                results.append(EvaluationResults(
                    question=question,
                    recall_at_k={},
                    precision_at_k={},
                    mrr=0.0,
                    ndcg_at_k={},
                    hit_rate_at_k={},
                    retrieved_docs=[],
                    expected_sources=expected_sources,
                    relevance_scores=[]
                ))
        
        return results

    def print_results(self, results: List[EvaluationResults], method_name: str):
        """Print evaluation results in a formatted manner"""
        if not results: 
            print("評価結果がありません")
            return
        
        print("\n" + "=" * 60)
        print(f"RAG評価結果 ({method_name})")
        print("=" * 60)
        
        for i, r in enumerate(results, 1):
            print(f"\n{i}. 質問: {r.question}")
            print(f"   MRR: {r.mrr:.4f}")
            
            for k in self.k_values:
                if k in r.recall_at_k:
                    print(f"   K={k}: Recall={r.recall_at_k.get(k,0):.4f}, "
                          f"Precision={r.precision_at_k.get(k,0):.4f}, "
                          f"nDCG={r.ndcg_at_k.get(k,0):.4f}, "
                          f"Hit Rate={r.hit_rate_at_k.get(k,0):.4f}")
        
        # Print average metrics
        avg = self.calculate_average_metrics(results)
        print(f"\n--- 平均指標 ({method_name}) ---")
        print(f"平均MRR: {avg.get('mrr', 0):.4f}")
        
        for k in self.k_values:
            print(f"K={k}: Recall={avg.get(f'recall_at_{k}',0):.4f}, "
                  f"Precision={avg.get(f'precision_at_{k}',0):.4f}, "
                  f"nDCG={avg.get(f'ndcg_at_{k}',0):.4f}, "
                  f"Hit Rate={avg.get(f'hit_rate_at_{k}',0):.4f}")
        print()

    def calculate_average_metrics(self, results: List[EvaluationResults]) -> Dict:
        """Calculate average metrics across all results"""
        if not results: 
            return {}
        
        avg = {}
        for k in self.k_values:
            valid_results = [r for r in results if k in r.recall_at_k]
            if valid_results:
                avg[f'recall_at_{k}'] = np.mean([r.recall_at_k[k] for r in valid_results])
                avg[f'precision_at_{k}'] = np.mean([r.precision_at_k[k] for r in valid_results])
                avg[f'ndcg_at_{k}'] = np.mean([r.ndcg_at_k[k] for r in valid_results])
                avg[f'hit_rate_at_{k}'] = np.mean([r.hit_rate_at_k[k] for r in valid_results])
        
        avg['mrr'] = np.mean([r.mrr for r in results])
        return avg

    def export_results_to_csv(self, 
                             all_results: Dict[str, List[EvaluationResults]], 
                             output_path: str):
        """Export evaluation results to CSV file"""
        df_list = []
        
        for method, results in all_results.items():
            for r in results:
                row = {
                    'method': method, 
                    'question': r.question, 
                    'MRR': r.mrr
                }
                
                for k in self.k_values:
                    if k in r.recall_at_k:
                        row.update({
                            f'Recall@{k}': r.recall_at_k.get(k, 0),
                            f'Precision@{k}': r.precision_at_k.get(k, 0),
                            f'nDCG@{k}': r.ndcg_at_k.get(k, 0),
                            f'Hit_Rate@{k}': r.hit_rate_at_k.get(k, 0)
                        })
                
                df_list.append(row)
        
        pd.DataFrame(df_list).to_csv(output_path, index=False, encoding='utf-8')
        print(f"全結果を {output_path} に保存しました")

    def create_evaluation_report(self, results: List[EvaluationResults]) -> EvaluationMetrics:
        """Create a comprehensive evaluation report"""
        avg_metrics = self.calculate_average_metrics(results)
        
        return EvaluationMetrics(
            mrr=avg_metrics.get('mrr', 0.0),
            recall_at_k={k: avg_metrics.get(f'recall_at_{k}', 0.0) for k in self.k_values},
            precision_at_k={k: avg_metrics.get(f'precision_at_{k}', 0.0) for k in self.k_values},
            ndcg_at_k={k: avg_metrics.get(f'ndcg_at_{k}', 0.0) for k in self.k_values},
            hit_rate_at_k={k: avg_metrics.get(f'hit_rate_at_{k}', 0.0) for k in self.k_values},
            num_questions=len(results),
            method=self.similarity_method,
            timestamp=datetime.now().isoformat()
        )


async def run_evaluation_example():
    """Example function showing how to use the evaluator"""
    from rag.config import Config
    
    # Initialize config
    config = Config()
    
    # Create evaluator with different similarity methods
    methods = ["azure_embedding", "azure_llm", "text_overlap", "hybrid"]
    all_results = {}
    
    for method in methods:
        print(f"\n評価方法: {method}")
        evaluator = RAGEvaluator(
            config=config,
            similarity_method=method,
            api_delay=0.2
        )
        
        # Example: Evaluate from CSV
        results = await evaluator.evaluate_csv("evaluation_data.csv")
        evaluator.print_results(results, method)
        all_results[method] = results
    
    # Export all results
    if all_results:
        evaluator.export_results_to_csv(all_results, "evaluation_results.csv")
        print("\n評価完了！")


if __name__ == "__main__":
    # Run example evaluation
    asyncio.run(run_evaluation_example())