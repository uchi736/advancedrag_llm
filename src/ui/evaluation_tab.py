"""
evaluation_tab.py - RAG System Evaluation Results Dashboard
==========================================================
This module provides a Streamlit interface for viewing and analyzing
RAG evaluation results from CSV files.
"""

import streamlit as st
import pandas as pd
import os
import asyncio
import math
from typing import Dict, Any, List, Optional
import plotly.graph_objects as go
import glob
from datetime import datetime
import numpy as np
from langchain_core.documents import Document

# Import from utils
from src.utils.helpers import cosine_similarity


def render_evaluation_tab(rag_system):
    """Render the evaluation results dashboard"""
    
    st.markdown("## 🎯 RAG評価結果ダッシュボード")
    st.markdown("---")
    
    # 説明テキスト
    st.info("""
    📋 **新しい評価フロー**: 
    1. Chatタブで一括質問実行（質問+想定引用元のCSV）→ 結果CSVダウンロード
    2. このタブで結果CSVをアップロード → 評価計算実行 → 即座に可視化
    
    📊 **このダッシュボード**: 一括質問結果CSVから評価指標を計算し、詳細な分析と可視化を提供します。
    """)
    
    # タブ分割
    eval_tabs = st.tabs(["🚀 評価実行", "📈 結果ダッシュボード", "📊 比較分析", "📝 レポート"])
    
    with eval_tabs[0]:
        render_evaluation_execution(rag_system)
    
    with eval_tabs[1]:
        render_results_dashboard()
    
    with eval_tabs[2]:
        render_comparison_analysis()
        
    with eval_tabs[3]:
        render_report_generation()


def render_results_dashboard():
    """結果ダッシュボードの表示"""
    
    st.subheader("📈 評価結果ダッシュボード")
    
    # 結果ファイルの読み込み（ファイル選択UI付き）
    results_df = load_evaluation_results_with_selection()
    
    if results_df is not None and not results_df.empty:
        display_results_summary(results_df)
        display_detailed_analysis(results_df)
        display_question_breakdown(results_df)
    else:
        st.info("評価結果CSVファイルをアップロードしてください。")
        st.markdown("""
        **評価結果ファイルの例**：
        - `evaluation_results_*.csv` 
        - `2025-08-28T15-26_export.csv`
        - その他の評価実行で生成されたCSVファイル
        """)


def load_evaluation_results_with_selection() -> Optional[pd.DataFrame]:
    """評価結果CSVファイルの読み込み（アップロードのみ）"""
    
    st.write("**評価結果CSVファイルをアップロードしてください**")
    
    uploaded_file = st.file_uploader(
        "評価結果CSVをアップロード",
        type=['csv'],
        help="評価実行で生成されたCSVファイル（例: 2025-08-28T15-26_export.csv）",
        key="results_dashboard_uploader_v3"
    )
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ {uploaded_file.name} を読み込みました（{len(df)}行）")
            return df
        except Exception as e:
            st.error(f"ファイル読み込みエラー: {e}")
    
    return None


def load_evaluation_results() -> Optional[pd.DataFrame]:
    """評価結果CSVファイルの読み込み"""
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("**結果ファイルの選択**")
        
        # 自動検出された結果ファイル
        result_files = glob.glob("evaluation_results*.csv")
        result_files.extend(glob.glob("*evaluation*.csv"))
        result_files = list(set(result_files))  # 重複除去
        
        if result_files:
            # 最新ファイルを先頭に
            result_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            
            selected_file = st.selectbox(
                "検出された結果ファイル",
                result_files,
                help="最新のファイルが上部に表示されます"
            )
            
            if selected_file:
                try:
                    df = pd.read_csv(selected_file)
                    file_info = os.stat(selected_file)
                    st.success(f"✅ {selected_file} を読み込みました（{len(df)}行、{file_info.st_size:,} bytes）")
                    
                    # ファイル情報の表示
                    mod_time = datetime.fromtimestamp(file_info.st_mtime)
                    st.caption(f"最終更新: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    return df
                except Exception as e:
                    st.error(f"ファイル読み込みエラー: {e}")
        else:
            st.warning("評価結果ファイルが見つかりません")
    
    with col2:
        st.write("**手動アップロード**")
        uploaded_file = st.file_uploader(
            "結果CSVをアップロード",
            type=['csv'],
            help="evaluate_rag.pyで生成されたCSVファイル"
        )
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ {len(df)}行のデータをアップロードしました")
                return df
            except Exception as e:
                st.error(f"アップロードエラー: {e}")
    
    return None


def display_results_summary(df: pd.DataFrame):
    """評価結果のサマリー表示"""
    
    st.markdown("### 📊 評価サマリー")
    
    # 基本統計の計算
    total_questions = len(df)
    
    # メトリクスの取得（カラム名を柔軟に対応）
    metrics = {}
    
    # MRRの取得
    if 'MRR' in df.columns:
        metrics['MRR'] = df['MRR'].mean()
    elif 'mrr' in df.columns:
        metrics['MRR'] = df['mrr'].mean()
    
    # Recall@Kの取得
    recall_cols = [col for col in df.columns if 'recall' in col.lower() and ('5' in col or '@5' in col)]
    if recall_cols:
        metrics['Recall@5'] = df[recall_cols[0]].mean()
    
    # Precision@Kの取得
    precision_cols = [col for col in df.columns if 'precision' in col.lower() and ('5' in col or '@5' in col)]
    if precision_cols:
        metrics['Precision@5'] = df[precision_cols[0]].mean()
    
    # nDCG@Kの取得
    ndcg_cols = [col for col in df.columns if 'ndcg' in col.lower() and ('5' in col or '@5' in col)]
    if ndcg_cols:
        metrics['nDCG@5'] = df[ndcg_cols[0]].mean()
    
    # Hit Rate@Kの取得
    hit_rate_cols = [col for col in df.columns if 'hit' in col.lower() and ('5' in col or '@5' in col)]
    if hit_rate_cols:
        metrics['Hit Rate@5'] = df[hit_rate_cols[0]].mean()
    
    # メトリクス表示
    cols = st.columns(len(metrics) + 1)
    
    with cols[0]:
        st.metric("総質問数", total_questions)
    
    for i, (metric_name, value) in enumerate(metrics.items(), 1):
        if i < len(cols):
            with cols[i]:
                st.metric(metric_name, f"{value:.3f}")
    
    # 性能レーティング
    if metrics:
        avg_score = sum(metrics.values()) / len(metrics)
        rating = get_performance_rating(avg_score)
        st.markdown(f"**総合評価**: {rating}")


def get_performance_rating(score: float) -> str:
    """性能スコアに基づくレーティング"""
    if score >= 0.9:
        return "🌟 優秀 (Excellent)"
    elif score >= 0.8:
        return "🚀 良好 (Good)"  
    elif score >= 0.7:
        return "👍 普通 (Fair)"
    elif score >= 0.6:
        return "⚠️ 要改善 (Needs Improvement)"
    else:
        return "❌ 不良 (Poor)"


def display_detailed_analysis(df: pd.DataFrame):
    """詳細分析の表示"""
    
    st.markdown("### 📈 詳細分析")
    
    # 指標の分布グラフ
    metric_cols = [col for col in df.columns if any(metric in col.lower() 
                   for metric in ['mrr', 'recall', 'precision', 'ndcg', 'hit'])]
    
    if metric_cols:
        col1, col2 = st.columns(2)
        
        with col1:
            # レーダーチャート用データ準備
            radar_metrics = {}
            for col in metric_cols[:6]:  # 最大6つの指標
                if df[col].notna().any():
                    radar_metrics[col] = df[col].mean()
            
            if radar_metrics:
                # レーダーチャート
                fig_radar = create_radar_chart(radar_metrics, "評価指標の総合比較")
                st.plotly_chart(fig_radar, use_container_width=True)
        
        with col2:
            # メトリクス比較棒グラフ
            if len(metric_cols) > 1:
                fig_bar = create_metrics_bar_chart(df, metric_cols[:6])
                st.plotly_chart(fig_bar, use_container_width=True)
        
        # 2行目のグラフ
        col3, col4 = st.columns(2)
        
        with col3:
            # 質問別成績ヒートマップ
            if len(df) > 1 and len(metric_cols) > 3:
                fig_heatmap = create_question_performance_heatmap(df, metric_cols[:4])
                st.plotly_chart(fig_heatmap, use_container_width=True)
        
        with col4:
            # K値別パフォーマンス比較
            k_metrics = [col for col in metric_cols if '@' in col]
            if k_metrics:
                fig_k_comparison = create_k_value_comparison(df, k_metrics)
                st.plotly_chart(fig_k_comparison, use_container_width=True)
        
        # 3行目：分布とトレンド
        st.markdown("#### 📊 指標分布とトレンド")
        
        col5, col6 = st.columns(2)
        
        with col5:
            # 評価指標の分布ヒストグラム（改善版）
            selected_metric = st.selectbox(
                "分布を表示する指標",
                metric_cols,
                key="distribution_metric_selector"
            )
            if selected_metric:
                fig_dist = create_metric_distribution(df, selected_metric)
                st.plotly_chart(fig_dist, use_container_width=True)
        
        with col6:
            # 累積パフォーマンス（質問数ごとの成績推移）
            if len(df) > 1:
                fig_cumulative = create_cumulative_performance(df, metric_cols[:3])
                st.plotly_chart(fig_cumulative, use_container_width=True)
        


def display_question_breakdown(df: pd.DataFrame):
    """質問別詳細結果の表示"""
    
    st.markdown("### 📝 質問別詳細結果")
    
    # 質問カラムの検出
    question_col = None
    for col in ['question', 'Question', '質問']:
        if col in df.columns:
            question_col = col
            break
    
    if question_col:
        # 質問選択
        questions = df[question_col].tolist()
        selected_idx = st.selectbox(
            "詳細を表示する質問",
            range(len(questions)),
            format_func=lambda x: f"Q{x+1}: {questions[x][:50]}..."
        )
        
        # 選択された質問の詳細表示
        selected_row = df.iloc[selected_idx]
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("**質問**")
            st.write(selected_row[question_col])
            
            # 評価指標
            st.write("**評価指標**")
            metrics_data = {}
            for col in df.columns:
                if col != question_col and pd.notna(selected_row[col]):
                    try:
                        if isinstance(selected_row[col], (int, float)):
                            metrics_data[col] = selected_row[col]
                    except:
                        pass
            
            if metrics_data:
                metrics_df = pd.DataFrame(list(metrics_data.items()), 
                                        columns=['指標', '値'])
                st.dataframe(metrics_df, hide_index=True)
        
        with col2:
            # 期待される引用元があれば表示
            expected_cols = [col for col in df.columns if '引用' in col or 'expected' in col.lower()]
            if expected_cols:
                st.write("**期待される引用元**")
                for col in expected_cols:
                    if pd.notna(selected_row[col]):
                        st.write(f"- {selected_row[col]}")
            
            # 取得された文書があれば表示
            retrieved_cols = [col for col in df.columns if 'retrieved' in col.lower() or '取得' in col]
            if retrieved_cols:
                st.write("**取得された文書**")
                for col in retrieved_cols:
                    if pd.notna(selected_row[col]):
                        st.write(f"- {selected_row[col]}")
    
    # 全データテーブル
    with st.expander("📋 全データを表示", expanded=False):
        st.dataframe(df, use_container_width=True)


def render_comparison_analysis():
    """比較分析の表示"""
    
    st.subheader("📊 比較分析")
    
    # 複数ファイルの選択
    result_files = glob.glob("evaluation_results*.csv")
    result_files.extend(glob.glob("*evaluation*.csv"))
    result_files = list(set(result_files))
    
    if len(result_files) < 2:
        st.warning("比較分析には2つ以上の結果ファイルが必要です。")
        return
    
    selected_files = st.multiselect(
        "比較するファイルを選択",
        result_files,
        default=result_files[:min(3, len(result_files))]  # 最大3ファイル
    )
    
    if len(selected_files) >= 2:
        comparison_data = load_multiple_results(selected_files)
        
        if comparison_data:
            display_comparison_charts(comparison_data)
            display_improvement_analysis(comparison_data)


def load_multiple_results(file_paths: List[str]) -> Dict[str, pd.DataFrame]:
    """複数の結果ファイルを読み込み"""
    
    results = {}
    for file_path in file_paths:
        try:
            df = pd.read_csv(file_path)
            # ファイル名から識別子を作成
            name = os.path.basename(file_path).replace('.csv', '')
            results[name] = df
        except Exception as e:
            st.error(f"{file_path} の読み込みに失敗: {e}")
    
    return results


def display_comparison_charts(comparison_data: Dict[str, pd.DataFrame]):
    """比較チャートの表示"""
    
    st.markdown("#### 📈 性能比較")
    
    # 各ファイルの平均指標を計算
    comparison_metrics = {}
    
    for name, df in comparison_data.items():
        metrics = {}
        
        # 主要指標の平均値を計算
        for col in df.columns:
            if any(metric in col.lower() for metric in ['mrr', 'recall', 'precision', 'ndcg', 'hit']):
                if df[col].dtype in ['float64', 'int64'] and df[col].notna().any():
                    metrics[col] = df[col].mean()
        
        comparison_metrics[name] = metrics
    
    if comparison_metrics:
        # 比較棒グラフ
        fig_comparison = create_comparison_bar_chart(comparison_metrics)
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # 比較レーダーチャート
        fig_radar = create_comparison_radar_chart(comparison_metrics)
        st.plotly_chart(fig_radar, use_container_width=True)


def display_improvement_analysis(comparison_data: Dict[str, pd.DataFrame]):
    """改善分析の表示"""
    
    st.markdown("#### 📊 改善分析")
    
    if len(comparison_data) >= 2:
        file_names = list(comparison_data.keys())
        
        # ベースラインとターゲットの選択
        col1, col2 = st.columns(2)
        
        with col1:
            baseline = st.selectbox("ベースライン", file_names, index=0)
        
        with col2:
            target = st.selectbox("比較対象", file_names, index=1)
        
        if baseline != target:
            improvement_analysis = calculate_improvement(
                comparison_data[baseline], 
                comparison_data[target]
            )
            
            if improvement_analysis:
                display_improvement_metrics(improvement_analysis)


def calculate_improvement(baseline_df: pd.DataFrame, target_df: pd.DataFrame) -> Dict[str, float]:
    """改善率の計算"""
    
    improvements = {}
    
    for col in baseline_df.columns:
        if (col in target_df.columns and 
            baseline_df[col].dtype in ['float64', 'int64'] and 
            baseline_df[col].notna().any() and target_df[col].notna().any()):
            
            baseline_mean = baseline_df[col].mean()
            target_mean = target_df[col].mean()
            
            if baseline_mean != 0:
                improvement = ((target_mean - baseline_mean) / baseline_mean) * 100
                improvements[col] = improvement
    
    return improvements


def display_improvement_metrics(improvements: Dict[str, float]):
    """改善指標の表示"""
    
    st.markdown("**改善率 (%)**")
    
    # 改善率のメトリクス表示
    metrics_cols = st.columns(min(len(improvements), 4))
    
    for i, (metric, improvement) in enumerate(improvements.items()):
        if i < len(metrics_cols):
            with metrics_cols[i]:
                delta_color = "normal" if improvement >= 0 else "inverse"
                st.metric(
                    metric, 
                    f"{improvement:+.1f}%",
                    delta=f"{improvement:+.1f}%",
                    delta_color=delta_color
                )


def render_evaluation_execution(rag_system):
    """評価実行セクション"""
    
    st.subheader("🚀 評価実行")
    
    st.markdown("""
    一括質問結果CSVをアップロードして、評価指標を計算します。
    
    **必要なCSVフォーマット**：
    - 質問, 想定の引用元1, 想定の引用元2, ..., 回答, 参照ソース, チャンク1, チャンク2, ...
    """)
    
    # 設定セクション
    with st.expander("⚙️ 評価設定", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            similarity_method = st.selectbox(
                "類似度計算手法",
                ["azure_embedding", "text_overlap", "azure_llm"],
                help="評価に使用する類似度計算手法を選択"
            )
        
        with col2:
            similarity_threshold = st.slider(
                "類似度閾値",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1,
                help="関連性判定の閾値"
            )
        
        with col3:
            k_values = st.multiselect(
                "評価K値",
                [1, 3, 5, 10],
                default=[1, 3, 5],
                help="Recall@K, Precision@Kなどで使用するK値"
            )
    
    # ファイルアップロード
    uploaded_file = st.file_uploader(
        "一括質問結果CSVをアップロード",
        type=['csv'],
        help="Chatタブの一括質問で生成されたCSVファイル",
        key="evaluation_execution_uploader"
    )
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ CSVファイルを読み込みました（{len(df)}行）")
            
            # データプレビュー
            with st.expander("📋 データプレビュー", expanded=False):
                st.dataframe(df.head(), use_container_width=True)
            
            # 評価実行ボタン
            if st.button("🎯 評価を実行", type="primary", use_container_width=True):
                run_evaluation(df, rag_system, similarity_method, similarity_threshold, k_values)
                
        except Exception as e:
            st.error(f"CSVファイルの読み込みに失敗しました: {e}")


def run_evaluation(df: pd.DataFrame, rag_system, similarity_method: str, similarity_threshold: float, k_values: list):
    """評価実行処理"""
    
    with st.spinner("評価計算を実行中..."):
        try:
            # RAGEvaluatorの初期化
            from rag.evaluator import RAGEvaluator
            evaluator = RAGEvaluator(
                config=rag_system.config,
                similarity_method=similarity_method,
                similarity_threshold=similarity_threshold,
                k_values=k_values
            )
            
            # 評価データの準備
            evaluation_results = []
            progress_bar = st.progress(0)
            total_rows = len(df)
            
            for i, row in df.iterrows():
                # 質問の取得
                question = row.get('質問', '')
                if not question:
                    continue
                
                # 想定引用元の取得
                expected_sources = []
                for col in df.columns:
                    if '想定の引用元' in col and pd.notna(row[col]):
                        expected_sources.append(str(row[col]))
                
                # 取得されたチャンクの取得
                retrieved_chunks = []
                for col in df.columns:
                    if 'チャンク' in col and pd.notna(row[col]):
                        chunk_content = str(row[col])
                        # チャンク内容からメタ情報を完全に除去
                        if '---\n' in chunk_content:
                            text_content = chunk_content.split('---\n', 1)[1]
                        elif '\n---\n' in chunk_content:
                            text_content = chunk_content.split('\n---\n', 1)[1]
                        else:
                            # メタ情報がない場合、最初の行（Source: xxx）を除去
                            lines = chunk_content.split('\n')
                            if lines[0].startswith('Source:'):
                                text_content = '\n'.join(lines[1:])
                            else:
                                text_content = chunk_content
                        
                        # 空白行を削除してクリーンアップ
                        text_content = text_content.strip()
                        if text_content:  # 空でない場合のみ追加
                            retrieved_chunks.append(text_content)
                
                if expected_sources and retrieved_chunks:
                    # 評価計算
                    result = calculate_single_question_metrics(
                        evaluator=evaluator,
                        question=question,
                        retrieved_documents=retrieved_chunks,
                        expected_sources=expected_sources
                    )
                    evaluation_results.append(result)
                
                progress_bar.progress((i + 1) / total_rows)
            
            # 結果の集約
            if evaluation_results:
                aggregated_metrics = evaluator.create_evaluation_report(evaluation_results)
                
                # セッション状態に保存
                st.session_state.evaluation_results = evaluation_results
                st.session_state.evaluation_metrics = aggregated_metrics
                st.session_state.evaluation_method = similarity_method
                
                # 結果の表示
                display_evaluation_results(aggregated_metrics, evaluation_results)
                
                # CSVエクスポート
                export_evaluation_results(evaluation_results, similarity_method)
            else:
                st.error("評価できるデータが見つかりませんでした。CSVの形式を確認してください。")
                
        except Exception as e:
            st.error(f"評価実行中にエラーが発生しました: {e}")


def calculate_single_question_metrics(evaluator, question: str, retrieved_documents: List[str], expected_sources: List[str]):
    """単一質問に対する評価指標計算（同期版）"""
    
    # Documentオブジェクトに変換
    retrieved_docs = [Document(page_content=doc) for doc in retrieved_documents]
    
    # 類似度行列の計算
    relevance_matrix = []
    
    for doc in retrieved_docs:
        chunk_content = doc.page_content
        chunk_relevances = []
        
        for expected_source in expected_sources:
            score = 0.0
            
            if evaluator.similarity_method == 'azure_embedding':
                # Azure Embeddingを同期的に処理
                try:
                    embedding1 = asyncio.run(evaluator.get_azure_embedding(chunk_content))
                    embedding2 = asyncio.run(evaluator.get_azure_embedding(expected_source))
                    if embedding1 is not None and embedding2 is not None:
                        score = cosine_similarity(
                            embedding1.reshape(1, -1), 
                            embedding2.reshape(1, -1)
                        )[0][0]
                except Exception as e:
                    st.warning(f"Embedding計算エラー: {e}")
                    score = 0.0
                        
            elif evaluator.similarity_method == 'azure_llm':
                # Azure LLMを同期的に処理
                try:
                    score = asyncio.run(evaluator.calculate_azure_llm_similarity(
                        question, expected_source, chunk_content
                    ))
                except Exception as e:
                    st.warning(f"LLM類似度計算エラー: {e}")
                    score = 0.0
                    
            elif evaluator.similarity_method == 'text_overlap':
                score = evaluator.calculate_text_overlap(expected_source, chunk_content)
                    
            elif evaluator.similarity_method == 'hybrid':
                # ハイブリッド計算
                try:
                    embedding1 = asyncio.run(evaluator.get_azure_embedding(chunk_content))
                    embedding2 = asyncio.run(evaluator.get_azure_embedding(expected_source))
                    embed_score = 0.0
                    if embedding1 is not None and embedding2 is not None:
                        embed_score = cosine_similarity(
                            embedding1.reshape(1, -1), 
                            embedding2.reshape(1, -1)
                        )[0][0]
                    overlap_score = evaluator.calculate_text_overlap(expected_source, chunk_content)
                    score = 0.7 * embed_score + 0.3 * overlap_score
                except Exception as e:
                    st.warning(f"ハイブリッド計算エラー: {e}")
                    score = evaluator.calculate_text_overlap(expected_source, chunk_content)
            
            is_relevant = score >= evaluator.similarity_threshold
            chunk_relevances.append((is_relevant, score))
        
        relevance_matrix.append(chunk_relevances)
    
    # メトリクス計算
    recall_at_k, precision_at_k, ndcg_at_k, hit_rate_at_k = {}, {}, {}, {}
    
    for k in evaluator.k_values:
        k_chunks = min(k, len(retrieved_docs))
        k_relevance_matrix = relevance_matrix[:k_chunks]
        
        # Recall@K
        found_sources = sum(
            1 for i in range(len(expected_sources)) 
            if any(k_relevance_matrix[j][i][0] for j in range(k_chunks))
        )
        recall_at_k[k] = found_sources / len(expected_sources) if expected_sources else 0.0
        
        # Precision@K
        relevant_chunks = sum(
            1 for i in range(k_chunks) 
            if any(is_rel for is_rel, _ in k_relevance_matrix[i])
        )
        precision_at_k[k] = relevant_chunks / k_chunks if k_chunks > 0 else 0.0
        
        # Hit Rate@K
        hit_rate_at_k[k] = 1.0 if relevant_chunks > 0 else 0.0
        
        # nDCG@K: 正規化された累積利得
        dcg = 0.0
        for i in range(k_chunks):
            if i < len(k_relevance_matrix):
                # 各位置での関連性判定（バイナリ: 1 if relevant, 0 if not）
                is_relevant = any(is_rel for is_rel, _ in k_relevance_matrix[i])
                relevance_score = 1.0 if is_relevant else 0.0
                dcg += relevance_score / math.log2(i + 2)
        
        # IDCG: 理想的なランキング（関連文書数の上位k個）
        num_relevant = sum(
            1 for r in relevance_matrix 
            if any(is_rel for is_rel, _ in r)
        )
        ideal_relevant_count = min(k_chunks, num_relevant, len(expected_sources))
        
        idcg = sum(
            1.0 / math.log2(i + 2) 
            for i in range(ideal_relevant_count)
        )
        
        ndcg_at_k[k] = dcg / idcg if idcg > 0 else 0.0
    
    # MRR
    mrr = next(
        (1.0 / (i + 1) for i, r in enumerate(relevance_matrix) 
         if any(is_rel for is_rel, _ in r)), 
        0.0
    )
    
    # 関連性スコア
    relevance_scores = [
        max(score for _, score in chunk_relevances) 
        for chunk_relevances in relevance_matrix
    ]
    
    # EvaluationResultsオブジェクトを作成
    from rag.evaluator import EvaluationResults
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


def display_evaluation_results(metrics, results):
    """評価結果の表示"""
    
    st.success("🎉 評価が完了しました！")
    
    # メトリクス表示
    st.markdown("### 📊 評価結果サマリー")
    
    cols = st.columns(5)
    with cols[0]:
        st.metric("総質問数", metrics.num_questions)
    with cols[1]:
        st.metric("MRR", f"{metrics.mrr:.3f}")
    with cols[2]:
        st.metric("Recall@5", f"{metrics.recall_at_k.get(5, 0):.3f}")
    with cols[3]:
        st.metric("Precision@5", f"{metrics.precision_at_k.get(5, 0):.3f}")
    with cols[4]:
        st.metric("nDCG@5", f"{metrics.ndcg_at_k.get(5, 0):.3f}")
    
    # レーダーチャート
    radar_metrics = {
        "MRR": metrics.mrr,
        "Recall@3": metrics.recall_at_k.get(3, 0),
        "Precision@3": metrics.precision_at_k.get(3, 0),
        "nDCG@3": metrics.ndcg_at_k.get(3, 0),
        "Hit Rate@3": metrics.hit_rate_at_k.get(3, 0)
    }
    
    fig_radar = create_radar_chart(radar_metrics, "評価指標結果")
    st.plotly_chart(fig_radar, use_container_width=True)


def export_evaluation_results(results, method):
    """評価結果のCSVエクスポート"""
    
    st.markdown("### 💾 結果保存")
    
    # 結果をDataFrameに変換
    export_data = []
    for result in results:
        row_data = {
            "question": result.question,
            "mrr": result.mrr
        }
        
        # K値別メトリクス
        for k in [1, 3, 5]:
            row_data[f"recall@{k}"] = result.recall_at_k.get(k, 0)
            row_data[f"precision@{k}"] = result.precision_at_k.get(k, 0)
            row_data[f"ndcg@{k}"] = result.ndcg_at_k.get(k, 0)
            row_data[f"hit_rate@{k}"] = result.hit_rate_at_k.get(k, 0)
        
        export_data.append(row_data)
    
    if export_data:
        export_df = pd.DataFrame(export_data)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"evaluation_results_{method}_{timestamp}.csv"
        
        csv_data = export_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"📥 結果をダウンロード ({filename})",
            data=csv_data,
            file_name=filename,
            mime="text/csv"
        )


def render_report_generation():
    """レポート生成機能"""
    
    st.subheader("📝 評価レポート生成")
    
    # レポート設定
    col1, col2 = st.columns([1, 1])
    
    with col1:
        report_type = st.selectbox(
            "レポートタイプ",
            ["標準レポート", "詳細分析レポート", "比較レポート"]
        )
        
        include_charts = st.checkbox("グラフを含める", value=True)
        include_raw_data = st.checkbox("生データを含める", value=False)
    
    with col2:
        export_format = st.selectbox(
            "出力形式", 
            ["CSV", "HTML", "Markdown"]
        )
        
        report_filename = st.text_input(
            "レポートファイル名",
            value=f"rag_evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    
    if st.button("📊 レポート生成", type="primary"):
        # 結果ファイルの読み込み
        results_df = load_evaluation_results()
        
        if results_df is not None:
            generate_evaluation_report(
                results_df, 
                report_type, 
                export_format,
                report_filename,
                include_charts,
                include_raw_data
            )
        else:
            st.error("レポート生成用のデータが見つかりません")


def generate_evaluation_report(df: pd.DataFrame, report_type: str, export_format: str, 
                             filename: str, include_charts: bool, include_raw_data: bool):
    """評価レポートの生成"""
    
    try:
        # レポート内容の生成
        report_content = create_report_content(df, report_type, include_charts, include_raw_data)
        
        if export_format == "CSV":
            # CSV形式での出力
            output_filename = f"{filename}.csv"
            df.to_csv(output_filename, index=False, encoding='utf-8-sig')
            
        elif export_format == "HTML":
            # HTML形式での出力
            output_filename = f"{filename}.html"
            with open(output_filename, 'w', encoding='utf-8') as f:
                f.write(report_content)
                
        elif export_format == "Markdown":
            # Markdown形式での出力
            output_filename = f"{filename}.md"
            with open(output_filename, 'w', encoding='utf-8') as f:
                f.write(report_content)
        
        st.success(f"✅ レポートを生成しました: {output_filename}")
        
        # ダウンロードリンク
        with open(output_filename, 'r', encoding='utf-8') as f:
            content = f.read()
            
        st.download_button(
            label=f"📥 {output_filename} をダウンロード",
            data=content,
            file_name=output_filename,
            mime="text/plain" if export_format == "Markdown" else "text/html"
        )
        
    except Exception as e:
        st.error(f"レポート生成エラー: {e}")


def create_report_content(df: pd.DataFrame, report_type: str, include_charts: bool, include_raw_data: bool) -> str:
    """レポート内容の作成"""
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    content = f"""
# RAGシステム評価レポート

**生成日時**: {timestamp}  
**レポートタイプ**: {report_type}  
**評価対象**: {len(df)}件の質問

## 📊 評価サマリー

"""
    
    # 基本統計の追加
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        content += "### 主要指標\n\n"
        for col in numeric_cols:
            mean_val = df[col].mean()
            content += f"- **{col}**: {mean_val:.3f}\n"
    
    content += f"\n### データ概要\n\n"
    content += f"- 総質問数: {len(df)}\n"
    content += f"- データカラム数: {len(df.columns)}\n"
    
    # 生データの追加
    if include_raw_data:
        content += f"\n## 📋 詳細データ\n\n"
        content += df.to_markdown(index=False)
    
    content += f"\n---\n*このレポートは自動生成されました*"
    
    return content


def show_evaluation_instructions():
    """評価実行手順の表示"""
    
    st.markdown("### 🔧 評価実行手順")
    
    st.code("""
# 1. 評価データCSVの準備
# 形式: 質問,想定の引用元1,想定の引用元2,想定の引用元3

# 2. コマンドラインで評価実行
python evaluate_rag.py

# 3. 生成された結果CSVをこのダッシュボードで確認
    """, language="bash")
    
    st.markdown("詳細な手順は [評価ガイド](../docs/evaluation_ui_guide.md) を参照してください。")


# ===== チャート作成関数 =====

def create_metrics_bar_chart(df: pd.DataFrame, metric_cols: List[str]) -> go.Figure:
    """メトリクス比較棒グラフの作成"""
    
    fig = go.Figure()
    
    metrics_data = {}
    for col in metric_cols:
        if col in df.columns and df[col].notna().any():
            metrics_data[col] = df[col].mean()
    
    if metrics_data:
        metrics = list(metrics_data.keys())
        values = list(metrics_data.values())
        
        # カラーグラデーション
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
        
        fig.add_trace(go.Bar(
            x=metrics,
            y=values,
            marker=dict(
                color=values,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="スコア", thickness=15)
            ),
            text=[f"{v:.3f}" for v in values],
            textposition='outside',
            textfont=dict(size=12, color='#333')
        ))
    
    fig.update_layout(
        title={
            'text': "📊 評価メトリクス比較",
            'x': 0.5,
            'xanchor': 'center',
            'font': dict(size=16)
        },
        xaxis_title="",
        yaxis_title="平均スコア",
        showlegend=False,
        plot_bgcolor='rgba(240, 240, 240, 0.1)',
        paper_bgcolor='white',
        yaxis=dict(
            gridcolor='rgba(200, 200, 200, 0.3)',
            range=[0, max(values) * 1.2] if values else [0, 1]
        ),
        xaxis=dict(tickangle=-45),
        margin=dict(t=60, b=100),
        height=400
    )
    
    return fig


def create_question_performance_heatmap(df: pd.DataFrame, metric_cols: List[str]) -> go.Figure:
    """質問別成績ヒートマップの作成"""
    
    # 質問列を取得
    question_col = None
    for col in ['question', 'Question', '質問']:
        if col in df.columns:
            question_col = col
            break
    
    if not question_col:
        # 質問列がない場合は行番号を使用
        questions = [f"Q{i+1}" for i in range(len(df))]
    else:
        questions = [q[:30] + "..." if len(str(q)) > 30 else str(q) for q in df[question_col]]
    
    # ヒートマップ用データ準備
    z_data = []
    for col in metric_cols:
        if col in df.columns:
            z_data.append(df[col].fillna(0).tolist())
    
    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=questions,
        y=metric_cols,
        colorscale='RdBu',
        colorbar=dict(
            title="スコア",
            thickness=15,
            len=0.7
        ),
        hoverongaps=False,
        text=[[f"{val:.3f}" for val in row] for row in z_data],
        texttemplate="%{text}",
        textfont={"size": 10}
    ))
    
    fig.update_layout(
        title={
            'text': "🔥 質問別パフォーマンスヒートマップ",
            'x': 0.5,
            'xanchor': 'center',
            'font': dict(size=16)
        },
        xaxis_title="",
        yaxis_title="",
        xaxis={'side': 'bottom'},
        plot_bgcolor='rgba(240, 240, 240, 0.1)',
        paper_bgcolor='white',
        height=400,
        margin=dict(t=60, b=100)
    )
    
    return fig


def create_k_value_comparison(df: pd.DataFrame, k_metrics: List[str]) -> go.Figure:
    """K値別パフォーマンス比較グラフの作成"""
    
    fig = go.Figure()
    
    # メトリクス種別ごとにグループ化
    metric_types = {}
    for col in k_metrics:
        if '@' in col:
            base_metric = col.split('@')[0].lower()
            if base_metric not in metric_types:
                metric_types[base_metric] = []
            metric_types[base_metric].append(col)
    
    colors = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)', 'rgb(44, 160, 44)', 'rgb(214, 39, 40)']
    
    for i, (metric_type, cols) in enumerate(metric_types.items()):
        k_values = []
        avg_values = []
        
        for col in sorted(cols):
            if col in df.columns:
                k_val = col.split('@')[1] if '@' in col else col
                k_values.append(f"K={k_val}")
                avg_values.append(df[col].mean())
        
        fig.add_trace(go.Scatter(
            x=k_values,
            y=avg_values,
            mode='lines+markers',
            name=metric_type.capitalize(),
            line=dict(color=colors[i % len(colors)], width=3),
            marker=dict(size=10, symbol='circle'),
            hovertemplate='%{y:.3f}<extra></extra>'
        ))
    
    fig.update_layout(
        title={
            'text': "📈 K値別パフォーマンス比較",
            'x': 0.5,
            'xanchor': 'center',
            'font': dict(size=16)
        },
        xaxis_title="K値",
        yaxis_title="平均スコア",
        showlegend=True,
        plot_bgcolor='rgba(240, 240, 240, 0.1)',
        paper_bgcolor='white',
        yaxis=dict(
            gridcolor='rgba(200, 200, 200, 0.3)',
            range=[0, 1.05]
        ),
        xaxis=dict(gridcolor='rgba(200, 200, 200, 0.3)'),
        legend=dict(
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='rgba(200, 200, 200, 0.5)',
            borderwidth=1
        ),
        height=400,
        margin=dict(t=60)
    )
    
    return fig


def create_metric_distribution(df: pd.DataFrame, metric: str) -> go.Figure:
    """指標分布ヒストグラムの作成"""
    
    if metric not in df.columns:
        return go.Figure()
    
    values = df[metric].dropna()
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=values,
        nbinsx=min(20, len(values)),
        marker=dict(
            color='#45B7D1',
            line=dict(color='#2C3E50', width=1.5)
        ),
        opacity=0.75,
        name=metric,
        hovertemplate='%{x:.3f}<br>頻度: %{y}<extra></extra>'
    ))
    
    # 統計情報を追加
    mean_val = values.mean()
    median_val = values.median()
    
    fig.add_vline(
        x=mean_val, 
        line_dash="dash", 
        line_color="#E74C3C",
        line_width=2,
        annotation_text=f"平均: {mean_val:.3f}",
        annotation_position="top right"
    )
    fig.add_vline(
        x=median_val, 
        line_dash="dash", 
        line_color="#27AE60",
        line_width=2,
        annotation_text=f"中央値: {median_val:.3f}",
        annotation_position="top left"
    )
    
    fig.update_layout(
        title={
            'text': f"📊 {metric} の分布",
            'x': 0.5,
            'xanchor': 'center',
            'font': dict(size=16)
        },
        xaxis_title=metric,
        yaxis_title="頻度",
        showlegend=False,
        plot_bgcolor='rgba(240, 240, 240, 0.1)',
        paper_bgcolor='white',
        bargap=0.1,
        yaxis=dict(gridcolor='rgba(200, 200, 200, 0.3)'),
        xaxis=dict(gridcolor='rgba(200, 200, 200, 0.3)'),
        height=400,
        margin=dict(t=60)
    )
    
    return fig


def create_cumulative_performance(df: pd.DataFrame, metric_cols: List[str]) -> go.Figure:
    """累積パフォーマンスの作成"""
    
    fig = go.Figure()
    
    # 16進数カラーコードに変更
    colors = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD', '#8C564B']
    
    for i, col in enumerate(metric_cols):
        if col in df.columns:
            # 累積平均を計算
            cumulative_avg = df[col].expanding().mean()
            
            fig.add_trace(go.Scatter(
                x=list(range(1, len(cumulative_avg) + 1)),
                y=cumulative_avg,
                mode='lines+markers',
                name=col,
                line=dict(
                    color=colors[i % len(colors)], 
                    width=3,
                    shape='spline'
                ),
                marker=dict(
                    size=8,
                    symbol='circle',
                    line=dict(width=2, color='white')
                ),
                hovertemplate='%{y:.3f}<extra></extra>'
            ))
    
    fig.update_layout(
        title={
            'text': "📊 累積パフォーマンス推移",
            'x': 0.5,
            'xanchor': 'center',
            'font': dict(size=16)
        },
        xaxis_title="質問数",
        yaxis_title="累積平均スコア",
        showlegend=True,
        plot_bgcolor='rgba(240, 240, 240, 0.1)',
        paper_bgcolor='white',
        yaxis=dict(
            gridcolor='rgba(200, 200, 200, 0.3)',
            range=[0, 1.05]
        ),
        xaxis=dict(gridcolor='rgba(200, 200, 200, 0.3)'),
        legend=dict(
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='rgba(200, 200, 200, 0.5)',
            borderwidth=1,
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        height=400,
        margin=dict(t=100)
    )
    
    return fig


def create_radar_chart(metrics: Dict[str, float], title: str) -> go.Figure:
    """レーダーチャートの作成"""
    
    categories = list(metrics.keys())
    values = list(metrics.values())
    
    # 閉じた形にするため最初の値を追加
    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]
    
    fig = go.Figure()
    
    # 背景のグリッドレイヤー
    for level in [0.2, 0.4, 0.6, 0.8, 1.0]:
        fig.add_trace(go.Scatterpolar(
            r=[level] * len(categories_closed),
            theta=categories_closed,
            fill=None,
            line=dict(color='rgba(200, 200, 200, 0.3)', width=1),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # メインデータ
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill='toself',
        name='評価指標',
        line=dict(color='#4ECDC4', width=3),
        fillcolor='rgba(78, 205, 196, 0.3)',
        marker=dict(size=10, color='#4ECDC4', symbol='circle'),
        hovertemplate='%{theta}: %{r:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        polar=dict(
            bgcolor='rgba(240, 240, 240, 0.1)',
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                showline=False,
                gridcolor='rgba(200, 200, 200, 0.3)',
                tickfont=dict(size=10)
            ),
            angularaxis=dict(
                gridcolor='rgba(200, 200, 200, 0.3)',
                tickfont=dict(size=12)
            )
        ),
        showlegend=False,
        title={
            'text': f"🎯 {title}",
            'x': 0.5,
            'xanchor': 'center',
            'font': dict(size=16)
        },
        paper_bgcolor='white',
        height=400,
        margin=dict(t=60, b=40, l=40, r=40)
    )
    
    return fig




def create_comparison_bar_chart(comparison_metrics: Dict[str, Dict[str, float]]) -> go.Figure:
    """比較棒グラフの作成"""
    
    # 共通指標の抽出
    all_metrics = set()
    for metrics in comparison_metrics.values():
        all_metrics.update(metrics.keys())
    
    common_metrics = list(all_metrics)[:6]  # 最大6つ
    
    fig = go.Figure()
    
    for name, metrics in comparison_metrics.items():
        values = [metrics.get(metric, 0) for metric in common_metrics]
        fig.add_trace(go.Bar(
            x=common_metrics,
            y=values,
            name=name
        ))
    
    fig.update_layout(
        title="評価指標比較",
        xaxis_title="指標",
        yaxis_title="値",
        barmode='group'
    )
    
    return fig


def create_comparison_radar_chart(comparison_metrics: Dict[str, Dict[str, float]]) -> go.Figure:
    """比較レーダーチャートの作成"""
    
    fig = go.Figure()
    
    # 共通指標の抽出
    all_metrics = set()
    for metrics in comparison_metrics.values():
        all_metrics.update(metrics.keys())
    
    common_metrics = list(all_metrics)[:6]  # 最大6つ
    
    colors = ['rgb(30, 144, 255)', 'rgb(255, 99, 71)', 'rgb(50, 205, 50)', 
              'rgb(255, 165, 0)', 'rgb(138, 43, 226)', 'rgb(255, 20, 147)']
    
    for i, (name, metrics) in enumerate(comparison_metrics.items()):
        values = [metrics.get(metric, 0) for metric in common_metrics]
        
        # 閉じた形にするため最初の値を追加
        values_closed = values + [values[0]]
        categories_closed = common_metrics + [common_metrics[0]]
        
        fig.add_trace(go.Scatterpolar(
            r=values_closed,
            theta=categories_closed,
            fill='toself',
            name=name,
            line=dict(color=colors[i % len(colors)])
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        title="評価指標比較 (レーダーチャート)"
    )
    
    return fig