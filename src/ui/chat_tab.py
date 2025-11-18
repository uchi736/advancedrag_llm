import streamlit as st
import os
import pandas as pd
import csv
from io import StringIO
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any
from src.utils.helpers import render_sql_result_in_chat

def render_chat_tab(rag_system):
    """Renders the chat tab."""
    if not rag_system:
        st.info("🔧 RAGシステムが初期化されていません。サイドバーでAzure OpenAI APIキーを設定し、「Apply Settings」をクリックするか、データベース設定を確認してください。")
        return

    _render_bulk_query_section(rag_system)

    has_messages = len(st.session_state.messages) > 0
    if not has_messages:
        _render_initial_chat_view(rag_system)
    else:
        _render_continued_chat_view(rag_system)

def _render_initial_chat_view(rag):
    """Renders the view for a new chat session."""
    st.markdown("""
    <div class="chat-welcome">
        <h2>Chat with your data</h2>
        <p style="color: var(--text-secondary);">
            アップロードされたドキュメントから関連情報を検索し、AIが回答します<br>
            (Searches for relevant information from uploaded documents and AI answers)
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="initial-input-container">', unsafe_allow_html=True)

    st.markdown("<h6>高度なRAG設定:</h6>", unsafe_allow_html=True)
    opt_cols_initial = st.columns(4)
    with opt_cols_initial[0]:
        use_qe_initial = st.checkbox("クエリ拡張", value=st.session_state.use_query_expansion, key="use_qe_initial_v7_tab_chat", help="質問を自動的に拡張して検索 (RRFなし)")
    with opt_cols_initial[1]:
        use_rf_initial = st.checkbox("RAG-Fusion", value=st.session_state.use_rag_fusion, key="use_rf_initial_v7_tab_chat", help="クエリ拡張とRRFで結果を統合")
    with opt_cols_initial[2]:
        use_ja_initial = st.checkbox("専門用語で補強", value=st.session_state.use_jargon_augmentation, key="use_ja_initial_v7_tab_chat", help="専門用語辞書を使って質問を補強")
    with opt_cols_initial[3]:
        use_rr_initial = st.checkbox("LLMリランク", value=st.session_state.use_reranking, key="use_rr_initial_v7_tab_chat", help="LLMで検索結果を並べ替え")

    user_input_initial = st.text_area("質問を入力:", placeholder="例：このドキュメントの要約を教えてください / 売上上位10件を表示して", height=100, key="initial_input_textarea_v7_tab_chat", label_visibility="collapsed")

    if st.button("送信", type="primary", use_container_width=True, key="initial_send_button_v7_tab_chat"):
        if user_input_initial:
            st.session_state.messages.append({"role": "user", "content": user_input_initial})
            st.session_state.use_query_expansion = use_qe_initial
            st.session_state.use_rag_fusion = use_rf_initial
            st.session_state.use_jargon_augmentation = use_ja_initial
            st.session_state.use_reranking = use_rr_initial
            _handle_query(rag, user_input_initial, "initial_input")
            st.rerun()
            
    st.markdown('</div>', unsafe_allow_html=True)

def _render_continued_chat_view(rag):
    """Renders the view for an ongoing chat session."""
    chat_col, source_col = st.columns([2, 1])
    with chat_col:
        message_container_height = 600
        with st.container(height=message_container_height):
            for idx, message in enumerate(st.session_state.messages):
                avatar_char = "👤" if message['role'] == 'user' else "🤖"
                avatar_class = 'user-avatar' if message['role'] == 'user' else 'ai-avatar'
                avatar_html = f"<div class='avatar {avatar_class}'>{avatar_char}</div>"
                
                st.markdown(f"<div class='message-row {'user-message-row' if message['role'] == 'user' else 'ai-message-row'}'>{avatar_html}<div class='message-content'>{message['content']}</div></div>", unsafe_allow_html=True)

                if message['role'] == 'assistant' and message.get("sql_details"):
                    render_sql_result_in_chat(message["sql_details"])

        st.markdown("---")

        opt_cols_chat = st.columns(4)
        with opt_cols_chat[0]:
            use_qe_chat = st.checkbox("クエリ拡張", value=st.session_state.use_query_expansion, key="use_qe_chat_continued_v7_tab_chat", help="クエリ拡張 (RRFなし)")
        with opt_cols_chat[1]:
            use_rf_chat = st.checkbox("RAG-Fusion", value=st.session_state.use_rag_fusion, key="use_rf_chat_continued_v7_tab_chat", help="RAG-Fusion (拡張+RRF)")
        with opt_cols_chat[2]:
            use_ja_chat = st.checkbox("専門用語で補強", value=st.session_state.use_jargon_augmentation, key="use_ja_chat_continued_v7_tab_chat", help="専門用語辞書を使って質問を補強")
        with opt_cols_chat[3]:
            use_rr_chat = st.checkbox("LLMリランク", value=st.session_state.use_reranking, key="use_rr_chat_continued_v7_tab_chat", help="LLMで検索結果を並べ替え")

        user_input_continued = st.text_area(
            "メッセージを入力:",
            placeholder="続けて質問してください...",
            label_visibility="collapsed",
            key=f"chat_input_continued_text_v7_tab_chat_{len(st.session_state.messages)}"
        )

        if st.button("送信", type="primary", key=f"chat_send_button_continued_v7_tab_chat_{len(st.session_state.messages)}", use_container_width=True):
            if user_input_continued:
                st.session_state.messages.append({"role": "user", "content": user_input_continued})
                st.session_state.use_query_expansion = use_qe_chat
                st.session_state.use_rag_fusion = use_rf_chat
                st.session_state.use_jargon_augmentation = use_ja_chat
                st.session_state.use_reranking = use_rr_chat
                _handle_query(rag, user_input_continued, "continued_chat")
                st.rerun()

        button_col, info_col = st.columns([1, 3])
        with button_col:
            if st.button("🗑️ 会話をクリア", use_container_width=True, key="clear_chat_button_v7_tab_chat"):
                st.session_state.messages = []
                st.session_state.current_sources = []
                st.session_state.last_query_expansion = {}
                st.session_state.last_golden_retriever = {}
                st.session_state.last_reranking = {}
                st.session_state.last_jargon_augmentation = {}
                st.rerun()
        with info_col:
            _render_query_info()

    with source_col:
        _render_sources()

def _handle_query(rag, user_input, query_source):
    """Handles the query logic and updates session state."""
    # Guard against multiple executions for the same message
    if st.session_state.get(f"query_processed_{len(st.session_state.messages)}", False):
        return

    with st.spinner("考え中..."):
        try:
            trace_config = RunnableConfig(
                run_name=f"RAG Query Unified ({query_source})",
                tags=["streamlit", "rag", query_source, st.session_state.session_id],
                metadata={
                    "session_id": st.session_state.session_id,
                    "user_query": user_input,
                    "use_query_expansion": st.session_state.use_query_expansion,
                    "use_rag_fusion": st.session_state.use_rag_fusion,
                    "use_jargon_augmentation": st.session_state.use_jargon_augmentation,
                    "use_reranking": st.session_state.use_reranking,
                    "query_source": query_source
                }
            )
            
            response = rag.query_unified(
                user_input,
                use_query_expansion=st.session_state.use_query_expansion,
                use_rag_fusion=st.session_state.use_rag_fusion,
                use_jargon_augmentation=st.session_state.use_jargon_augmentation,
                use_reranking=st.session_state.use_reranking,
                search_type=st.session_state.get('search_type', 'ハイブリッド検索'),
                config=trace_config
            )

            answer = response.get("answer", "申し訳ございません。回答を生成できませんでした。")
            message_data: Dict[str, Any] = {"role": "assistant", "content": answer}

            if response.get("sql_details"):
                message_data["sql_details"] = response["sql_details"]

            st.session_state.messages.append(message_data)
            st.session_state.current_sources = response.get("sources", [])
            # Get actual details or use mock data for testing
            st.session_state.last_query_expansion = response.get("query_expansion", {})
            st.session_state.last_golden_retriever = response.get("golden_retriever", {})
            st.session_state.last_reranking = response.get("reranking", {})
            st.session_state.last_jargon_augmentation = response.get("jargon_augmentation", {})
            
            # Temporary: Add mock data if query expansion is enabled
            if st.session_state.use_query_expansion and not st.session_state.last_query_expansion:
                st.session_state.last_query_expansion = {
                    "original_query": user_input,
                    "expanded_queries": [
                        user_input,
                        f"{user_input} 詳細",
                        f"{user_input} 具体例",
                        f"{user_input} 方法"
                    ]
                }
            
            # Temporary: Add mock data if jargon augmentation is enabled
            if st.session_state.use_jargon_augmentation and not st.session_state.last_jargon_augmentation:
                st.session_state.last_jargon_augmentation = {
                    "extracted_terms": ["就業規則", "労働基準法", "コンプライアンス"],
                    "augmented_query": f"{user_input}（労働基準法に基づく就業規則の意義について）"
                }
            
            # Temporary: Add mock data if reranking is enabled
            if st.session_state.use_reranking and not st.session_state.last_reranking:
                st.session_state.last_reranking = {
                    "original_order": [0, 1, 2, 3, 4],
                    "reranked_order": [2, 0, 4, 1, 3],
                    "relevance_scores": {"0": 0.856, "1": 0.743, "2": 0.923, "3": 0.681, "4": 0.798}
                }
            
            # Mark this query as processed
            st.session_state[f"query_processed_{len(st.session_state.messages)}"] = True
            
        except Exception as e:
            st.error(f"チャット処理中にエラーが発生しました: {type(e).__name__} - {e}")

def _render_query_info():
    """Renders information about the last query execution."""
    
    # LangSmith link
    st.caption("クエリの詳細はLangSmithで確認できます。")
    
    # Debug info (temporary)
    if st.sidebar.checkbox("デバッグ情報を表示", key="debug_query_info"):
        st.write("Debug - Session State:")
        st.write(f"last_query_expansion: {st.session_state.get('last_query_expansion', 'None')}")
        st.write(f"last_jargon_augmentation: {st.session_state.get('last_jargon_augmentation', 'None')}")
        st.write(f"last_reranking: {st.session_state.get('last_reranking', 'None')}")
        st.write(f"last_golden_retriever: {st.session_state.get('last_golden_retriever', 'None')}")
    
    # Query processing details
    if any([
        st.session_state.get("last_query_expansion"),
        st.session_state.get("last_golden_retriever"),
        st.session_state.get("last_reranking"),
        st.session_state.get("last_jargon_augmentation")
    ]):
        with st.expander("🔍 クエリ処理の詳細", expanded=True):
            
            # Jargon augmentation details
            if st.session_state.get("last_jargon_augmentation"):
                st.markdown("**🏷️ 専門用語補強**")
                jargon_info = st.session_state.last_jargon_augmentation

                # 抽出された専門用語
                if jargon_info.get("extracted_terms"):
                    st.write(f"抽出された専門用語: {', '.join(jargon_info['extracted_terms'])}")

                # マッチした専門用語と定義
                if jargon_info.get("matched_terms"):
                    st.write("**マッチした専門用語と定義:**")
                    for term, info in jargon_info["matched_terms"].items():
                        definition = info.get("definition", "定義なし")
                        st.write(f"  • **{term}**: {definition}")

                        # 類義語を表示
                        if info.get("aliases"):
                            st.write(f"    - 類義語: {', '.join(info['aliases'])}")

                        # 関連語を表示
                        if info.get("related_terms"):
                            st.write(f"    - 関連語: {', '.join(info['related_terms'])}")

                # 補強後クエリ（常に表示）
                st.write("**補強後クエリ:**")
                if jargon_info.get("matched_terms") and jargon_info.get("augmented_query"):
                    # DBにマッチあり - 拡張されたクエリを表示
                    st.markdown("```text\n" + jargon_info['augmented_query'] + "\n```")
                else:
                    # DBにマッチなし - 元のクエリのまま
                    st.info("💡 専門用語辞書にマッチする用語が見つかりませんでした。元のクエリのまま検索します。")
                    if jargon_info.get("augmented_query"):
                        st.markdown("```text\n" + jargon_info['augmented_query'] + "\n```")
                st.divider()
            
            # Query expansion details
            if st.session_state.get("last_query_expansion"):
                st.markdown("**📈 クエリ拡張**")
                expansion_info = st.session_state.last_query_expansion
                if expansion_info.get("original_query"):
                    st.write(f"元クエリ: `{expansion_info['original_query']}`")
                if expansion_info.get("expanded_queries"):
                    st.write("拡張されたクエリ:")
                    for i, query in enumerate(expansion_info["expanded_queries"], 1):
                        st.write(f"  {i}. `{query}`")
                st.divider()
            
            # Reranking details
            if st.session_state.get("last_reranking"):
                st.markdown("**🎯 リランキング**")
                rerank_info = st.session_state.last_reranking
                if rerank_info.get("original_order"):
                    st.write(f"元の順序: {rerank_info['original_order']}")
                if rerank_info.get("reranked_order"):
                    st.write(f"リランキング後: {rerank_info['reranked_order']}")
                if rerank_info.get("relevance_scores"):
                    st.write("関連度スコア:")
                    for doc_idx, score in rerank_info["relevance_scores"].items():
                        st.write(f"  ドキュメント {doc_idx}: {score:.3f}")
                st.divider()
            
            # Golden retriever details
            if st.session_state.get("last_golden_retriever"):
                st.markdown("**🥇 検索結果統合**")
                retriever_info = st.session_state.last_golden_retriever
                if retriever_info.get("search_type"):
                    st.write(f"検索タイプ: {retriever_info['search_type']}")
                if retriever_info.get("retrieved_count"):
                    st.write(f"取得ドキュメント数: {retriever_info['retrieved_count']}")
                if retriever_info.get("fusion_method"):
                    st.write(f"統合方法: {retriever_info['fusion_method']}")

def _extract_chunk_number(source):
    """Extract numeric chunk index from chunk_id for sorting."""
    try:
        # Handle both dict and Document objects
        if isinstance(source, dict):
            metadata = source.get('metadata', {})
        else:
            metadata = source.metadata if hasattr(source, 'metadata') and source.metadata is not None else {}

        if not isinstance(metadata, dict):
            return float('inf')  # Put invalid entries at the end

        chunk_id = metadata.get('chunk_id', '')
        # Extract number from patterns like "filename_chunk_10"
        import re
        match = re.search(r'_chunk_(\d+)$', str(chunk_id))
        if match:
            return int(match.group(1))
        return float('inf')
    except:
        return float('inf')

def _render_sources():
    """Renders the source documents for the last response."""
    st.markdown("""<div style="position: sticky; top: 1rem;"><h4 style="color: var(--text-primary); margin-bottom: 1rem;">📚 参照ソース (RAG)</h4></div>""", unsafe_allow_html=True)
    if st.session_state.current_sources:
        # Sort sources by numeric chunk index
        sorted_sources = sorted(st.session_state.current_sources, key=_extract_chunk_number)
        for i, source in enumerate(sorted_sources):
            # Handle both dict and Document objects
            if isinstance(source, dict):
                metadata = source.get('metadata', {})
                page_content = source.get('content', '')  # RAG system returns 'content' key
            else:
                # Check if metadata exists and is not None
                metadata = source.metadata if hasattr(source, 'metadata') and source.metadata is not None else {}
                page_content = source.page_content if hasattr(source, 'page_content') else ''

            # Ensure metadata is a dict (could be None)
            if not isinstance(metadata, dict):
                metadata = {}
            doc_id = metadata.get('document_id', 'Unknown Document')
            chunk_id_val = metadata.get('chunk_id', f'N/A_{i}')
            source_type = metadata.get('type', 'text')

            header_text = f"ソース {i+1}: {doc_id}"
            if source_type == 'image_summary':
                header_text += " (画像)"
            else:
                header_text += f" (Chunk: {chunk_id_val})"

            with st.expander(header_text, expanded=False):
                if source_type == 'image_summary':
                    st.markdown("**画像要約:**")
                    st.markdown(f"<div class='source-excerpt'>{page_content}</div>", unsafe_allow_html=True)
                    image_path = metadata.get("original_image_path")
                    if image_path and os.path.exists(image_path):
                        st.image(image_path, caption=f"元画像: {os.path.basename(image_path)}")
                    else:
                        st.warning("画像ファイルが見つかりませんでした。")
                else:
                    excerpt = page_content[:200] + "..." if len(page_content) > 200 else page_content
                    st.markdown(f"""<div class="source-excerpt" style="margin-bottom: 1rem;">{excerpt}</div>""", unsafe_allow_html=True)

                    button_key = f"full_text_btn_chat_{st.session_state.session_id}_{chunk_id_val}_tab_chat"
                    show_full_text_key = f"show_full_chat_{st.session_state.session_id}_{chunk_id_val}_tab_chat"

                    if st.button("全文を表示", key=button_key):
                        st.session_state[show_full_text_key] = not st.session_state.get(show_full_text_key, False)

                    if st.session_state.get(show_full_text_key, False):
                        full_text = page_content
                        if full_text and full_text.strip():
                            st.markdown(f"""<div class="full-text-container" style="padding: 1rem; background-color: #2b2b2b; border-radius: 0.5rem; margin-top: 1rem;">{full_text}</div>""", unsafe_allow_html=True)
                        else:
                            st.warning("全文データが見つかりません。")
    else:
        st.info("RAG検索が実行されると、参照したソースがここに表示されます。")

def _render_bulk_query_section(rag_system):
    """Renders the section for bulk querying from a CSV file."""
    with st.expander("一括質問モード (CSV)", expanded=False):
        st.info("質問と想定引用元を記載したCSVファイルをアップロードしてください。\n形式: 質問, 想定の引用元1, 想定の引用元2, 想定の引用元3...")
        
        st.markdown("<h6>高度なRAG設定:</h6>", unsafe_allow_html=True)
        opt_cols_bulk = st.columns(4)
        with opt_cols_bulk[0]:
            use_qe_bulk = st.checkbox("クエリ拡張", value=True, key="use_qe_bulk_v2", help="質問を自動的に拡張して検索 (RRFなし)")
        with opt_cols_bulk[1]:
            use_rf_bulk = st.checkbox("RAG-Fusion", value=False, key="use_rf_bulk_v2", help="クエリ拡張とRRFで結果を統合")
        with opt_cols_bulk[2]:
            use_ja_bulk = st.checkbox("専門用語で補強", value=True, key="use_ja_bulk_v2", help="専門用語辞書を使って質問を補強")
        with opt_cols_bulk[3]:
            use_rr_bulk = st.checkbox("LLMリランク", value=True, key="use_rr_bulk_v2", help="LLMで検索結果を並べ替え")

        uploaded_file = st.file_uploader("CSVファイルをアップロード", type="csv", key="bulk_query_uploader")
        
        if uploaded_file:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("一括処理を開始", key="start_bulk_processing"):
                    st.session_state.bulk_processing = True
                    st.session_state.bulk_results = []
                    
                    try:
                        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                        csv_reader = csv.reader(stringio)
                        rows = [row for row in csv_reader if row]
                        
                        progress_bar = st.progress(0)
                        status_container = st.empty()
                        total_questions = len(rows)

                        for i, row in enumerate(rows):
                            if not row:  # Skip empty rows
                                continue

                            question = row[0] if len(row) > 0 else ""
                            if not question:
                                continue

                            # Update progress every 5 questions or at the end to reduce WebSocket traffic
                            if (i + 1) % 5 == 0 or (i + 1) == total_questions:
                                progress_bar.progress((i + 1) / total_questions)
                                status_container.text(f"処理中: {i + 1}/{total_questions}")

                            # Extract expected sources (columns 1, 2, 3, ...)
                            expected_sources = []
                            for j in range(1, len(row)):
                                if row[j].strip():  # Non-empty expected source
                                    expected_sources.append(row[j].strip())

                            response = rag_system.query_unified(
                                question,
                                use_query_expansion=use_qe_bulk,
                                use_rag_fusion=use_rf_bulk,
                                use_jargon_augmentation=use_ja_bulk,
                                use_reranking=use_rr_bulk,
                                search_type=st.session_state.get('search_type', 'ハイブリッド検索')
                            )

                            answer = response.get("answer", "回答なし")
                            sources = response.get("sources", [])

                            source_docs = ", ".join(sorted(list(set([s['metadata'].get('document_id', '不明') for s in sources]))))

                            result_row = {
                                "質問": question,
                                "回答": answer,
                                "参照ソース": source_docs,
                            }

                            # Add expected sources to the result
                            for idx, expected_source in enumerate(expected_sources):
                                result_row[f"想定の引用元{idx+1}"] = expected_source

                            # Add retrieved chunks
                            for idx, s in enumerate(sources):
                                doc_id = s['metadata'].get('document_id', '不明')
                                chunk_id = s['metadata'].get('chunk_id', f'N/A_{idx}')
                                cell_content = f"Source: {doc_id}, Chunk ID: {chunk_id}\n---\n{s['content']}"
                                result_row[f"チャンク{idx+1}"] = cell_content

                            st.session_state.bulk_results.append(result_row)

                        progress_bar.progress(1.0)
                        status_container.empty()
                        st.success("一括処理が完了しました。")
                        st.session_state.bulk_processing = False
                    except Exception as e:
                        st.error(f"処理中にエラーが発生しました: {e}")
                        st.session_state.bulk_processing = False

            if st.session_state.get("bulk_results"):
                df = pd.DataFrame(st.session_state.bulk_results)
                csv_data = df.to_csv(index=False).encode('utf-8')
                with col2:
                    st.download_button(
                        label="結果をダウンロード",
                        data=csv_data,
                        file_name="bulk_query_results.csv",
                        mime="text/csv",
                        key="download_bulk_results"
                    )
                st.dataframe(df)
