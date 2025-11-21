import streamlit as st
import pandas as pd
import time
from sqlalchemy import text
from src.utils.helpers import _persist_uploaded_file, get_documents_dataframe, create_empty_collection, delete_collection

def _get_available_collections(rag_system):
    """Get list of available collections from database"""
    try:
        with rag_system.engine.connect() as conn:
            result = conn.execute(text("""
                SELECT DISTINCT collection_name
                FROM document_chunks
                ORDER BY collection_name
            """))
            collections = [row[0] for row in result]
            return collections if collections else [rag_system.config.collection_name]
    except Exception as e:
        return [rag_system.config.collection_name]

def render_documents_tab(rag_system):
    """Renders the document management tab."""
    if not rag_system:
        st.info("RAGシステムが初期化されていません。サイドバーで設定を確認してください。")
        return

    st.markdown("### 📤 ドキュメントアップロード")

    # Collection management UI
    st.markdown("#### 📂 コレクション管理")

    available_collections = _get_available_collections(rag_system)
    current_collection = st.session_state.get("selected_collection", rag_system.config.collection_name)

    col1, col2, col3 = st.columns([4, 1, 1])

    with col1:
        selected_collection = st.selectbox(
            "保存先コレクションを選択",
            available_collections,
            index=available_collections.index(current_collection) if current_collection in available_collections else 0,
            key="collection_selector"
        )

    with col2:
        if st.button("➕ 新規作成", use_container_width=True, key="create_collection_btn"):
            st.session_state.show_create_dialog = True

    with col3:
        if st.button("🗑️ 削除", use_container_width=True, key="delete_collection_btn", type="secondary"):
            st.session_state.show_delete_dialog = True

    # Create collection dialog
    if st.session_state.get("show_create_dialog", False):
        @st.dialog("新規コレクション作成")
        def create_collection_dialog():
            new_collection_name = st.text_input(
                "コレクション名を入力",
                placeholder="例: 技術文書, 営業資料, 法務文書",
                key="new_collection_input"
            )

            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("作成", type="primary", use_container_width=True):
                    if new_collection_name:
                        if new_collection_name in available_collections:
                            st.error(f"コレクション '{new_collection_name}' は既に存在します。")
                        else:
                            if create_empty_collection(rag_system, new_collection_name):
                                st.success(f"コレクション '{new_collection_name}' を作成しました。")
                                st.session_state.selected_collection = new_collection_name
                                st.session_state.show_create_dialog = False
                                st.session_state.force_collection_switch = True  # Force collection switch
                                if "rag_system" in st.session_state:
                                    del st.session_state["rag_system"]
                                st.rerun()
                    else:
                        st.error("コレクション名を入力してください。")
            with col_b:
                if st.button("キャンセル", use_container_width=True):
                    st.session_state.show_create_dialog = False
                    st.rerun()

        create_collection_dialog()

    # Delete collection dialog
    if st.session_state.get("show_delete_dialog", False):
        @st.dialog("コレクション削除")
        def delete_collection_dialog():
            st.warning(f"⚠️ コレクション **'{current_collection}'** を削除しますか？")
            st.error("この操作は取り消せません。コレクション内のすべてのドキュメントとデータが削除されます。")

            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("削除する", type="primary", use_container_width=True):
                    delete_collection(rag_system, current_collection)
                    st.session_state.show_delete_dialog = False
                    # Switch to first available collection or default
                    remaining_collections = _get_available_collections(rag_system)
                    if remaining_collections:
                        st.session_state.selected_collection = remaining_collections[0]
                    else:
                        st.session_state.selected_collection = "documents"
                    if "rag_system" in st.session_state:
                        del st.session_state["rag_system"]
                    st.rerun()
            with col_b:
                if st.button("キャンセル", use_container_width=True):
                    st.session_state.show_delete_dialog = False
                    st.rerun()

        delete_collection_dialog()

    # Show current collection status
    st.info(f"**現在の保存先:** {current_collection}")

    # Collection change handling
    if (selected_collection and selected_collection != st.session_state.get("selected_collection")) or \
       st.session_state.get("force_collection_switch", False):
        st.session_state.selected_collection = selected_collection
        st.session_state.force_collection_switch = False  # Clear flag after processing
        # Clear RAG system to reinitialize with new collection
        if "rag_system" in st.session_state:
            del st.session_state["rag_system"]
        st.rerun()

    st.markdown("---")

    # PDF処理方式の表示
    st.info(f"📑 PDF処理方式: **Azure Document Intelligence**")

    uploaded_docs = st.file_uploader(
        "ファイルを選択またはドラッグ&ドロップ (.pdf)",
        accept_multiple_files=True,
        type=["pdf"],
        label_visibility="collapsed",
        key=f"doc_uploader_v7_tab_documents_{rag_system.config.collection_name}"
    )

    if uploaded_docs:
        st.markdown(f"#### 選択されたファイル ({len(uploaded_docs)})")
        file_info = [{"ファイル名": f.name, "サイズ": f"{f.size / 1024:.1f} KB", "タイプ": f.type or "不明"} for f in uploaded_docs]
        st.dataframe(pd.DataFrame(file_info), use_container_width=True, hide_index=True)

        if st.button("🚀 ドキュメントを処理 (インジェスト)", type="primary", use_container_width=True, key="process_docs_button_v7_tab_documents"):
            progress_bar = st.progress(0, text="処理開始...")
            status_text = st.empty()
            try:
                paths_to_ingest = []
                for i, file in enumerate(uploaded_docs):
                    status_text.info(f"一時保存中: {file.name}")
                    paths_to_ingest.append(str(_persist_uploaded_file(file)))
                    progress_bar.progress((i + 1) / (len(uploaded_docs) * 2), text=f"一時保存完了: {file.name}")

                status_text.info(f"インデックスを構築中... ({len(paths_to_ingest)}件のファイル)")
                rag_system.ingest_documents(paths_to_ingest)
                progress_bar.progress(1.0, text="インジェスト完了！")
                st.success(f"✅ {len(uploaded_docs)}個のファイルが正常に処理されました！")
                time.sleep(1)
                st.balloons()
                st.rerun()
            except Exception as e:
                st.error(f"ドキュメント処理中にエラーが発生しました: {type(e).__name__} - {e}")
            finally:
                progress_bar.empty()
                status_text.empty()

    st.markdown("### 📚 登録済みドキュメント")
    docs_df = get_documents_dataframe(rag_system)
    
    if 'doc_to_show_chunks' not in st.session_state:
        st.session_state.doc_to_show_chunks = None

    if not docs_df.empty:
        # Use dataframe for efficient display instead of loop
        display_df = docs_df.copy()
        display_df['表示'] = False

        # Add an editable column for viewing chunks
        edited_df = st.data_editor(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Document ID": st.column_config.TextColumn("Document ID", width="large"),
                "Chunks": st.column_config.NumberColumn("Chunks", width="small"),
                "Last Updated": st.column_config.TextColumn("Last Updated", width="medium"),
                "表示": st.column_config.CheckboxColumn(
                    "チャンク表示",
                    help="チャンクを表示する場合はチェック",
                    default=False,
                    width="small"
                )
            },
            disabled=["Document ID", "Chunks", "Last Updated"],
            key="docs_table_editor"
        )

        # Show chunks for checked documents
        docs_to_show = edited_df[edited_df['表示'] == True]
        if not docs_to_show.empty:
            st.markdown("---")
            st.markdown("### 📄 チャンク詳細")
            for _, row in docs_to_show.iterrows():
                doc_id = row["Document ID"]
                with st.expander(f"📋 {doc_id} のチャンク ({row['Chunks']}個)", expanded=True):
                    with st.spinner(f"'{doc_id}'のチャンクを取得中..."):
                        chunks_df = rag_system.get_chunks_by_document_id(doc_id)

                    if not chunks_df.empty:
                        csv = chunks_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="💾 全チャンクをCSVでダウンロード",
                            data=csv,
                            file_name=f"chunks_{doc_id}.csv",
                            mime="text/csv",
                            key=f"download_chunks_{doc_id}"
                        )

                        # Display chunks in a dataframe instead of loop for better performance
                        chunk_display_df = chunks_df[['chunk_id', 'content']].copy()
                        chunk_display_df['content'] = chunk_display_df['content'].str[:200] + '...'

                        st.dataframe(
                            chunk_display_df,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "chunk_id": "Chunk ID",
                                "content": st.column_config.TextColumn("Content (Preview)", width="large")
                            }
                        )

                        # Optional: Show full content for selected chunk
                        selected_chunk = st.selectbox(
                            "全文を表示するチャンクを選択:",
                            ["選択してください..."] + chunks_df['chunk_id'].tolist(),
                            key=f"chunk_selector_{doc_id}"
                        )

                        if selected_chunk != "選択してください...":
                            full_content = chunks_df[chunks_df['chunk_id'] == selected_chunk]['content'].iloc[0]
                            st.markdown(
                                f"""
                                <div style="background-color: #262730; border-radius: 0.5rem; padding: 10px; max-height: 400px; overflow-y: auto; border: 1px solid #333;">
                                    <pre style="white-space: pre-wrap; word-wrap: break-word; color: #FAFAFA;">{full_content}</pre>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                    else:
                        st.warning("このドキュメントにはチャンクデータが見つかりませんでした。")
        
        st.markdown("---")
        st.markdown("### 🗑️ ドキュメント削除")
        doc_ids_for_deletion = ["選択してください..."] + docs_df["Document ID"].tolist()
        doc_to_delete = st.selectbox(
            "削除するドキュメントIDを選択:",
            doc_ids_for_deletion,
            label_visibility="collapsed",
            key=f"doc_delete_selectbox_v7_tab_documents_{rag_system.config.collection_name}"
        )
        if doc_to_delete != "選択してください...":
            st.warning(f"**警告:** ドキュメント '{doc_to_delete}' を削除すると、関連する全てのチャンクがデータベースとベクトルストアから削除されます。この操作は元に戻せません。")
            if st.button(f"'{doc_to_delete}' を削除実行", type="secondary", key="doc_delete_button_v7_tab_documents"):
                try:
                    with st.spinner(f"削除中: {doc_to_delete}"):
                        success, message = rag_system.delete_document_by_id(doc_to_delete)
                    if success:
                        st.success(message)
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(message)
                except Exception as e:
                    st.error(f"ドキュメント削除中にエラーが発生しました: {type(e).__name__} - {e}")
    else:
        st.info("まだドキュメントが登録されていません。上のセクションからアップロードしてください。")
