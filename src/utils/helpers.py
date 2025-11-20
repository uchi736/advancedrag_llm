import streamlit as st
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from datetime import datetime
from typing import Dict, Any

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# This is a forward declaration. The actual RAGSystem class will be imported where needed.
# This avoids circular dependencies.
class RAGSystem:
    pass

def _persist_uploaded_file(uploaded_file) -> Path:
    if uploaded_file is None:
        raise ValueError("Uploaded file cannot be None")
    tmp_dir = Path(tempfile.gettempdir()) / "rag_uploads"
    tmp_dir.mkdir(exist_ok=True)
    tmp_path = tmp_dir / uploaded_file.name
    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return tmp_path

def get_collection_statistics(rag: RAGSystem) -> Dict[str, Any]:
    if not rag:
        return {"documents": 0, "chunks": 0, "collection_name": "N/A"}
    try:
        engine = create_engine(rag.connection_string)
        with engine.connect() as conn:
            query = text("SELECT COUNT(DISTINCT document_id) AS num_documents, COUNT(*) AS num_chunks FROM document_chunks WHERE collection_name = :collection")
            result = conn.execute(query, {"collection": rag.config.collection_name}).first()
        return {
            "documents": result.num_documents if result else 0,
            "chunks": result.num_chunks if result else 0,
            "collection_name": rag.config.collection_name
        }
    except Exception as e:
        st.error(f"統計情報の取得に失敗: {e}")
        return {"documents": 0, "chunks": 0, "collection_name": rag.config.collection_name if rag else "N/A"}

def get_documents_dataframe(rag: RAGSystem) -> pd.DataFrame:
    if not rag:
        return pd.DataFrame()
    try:
        engine = create_engine(rag.connection_string)
        with engine.connect() as conn:
            query = text("""
                SELECT document_id, COUNT(*) as chunk_count, MAX(created_at) as last_updated
                FROM document_chunks
                WHERE collection_name = :collection
                  AND document_id NOT LIKE '__collection_marker_%'
                GROUP BY document_id
                ORDER BY last_updated DESC
            """)
            result = conn.execute(query, {"collection": rag.config.collection_name})
            df = pd.DataFrame(result.fetchall(), columns=["Document ID", "Chunks", "Last Updated"])
        if not df.empty and "Last Updated" in df.columns:
            df["Last Updated"] = pd.to_datetime(df["Last Updated"]).dt.strftime("%Y-%m-%d %H:%M")
        return df
    except Exception as e:
        st.error(f"登録済みドキュメントリストの取得に失敗: {e}")
        return pd.DataFrame()

def get_query_history_data(days: int = 30) -> pd.DataFrame:
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    queries = [20 + int(10 * abs(np.sin(i / 5.0))) + np.random.randint(-3, 4) for i in range(days)]
    queries = [max(0, q) for q in queries]
    return pd.DataFrame({'Date': dates, 'Queries': queries})

def render_simple_chart(df: pd.DataFrame):
    """簡単なチャート描画"""
    try:
        if df.empty:
            st.info("チャートを描画するデータがありません。")
            return

        if not PLOTLY_AVAILABLE:
            st.warning("Plotlyライブラリがインストールされていないため、チャートを表示できません。`pip install plotly plotly-express`でインストールしてください。")
            return

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if not numeric_cols:
            st.info("数値型の列がないため、チャートを描画できません。")
            return

        categorical_cols = df.select_dtypes(include=['object', 'category', 'datetime64']).columns.tolist()

        chart_type_options = ["なし"]
        if len(df.columns) >= 2 and categorical_cols and numeric_cols:
            chart_type_options.append("棒グラフ")
        if numeric_cols:
            chart_type_options.append("折れ線グラフ")
        if len(numeric_cols) >= 2:
            chart_type_options.append("散布図")

        if len(chart_type_options) == 1:
            st.info("適切なデータ形式ではないため、チャートタイプを選択できません。")
            return

        chart_type = st.selectbox("可視化タイプを選択:", chart_type_options, key=f"sql_chart_type_selector_{df.shape[0]}_{df.shape[1]}")

        if chart_type == "棒グラフ":
            if categorical_cols and numeric_cols:
                x_col_bar = st.selectbox("X軸 (カテゴリ/日付)", categorical_cols, key=f"bar_x_sql_{df.shape[0]}")
                y_col_bar = st.selectbox("Y軸 (数値)", numeric_cols, key=f"bar_y_sql_{df.shape[0]}")
                if x_col_bar and y_col_bar:
                    fig = px.bar(df.head(25), x=x_col_bar, y=y_col_bar, title=f"{y_col_bar} by {x_col_bar}")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("棒グラフにはカテゴリ列と数値列が必要です。")

        elif chart_type == "折れ線グラフ":
            y_cols_line = st.multiselect("Y軸 (数値 - 複数選択可)", numeric_cols, default=numeric_cols[0] if numeric_cols else None, key=f"line_y_sql_{df.shape[0]}")
            date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
            x_col_line_options = ["(インデックス)"] + categorical_cols
            
            chosen_x_col = None
            if date_cols:
                x_col_line_options = ["(インデックス)"] + date_cols + [c for c in categorical_cols if c not in date_cols]
                chosen_x_col = date_cols[0]
            elif categorical_cols:
                chosen_x_col = categorical_cols[0]

            x_col_line = st.selectbox("X軸", x_col_line_options, index=x_col_line_options.index(chosen_x_col) if chosen_x_col and chosen_x_col in x_col_line_options else 0, key=f"line_x_sql_{df.shape[0]}")

            if y_cols_line:
                title_ys = ", ".join(y_cols_line)
                if x_col_line and x_col_line != "(インデックス)":
                    fig = px.line(df.head(100), x=x_col_line, y=y_cols_line, title=f"{title_ys} over {x_col_line}", markers=True)
                else:
                    fig = px.line(df.head(100), y=y_cols_line, title=f"{title_ys} Trend", markers=True)
                st.plotly_chart(fig, use_container_width=True)
            
        elif chart_type == "散布図":
            if len(numeric_cols) >= 2:
                x_col_scatter = st.selectbox("X軸 (数値)", numeric_cols, key=f"scatter_x_sql_{df.shape[0]}")
                y_col_scatter = st.selectbox("Y軸 (数値)", [nc for nc in numeric_cols if nc != x_col_scatter], key=f"scatter_y_sql_{df.shape[0]}")
                color_col_scatter_options = ["なし"] + categorical_cols + [nc for nc in numeric_cols if nc != x_col_scatter and nc != y_col_scatter]
                color_col_scatter = st.selectbox("色分け (任意)", color_col_scatter_options, key=f"scatter_color_sql_{df.shape[0]}")

                if x_col_scatter and y_col_scatter:
                    fig = px.scatter(
                        df.head(500),
                        x=x_col_scatter, 
                        y=y_col_scatter, 
                        color=color_col_scatter if color_col_scatter != "なし" else None,
                        title=f"{y_col_scatter} vs {x_col_scatter}" + (f" by {color_col_scatter}" if color_col_scatter != "なし" else "")
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("散布図には少なくとも2つの数値列が必要です。")

    except Exception as e:
        st.error(f"チャート描画エラー: {type(e).__name__} - {e}")

def render_sql_result_in_chat(sql_details_dict: Dict[str, Any]):
    """チャット内でのSQL関連情報表示"""
    if not sql_details_dict or not isinstance(sql_details_dict, dict):
        st.warning("チャット表示用のSQL詳細情報がありません。")
        return

    with st.expander("🔍 実行されたSQL (チャット内)", expanded=False):
        st.code(sql_details_dict.get("generated_sql", "SQLが生成されませんでした。"), language="sql")

    results_data_preview = sql_details_dict.get("results_preview")
    if results_data_preview and isinstance(results_data_preview, list) and len(results_data_preview) > 0:
        with st.expander("📊 SQL実行結果プレビュー (チャット内)", expanded=False):
            try:
                df_chat_preview = pd.DataFrame(results_data_preview)
                st.dataframe(df_chat_preview, use_container_width=True, height = min(300, (len(df_chat_preview) + 1) * 35 + 3))
                
                total_fetched = sql_details_dict.get("row_count_fetched", 0)
                preview_count = len(results_data_preview)
                if total_fetched > preview_count:
                    st.caption(f"結果の最初の{preview_count}件を表示（全{total_fetched}件取得）。")
                elif total_fetched > 0:
                    st.caption(f"全{total_fetched}件の結果を表示。")
            except Exception as e:
                st.error(f"チャット内でのSQL結果プレビュー表示エラー: {e}")
    elif sql_details_dict.get("success"):
        with st.expander("📊 SQL実行結果プレビュー (チャット内)", expanded=False):
            st.info("SQLクエリは成功しましたが、該当するデータはありませんでした。")

@st.cache_data(ttl=60, show_spinner=False)
def load_terms_from_db(pg_url: str, jargon_table_name: str, keyword: str = "") -> pd.DataFrame:
    """PostgreSQLから用語辞書を読み込む"""
    if not pg_url:
        return pd.DataFrame()
    
    try:
        engine = create_engine(pg_url)
        
        # テーブルの存在確認
        with engine.connect() as conn:
            check_table = text(f"""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = '{jargon_table_name}'
                );
            """)
            table_exists = conn.execute(check_table).scalar()
            
            if not table_exists:
                return pd.DataFrame()
        
        # 用語データの取得
        if keyword:
            # Use %(name)s style for psycopg2 compatibility with pandas
            query = f"""
                SELECT term, definition, domain, aliases, related_terms, confidence_score, updated_at
                FROM {jargon_table_name}
                WHERE term ILIKE %(keyword)s
                   OR definition ILIKE %(keyword)s
                   OR EXISTS (
                       SELECT 1 FROM unnest(aliases) AS s 
                       WHERE s ILIKE %(keyword)s
                   )
                ORDER BY term
            """
            params = {"keyword": f"%{keyword}%"}
        else:
            query = f"""
                SELECT term, definition, domain, aliases, related_terms, confidence_score, updated_at
                FROM {jargon_table_name}
                ORDER BY term
            """
            params = {}
        
        df = pd.read_sql(query, engine, params=params)
        
        if not df.empty and "updated_at" in df.columns:
            df["updated_at"] = pd.to_datetime(df["updated_at"]).dt.strftime("%Y-%m-%d %H:%M")
        
        return df
        
    except Exception as e:
        st.error(f"用語辞書の読み込みエラー: {e}")
        return pd.DataFrame()

def render_term_card(term_data: pd.Series):
    """用語カードのレンダリング"""
    st.markdown(f"""
    <div class="term-card">
        <div class="term-headword">{term_data['term']}</div>
        <div class="term-definition">{term_data['definition']}</div>
        <div class="term-meta">
            <strong>分野:</strong> {term_data.get('domain', 'N/A')}
        </div>
        <div class="term-meta">
            <strong>類義語:</strong> {', '.join(term_data['aliases']) if term_data['aliases'] else 'なし'}
        </div>
        <div class="term-meta">
            <strong>関連語:</strong> {', '.join(term_data['related_terms']) if term_data['related_terms'] else 'なし'}
        </div>
    </div>
    """, unsafe_allow_html=True)

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors using numpy

    Args:
        vec1: First vector (numpy array or list)
        vec2: Second vector (numpy array or list)

    Returns:
        Cosine similarity score(s) as numpy array
    """
    # Convert to numpy arrays if needed
    if isinstance(vec1, list):
        vec1 = np.array(vec1)
    if isinstance(vec2, list):
        vec2 = np.array(vec2)

    # Reshape if needed (ensure 2D arrays)
    if vec1.ndim == 1:
        vec1 = vec1.reshape(1, -1)
    if vec2.ndim == 1:
        vec2 = vec2.reshape(1, -1)

    # Calculate cosine similarity
    dot_product = np.dot(vec1, vec2.T)
    norm1 = np.linalg.norm(vec1, axis=1, keepdims=True)
    norm2 = np.linalg.norm(vec2, axis=1, keepdims=True)
    similarity = dot_product / (norm1 @ norm2.T + 1e-10)

    return similarity

def create_empty_collection(rag_system, collection_name: str) -> bool:
    """
    Create an empty collection by inserting a dummy record into document_chunks.
    Returns True if successful, False otherwise.
    """
    try:
        with rag_system.engine.connect() as conn:
            # Insert a dummy record with NULL content to mark collection existence
            conn.execute(text("""
                INSERT INTO document_chunks (document_id, chunk_id, content, collection_name, created_at)
                VALUES (:doc_id, :chunk_id, :content, :collection_name, :created_at)
            """), {
                "doc_id": f"__collection_marker_{collection_name}__",
                "chunk_id": f"__collection_marker_{collection_name}___chunk_0",
                "content": "",  # Empty string instead of NULL to satisfy NOT NULL constraint
                "collection_name": collection_name,
                "created_at": datetime.now()
            })
            conn.commit()
        return True
    except Exception as e:
        st.error(f"コレクション作成エラー: {e}")
        return False

def delete_collection(rag_system, collection_name: str) -> int:
    """
    Delete all records for a collection from all relevant tables.
    Returns the total number of records deleted.
    """
    try:
        total_deleted = 0

        with rag_system.engine.connect() as conn:
            # Delete from document_chunks
            result = conn.execute(text("""
                DELETE FROM document_chunks WHERE collection_name = :collection_name
            """), {"collection_name": collection_name})
            chunks_deleted = result.rowcount
            total_deleted += chunks_deleted

            # Delete from langchain_pg_embedding (if collection_name is stored in cmetadata)
            result = conn.execute(text("""
                DELETE FROM langchain_pg_embedding
                WHERE cmetadata->>'collection_name' = :collection_name
            """), {"collection_name": collection_name})
            embeddings_deleted = result.rowcount
            total_deleted += embeddings_deleted

            # Delete from jargon_dictionary if it has collection_name column
            try:
                result = conn.execute(text("""
                    DELETE FROM jargon_dictionary WHERE collection_name = :collection_name
                """), {"collection_name": collection_name})
                jargon_deleted = result.rowcount
                total_deleted += jargon_deleted
            except Exception:
                # Table might not have collection_name column, skip
                pass

            conn.commit()

        st.success(f"コレクション '{collection_name}' を削除しました (削除レコード数: {total_deleted})")
        return total_deleted
    except Exception as e:
        st.error(f"コレクション削除エラー: {e}")
        return 0
