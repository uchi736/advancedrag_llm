import streamlit as st
import os
from sqlalchemy import text

def check_vector_store_has_data(rag_system):
    """Check if vector store has any documents."""
    try:
        if not rag_system or not hasattr(rag_system, 'engine'):
            return False
        with rag_system.engine.connect() as conn:
            result = conn.execute(text(
                "SELECT COUNT(*) FROM langchain_pg_embedding"
            ))
            count = result.scalar()
            return count > 0
    except:
        # If table doesn't exist or any error, assume no data
        return False

def render_sidebar(rag_system, env_defaults):
    """Renders the sidebar and handles configuration updates."""
    with st.sidebar:
        st.markdown("<h2 style='color: var(--text-primary);'>⚙️ Configuration</h2>", unsafe_allow_html=True)
        if rag_system:
            st.success(f"✅ System Online - Collection: **{rag_system.config.collection_name}**")

            # LLM Provider info
            provider = (rag_system.config.llm_provider or "azure").upper()
            if provider == "VLLM":
                endpoint = rag_system.config.vllm_endpoint or "未設定"
                st.info(f"🤖 LLM: **{provider}**\n\n`{endpoint}`")
            elif provider == "AZURE":
                deployment = rag_system.config.azure_openai_chat_deployment_name or "未設定"
                st.info(f"🤖 LLM: **{provider}**\n\n`{deployment}`")
            elif provider == "HUGGINGFACE":
                model = rag_system.config.hf_model_id or "未設定"
                st.info(f"🤖 LLM: **{provider}**\n\n`{model}`")
            else:
                st.info(f"🤖 LLM: **{provider}**")
        else:
            st.warning("⚠️ System Offline")

        st.info("すべての設定は「詳細設定」タブで行えます。")

        # Add search type selection
        st.session_state.search_type = st.radio(
            "検索タイプを選択",
            ('ハイブリッド検索', 'ベクトル検索'),
            index=0 if st.session_state.get('search_type', 'ハイブリッド検索') == 'ハイブリッド検索' else 1,
            key='search_type_radio'
        )
        
        # PDF処理方式の表示
        st.markdown("---")
        st.markdown("#### 📑 PDF処理方式")
        st.info("Azure Document Intelligence (固定)")


def render_langsmith_info():
    """Renders LangSmith tracing info in the sidebar."""
    langsmith_api_key = os.getenv("LANGCHAIN_API_KEY")
    langsmith_project = os.getenv("LANGCHAIN_PROJECT")
    if langsmith_api_key:
        st.sidebar.success(f"ιχ LangSmith Tracing: ENABLED{' (Project: ' + langsmith_project + ')' if langsmith_project else ''}")
    else:
        st.sidebar.info("ιχ LangSmith Tracing: DISABLED (環境変数を設定してください)")
