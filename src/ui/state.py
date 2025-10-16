import streamlit as st
import os
import uuid
from src.core.rag_system import Config, RAGSystem

def initialize_session_state():
    """Initializes the Streamlit session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_sources" not in st.session_state:
        st.session_state.current_sources = []
    if "last_query_expansion" not in st.session_state:
        st.session_state.last_query_expansion = {}
    if "last_golden_retriever" not in st.session_state:
        st.session_state.last_golden_retriever = {}
    if "last_reranking" not in st.session_state:
        st.session_state.last_reranking = {}
    if "use_query_expansion" not in st.session_state:
        st.session_state.use_query_expansion = False
    if "use_rag_fusion" not in st.session_state:
        st.session_state.use_rag_fusion = False
    if "use_jargon_augmentation" not in st.session_state:
        st.session_state.use_jargon_augmentation = False
    if "use_reranking" not in st.session_state:
        st.session_state.use_reranking = os.getenv("ENABLE_RERANKING", "false").lower() == "true"
    if "search_type" not in st.session_state:
        st.session_state.search_type = "ハイブリッド検索"
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

@st.cache_resource(show_spinner=False)
def initialize_rag_system(config_obj: Config) -> RAGSystem:
    """Cached function to initialize the RAGSystem."""
    return RAGSystem(config_obj)

def get_rag_system():
    """
    Initializes and retrieves the RAG system from session state.
    Returns the RAG system instance or None if initialization fails.
    """
    if "rag_system" not in st.session_state:
        # Load environment variables and defaults
        azure_api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        azure_chat_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "")
        azure_embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", "")

        if all([azure_api_key, azure_endpoint, azure_chat_deployment, azure_embedding_deployment]):
            try:
                app_config = Config(
                    azure_openai_api_key=azure_api_key,
                    azure_openai_endpoint=azure_endpoint,
                    azure_openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
                    azure_openai_chat_deployment_name=azure_chat_deployment,
                    azure_openai_embedding_deployment_name=azure_embedding_deployment,
                    db_password=os.getenv("DB_PASSWORD", "postgres"),
                    llm_model_identifier=os.getenv("LLM_MODEL_IDENTIFIER", "gpt-4.1-mini"),
                    embedding_model_identifier=os.getenv("EMBEDDING_MODEL_IDENTIFIER", "text-embedding-ada-002"),
                    collection_name=os.getenv("COLLECTION_NAME", "documents"),
                    final_k=int(os.getenv("FINAL_K", 5)),
                    enable_jargon_augmentation=st.session_state.use_jargon_augmentation
                )
                st.session_state.rag_system = initialize_rag_system(app_config)
                st.toast("✅ RAGシステムがAzure OpenAIで正常に初期化されました", icon="🎉")
            except Exception as e:
                st.error(f"Azure RAGシステムの初期化中にエラーが発生しました: {type(e).__name__} - {e}")
                st.warning("""
    ### 🔧 Azure OpenAI 接続エラーの解決方法 (一般的な例)

    1.  **.envファイルまたは環境変数を確認してください**:
        `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_VERSION`,
        `AZURE_OPENAI_CHAT_DEPLOYMENT_NAME`, `AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME`
        が正しく設定されているか確認してください。
    2.  **デプロイメント名**: Azureポータルで設定したチャットモデルと埋め込みモデルのデプロイメント名が正確であることを確認してください。
    3.  **エンドポイントとAPIバージョン**: エンドポイントのURLとAPIバージョンが正しいか確認してください。
    4.  **ネットワーク接続**: Azure OpenAIサービスへのネットワークアクセスが可能であることを確認してください（ファイアウォール、プロキシ設定など）。
                """)
                st.session_state.rag_system = None
        else:
            st.warning("Azure OpenAIのAPIキーと関連設定がされていません。チャット機能を利用できません。サイドバーから設定してください。")
            st.session_state.rag_system = None
            
    return st.session_state.get("rag_system")
