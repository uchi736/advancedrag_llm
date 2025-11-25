import streamlit as st
import os
import time
from src.core.rag_system import Config

def render_settings_tab(rag_system, env_defaults):
    """Renders the detailed settings tab."""
    st.markdown("### ⚙️ システム詳細設定")
    st.caption("RAGシステムの詳細な設定を行います。変更後は「設定を適用」ボタンをクリックしてください。システムの再初期化が必要な場合があります。")

    temp_default_cfg = Config()
    current_values = {}
    if rag_system and hasattr(rag_system, 'config'):
        current_values = rag_system.config.__dict__.copy()
    else:
        current_values = temp_default_cfg.__dict__.copy()
        for key, value in env_defaults.items():
            if key.lower() in current_values:
                current_values[key.lower()] = value
            if key.lower() == "openai_api_key":
                current_values[key.lower()] = None

    for key in temp_default_cfg.__dict__:
        if key not in current_values:
            current_values[key] = getattr(temp_default_cfg, key)
        if key == "openai_api_key":
            current_values[key] = None

    # LLM Provider Selection (outside form for real-time switching)
    st.markdown("### 🤖 LLMプロバイダー選択")
    provider_options = ["azure", "huggingface"]
    provider_labels = {"azure": "Azure OpenAI", "huggingface": "Hugging Face (ローカルLLM)"}

    # Initialize session state for provider selection
    if "selected_llm_provider" not in st.session_state:
        current_provider = current_values.get("llm_provider", "azure")
        st.session_state.selected_llm_provider = current_provider

    # Ensure selected provider is valid
    current_provider_value = st.session_state.selected_llm_provider
    if current_provider_value not in provider_options:
        current_provider_value = "azure"
        st.session_state.selected_llm_provider = current_provider_value

    selected_provider = st.radio(
        "LLMプロバイダー",
        provider_options,
        index=provider_options.index(current_provider_value),
        format_func=lambda x: provider_labels[x],
        key="llm_provider_radio_v7",
        horizontal=True,
        help="Azure OpenAIまたはローカルのHugging Faceモデルを選択"
    )
    st.session_state.selected_llm_provider = selected_provider
    st.markdown("---")

    with st.form("detailed_settings_form_v7_tab_settings"):
        col1, col2 = st.columns(2)
        with col1:
            _render_azure_settings(current_values)
            _render_huggingface_settings(current_values, temp_default_cfg)
            _render_chunking_settings(current_values, temp_default_cfg)
        with col2:
            _render_search_rag_settings(current_values, temp_default_cfg)
            _render_pdf_processing_settings(current_values, temp_default_cfg)

        st.markdown("---")
        st.markdown("#### 🗄️ データベース設定 (変更には注意が必要です)")
        db_col1, db_col2 = st.columns(2)
        with db_col1:
            _render_db_connection_settings(current_values, temp_default_cfg)
        with db_col2:
            _render_db_auth_settings(current_values, temp_default_cfg)

        s_col, r_col = st.columns([3, 1])
        apply_button = s_col.form_submit_button("🔄 設定を適用", type="primary", use_container_width=True)
        reset_button = r_col.form_submit_button("↩️ デフォルトにリセット", use_container_width=True)

    if apply_button:
        _apply_settings(st.session_state.form_values)
    
    if reset_button:
        _reset_to_defaults(env_defaults)

    st.markdown("---")
    st.markdown("### 📋 現在の有効な設定")
    _display_current_config(rag_system)

def _render_azure_settings(values):
    # Check session state for provider selection
    if st.session_state.get("selected_llm_provider", "azure") != "azure":
        return

    st.markdown("#### 🔑 Azure OpenAI 設定")
    if 'form_values' not in st.session_state:
        st.session_state.form_values = {}

    st.session_state.form_values['llm_provider'] = st.session_state.selected_llm_provider
    st.session_state.form_values['azure_openai_api_key'] = st.text_input("Azure OpenAI APIキー", value=values.get("azure_openai_api_key", ""), type="password", key="setting_azure_key_v7")
    st.session_state.form_values['azure_openai_endpoint'] = st.text_input("Azure OpenAI エンドポイント", value=values.get("azure_openai_endpoint", ""), key="setting_azure_endpoint_v7")
    st.session_state.form_values['azure_openai_api_version'] = st.text_input("Azure OpenAI APIバージョン", value=values.get("azure_openai_api_version", ""), key="setting_azure_version_v7")
    st.session_state.form_values['azure_openai_chat_deployment_name'] = st.text_input("Azure チャットデプロイメント名", value=values.get("azure_openai_chat_deployment_name", ""), key="setting_azure_chat_deploy_v7")
    st.session_state.form_values['azure_openai_embedding_deployment_name'] = st.text_input("Azure 埋め込みデプロイメント名", value=values.get("azure_openai_embedding_deployment_name", ""), key="setting_azure_embed_deploy_v7")

def _render_huggingface_settings(values, defaults):
    # Check session state for provider selection
    if st.session_state.get("selected_llm_provider", "azure") != "huggingface":
        return

    if 'form_values' not in st.session_state:
        st.session_state.form_values = {}

    st.session_state.form_values['llm_provider'] = st.session_state.selected_llm_provider

    st.markdown("#### 🤗 Hugging Face 設定")

    st.session_state.form_values['hf_model_id'] = st.text_input(
        "LLMモデルID",
        value=values.get("hf_model_id", defaults.hf_model_id if hasattr(defaults, "hf_model_id") else "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3"),
        key="setting_hf_model_id_v7",
        help="Hugging FaceのモデルID（例: tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3）"
    )

    st.session_state.form_values['hf_embedding_model_id'] = st.text_input(
        "埋め込みモデルID",
        value=values.get("hf_embedding_model_id", defaults.hf_embedding_model_id if hasattr(defaults, "hf_embedding_model_id") else "intfloat/multilingual-e5-large"),
        key="setting_hf_embedding_model_id_v7",
        help="埋め込み用モデルID（例: intfloat/multilingual-e5-large）"
    )

    device_options = ["cuda", "cpu", "mps"]
    device_labels = {"cuda": "CUDA (NVIDIA GPU)", "cpu": "CPU", "mps": "MPS (Apple Silicon)"}
    current_device = values.get("hf_device", defaults.hf_device if hasattr(defaults, "hf_device") else "cuda")
    device_idx = device_options.index(current_device) if current_device in device_options else 0

    st.session_state.form_values['hf_device'] = st.selectbox(
        "デバイス",
        device_options,
        index=device_idx,
        format_func=lambda x: device_labels[x],
        key="setting_hf_device_v7",
        help="推論に使用するデバイス"
    )

    col1, col2 = st.columns(2)
    with col1:
        st.session_state.form_values['hf_load_in_4bit'] = st.checkbox(
            "4-bit量子化",
            value=values.get("hf_load_in_4bit", defaults.hf_load_in_4bit if hasattr(defaults, "hf_load_in_4bit") else True),
            key="setting_hf_load_in_4bit_v7",
            help="メモリ使用量を削減（精度は若干低下）"
        )
    with col2:
        st.session_state.form_values['hf_load_in_8bit'] = st.checkbox(
            "8-bit量子化",
            value=values.get("hf_load_in_8bit", defaults.hf_load_in_8bit if hasattr(defaults, "hf_load_in_8bit") else False),
            key="setting_hf_load_in_8bit_v7",
            help="4-bitより精度高いが、メモリ使用量は多い"
        )

    st.session_state.form_values['hf_max_new_tokens'] = st.number_input(
        "最大トークン数",
        min_value=128,
        max_value=8192,
        value=int(values.get("hf_max_new_tokens", defaults.hf_max_new_tokens if hasattr(defaults, "hf_max_new_tokens") else 2048)),
        step=128,
        key="setting_hf_max_new_tokens_v7"
    )

    col1, col2 = st.columns(2)
    with col1:
        st.session_state.form_values['hf_temperature'] = st.number_input(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=float(values.get("hf_temperature", defaults.hf_temperature if hasattr(defaults, "hf_temperature") else 0.0)),
            step=0.1,
            key="setting_hf_temperature_v7"
        )
    with col2:
        st.session_state.form_values['hf_top_p'] = st.number_input(
            "Top P",
            min_value=0.0,
            max_value=1.0,
            value=float(values.get("hf_top_p", defaults.hf_top_p if hasattr(defaults, "hf_top_p") else 0.9)),
            step=0.05,
            key="setting_hf_top_p_v7"
        )

    st.session_state.form_values['hf_top_k'] = st.number_input(
        "Top K",
        min_value=1,
        max_value=200,
        value=int(values.get("hf_top_k", defaults.hf_top_k if hasattr(defaults, "hf_top_k") else 50)),
        step=10,
        key="setting_hf_top_k_v7"
    )

def _render_chunking_settings(values, defaults):
    st.markdown("#### 📄 チャンク設定")

    st.session_state.form_values['chunk_size'] = st.number_input("チャンクサイズ", 100, 5000, int(values.get("chunk_size", defaults.chunk_size)), 100, key="setting_chunk_size_v7")
    st.session_state.form_values['chunk_overlap'] = st.number_input("チャンクオーバーラップ", 0, 1000, int(values.get("chunk_overlap", defaults.chunk_overlap)), 50, key="setting_chunk_overlap_v7")

def _render_search_rag_settings(values, defaults):
    st.markdown("#### 🔍 検索・RAG設定")
    st.session_state.form_values['collection_name'] = st.text_input("コレクション名", values.get("collection_name", defaults.collection_name), key="setting_collection_name_v7")

    # デフォルト検索モード選択
    search_mode_options = ["hybrid", "vector", "keyword"]
    search_mode_labels = {"hybrid": "ハイブリッド検索", "vector": "ベクトル検索", "keyword": "キーワード検索"}
    current_mode = values.get("default_search_type", defaults.default_search_type)
    mode_idx = search_mode_options.index(current_mode) if current_mode in search_mode_options else 0
    selected_mode = st.selectbox(
        "デフォルト検索モード",
        search_mode_options,
        index=mode_idx,
        format_func=lambda x: search_mode_labels[x],
        key="setting_default_search_type_v7",
        help="チャットで使用するデフォルトの検索モードを選択します"
    )
    st.session_state.form_values['default_search_type'] = selected_mode

    st.session_state.form_values['final_k'] = st.slider("最終検索結果数 (Final K)", 1, 20, int(values.get("final_k", defaults.final_k)), key="setting_final_k_v7")
    st.session_state.form_values['vector_search_k'] = st.number_input("ベクトル検索数 (Vector K)", 1, 50, int(values.get("vector_search_k", defaults.vector_search_k)), key="setting_vector_k_v7")
    st.session_state.form_values['keyword_search_k'] = st.number_input("キーワード検索数 (Keyword K)", 1, 50, int(values.get("keyword_search_k", defaults.keyword_search_k)), key="setting_keyword_k_v7")
    st.session_state.form_values['rrf_k_for_fusion'] = st.number_input("RAG-Fusion用RRF係数 (k)", 1, 100, int(values.get("rrf_k_for_fusion", defaults.rrf_k_for_fusion)), key="setting_rrf_k_v7")

def _render_db_connection_settings(values, defaults):
    st.session_state.form_values['db_host'] = st.text_input("DBホスト", values.get("db_host", defaults.db_host), key="setting_db_host_v7")
    st.session_state.form_values['db_name'] = st.text_input("DB名", values.get("db_name", defaults.db_name), key="setting_db_name_v7")
    st.session_state.form_values['db_user'] = st.text_input("DBユーザー", values.get("db_user", defaults.db_user), key="setting_db_user_v7")

def _render_db_auth_settings(values, defaults):
    st.session_state.form_values['db_port'] = st.text_input("DBポート", str(values.get("db_port", defaults.db_port)), key="setting_db_port_v7")
    st.session_state.form_values['db_password'] = st.text_input("DBパスワード", values.get("db_password", defaults.db_password), type="password", key="setting_db_pass_v7")
    fts_opts = ["english", "japanese", "simple", "german", "french"]
    current_fts = values.get("fts_language", defaults.fts_language)
    fts_idx = fts_opts.index(current_fts) if current_fts in fts_opts else 0
    st.session_state.form_values['fts_language'] = st.selectbox("FTS言語", fts_opts, index=fts_idx, key="setting_fts_lang_v7")

def _render_pdf_processing_settings(values, defaults):
    st.markdown("#### 📑 PDF処理設定")
    
    # Azure Document Intelligence設定
    st.markdown("##### Azure Document Intelligence 設定")

    st.session_state.form_values['azure_di_endpoint'] = st.text_input(
        "Azure DI エンドポイント",
        value=values.get("azure_di_endpoint", ""),
        key="setting_azure_di_endpoint_v7",
        placeholder="https://your-resource.cognitiveservices.azure.com/"
    )
    st.session_state.form_values['azure_di_api_key'] = st.text_input(
        "Azure DI APIキー",
        value=values.get("azure_di_api_key", ""),
        type="password",
        key="setting_azure_di_key_v7"
    )

    model_options = ["prebuilt-layout", "prebuilt-document", "prebuilt-read"]
    current_model = values.get("azure_di_model", defaults.azure_di_model)
    if current_model not in model_options:
        current_model = "prebuilt-layout"

    st.session_state.form_values['azure_di_model'] = st.selectbox(
        "使用モデル",
        options=model_options,
        index=model_options.index(current_model),
        key="setting_azure_di_model_v7",
        help="prebuilt-layout: 高精度なレイアウト解析、prebuilt-document: 汎用文書処理、prebuilt-read: OCR特化"
    )

    st.session_state.form_values['save_markdown'] = st.checkbox(
        "Markdownファイルとして保存",
        value=values.get("save_markdown", defaults.save_markdown),
        key="setting_save_markdown_v7",
        help="処理結果をMarkdownファイルとして保存します"
    )

def _apply_settings(form_values):
    from src.ui.state import initialize_rag_system
    try:
        form_values["openai_api_key"] = None
        new_config = Config(**form_values)
        
        with st.spinner("設定を適用し、システムを初期化しています..."):
            if "rag_system" in st.session_state:
                del st.session_state["rag_system"]
            st.cache_resource.clear()
            collection_name = new_config.collection_name
            st.session_state.rag_system = initialize_rag_system(collection_name, new_config)
        st.success("✅ 設定が正常に適用され、システムが初期化されました！")
        time.sleep(1)
        st.rerun()
    except Exception as e:
        st.error(f"❌ 設定の適用中にエラーが発生しました: {type(e).__name__} - {e}")

def _reset_to_defaults(env_defaults):
    from src.ui.state import initialize_rag_system
    st.info("設定をデフォルト値にリセットし、システムを再初期化します...")
    
    default_config = Config()
    for key, value in env_defaults.items():
        if hasattr(default_config, key.lower()):
            setattr(default_config, key.lower(), value)
    default_config.openai_api_key = None

    with st.spinner("デフォルト設定でシステムを初期化しています..."):
        if "rag_system" in st.session_state:
            del st.session_state["rag_system"]
        st.cache_resource.clear()
        collection_name = default_config.collection_name
        st.session_state.rag_system = initialize_rag_system(collection_name, default_config)
    st.success("✅ 設定がデフォルトにリセットされ、システムが初期化されました！")
    time.sleep(1)
    st.rerun()

def _display_current_config(rag_system):
    if rag_system and hasattr(rag_system, 'config'):
        config_dict = rag_system.config.__dict__.copy()
        sensitive_keys = ["db_password", "openai_api_key", "azure_openai_api_key"]
        for key in sensitive_keys:
            if key in config_dict and config_dict[key]:
                value = str(config_dict[key])
                config_dict[key] = f"***{value[-4:]}" if len(value) > 7 else "********"
            elif key == "openai_api_key" and key in config_dict:
                config_dict[key] = "None (Fallback Disabled)"

        col1, col2 = st.columns(2)
        items = list(config_dict.items())
        midpoint = (len(items) + 1) // 2
        with col1:
            for k, v in items[:midpoint]:
                st.markdown(f"**{k.replace('_', ' ').capitalize()}:** `{str(v)}`")
        with col2:
            for k, v in items[midpoint:]:
                st.markdown(f"**{k.replace('_', ' ').capitalize()}:** `{str(v)}`")
    else:
        st.info("システムが初期化されていません。上記フォームから設定を適用してください。")
